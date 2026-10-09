#!/usr/bin/env python3
"""Materialise the final Pythia checkpoint of every size as a model directory
run_liars_bench_honesty.py can be pointed at with --model-id.

The size ablation needs one checkpoint per model, the last one, read the same
way at every size.  Three things make that less trivial than a loop over
AutoModelForCausalLM.from_pretrained, and this script handles each:

1. WHICH REVISION IS FINAL.  The Pythia repos keep every intermediate
   checkpoint on a `step<N>` branch, and `main` is the end of training.  We pin
   `step143000` rather than `main` so the revision is explicit in the
   provenance record, and --verify checks the two agree.  They do: on
   pythia-70m-deduped all 76 parameters are bit-identical across the two
   revisions, and `main` differs only by 18 non-persistent buffers
   (attention.bias, attention.masked_bias, rotary_emb.inv_freq) that the newer
   safetensors conversion drops.  The blob hash of `pytorch_model.bin` is
   identical on both revisions at every size, which --verify reports without
   downloading the weights.

2. WHICH WEIGHT FILE TO FETCH.  Resolved per (repo, revision), not per repo,
   because the formats are not uniform: 70m-1b and 1.4b ship a single
   `model.safetensors` at step143000, while 2.8b, 6.9b and 12b have only
   `pytorch_model*.bin` there.  pythia-1b's step143000 branch also carries an
   `optimizer.pt` several GB wide, which the allow-list keeps out.  The
   allow-list also leaves out generation_config.json, which only pythia-1b
   happens to publish: taking it would give that one size different decoding
   defaults from its siblings, which is the opposite of what a size ablation
   wants.  This
   follows ptm_ftm/eval_pythia_scaling.py:patterns_for, whose warning about
   stale `.bin` blobs on intermediate branches does not bite here because we
   only ever read the final step, where `.bin` and `main` coincide.

3. PYTHIA IS A BASE MODEL WITH NO CHAT TEMPLATE.  run_liars_bench_honesty.py
   renders every item through tok.apply_chat_template (via
   run_reversed_contrast.generate_batched), which raises on a tokenizer whose
   chat_template is None -- that is every Pythia tokenizer.  The prepared
   directory therefore carries a tokenizer_config.json with an explicit
   plain-text template baked in, so the runner needs no change and every size
   is prompted identically.  CHAT_TEMPLATE below is the one knob worth
   discussing before the ablation runs; see its docstring.

Two one-line changes in did_estimators.py are also required, and have been
made: decoder_layers() did not know GPTNeoX keeps its stack at
`gpt_neox.layers`, and target_layers()'s skip list named `lm_head`, which is
called `embed_out` here -- without that the contrast would have selected
weights inside the output embedding.  --check verifies both from this side.

  python prepare_pythia_checkpoints.py --list
  python prepare_pythia_checkpoints.py --verify
  python prepare_pythia_checkpoints.py --sizes 70m 160m 410m 1b 1.4b 2.8b
  python prepare_pythia_checkpoints.py --sizes 70m --check

Then, per size, exactly the main-table protocol:

  GPU=0 MID=$(python prepare_pythia_checkpoints.py --sizes 1.4b --print-path) \
    ARMS=unsteered,magmatch100,poolmix100,ours075,prompt2 \
    EXTRA="--polarity-arms" TAGX=pythia1p4b bash lb_local.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

FINAL_STEP = 143000
FINAL_REV = f"step{FINAL_STEP}"

# layers/hidden/heads read from each repo's config.json at step143000, not from
# memory.  `batch` and `score_batch` are starting points for an 80 GB card at
# fp16 with the mask build resident; the runner's own OOM behaviour is not
# adaptive, so halve them if a size dies.  `steer` is the mid-depth layer, a
# sensible centre for the activation-steering baseline's layer sweep -- it is
# not a substitute for sweeping (the main table chooses every method's setting
# on the validation half, which for steering includes the layer).
SIZES = {
    "70m":  dict(layers=6,  hidden=512,  heads=8,  batch=16, score_batch=8),
    "160m": dict(layers=12, hidden=768,  heads=12, batch=16, score_batch=8),
    "410m": dict(layers=24, hidden=1024, heads=16, batch=16, score_batch=8),
    "1b":   dict(layers=16, hidden=2048, heads=8,  batch=16, score_batch=4),
    "1.4b": dict(layers=24, hidden=2048, heads=16, batch=16, score_batch=4),
    "2.8b": dict(layers=32, hidden=2560, heads=32, batch=8,  score_batch=2),
    "6.9b": dict(layers=32, hidden=4096, heads=32, batch=8,  score_batch=2),
    "12b":  dict(layers=36, hidden=5120, heads=40, batch=4,  score_batch=2),
}
for _s, _c in SIZES.items():
    _c["steer"] = _c["layers"] // 2

DEFAULT_CACHE = os.environ.get("LIE_CACHE", "/scratch/aebrahim/hf_cache")
DEFAULT_ROOT = os.environ.get("PYTHIA_ROOT", "/scratch/aebrahim/pythia_final")

# A base model has no turn format of its own, so one has to be chosen, and the
# choice is part of the experiment: it decides how plainly the deception
# instruction reaches the model.  This is the conventional plain-text framing --
# the system prompt as a bare preamble, then labelled turns -- which keeps the
# LIARS' BENCH and AI-LIAR system prompts intact and leaves the model to
# continue after "Assistant:".  It also round-trips the pre-filled assistant
# turn that MASK's doubling_down items carry.  Change it here, not per size, or
# the sizes stop being comparable.
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}{{ m['content'] + '\n\n' }}"
    "{% elif m['role'] == 'user' %}{{ 'User: ' + m['content'] + '\n' }}"
    "{% elif m['role'] == 'assistant' %}{{ 'Assistant: ' + m['content'] + '\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}"
)

_PATTERNS: dict = {}


def repo_id(size: str, deduped: bool = True) -> str:
    return f"EleutherAI/pythia-{size}{'-deduped' if deduped else ''}"


def dest_dir(size: str, deduped: bool, root: str) -> Path:
    return Path(root) / f"pythia-{size}{'-deduped' if deduped else ''}-{FINAL_REV}"


def allow_patterns(repo: str, rev: str) -> list:
    """One weight format only, preferring safetensors, resolved per revision.

    Keyed on (repo, rev) because the format is not a property of the repo: see
    the module docstring.  Tokenizer files come from the same revision, which
    carries them at every size.
    """
    key = (repo, rev)
    if key not in _PATTERNS:
        from huggingface_hub import HfApi
        files = {f.rfilename for f in HfApi().repo_info(repo, revision=rev).siblings}
        weights = (["*.safetensors", "model.safetensors.index.json"]
                   if any(f.endswith(".safetensors") for f in files) else
                   ["pytorch_model*.bin", "pytorch_model.bin.index.json"])
        _PATTERNS[key] = ["config.json", "tokenizer.json",
                          "tokenizer_config.json", "special_tokens_map.json",
                          *weights]
    return _PATTERNS[key]


def verify(size: str, deduped: bool) -> dict:
    """Confirm step143000 is the end of training, without downloading weights.

    Compares the LFS digest of every weight file the two revisions share.  The
    `.bin` is present on both at every size, so this is always a real check;
    `model.safetensors` may legitimately differ in bytes while holding
    identical parameters, because `main`'s older conversion also stores
    non-persistent buffers, so a safetensors mismatch is reported rather than
    treated as a failure.
    """
    from huggingface_hub import HfApi
    api = HfApi()
    repo = repo_id(size, deduped)

    def w(rev):
        info = api.repo_info(repo, revision=rev, files_metadata=True)
        return {f.rfilename: (f.lfs.sha256 if getattr(f, "lfs", None) else None)
                for f in info.siblings
                if f.rfilename.endswith((".safetensors", ".bin"))}

    m, s = w("main"), w(FINAL_REV)
    shared = sorted(set(m) & set(s))
    bins = [f for f in shared if f.endswith(".bin")]
    out = {"repo": repo, "shared": {f: m[f] == s[f] for f in shared},
           "bin_identical": bool(bins) and all(m[f] == s[f] for f in bins),
           "only_main": sorted(set(m) - set(s)),
           "only_final": sorted(set(s) - set(m))}
    return out


def fetch(size: str, deduped: bool, cache_dir: str) -> str:
    from huggingface_hub import snapshot_download
    repo, rev = repo_id(size, deduped), FINAL_REV
    return snapshot_download(repo, revision=rev, cache_dir=cache_dir,
                             allow_patterns=allow_patterns(repo, rev),
                             max_workers=8)


def materialize(size: str, deduped: bool, cache_dir: str, root: str,
                force: bool = False) -> Path:
    """Build the directory the runner is pointed at.

    Weights are symlinked out of the HF cache rather than copied -- the full
    set is about 46 GB -- and only tokenizer_config.json is rewritten, to carry
    the chat template, the pad token and left padding.  PROVENANCE.json records
    the repo, the revision and its commit, so a results file can always be
    traced back to a checkpoint.
    """
    from huggingface_hub import HfApi
    snap = Path(fetch(size, deduped, cache_dir))
    dest = dest_dir(size, deduped, root)
    if dest.exists() and force:
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    for src in sorted(snap.iterdir()):
        if src.name == "tokenizer_config.json" or src.name.startswith("."):
            continue
        link = dest / src.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve())

    tc = json.loads((snap / "tokenizer_config.json").read_text())
    tc.update(chat_template=CHAT_TEMPLATE, padding_side="left",
              model_max_length=2048,
              pad_token="<|padding|>", bos_token="<|endoftext|>",
              eos_token="<|endoftext|>", unk_token="<|endoftext|>")
    (dest / "tokenizer_config.json").write_text(json.dumps(tc, indent=1))

    sha = HfApi().repo_info(repo_id(size, deduped), revision=FINAL_REV).sha
    (dest / "PROVENANCE.json").write_text(json.dumps({
        "repo": repo_id(size, deduped), "revision": FINAL_REV,
        "commit": sha, "snapshot": str(snap), "config": SIZES[size],
        "chat_template": CHAT_TEMPLATE,
        "prepared_by": "prepare_pythia_checkpoints.py"}, indent=1))
    return dest


def check(size: str, deduped: bool, root: str) -> None:
    """Load the prepared directory and assert the pipeline can use it.

    Covers the three things that silently break on GPTNeoX: the chat template,
    the decoder-stack path decoder_layers() walks, and whether target_layers()
    excludes the output embedding.  CPU only, so it is cheap at 70m and
    pointless above about 1b.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from did_estimators import decoder_layers, target_layers

    d = dest_dir(size, deduped, root)
    tok = AutoTokenizer.from_pretrained(d)
    txt = tok.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."},
         {"role": "user", "content": "What is the capital of France?"}],
        tokenize=False, add_generation_prompt=True)
    print(f"  chat template ok, renders:\n    {txt!r}")
    assert tok.padding_side == "left" and tok.pad_token_id is not None

    model = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.float32)
    stack = decoder_layers(model)
    assert len(stack) == SIZES[size]["layers"], (len(stack), SIZES[size])
    print(f"  decoder_layers -> {len(stack)} blocks")
    names = target_layers(model)
    bad = [n for n in names if n in ("embed_out", "gpt_neox.embed_in")]
    assert not bad, f"output/input embedding reached target_layers: {bad}"
    print(f"  target_layers -> {len(names)} Linear modules, "
          f"e.g. {names[:2]}, embeddings excluded")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", nargs="+", default=list(SIZES),
                    choices=list(SIZES), metavar="SIZE")
    ap.add_argument("--no-deduped", action="store_true",
                    help="use the standard rather than the deduped series")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--out-root", default=DEFAULT_ROOT)
    ap.add_argument("--force", action="store_true", help="rebuild the directory")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--verify", action="store_true",
                    help="check step143000 against main, no weights downloaded")
    ap.add_argument("--check", action="store_true",
                    help="load each prepared directory and test the pipeline hooks")
    ap.add_argument("--print-path", action="store_true",
                    help="print only the prepared path, for use in a shell variable")
    ap.add_argument("--emit-env", action="store_true",
                    help="print shell assignments for one size, for eval in a driver")
    args = ap.parse_args()
    ded = not args.no_deduped

    if args.list:
        print(f"{'size':6} {'repo':34} {'layers':>6} {'hidden':>6} "
              f"{'steer':>5} {'batch':>5} {'sb':>3}")
        for s in args.sizes:
            c = SIZES[s]
            print(f"{s:6} {repo_id(s, ded):34} {c['layers']:6} {c['hidden']:6} "
                  f"{c['steer']:5} {c['batch']:5} {c['score_batch']:3}")
        return

    if args.verify:
        for s in args.sizes:
            v = verify(s, ded)
            flag = "OK " if v["bin_identical"] else "?? "
            print(f"{flag}{v['repo']:34} {FINAL_REV} vs main: " +
                  ", ".join(f"{f}={'same' if ok else 'DIFFERS'}"
                            for f, ok in v["shared"].items()))
            for k in ("only_main", "only_final"):
                if v[k]:
                    print(f"     {k}: {v[k]}")
        return

    os.environ.pop("HF_HUB_OFFLINE", None)      # the revisions must be resolved
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    for s in args.sizes:
        d = materialize(s, ded, args.cache_dir, args.out_root, args.force)
        if args.emit_env:
            c = SIZES[s]
            print(f"MID={d}\nBS={c['batch']}\nSB={c['score_batch']}\n"
                  f"STEER={c['steer']}\nNLAYER={c['layers']}\nSIZE={s}")
        elif args.print_path:
            print(d)
        else:
            n = len([x for x in d.iterdir() if x.name.endswith((".safetensors", ".bin"))])
            print(f"{s:6} -> {d}  ({n} weight file(s))")
        if args.check:
            check(s, ded, args.out_root)


if __name__ == "__main__":
    main()
