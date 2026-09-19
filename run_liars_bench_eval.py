#!/usr/bin/env python3
"""
On-policy LIARS' BENCH generation for baseline vs LieNeuro-DID-steered models.

For every benchmark prompt context the final assistant turn is regenerated with
the local model, so the resulting transcripts are on-policy (the released
`deceptive` labels belong to the 24B-72B models that produced them).

Two passes are run:

  1. belief  — a neutral, pressure-free probe of each underlying question, run on
               the *unsteered* weights.  This establishes what the model actually
               believes, which is what a lie has to contradict.
  2. main    — the benchmark context itself, run with steering applied (or not).

Validated operating points (from run_lieneuro_did_sweep.py):
    Qwen  : rho=0.06  alpha=4.0  lie-truth-ratio=0.35
    Llama : rho=0.04  alpha=4.0  lie-truth-ratio=0.41

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python run_liars_bench_eval.py --model llama --steered  --device cuda:0
  CUDA_VISIBLE_DEVICES=1 python run_liars_bench_eval.py --model llama --baseline --device cuda:0
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

CACHE_DIR = "/scratch/aebrahim/hf_cache"
MODE      = "score_diff"
N_TRUTH   = 25          # matches the cached score files and the validated sweeps

MODEL_CONFIGS = {
    "llama": {
        "model_id":    "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "model_short": "llama_8b",
        "rho":   0.04,
        "alpha": 4.0,
        "ratio": 0.37,
    },
    "qwen": {
        "model_id":    "Qwen/Qwen2.5-7B-Instruct",
        "model_short": "qwen2p5_7b",
        "rho":   0.06,
        "alpha": 4.0,
        "ratio": 0.35,
    },
}

# max_new_tokens / temperature per dataset, matching the paper's generation setup
# (HP-C, HP-KR and ID were sampled at 0.7; the convincing game at 1.0).
GEN_SETTINGS = {
    "harm-pressure-choice":           {"max_new_tokens":   8, "temperature": 0.7},
    "harm-pressure-knowledge-report": {"max_new_tokens":  48, "temperature": 0.7},
    "instructed-deception":           {"max_new_tokens": 128, "temperature": 0.7},
    "convincing-game":                {"max_new_tokens": 400, "temperature": 1.0},
    "alpaca":                         {"max_new_tokens": 400, "temperature": 0.7},
}

BELIEF_SETTINGS = {
    "mc_letter":  {"max_new_tokens":   8, "temperature": 0.0},
    "true_false": {"max_new_tokens": 128, "temperature": 0.0},
}


def load_items(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


@torch.no_grad()
def generate_batched(model, tok, chats, device, max_new_tokens, temperature,
                     batch_size, desc=""):
    """Chat-template + left-padded batched generation, sorted by length.

    Sorting by prompt length keeps padding waste low; the original order is
    restored before returning.
    """
    texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
             for c in chats]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = [None] * len(texts)

    t0 = time.time()
    for start in range(0, len(order), batch_size):
        idx   = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        enc   = tok(batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=3072,
                    add_special_tokens=False).to(device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0.0),
            temperature=temperature if temperature > 0.0 else 1.0,
            top_p=1.0,
            top_k=0,
            pad_token_id=tok.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1]:]
        for i, row in zip(idx, new):
            out[i] = tok.decode(row, skip_special_tokens=True).strip()

        done = min(start + batch_size, len(order))
        el   = time.time() - t0
        print(f"    [{desc}] {done}/{len(order)}  "
              f"{el:.0f}s elapsed, {el / done * (len(order) - done):.0f}s left",
              flush=True)
    return out


def build_steered_model(model, tok, cfg_name, device):
    """Apply the validated LieNeuro-DID intervention in place."""
    c = MODEL_CONFIGS[cfg_name]
    # skip_modules=[] reproduces the validated runs, which included lm_head.
    # This script steers once and never restores (the belief pass runs before
    # steering, on the unsteered weights), so keeping a GPU-side copy of every
    # original weight would waste a second full model — ~16 GB for the 8B —
    # and pushes apply_steering() into OOM on a shared GPU.
    ln_cfg = LieNeuroConfig(importance_threshold=c["rho"], alpha=c["alpha"],
                            device=device, skip_modules=[],
                            save_original_weights=False)
    ln = LieNeuro(model, tok, ln_cfg)

    scores_path = Path(f"outputs/scores_{c['model_short']}_ntruth{N_TRUTH}.pt")
    if scores_path.exists():
        print(f"Loading cached importance scores from {scores_path} …", flush=True)
        saved = torch.load(scores_path, map_location="cpu", weights_only=False)
        ln._S_lie, ln._S_truth = saved["S_lie"], saved["S_truth"]
    else:
        print("Computing importance scores (no cache found) …", flush=True)
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT[:N_TRUTH], desc="S_truth")
        ln._S_lie = ln._compute_scores_for_prompts(
            DIRECT_LIE[:N_TRUTH], desc="S_lie")

    ln.build_mask(mode=MODE, n_prompts=N_TRUTH, lie_truth_ratio=c["ratio"])
    ln.apply_steering()
    print(f"Steering applied: rho={c['rho']} alpha={c['alpha']} "
          f"ratio={c['ratio']} (lm_head included)", flush=True)
    return ln


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--steered",  action="store_true")
    grp.add_argument("--baseline", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--eval-set", default="outputs/liars_bench/eval_set.jsonl")
    ap.add_argument("--out-dir",  default="outputs/liars_bench")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--datasets", default=None,
                    help="Comma-separated subset of datasets to run.")
    # Operating-point overrides, for trying a config without editing MODEL_CONFIGS.
    # --arm-tag renames the output arm so a trial run sits alongside the
    # validated one instead of overwriting it; it must contain no underscore,
    # since the scorer splits "gen_<model_short>_<arm>" on the last one.
    ap.add_argument("--rho",   type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--arm-tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = MODEL_CONFIGS[args.model]
    for k in ("rho", "alpha", "ratio"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)      # build_steered_model reads the same dict
    arm = args.arm_tag or ("steered" if args.steered else "baseline")
    assert "_" not in arm, "arm tag must not contain '_' (scorer splits on it)"
    tag     = f"{cfg['model_short']}_{arm}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(args.eval_set)
    if args.datasets:
        keep  = set(args.datasets.split(","))
        items = [it for it in items if it["dataset"] in keep]
    print(f"[{tag}] {len(items)} items", flush=True)

    print(f"[{tag}] loading {cfg['model_id']} …", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float16,
        device_map=args.device, cache_dir=CACHE_DIR)
    model.eval()

    # ── Pass 1: belief probes on the UNSTEERED model ─────────────────────────
    # A lie is a claim that contradicts what the model actually believes, so the
    # belief has to be read off the model before the intervention touches it.
    # Per-arm filename so the two arms of a model can run concurrently without
    # racing on the same file; both probe unsteered weights greedily, so the
    # contents are identical.  The scorer reads the baseline arm's copy.
    belief_path = out_dir / f"beliefs_{tag}.jsonl"
    if belief_path.exists():
        print(f"[{tag}] reusing beliefs from {belief_path}", flush=True)
    else:
        probes = [it for it in items if it.get("belief_ctx")]
        print(f"[{tag}] belief pass: {len(probes)} probes (unsteered)", flush=True)
        records = []
        for btype, settings in BELIEF_SETTINGS.items():
            sub = [it for it in probes if it["belief_type"] == btype]
            if not sub:
                continue
            resp = generate_batched(
                model, tok, [it["belief_ctx"] for it in sub], args.device,
                settings["max_new_tokens"], settings["temperature"],
                args.batch_size, desc=f"belief/{btype}")
            records += [{"uid": it["uid"], "belief_type": btype, "response": r}
                        for it, r in zip(sub, resp)]
        with open(belief_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"[{tag}] wrote {len(records)} beliefs to {belief_path}", flush=True)

    # ── Pass 2: main benchmark generation ────────────────────────────────────
    if args.steered:
        build_steered_model(model, tok, args.model, args.device)

    main_path = out_dir / f"gen_{tag}.jsonl"
    with open(main_path, "w") as fout:
        for ds, settings in GEN_SETTINGS.items():
            sub = [it for it in items if it["dataset"] == ds]
            if not sub:
                continue
            print(f"[{tag}] {ds}: {len(sub)} items", flush=True)
            resp = generate_batched(
                model, tok, [it["ctx"] for it in sub], args.device,
                settings["max_new_tokens"], settings["temperature"],
                args.batch_size, desc=f"{tag}/{ds}")
            for it, r in zip(sub, resp):
                fout.write(json.dumps({
                    "uid":      it["uid"],
                    "dataset":  ds,
                    "arm":      arm,
                    "model":    cfg["model_short"],
                    "response": r,
                }) + "\n")
            fout.flush()
    print(f"[{tag}] done → {main_path}", flush=True)


if __name__ == "__main__":
    main()
