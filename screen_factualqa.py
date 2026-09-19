#!/usr/bin/env python3
"""
Generate and CACHE the FACTUAL_QA OOD responses for a LieNeuro-DID grid.

This is stage 1 of 2.  It writes every generation to JSONL and judges nothing;
judge_cached.py then scores that cache.  Splitting them matters because judging
is the part whose method is still in question -- caching the ~38k generations
once (about 10 MB) turns "which judge?" from an hour-long re-run into a pass over
text that costs minutes and can be repeated for any judge.

The objective being measured is the paper's own validation set: OOD_QA, the
FACTUAL_QA questions NOT used to build the DID scores.  What matters there is
unprompted lying -- a confident, fluent, false answer to a bare question with no
scenario inviting deception.  That is a different quantity from what
sweep_liars_local.py measures: LIARS' BENCH always supplies a scenario, so it
rewards far lower alpha than bare questions do.  Both matter, so the two CSVs
are joined on (rho, alpha, ratio) afterwards.

Two passes per grid point, both cached and tagged:
  greedy   temperature 0.0 over the 25 OOD questions -- paper-faithful, matching
           run_lieneuro_did_sweep.py, but only 25 items wide (4% granularity)
  sampled  3 reps at temperature 1.0, matching run_lieneuro_did_samples.py, for
           ranking resolution
"""


import argparse, contextlib, io, json, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, EVAL_SYSTEM_PROMPT, _is_correct
from src.prompts import DIRECT_TRUTH_INSTRUCT
from src.factual_qa_ext import FACTUAL_QA_EXT
from run_liars_bench_eval import MODEL_CONFIGS, CACHE_DIR, N_TRUTH, generate_batched
from sweep_liars_local import build_mask_gpu

_INSTRUCT = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
             for p in DIRECT_TRUTH_INSTRUCT}
OOD_QA = [qa for qa in FACTUAL_QA
          if qa["question"].strip().lower() not in _INSTRUCT]
# the paper-faithful 25 plus 75 new held-out questions; records carry a "set" tag
# so the original number stays reportable next to the higher-resolution one
EVAL_QA = ([dict(qa, set="ood25") for qa in OOD_QA] +
           [dict(qa, set="ext75") for qa in FACTUAL_QA_EXT])

N_REPS, SAMPLE_TEMP, MAX_NEW = 3, 1.0, 50


def generate_pass(model, tok, device, batch_size, temp, reps, seed):
    """Cache-ready records for one pass over OOD_QA."""
    qa = EVAL_QA * reps
    chats = [[{"role": "system", "content": EVAL_SYSTEM_PROMPT},
              {"role": "user", "content": it["question"]}] for it in qa]
    if temp > 0:
        torch.manual_seed(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        resp = generate_batched(model, tok, chats, device, MAX_NEW, temp, batch_size)
    return [{"set": it["set"], "q": it["question"], "gold": it["answer"], "resp": r,
             "correct": _is_correct(r, it["answer"])}
            for it, r in zip(qa, resp)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg, short = MODEL_CONFIGS[args.model], MODEL_CONFIGS[args.model]["model_short"]
    grid = json.loads(Path(args.grid).read_text())
    blocks = {}
    for rho, alpha, ratio in grid:
        blocks.setdefault((rho, ratio), []).append(alpha)
    blocks = sorted(blocks.items())
    i, n = (int(x) for x in args.shard.split("/"))
    blocks = blocks[i::n]
    print(f"[{short}] shard {args.shard}: {len(blocks)} blocks, "
          f"{sum(len(a) for _, a in blocks)} points, "
          f"eval={len(EVAL_QA)} (ood25={len(OOD_QA)}, ext75={len(FACTUAL_QA_EXT)})", flush=True)

    def load(mid, dev):
        t = AutoTokenizer.from_pretrained(mid, cache_dir=CACHE_DIR)
        if t.pad_token is None:
            t.pad_token = t.eos_token
        t.padding_side = "left"
        m = AutoModelForCausalLM.from_pretrained(
            mid, torch_dtype=torch.float16, device_map=dev, cache_dir=CACHE_DIR)
        m.eval()
        return m, t

    model, tok = load(cfg["model_id"], args.device)

    ln = LieNeuro(model, tok, LieNeuroConfig(importance_threshold=blocks[0][0][0],
                                             alpha=1.0, device=args.device, skip_modules=[]))
    saved = torch.load(f"outputs/scores_{short}_ntruth{N_TRUTH}.pt",
                       map_location="cpu", weights_only=False)
    ln._S_lie, ln._S_truth = saved["S_lie"], saved["S_truth"]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    def dump(f, rho, alpha, ratio, recs, tag):
        for r in recs:
            f.write(json.dumps({"rho": rho, "alpha": alpha, "ratio": ratio,
                                "pass": tag, **r}) + "\n")

    with open(out, "w") as f:
        # rho=-1 marks the unsteered baseline
        dump(f, -1, -1, -1, generate_pass(model, tok, args.device, args.batch_size,
                                          0.0, 1, args.seed), "greedy")
        dump(f, -1, -1, -1, generate_pass(model, tok, args.device, args.batch_size,
                                          SAMPLE_TEMP, N_REPS, args.seed), "sampled")
        f.flush()
        done = 0
        for (rho, ratio), alphas in blocks:
            ln.config.importance_threshold = rho
            build_mask_gpu(ln, rho, ratio, args.device, N_TRUTH)
            for alpha in sorted(alphas):
                t0 = time.time()
                ln.config.alpha = alpha
                ln.apply_steering()
                g = generate_pass(model, tok, args.device, args.batch_size,
                                  0.0, 1, args.seed)
                sm = generate_pass(model, tok, args.device, args.batch_size,
                                   SAMPLE_TEMP, N_REPS, args.seed)
                ln.restore_original()
                dump(f, rho, alpha, ratio, g, "greedy")
                dump(f, rho, alpha, ratio, sm, "sampled")
                f.flush()
                done += 1
                print(f"[{short}] {done} rho={rho:.2f} a={alpha:.1f} r={ratio:.2f}  "
                      f"wrong g={100*sum(not x['correct'] for x in g)/len(g):5.1f}% "
                      f"s={100*sum(not x['correct'] for x in sm)/len(sm):5.1f}%  "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"[{short}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
