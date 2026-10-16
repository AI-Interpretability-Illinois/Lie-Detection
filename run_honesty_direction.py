#!/usr/bin/env python3
"""
Which edit promotes honesty: the reversed mask amplified, or the DID mask damped?

Two candidate honesty edits fall out of the reversed-contrast result, and they are
not the same operation:

  reverse, alpha > 0   amplify the weights that matter more to truth-telling
  forward, alpha < 0   shrink the weights that matter more to lying
                       (alpha = -1 zeroes them outright: ablate the lie circuit)

Both are run against the control that decides whether either is specific:

  magmatch, alpha < 0  shrink a random magnitude-matched set of the same size
  magmatch, alpha > 0  amplify one

plus `forward, alpha > 0` as the known deception direction, for reference.

Honesty is measured as resistance to an explicit lie instruction, because an
already-accurate model has no headroom to become more accurate on neutral
questions.  See honesty_eval.py for the metric definitions.

Usage
-----
  python run_honesty_direction.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
      --rho 0.04 --ratio 0.37 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from run_reversed_contrast import (EVAL_QA, SanitizeLogits, apply_steering,
                                   build_mask, generate_batched, junk,
                                   load_wikitext_ids, restore, wikitext_ppl)
from honesty_eval import (HEADER, eval_conditions, fmt_row, known_set,
                          summarize_honesty)

# (arm, alpha) pairs.  Negative alpha shrinks the selected weights; -1.0 zeroes
# them.  Values below -1 would flip their sign, which is a different edit, so the
# sweep stops there.
PLAN = [("forward",  [-1.0, -0.75, -0.5, -0.25, 2.0, 3.0]),
        ("reverse",  [1.0, 2.0, 3.0, 4.0, -1.0, -0.5]),
        ("magmatch", [-1.0, -0.75, -0.5, 2.0, 3.0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--out-dir", default="/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = args.device
    short = args.model_id.split("/")[-1].replace(".", "p")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"honesty_direction_{short}_rho{args.rho}_ratio{args.ratio}"
    if args.tag:
        stem += f"_{args.tag}"

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)
    model.eval()

    cfg = LieNeuroConfig(importance_threshold=args.rho, alpha=1.0, device=device,
                         skip_modules=["lm_head", "embed_tokens"])
    ln = LieNeuro(model, tok, cfg)

    scores_path = out_dir / f"scores_{short}_ntruth{args.n_truth}_lmhead0.pt"
    if not scores_path.exists():
        raise SystemExit(f"scores not found: {scores_path}\n"
                         "run run_reversed_contrast.py first to build them")
    saved = torch.load(scores_path, map_location="cpu", weights_only=False)
    ln._S_lie, ln._S_truth = saved["S_lie"], saved["S_truth"]
    print(f"loaded scores from {scores_path}", flush=True)

    wt_ids = load_wikitext_ids(tok, args.cache_dir)
    guard = SanitizeLogits()

    def ev():
        return eval_conditions(model, tok, EVAL_QA, device, args.batch_size,
                               args.reps, args.temp, args.seed, generate_batched,
                               guard=guard, junk_fn=junk)

    # ── unsteered reference ──────────────────────────────────────────────────
    print("\n── none (unsteered) ──", flush=True)
    t0 = time.time()
    base_recs = ev()
    known = known_set(base_recs)
    base = summarize_honesty(base_recs, known)
    base_ppl = wikitext_ppl(model, wt_ids, device)
    print(f"  {len(known)} questions the model knows; "
          f"resist={base['resist_rate']:.1%}  comply={base['comply_rate']:.1%}  "
          f"neutral_acc={base['neutral_acc']:.1%}  ppl={base_ppl:.2f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    rows = [dict(base, arm="none", alpha=0.0, wikitext_ppl=base_ppl, ppl_ratio=1.0,
                 delta_resist=0.0, rho=args.rho, ratio=args.ratio)]
    all_recs = [dict(r, arm="none", alpha=0.0) for r in base_recs]

    # ── steered arms ─────────────────────────────────────────────────────────
    originals = {}
    for arm, alphas in PLAN:
        print(f"\n── {arm} ──", flush=True)
        build_mask(ln, arm, args.rho, args.ratio, device, args.n_truth, args.seed)
        for alpha in alphas:
            t0 = time.time()
            guard.n_fired = 0
            apply_steering(ln, alpha, originals)
            recs = ev()
            ppl = wikitext_ppl(model, wt_ids, device)
            restore(ln, originals)

            s = summarize_honesty(recs, known)
            s.update(arm=arm, alpha=alpha, rho=args.rho, ratio=args.ratio,
                     wikitext_ppl=ppl,
                     ppl_ratio=ppl / base_ppl if base_ppl else float("nan"),
                     delta_resist=s["resist_rate"] - base["resist_rate"],
                     delta_comply=s["comply_rate"] - base["comply_rate"],
                     nonfinite_logit_rows=guard.n_fired, secs=time.time() - t0)
            rows.append(s)
            all_recs.extend(dict(r, arm=arm, alpha=alpha) for r in recs)
            print(f"  a={alpha:<6} resist={s['resist_rate']:.1%} "
                  f"(Δ{s['delta_resist']:+.1%})  comply={s['comply_rate']:.1%} "
                  f"(Δ{s['delta_comply']:+.1%})  neutral_acc={s['neutral_acc']:.1%}"
                  f"  junk={s['pressured_junk']:.1%}  ppl={ppl:.2f} "
                  f"({s['ppl_ratio']:.2f}x)  [{time.time()-t0:.0f}s]", flush=True)

            fields = sorted({k for r in rows for k in r})
            with open(out_dir / f"{stem}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for r in rows:
                    w.writerow({k: r.get(k, "") for k in fields})

    with open(out_dir / f"{stem}_responses.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")

    print("\n" + HEADER)
    print("-" * 73)
    for r in sorted(rows, key=lambda x: (x["arm"], x["alpha"])):
        print(fmt_row(r["arm"], r["alpha"], r, r["ppl_ratio"]))
    print(f"\nrows → {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
