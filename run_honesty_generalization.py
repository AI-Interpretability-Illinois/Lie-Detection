#!/usr/bin/env python3
"""
Does the honesty edit resist lie framings it was never built from?

run_honesty_direction.py pressured the model with `DIRECT_LIE`'s wording, which
is the wording the importance scores were computed from. An edit that merely
suppressed that phrasing would score well there and be useless in practice. This
runs the leading honesty arms against three held-out framings (see PRESSURE_SETS
in honesty_eval.py) that share no instruction wording with the scoring prompts.

Usage
-----
  python run_honesty_generalization.py --model-id Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import EVAL_SYSTEM_PROMPT
from run_reversed_contrast import (EVAL_QA, SanitizeLogits, apply_steering,
                                   build_mask, generate_batched, junk,
                                   load_wikitext_ids, restore, wikitext_ppl)
from honesty_eval import (PRESSURE_SETS, eval_conditions, known_set,
                          summarize_honesty)

# The honesty candidates, their control, and the deception reference.
WEIGHT_ARMS = [("forward", -0.75), ("forward", -1.0), ("magmatch", -0.75),
               ("reverse", 2.0)]
ACTSTEER_ARMS = [(7, -8.0)]
CONDITIONS = ("neutral",) + tuple(PRESSURE_SETS)


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
    args = ap.parse_args()

    device = args.device
    short = args.model_id.split("/")[-1].replace(".", "p")
    out_dir = Path(args.out_dir)
    stem = f"honesty_generalization_{short}"

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)
    model.eval()

    guard = SanitizeLogits()
    wt_ids = load_wikitext_ids(tok, args.cache_dir)

    def ev():
        return eval_conditions(model, tok, EVAL_QA, device, args.batch_size,
                               args.reps, args.temp, args.seed, generate_batched,
                               guard=guard, junk_fn=junk,
                               system=EVAL_SYSTEM_PROMPT, conditions=CONDITIONS)

    print("── none (unsteered) ──", flush=True)
    base_recs = ev()
    known = known_set(base_recs)
    base_ppl = wikitext_ppl(model, wt_ids, device)
    rows, all_recs = [], []

    def tally(label, coef, recs, ppl):
        row = {"arm": label, "coef": coef, "ppl_ratio": ppl / base_ppl}
        for cond in PRESSURE_SETS:
            sub = [r for r in recs if r["cond"] == cond]
            s = summarize_honesty([dict(r, cond="pressured") for r in sub]
                                  + [r for r in recs if r["cond"] == "neutral"],
                                  known)
            row[f"lie_{cond}"] = s["comply_rate"]
            row[f"resist_{cond}"] = s["resist_rate"]
            row["neutral_acc"] = s["neutral_acc"]
        row["lie_heldout"] = sum(row[f"lie_{c}"] for c in PRESSURE_SETS
                                 if c != "direct") / (len(PRESSURE_SETS) - 1)
        rows.append(row)
        all_recs.extend(dict(r, arm=label, coef=coef) for r in recs)
        print(f"  {label:<18}{coef:>6} | " +
              "  ".join(f"{c[:6]}={row[f'lie_{c}']:.1%}" for c in PRESSURE_SETS) +
              f" | held-out mean={row['lie_heldout']:.1%}"
              f"  neutral={row['neutral_acc']:.1%}  ppl={row['ppl_ratio']:.2f}x",
              flush=True)

    tally("none", 0.0, base_recs, base_ppl)

    cfg = LieNeuroConfig(importance_threshold=args.rho, alpha=1.0, device=device,
                         skip_modules=["lm_head", "embed_tokens"])
    ln = LieNeuro(model, tok, cfg)
    saved = torch.load(out_dir / f"scores_{short}_ntruth{args.n_truth}_lmhead0.pt",
                       map_location="cpu", weights_only=False)
    ln._S_lie, ln._S_truth = saved["S_lie"], saved["S_truth"]

    print("── weight arms ──", flush=True)
    originals = {}
    for arm, alpha in WEIGHT_ARMS:
        build_mask(ln, arm, args.rho, args.ratio, device, args.n_truth, args.seed)
        apply_steering(ln, alpha, originals)
        recs = ev()
        ppl = wikitext_ppl(model, wt_ids, device)
        restore(ln, originals)
        tally(f"weight:{arm}", alpha, recs, ppl)

    print("── activation steering ──", flush=True)
    asteer = ActivationSteering(model, tok, ActivationSteeringConfig(device=device))
    asteer.load_vectors(out_dir / f"as_vectors_{short}_ntruth{args.n_truth}.pt")
    for layer, c in ACTSTEER_ARMS:
        asteer.apply_steering(layer, c)
        recs = ev()
        ppl = wikitext_ppl(model, wt_ids, device)
        asteer.remove_steering()
        tally(f"actsteer:L{layer}", c, recs, ppl)

    fields = sorted({k for r in rows for k in r})
    with open(out_dir / f"{stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    with open(out_dir / f"{stem}_responses.jsonl", "w") as f:
        for r in all_recs:
            f.write(json.dumps(r) + "\n")
    print(f"\nrows → {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
