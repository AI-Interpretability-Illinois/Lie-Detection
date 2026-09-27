#!/usr/bin/env python3
"""
Placebo-treatment test: does a contrast with no deception signal produce the same
mask behaviour as the real one?

This is the falsification test that has to pass before any refinement of the
contrast is worth building. The method selects weights by

    colstat_j = v * n_lie_j - (1 - v) * n_truth_j,    v = ratio / (1 + ratio)

and the claim is that this isolates deception. If the identical pipeline run on a
contrast that carries no deception at all produces a mask that steers behaviour
just as well, the claim is false and no second difference will rescue it.

Three placebo treatments, each contrasted against the same uniform
DIRECT_TRUTH_INSTRUCT control the real method uses, so only the treatment arm
changes:

  real       DIRECT_LIE            six rotating lie instructions
  para       DIRECT_TRUTH_PARA     six rotating TRUTHFUL instructions. Pure null:
                                   no deception, only instruction-phrasing
                                   variance. This one also controls a confound the
                                   real contrast has and nobody has named: the lie
                                   arm rotates six phrasings while the truth arm
                                   uses one, so phrasing diversity is already
                                   inside the real contrast.
  fiction    DIRECT_FICTION_FALSE  assert a falsehood inside a fiction frame:
                                   knowledge-contradiction on instruction, no
                                   intent to deceive
  error      DIRECT_HONEST_ERROR   produce a wrong answer framed as an honest
                                   slip: knowledge-contradiction, no fiction frame

Each mask is evaluated in both directions, damped for honesty and amplified for
deception, against the unsteered model and the magnitude-matched control of the
real mask. Mask diagnostics (overlap with the real mask, median |W|) are printed
too, because a placebo that selects the same weights is a different finding from
one that selects different weights and behaves the same.

Interpretation:
  para behaves like real    -> the estimator reads phrasing variance, not deception
  fiction/error like real   -> it reads instructed knowledge-contradiction
  none behave like real     -> the contrast is doing something specific; refining
                               it (a second difference) becomes worth building

Usage
-----
  python run_placebo_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.prompts import (DIRECT_LIE, DIRECT_TRUTH_INSTRUCT, DIRECT_TRUTH_PARA,
                         DIRECT_FICTION_FALSE, DIRECT_HONEST_ERROR)
from did_estimators import (cell_mean_sd, collect_column_norms, get_module,
                           load_lm, target_layers)
from run_reversed_contrast import (EVAL_QA, SanitizeLogits, generate_batched,
                                   junk, load_wikitext_ids, wikitext_ppl)
from run_real_did import apply_mask, magnitude_matched, restore
from honesty_eval import PRESSURE_SETS, eval_conditions, known_set, summarize_honesty

CONDITIONS = ("neutral",) + tuple(PRESSURE_SETS)

TREATMENTS = {
    "real":    DIRECT_LIE,
    "para":    DIRECT_TRUTH_PARA,
    "fiction": DIRECT_FICTION_FALSE,
    "error":   DIRECT_HONEST_ERROR,
}


@torch.no_grad()
def mask_from_contrast(model, layers, n_treat, n_ctrl, rho, ratio, device):
    """top-rho per layer of |W_ij| * (v*n_treat_j - (1-v)*n_ctrl_j)."""
    v = ratio / (1.0 + ratio) if ratio > 0 else 0.0
    masks = {}
    for name in layers:
        W = get_module(model, name).weight.detach().float().abs().to(device)
        col = (v * n_treat[name] - (1.0 - v) * n_ctrl[name]).to(device)
        s = (W * col.unsqueeze(0)).view(-1)
        k = max(1, int(rho * s.numel()))
        m = torch.zeros(s.numel(), dtype=torch.bool, device=device)
        m[s.topk(k).indices] = True
        masks[name] = m.view(W.shape).cpu()
        del W, col, s, m
    return masks


@torch.no_grad()
def mask_stats(model, layers, masks, ref=None):
    """Median |W| of the selected weights, and Jaccard against a reference mask."""
    sel, all_w = [], []
    inter = union = 0
    for n in layers:
        W = get_module(model, n).weight.detach().float().abs()
        m = masks[n].to(W.device)
        s = W[m]
        sel.append(s[:: max(1, s.numel() // 20000)].cpu())
        all_w.append(W.view(-1)[::997].cpu())
        if ref is not None:
            r = ref[n].to(W.device)
            inter += int((m & r).sum().item())
            union += int((m | r).sum().item())
    med_sel = torch.cat(sel).median().item()
    med_all = torch.cat(all_w).median().item()
    return med_sel / med_all, (inter / max(1, union) if ref is not None else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--out-dir", default="/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--damp", type=float, default=-0.75)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--score-batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = args.device
    short = args.model_id.split("/")[-1].replace(".", "p")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"placebo_contrast_{short}_rho{args.rho}" + (f"_{args.tag}" if args.tag else "")

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_lm(args.model_id, args.cache_dir, device)
    layers = target_layers(model)
    guard = SanitizeLogits()
    wt_ids = load_wikitext_ids(tok, args.cache_dir)

    # ── column norms: one control arm, four treatment arms ───────────────────
    def norms_for(prompts, label):
        t0 = time.time()
        cols = collect_column_norms(
            model, tok, [{"prompt": p} for p in prompts[:args.n_truth]],
            layers, device, position="prompt", batch_size=args.score_batch)
        mean, _, _ = cell_mean_sd(cols)
        print(f"  norms[{label}] ({time.time()-t0:.0f}s)", flush=True)
        del cols
        return mean

    n_ctrl = norms_for(DIRECT_TRUTH_INSTRUCT, "truth (control arm)")
    n_treat = {k: norms_for(v, k) for k, v in TREATMENTS.items()}

    # ── masks and their diagnostics ──────────────────────────────────────────
    masks = {k: mask_from_contrast(model, layers, n_treat[k], n_ctrl,
                                   args.rho, args.ratio, device)
             for k in TREATMENTS}
    masks["magmatch"] = magnitude_matched(model, masks["real"], device, args.seed)

    print("\nmask diagnostics (median |W| relative to all maskable weights; "
          "Jaccard against the real mask)", flush=True)
    diag = {}
    for k, m in masks.items():
        rel, jac = mask_stats(model, layers, m, ref=masks["real"] if k != "real" else None)
        diag[k] = (rel, jac)
        j = "1.000 (self)" if k == "real" else f"{jac:.3f}"
        print(f"  {k:<10} median|W| = {rel:.3f}x    Jaccard vs real = {j}", flush=True)

    # ── evaluation ───────────────────────────────────────────────────────────
    def ev():
        return eval_conditions(model, tok, EVAL_QA, device, args.batch_size,
                               args.reps, args.temp, args.seed, generate_batched,
                               guard=guard, junk_fn=junk, conditions=CONDITIONS)

    rows, originals = [], {}
    base_ppl = wikitext_ppl(model, wt_ids, device)
    print(f"\nbaseline wikitext ppl = {base_ppl:.3f}", flush=True)

    base_recs = ev()
    known = known_set(base_recs)
    base = summarize_honesty(base_recs, known)

    def tally(label, knob, recs, ppl):
        row = {"arm": label, "knob": knob, "rho": args.rho, "ratio": args.ratio,
               "ppl_ratio": ppl / base_ppl,
               "median_W_rel": diag.get(label, (float("nan"),))[0],
               "jaccard_vs_real": diag.get(label, (0, float("nan")))[1]}
        for cond in PRESSURE_SETS:
            sub = [dict(r, cond="pressured") for r in recs if r["cond"] == cond]
            s = summarize_honesty(sub + [r for r in recs if r["cond"] == "neutral"],
                                  known)
            row[f"lie_{cond}"] = s["comply_rate"]
            row["neutral_acc"] = s["neutral_acc"]
        row["lie_heldout"] = sum(row[f"lie_{c}"] for c in PRESSURE_SETS
                                 if c != "direct") / (len(PRESSURE_SETS) - 1)
        rows.append(row)
        print(f"  {label:<10} knob={knob:<6} direct={row['lie_direct']:>6.1%} "
              f"held-out={row['lie_heldout']:>6.1%} neutral={row['neutral_acc']:>6.1%} "
              f"ppl={row['ppl_ratio']:>5.2f}x", flush=True)
        with open(out_dir / f"{stem}.csv", "w", newline="") as f:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    print("\n── unsteered ──", flush=True)
    tally("none", 0.0, base_recs, base_ppl)

    for knob in (args.damp, args.amp):
        print(f"\n── knob = {knob} ──", flush=True)
        for k in list(TREATMENTS) + ["magmatch"]:
            apply_mask(model, masks[k], knob, originals)
            recs = ev()
            ppl = wikitext_ppl(model, wt_ids, device)
            restore(model, originals)
            tally(k, knob, recs, ppl)

    print(f"\n{'arm':<10}{'knob':>7}{'direct':>9}{'held-out':>10}{'neutral':>9}"
          f"{'ppl':>7}{'med|W|':>9}{'Jac':>7}")
    print("-" * 68)
    for r in sorted(rows, key=lambda r: (r["knob"], r["lie_heldout"])):
        print(f"{r['arm']:<10}{r['knob']:>7.2f}{r['lie_direct']:>9.1%}"
              f"{r['lie_heldout']:>10.1%}{r['neutral_acc']:>9.1%}"
              f"{r['ppl_ratio']:>7.2f}{r['median_W_rel']:>9.3f}"
              f"{r['jaccard_vs_real']:>7.3f}")
    print(f"\nrows → {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
