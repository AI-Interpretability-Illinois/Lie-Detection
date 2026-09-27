#!/usr/bin/env python3
"""
The proposal's selection-side DiD, run the way the magnitude diagnostic says it
must be, plus a check on its algebra.

## The algebra problem

The proposal defines

    DID_2 = [n_lie - n_truth] - [n_cplus - n_cminus]

with (c+, c-) a control pair "sharing the surface structure of (lie, truth) but
carrying no deceptive intent", and suggests c- = "assert the fact". But asserting
the fact IS the truth arm. With c- = truth the two brackets share a term and

    DID_2 = n_lie - n_cplus,

a single difference against a different control, not a difference of differences.
That is the same collapse that sank the naive neutral-baseline 2x2. To get a
genuine second difference the control pair's own baseline has to differ from the
treatment pair's, so this script runs both readings:

  lie_vs_fiction   n_lie - n_fiction          the collapsed form: deceptive-intent
                                              falsehood net of licensed falsehood
  did2_fiction     (n_lie - n_truth) - (n_fiction - n_para)
  did2_error       (n_lie - n_truth) - (n_error  - n_para)

where `para` is a second, distinct set of truthful instructions, so no term is
shared between the brackets.

## Why stratification

Ranking by |W| * colstat is dominated by magnitude, not by the contrast. Measured
on Llama-8B: the working single-difference mask selects weights whose median |W| is
0.04x the global median, because the ratio weighting makes the statistic negative
for 99.5% of columns; every second-difference variant tried so far is sign-mixed,
lands on weights 2.1-2.7x the global median, and destroys the model before changing
behaviour. So a second difference cannot be evaluated until magnitude is held
fixed.

Each contrast is therefore built two ways:

  plain        top-rho of |W| * colstat            (the method's own rule)
  stratified   top-rho of colstat, restricted to weights in the bottom --wpct
               percent of |W| within each layer, so the contrast chooses and
               magnitude does not

If the stratified second differences beat the stratified single difference, the
proposal's Part 1 is worth building. If stratification is what matters and the
contrast barely moves the result, the estimand was a magnitude regime all along.

Usage
-----
  python run_stratified_did2.py --model-id Qwen/Qwen2.5-1.5B-Instruct --wpct 10
"""

from __future__ import annotations

import argparse
import csv
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


def contrasts(n, v):
    """Per-layer column statistics. n maps arm name -> {layer: vector}."""
    L = list(n["lie"].keys())
    single = {k: v * n["lie"][k] - (1.0 - v) * n["truth"][k] for k in L}
    return {
        # the method
        "single":          single,
        # the proposal's form with c- = the truth arm, which collapses
        "lie_vs_fiction":  {k: v * n["lie"][k] - (1.0 - v) * n["fiction"][k] for k in L},
        # genuine second differences: the control pair has its own baseline
        "did2_fiction":    {k: single[k] - (v * n["fiction"][k]
                                            - (1.0 - v) * n["para"][k]) for k in L},
        "did2_error":      {k: single[k] - (v * n["error"][k]
                                            - (1.0 - v) * n["para"][k]) for k in L},
    }


@torch.no_grad()
def build_mask(model, layers, colstat, rho, device, wpct=None):
    """top-rho per layer. wpct=None ranks |W|*colstat; otherwise ranks colstat
    among weights below the wpct-th percentile of |W| in that layer."""
    masks = {}
    for name in layers:
        W = get_module(model, name).weight.detach().float().abs().to(device)
        col = colstat[name].to(device)
        n = W.numel()
        k = max(1, int(rho * n))
        flat_w = W.view(-1)
        if wpct is None:
            s = (W * col.unsqueeze(0)).view(-1)
            idx = s.topk(k).indices
        else:
            sample = flat_w[torch.randperm(n, device=device)[:min(n, 200_000)]]
            thresh = torch.quantile(sample, wpct / 100.0)
            eligible = (flat_w <= thresh).nonzero(as_tuple=True)[0]
            if eligible.numel() < k:          # stratum too small: take all of it
                idx = eligible
            else:
                s_el = col.unsqueeze(0).expand_as(W).reshape(-1)[eligible]
                idx = eligible[s_el.topk(k).indices]
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[idx] = True
        masks[name] = m.view(W.shape).cpu()
        del W, col, flat_w, m, idx
    return masks


@torch.no_grad()
def med_w_rel(model, layers, masks):
    sel, allw = [], []
    for nm in layers:
        W = get_module(model, nm).weight.detach().float().abs()
        s = W[masks[nm].to(W.device)]
        if s.numel():
            sel.append(s[:: max(1, s.numel() // 20000)].cpu())
        allw.append(W.view(-1)[::997].cpu())
    if not sel:
        return float("nan")
    return torch.cat(sel).median().item() / torch.cat(allw).median().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--out-dir", default="/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--wpct", type=float, default=10.0,
                    help="|W| percentile defining the low-magnitude stratum.")
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--damp", type=float, default=-0.75)
    ap.add_argument("--amp", type=float, default=3.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = args.device
    short = args.model_id.split("/")[-1].replace(".", "p")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"stratified_did2_{short}_rho{args.rho}_w{args.wpct:g}"
            + (f"_{args.tag}" if args.tag else ""))

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_lm(args.model_id, args.cache_dir, device)
    layers = target_layers(model)
    guard = SanitizeLogits()
    wt_ids = load_wikitext_ids(tok, args.cache_dir)
    v = args.ratio / (1.0 + args.ratio) if args.ratio > 0 else 0.0

    arms = {"lie": DIRECT_LIE, "truth": DIRECT_TRUTH_INSTRUCT,
            "para": DIRECT_TRUTH_PARA, "fiction": DIRECT_FICTION_FALSE,
            "error": DIRECT_HONEST_ERROR}
    n = {}
    for k, prompts in arms.items():
        cols = collect_column_norms(
            model, tok, [{"prompt": p} for p in prompts[:args.n_truth]],
            layers, device, position="prompt", batch_size=args.score_batch)
        n[k], _, _ = cell_mean_sd(cols)
        print(f"  norms[{k}]", flush=True)
        del cols

    cs = contrasts(n, v)
    masks, diag = {}, {}
    for cname, col in cs.items():
        for mode, wp in (("plain", None), ("strat", args.wpct)):
            label = f"{cname}:{mode}"
            masks[label] = build_mask(model, layers, col, args.rho, device, wp)
            diag[label] = med_w_rel(model, layers, masks[label])
    masks["magmatch"] = magnitude_matched(model, masks["single:plain"], device, args.seed)
    diag["magmatch"] = med_w_rel(model, layers, masks["magmatch"])

    print("\nmedian |W| of selected weights, relative to all maskable weights", flush=True)
    for k in masks:
        print(f"  {k:<26} {diag[k]:.4f}x", flush=True)

    def ev():
        return eval_conditions(model, tok, EVAL_QA, device, args.batch_size,
                               args.reps, args.temp, args.seed, generate_batched,
                               guard=guard, junk_fn=junk, conditions=CONDITIONS)

    rows = []
    base_ppl = wikitext_ppl(model, wt_ids, device)
    base_recs = ev()
    known = known_set(base_recs)

    def tally(label, knob, recs, ppl):
        row = {"arm": label, "knob": knob, "rho": args.rho, "ratio": args.ratio,
               "wpct": args.wpct, "ppl_ratio": ppl / base_ppl,
               "median_W_rel": diag.get(label, float("nan"))}
        for cond in PRESSURE_SETS:
            sub = [dict(r, cond="pressured") for r in recs if r["cond"] == cond]
            s = summarize_honesty(sub + [r for r in recs if r["cond"] == "neutral"],
                                  known)
            row[f"lie_{cond}"] = s["comply_rate"]
            row["neutral_acc"] = s["neutral_acc"]
        row["lie_heldout"] = sum(row[f"lie_{c}"] for c in PRESSURE_SETS
                                 if c != "direct") / (len(PRESSURE_SETS) - 1)
        rows.append(row)
        print(f"  {label:<26} knob={knob:<6} direct={row['lie_direct']:>6.1%} "
              f"held-out={row['lie_heldout']:>6.1%} neutral={row['neutral_acc']:>6.1%} "
              f"ppl={row['ppl_ratio']:>7.2f}x", flush=True)
        with open(out_dir / f"{stem}.csv", "w", newline="") as f:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    print(f"\nbaseline ppl = {base_ppl:.3f}", flush=True)
    tally("none", 0.0, base_recs, base_ppl)

    originals = {}
    for knob in (args.damp, args.amp):
        print(f"\n── knob = {knob} ──", flush=True)
        for label in list(masks):
            apply_mask(model, masks[label], knob, originals)
            recs = ev()
            ppl = wikitext_ppl(model, wt_ids, device)
            restore(model, originals)
            tally(label, knob, recs, ppl)

    print(f"\nrows → {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
