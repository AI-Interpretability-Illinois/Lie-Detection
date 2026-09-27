#!/usr/bin/env python3
"""
Evaluation-side DiD: dose-response curves for each mask family, matched on damage.

The estimand is the behaviour attributable to WHICH weights were selected, net of
the damage amplification causes by itself. For a mask family m, trace the curve
L_m(d): the lie rate reached when m is scaled by whatever knob yields damage level
d. The contrast of interest is the vertical gap between curves,

    tau(d) = L_DID(d) - L_placebo(d),

with damage as the matching variable rather than alpha. tau(d) > 0 across the range
of d is the evidence that selection rather than amplification drives the behaviour.
Negative knobs give the necessity direction (ablation at matched capability loss).

Mask families, all at the same per-layer rho:

  did         v*n_lie - (1-v)*n_truth, the method
  magmatch    random, matched to the did mask's per-layer |W| decile histogram
  random      uniform random per layer, same k
  truth_only  top-rho by |W| * n_truth alone
  lie_only    top-rho by |W| * n_lie alone

Damage is recorded as three quantities, not one, because perplexity alone is
demonstrably insufficient: on Falcon3-3B the harmful arms sat at 0.93x perplexity,
BELOW baseline, while collapsing the honest rate. So each row carries the
perplexity ratio, neutral accuracy (capability) and the junk rate
(degeneration), and the matching variable is chosen at analysis time.

Usage
-----
  # split the knob grid across GPUs; each job rebuilds the (cheap) masks
  python run_dose_response.py --knobs "0.5,1.0,2.0" --tag k1
  python run_dose_response.py --knobs "-0.5,-0.75,-1.0" --tag k3
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT
from did_estimators import (cell_mean_sd, collect_column_norms, get_module,
                            load_lm, target_layers)
from run_reversed_contrast import (EVAL_QA, SanitizeLogits, generate_batched,
                                   junk, load_wikitext_ids, wikitext_ppl)
from run_real_did import apply_mask, magnitude_matched, restore
from honesty_eval import HELDOUT3, PRESSURE_SETS, eval_conditions, known_set, summarize_honesty

CONDITIONS = ("neutral",) + tuple(PRESSURE_SETS)
FAMILIES = ["did", "magmatch", "random", "truth_only", "lie_only"]


@torch.no_grad()
def topk_mask(model, layers, colstat, rho, device):
    masks = {}
    for name in layers:
        W = get_module(model, name).weight.detach().float().abs().to(device)
        s = (W * colstat[name].to(device).unsqueeze(0)).view(-1)
        k = max(1, int(rho * s.numel()))
        m = torch.zeros(s.numel(), dtype=torch.bool, device=device)
        m[s.topk(k).indices] = True
        masks[name] = m.view(W.shape).cpu()
        del W, s, m
    return masks


@torch.no_grad()
def uniform_random_mask(model, layers, rho, device, seed=0):
    masks = {}
    for li, name in enumerate(layers):
        n = get_module(model, name).weight.numel()
        k = max(1, int(rho * n))
        g = torch.Generator(device=device).manual_seed(seed * 100003 + li)
        idx = torch.randperm(n, generator=g, device=device)[:k]
        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[idx] = True
        masks[name] = m.view(get_module(model, name).weight.shape).cpu()
        del m, idx
    return masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--out-dir", default="/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--knobs", required=True,
                    help="Comma-separated alphas. Use = syntax for negatives: "
                         "--knobs=\"-0.5,-0.75\"")
    ap.add_argument("--families", default=",".join(FAMILIES))
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
    stem = f"dose_response_{short}_rho{args.rho}" + (f"_{args.tag}" if args.tag else "")
    knobs = [float(x) for x in args.knobs.split(",")]
    fams = [f.strip() for f in args.families.split(",") if f.strip()]

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_lm(args.model_id, args.cache_dir, device)
    layers = target_layers(model)
    guard = SanitizeLogits()
    wt_ids = load_wikitext_ids(tok, args.cache_dir)
    v = args.ratio / (1.0 + args.ratio) if args.ratio > 0 else 0.0

    def norms_for(prompts, label):
        cols = collect_column_norms(
            model, tok, [{"prompt": p} for p in prompts[:args.n_truth]],
            layers, device, position="prompt", batch_size=args.score_batch)
        mean, _, _ = cell_mean_sd(cols)
        print(f"  norms[{label}]", flush=True)
        del cols
        return mean

    n_lie = norms_for(DIRECT_LIE, "lie")
    n_truth = norms_for(DIRECT_TRUTH_INSTRUCT, "truth")

    masks = {}
    if "did" in fams or "magmatch" in fams:
        did = topk_mask(model, layers,
                        {k: v * n_lie[k] - (1.0 - v) * n_truth[k] for k in layers},
                        args.rho, device)
        if "did" in fams:
            masks["did"] = did
        if "magmatch" in fams:
            masks["magmatch"] = magnitude_matched(model, did, device, args.seed)
    if "random" in fams:
        masks["random"] = uniform_random_mask(model, layers, args.rho, device, args.seed)
    if "truth_only" in fams:
        masks["truth_only"] = topk_mask(model, layers, n_truth, args.rho, device)
    if "lie_only" in fams:
        masks["lie_only"] = topk_mask(model, layers, n_lie, args.rho, device)
    print(f"masks built: {list(masks)}", flush=True)

    def ev():
        return eval_conditions(model, tok, EVAL_QA, device, args.batch_size,
                               args.reps, args.temp, args.seed, generate_batched,
                               guard=guard, junk_fn=junk, conditions=CONDITIONS)

    rows = []
    base_ppl = wikitext_ppl(model, wt_ids, device)
    base_recs = ev()
    known = known_set(base_recs)

    all_recs = []    # saved so any later grader fix can re-score offline

    def tally(family, knob, recs, ppl):
        row = {"family": family, "knob": knob, "rho": args.rho, "ratio": args.ratio,
               "ppl_ratio": ppl / base_ppl}
        for cond in PRESSURE_SETS:
            sub = [dict(r, cond="pressured") for r in recs if r["cond"] == cond]
            s = summarize_honesty(sub + [r for r in recs if r["cond"] == "neutral"],
                                  known)
            row[f"lie_{cond}"] = s["comply_rate"]
            row["neutral_acc"] = s["neutral_acc"]
            row["junk_rate"] = s["pressured_junk"]
        # lie_heldout is the three original held-out framings, so every run stays
        # comparable with the reports. PRESSURE_SETS grew from four framings to six
        # on 2026-09-26 and this line silently became a five-framing mean for any
        # run started after that. lie_heldout_all is the five-framing mean.
        row["lie_heldout"] = sum(row[f"lie_{c}"] for c in HELDOUT3) / len(HELDOUT3)
        _nd = [c for c in PRESSURE_SETS if c != "direct"]
        row["lie_heldout_all"] = sum(row[f"lie_{c}"] for c in _nd) / len(_nd)
        rows.append(row)
        all_recs.extend(dict(r, arm=family, knob=knob) for r in recs)
        with open(out_dir / f"{stem}_responses.jsonl", "w") as f:
            for r in all_recs:
                f.write(json.dumps(r) + "\n")
        print(f"  {family:<11} knob={knob:<6} direct={row['lie_direct']:>6.1%} "
              f"held-out={row['lie_heldout']:>6.1%} | damage: ppl={row['ppl_ratio']:>7.2f}x "
              f"neutral={row['neutral_acc']:>6.1%} junk={row['junk_rate']:>5.1%}",
              flush=True)
        with open(out_dir / f"{stem}.csv", "w", newline="") as f:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})

    print(f"\nbaseline ppl = {base_ppl:.3f}", flush=True)
    tally("none", 0.0, base_recs, base_ppl)

    originals = {}
    for knob in knobs:
        print(f"\n── knob = {knob} ──", flush=True)
        for fam in fams:
            if fam not in masks:
                continue
            t0 = time.time()
            apply_mask(model, masks[fam], knob, originals)
            recs = ev()
            ppl = wikitext_ppl(model, wt_ids, device)
            restore(model, originals)
            tally(fam, knob, recs, ppl)

    print(f"\nrows → {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
