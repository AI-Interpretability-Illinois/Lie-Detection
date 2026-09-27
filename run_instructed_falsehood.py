#!/usr/bin/env python3
"""
Instructed falsehood as the target behaviour, for both selection and evaluation.

The placebo test settled what the contrast actually tracks. Across five models a
fiction-framed or mistake-framed request builds a mask as good as, or better than,
a lie instruction, while a truthful-paraphrase control with matched prefix
diversity does nothing:

  Llama-8B held-out lying, damped:  lie 34.2%  fiction 8.9%  error 21.2%
                                    paraphrase 59.1%  control 59.8%  unsteered 60.9%

So deceptive intent is not the axis. Being instructed to assert a falsehood is.
That has two consequences this script acts on.

SELECTION. If the target is instructed falsehood, the treatment arm should span
falsehood instructions rather than lie instructions alone. Arms compared, all
against the same uniform truthful control, all with matched prefix diversity:

  lie        DIRECT_LIE                    the method as published
  fiction    DIRECT_FICTION_FALSE          falsehood licensed by a fiction frame
  error      DIRECT_HONEST_ERROR           falsehood framed as a mistake
  pooled     all three averaged            the design this evidence implies
  para       DIRECT_TRUTH_PARA             null: truthful paraphrases
  magmatch   random matched to lie's |W|   specificity control

EVALUATION. The metric now pressures the model six ways, adding a fiction framing
and a mistake framing to the four lie-flavoured ones, with wording held out from
every treatment arm. Three aggregates are reported: the original three held-out
framings (comparable with every earlier report), the two new falsehood framings,
and all five non-direct framings.

Damping is swept to full ablation, because -1.0 beat -0.75 by 27 points on
Llama-8B and the earlier sweeps stopped short.

Masks are carried as int32 index sets, not boolean tensors. Six boolean masks over
Llama-8B's maskable weights cost ~39 GB of host RAM; the same six index sets cost
~6 GB, which is what makes it safe to run beside another job on a shared node.
Only the selected weights are saved and rolled back, so nothing copies a whole
layer either.

Usage
-----
  python run_instructed_falsehood.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
      --rho 0.04 --ratio 0.37 --knobs="-0.5,-0.75,-1.0"
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
from honesty_eval import (FALSEHOOD_FRAMINGS, HELDOUT3, PRESSURE_SETS,
                          eval_conditions, known_set, summarize_honesty)

CONDITIONS = ("neutral",) + tuple(PRESSURE_SETS)
TREAT_SETS = {"lie": DIRECT_LIE, "fiction": DIRECT_FICTION_FALSE,
              "error": DIRECT_HONEST_ERROR, "para": DIRECT_TRUTH_PARA}
ARMS = ("lie", "fiction", "error", "pooled", "para", "magmatch")


# ── index-set masks ───────────────────────────────────────────────────────────

@torch.no_grad()
def idx_from_colstat(model, layers, colstat, rho, device):
    """Per layer, the top-rho indices of |W_ij| * colstat_j (separable Wanda)."""
    out = {}
    for name in layers:
        W = get_module(model, name).weight
        n = W.numel()
        k = max(1, int(rho * n))
        s = (W.detach().float().abs().to(device)
             * colstat[name].to(device).unsqueeze(0)).view(-1)
        out[name] = s.topk(k).indices.to(torch.int32).cpu()
        del s
    return out


@torch.no_grad()
def idx_magnitude_matched(model, ref, device, seed=0):
    """Random index set reproducing ref's per-layer |W| decile histogram."""
    out = {}
    for li, (name, idx) in enumerate(ref.items()):
        W = get_module(model, name).weight.detach().float().abs().view(-1).to(device)
        n = W.numel()
        g = torch.Generator(device=device).manual_seed(seed * 100003 + li)
        sample = W[torch.randperm(n, generator=g, device=device)[:min(n, 100_000)]]
        qs = torch.quantile(sample, torch.linspace(0, 1, 11, device=device))
        qs[0], qs[-1] = -float("inf"), float("inf")
        bins = torch.bucketize(W, qs[1:-1].contiguous())
        want = torch.bincount(bins[idx.to(device).long()], minlength=10)
        picks = []
        for b in range(10):
            pool = (bins == b).nonzero(as_tuple=True)[0]
            take = int(min(int(want[b].item()), pool.numel()))
            if take:
                sel = torch.randperm(pool.numel(), generator=g, device=device)[:take]
                picks.append(pool[sel])
        out[name] = (torch.cat(picks) if picks
                     else torch.empty(0, dtype=torch.long, device=device)
                     ).to(torch.int32).cpu()
        del W, bins
    return out


@torch.no_grad()
def apply_idx(model, idxs, alpha, saved):
    """W[i] *= (1 + alpha) for the selected entries; stash the old values."""
    for name, idx in idxs.items():
        flat = get_module(model, name).weight.data.view(-1)
        i = idx.to(flat.device).long()
        saved[name] = flat[i].detach().to("cpu", copy=True)
        flat[i] = flat[i] * (1.0 + alpha)
        del i


@torch.no_grad()
def restore_idx(model, idxs, saved):
    for name, vals in saved.items():
        flat = get_module(model, name).weight.data.view(-1)
        i = idxs[name].to(flat.device).long()
        flat[i] = vals.to(flat.device)
        del i
    saved.clear()


@torch.no_grad()
def idx_diag(model, idxs, ref=None):
    """Median |W| of the selection relative to all weights, and Jaccard vs ref."""
    sel, allw = [], []
    inter = union = 0
    for name, idx in idxs.items():
        W = get_module(model, name).weight.detach().float().abs().view(-1)
        i = idx.to(W.device).long()
        s = W[i]
        if s.numel():
            sel.append(s[:: max(1, s.numel() // 20000)].cpu())
        allw.append(W[::997].cpu())
        if ref is not None:
            r = ref[name].to(W.device)
            hit = int(torch.isin(idx.to(W.device), r).sum().item())
            inter += hit
            union += idx.numel() + r.numel() - hit
        del W, i, s
    rel = (torch.cat(sel).median().item() / torch.cat(allw).median().item()
           if sel else float("nan"))
    return rel, (inter / max(1, union) if ref is not None else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--out-dir", default="/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--knobs", default="-0.5,-0.75,-1.0",
                    help="Use = syntax for negatives: --knobs=\"-0.75,-1.0\"")
    ap.add_argument("--arms", default=",".join(ARMS))
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
    stem = (f"instructed_falsehood_{short}_rho{args.rho}"
            + (f"_{args.tag}" if args.tag else ""))
    knobs = [float(x) for x in args.knobs.split(",")]
    want = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"instructed falsehood: {args.model_id}  rho={args.rho} ratio={args.ratio}")
    print(f"arms={want}  knobs={knobs}")
    print(f"eval framings: {list(PRESSURE_SETS)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = load_lm(args.model_id, args.cache_dir, device)
    layers = target_layers(model)
    guard = SanitizeLogits()
    wt_ids = load_wikitext_ids(tok, args.cache_dir)
    v = args.ratio / (1.0 + args.ratio) if args.ratio > 0 else 0.0
    print(f"{len(layers)} maskable matrices; v={v:.4f}", flush=True)

    # ── column norms: one control arm, four treatment arms ────────────────────
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
    n_treat = {k: norms_for(p, k) for k, p in TREAT_SETS.items()}
    # the design the placebo evidence implies: pool every falsehood instruction
    n_treat["pooled"] = {ly: (n_treat["lie"][ly] + n_treat["fiction"][ly]
                              + n_treat["error"][ly]) / 3.0 for ly in layers}

    colstat = {a: {ly: v * n_treat[a][ly] - (1.0 - v) * n_ctrl[ly] for ly in layers}
               for a in n_treat}

    def build(arm):
        if arm == "magmatch":
            return idx_magnitude_matched(
                model, idx_from_colstat(model, layers, colstat["lie"], args.rho,
                                        device), device, args.seed)
        return idx_from_colstat(model, layers, colstat[arm], args.rho, device)

    # ── mask diagnostics: one extra mask alive at a time ──────────────────────
    print("\nmask diagnostics (median |W| relative to all maskable weights; "
          "Jaccard against the lie mask)", flush=True)
    diag = {}
    lie_idx = build("lie")
    for arm in want:
        t0 = time.time()
        idxs = lie_idx if arm == "lie" else build(arm)
        rel, jac = idx_diag(model, idxs, ref=None if arm == "lie" else lie_idx)
        diag[arm] = (rel, jac)
        j = "1.000 (self)" if arm == "lie" else f"{jac:.3f}"
        print(f"  {arm:<10} median|W| = {rel:.4f}x   Jaccard vs lie = {j}"
              f"   ({time.time()-t0:.0f}s)", flush=True)
        del idxs
    del lie_idx

    # ── evaluation ────────────────────────────────────────────────────────────
    def ev():
        return eval_conditions(model, tok, EVAL_QA, device, args.batch_size,
                               args.reps, args.temp, args.seed, generate_batched,
                               guard=guard, junk_fn=junk, conditions=CONDITIONS)

    rows, all_recs = [], []
    base_ppl = wikitext_ppl(model, wt_ids, device)
    base_recs = ev()
    known = known_set(base_recs)
    print(f"\nbaseline ppl = {base_ppl:.3f}; {len(known)} known questions", flush=True)

    def tally(arm, knob, recs, ppl):
        row = {"arm": arm, "knob": knob, "rho": args.rho, "ratio": args.ratio,
               "ppl_ratio": ppl / base_ppl,
               "median_W_rel": diag.get(arm, (float("nan"), 0))[0],
               "jaccard_vs_lie": diag.get(arm, (0, float("nan")))[1]}
        for cond in PRESSURE_SETS:
            sub = [dict(r, cond="pressured") for r in recs if r["cond"] == cond]
            s = summarize_honesty(sub + [r for r in recs if r["cond"] == "neutral"],
                                  known)
            row[f"lie_{cond}"] = s["comply_rate"]
            row["neutral_acc"] = s["neutral_acc"]
            row["junk_rate"] = s["pressured_junk"]
        row["heldout3"] = sum(row[f"lie_{c}"] for c in HELDOUT3) / len(HELDOUT3)
        row["falsehood2"] = (sum(row[f"lie_{c}"] for c in FALSEHOOD_FRAMINGS)
                             / len(FALSEHOOD_FRAMINGS))
        nd = [c for c in PRESSURE_SETS if c != "direct"]
        row["heldout_all"] = sum(row[f"lie_{c}"] for c in nd) / len(nd)
        rows.append(row)
        all_recs.extend(dict(r, arm=arm, knob=knob) for r in recs)
        print(f"  {arm:<10} knob={knob:<6} direct={row['lie_direct']:>6.1%} "
              f"heldout3={row['heldout3']:>6.1%} falsehood2={row['falsehood2']:>6.1%} "
              f"all={row['heldout_all']:>6.1%} neutral={row['neutral_acc']:>6.1%} "
              f"ppl={row['ppl_ratio']:>6.2f}x", flush=True)
        with open(out_dir / f"{stem}.csv", "w", newline="") as f:
            fields = sorted({k for r in rows for k in r})
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in fields})
        with open(out_dir / f"{stem}_responses.jsonl", "w") as f:
            for r in all_recs:
                f.write(json.dumps(r) + "\n")

    tally("none", 0.0, base_recs, base_ppl)

    for knob in knobs:
        print(f"\n── knob = {knob} ──", flush=True)
        for arm in want:
            t0 = time.time()
            idxs, saved = build(arm), {}
            apply_idx(model, idxs, knob, saved)
            recs = ev()
            ppl = wikitext_ppl(model, wt_ids, device)
            restore_idx(model, idxs, saved)
            del idxs
            tally(arm, knob, recs, ppl)

    print(f"\n{'arm':<10}{'knob':>7}{'direct':>9}{'heldout3':>10}"
          f"{'falsehood2':>12}{'all':>8}{'neutral':>9}{'ppl':>8}")
    print("-" * 73)
    for r in sorted(rows, key=lambda r: (r["knob"], r["heldout_all"])):
        print(f"{r['arm']:<10}{r['knob']:>7.2f}{r['lie_direct']:>9.1%}"
              f"{r['heldout3']:>10.1%}{r['falsehood2']:>12.1%}"
              f"{r['heldout_all']:>8.1%}{r['neutral_acc']:>9.1%}"
              f"{r['ppl_ratio']:>8.2f}")
    print(f"\nrows → {out_dir / (stem + '.csv')}")


if __name__ == "__main__":
    main()
