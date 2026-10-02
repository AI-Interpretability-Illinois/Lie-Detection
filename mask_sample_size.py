#!/usr/bin/env python3
"""
Test (b): does the mask depend on WHICH 25 prompts it is built from, and does it
settle as the number of prompts grows?

Prompts are matched pairs: a question wrapped in one of the six falsehood
prefixes (F) or in "Answer truthfully:" (T), exactly as src/prompts.py builds the
25-prompt sets, but drawn from a 691-question pool in the same style
(build_mask_pool.py). Per-prompt input-channel norms are collected once; every
mask below is then the runner's rule (top rho of |W_ij| * c_j per matrix, with
c_j = v mean_F - (1 - v) mean_T, v = r / (1 + r)) on a subset of prompts.

Reported per family (fiction; pooled = lie + fiction + error prefixes):
  split-half   two disjoint draws of n prompts each; weight-weighted Jaccard of
               their masks, n in --ns, --reps repeats
  vs ref       each draw against the mask from all --pool prompts
  orig         the mask from the 25 hand-written prompts, against the pool mask
               and against pool draws of 25
  signal       the same overlaps restricted to selected weights with a positive
               score; the rest of the budget is filled by near-zero scores
               ("padding"), i.e. the smallest weights, which any draw picks alike
  lambda       for the pool mask, the padding share and overlap with r = 0.37 at
               other r (lambda = 1/r)
Forward passes only. Writes outputs/controls/mask_sample_size_<model>.json.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from did_estimators import collect_column_norms, get_module, load_lm, target_layers
from src.prompts import (DIRECT_FICTION_FALSE, DIRECT_HONEST_ERROR, DIRECT_LIE,
                         DIRECT_TRUTH_INSTRUCT, _TF_PREFIX)

OUT = Path(os.environ.get("LIE_CONTROLS", "/u/xiaocong/lie_detection/outputs/controls"))  # other hosts
POOL = Path(__file__).parent / "extra_prompts" / "mask_pool_questions.json"


def wrap(questions):
    """F prompts per style and T prompts, index-matched, prefixes rotated as in src/prompts.py."""
    F = {st: [f"{_TF_PREFIX[st][i % 6]} {q}" for i, q in enumerate(questions)]
         for st in ("lie", "fiction", "error")}
    T = [f"Answer truthfully: {q}" for q in questions]
    return F, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--ns", default="10,25,50,100,200")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--ratios", default="1.0,0.667,0.5,0.25")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",")]
    assert 2 * max(ns) <= args.pool, "split-half needs 2n <= pool"
    device = "cuda"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_lm(args.model_id, args.cache_dir, device)
    layers = target_layers(model)

    qs = json.loads(POOL.read_text())[:args.pool]
    F, T = wrap(qs)
    sets = {"F_" + k: v for k, v in F.items()}
    sets["T"] = T
    sets["o_lie"], sets["o_fiction"] = DIRECT_LIE[:25], DIRECT_FICTION_FALSE[:25]
    sets["o_error"], sets["o_T"] = DIRECT_HONEST_ERROR[:25], DIRECT_TRUTH_INSTRUCT[:25]
    norms = {}                      # set -> layer -> (n_prompts, in_features) fp16
    with torch.no_grad():
        for k, prompts in sets.items():
            cols = collect_column_norms(model, tok, [{"prompt": p} for p in prompts], layers,
                                        device, position="prompt", batch_size=args.batch)
            norms[k] = {ly: t.half() for ly, t in cols.items()}
            print(f"  norms[{k}] {len(prompts)} prompts  ({time.time()-t0:.0f}s)", flush=True)
    fam_F = {"fiction": lambda s, ly: norms[s + "fiction"][ly].float(),
             "pooled": lambda s, ly: (norms[s + "lie"][ly].float() + norms[s + "fiction"][ly].float()
                                      + norms[s + "error"][ly].float()) / 3}

    def col(fam, ly, idx, r, orig=False):
        v = r / (1 + r)
        if orig:
            f, t = fam_F[fam]("o_", ly), norms["o_T"][ly].float()
        else:
            f, t = fam_F[fam]("F_", ly)[idx], norms["T"][ly].float()[idx]
        return v * f.mean(0) - (1 - v) * t.mean(0)

    rng = random.Random(args.seed)
    draws = [(n, rep, rng.sample(range(args.pool), 2 * n)) for n in ns for rep in range(args.reps)]
    allidx = list(range(args.pool))
    ratios = [float(x) for x in args.ratios.split(",")]
    # accumulators: key -> [intersection, union] summed over layers (weight-weighted)
    acc = {}

    def add(key, a, b):
        x = acc.setdefault(key, [0, 0])
        x[0] += int((a & b).sum()); x[1] += int((a | b).sum())
    pad = {}                        # (fam, r) -> [padding weights, budget]

    def mask(W, c, k):
        s = W * c.to(W.device).unsqueeze(0)
        m = torch.zeros(W.numel(), dtype=torch.bool, device=W.device)
        m[s.view(-1).topk(k).indices] = True
        return m, m & (s.view(-1) > 0)

    for li, ly in enumerate(layers):
        W = get_module(model, ly).weight.detach().float().abs()
        k = max(1, int(args.rho * W.numel()))
        for fam in fam_F:
            ref, ref_p = mask(W, col(fam, ly, allidx, args.ratio), k)
            p = pad.setdefault((fam, args.ratio), [0, 0]); p[0] += k - int(ref_p.sum()); p[1] += k
            for r in ratios:
                m_r, m_rp = mask(W, col(fam, ly, allidx, r), k)
                add((fam, "lambda", r), m_r, ref)
                p = pad.setdefault((fam, r), [0, 0]); p[0] += k - int(m_rp.sum()); p[1] += k
                del m_r, m_rp
            o, o_p = mask(W, col(fam, ly, None, args.ratio, orig=True), k)
            add((fam, "orig_vs_ref"), o, ref); add((fam, "orig_vs_ref_signal"), o_p, ref_p)
            p = pad.setdefault((fam, "orig"), [0, 0]); p[0] += k - int(o_p.sum()); p[1] += k
            for n, rep, idx in draws:
                a, a_p = mask(W, col(fam, ly, idx[:n], args.ratio), k)
                b, b_p = mask(W, col(fam, ly, idx[n:], args.ratio), k)
                add((fam, "split", n), a, b); add((fam, "split_signal", n), a_p, b_p)
                add((fam, "vs_ref", n), a, ref); add((fam, "vs_ref_signal", n), a_p, ref_p)
                if n == 25:
                    add((fam, "orig_vs_draw25"), o, a)
                    p = pad.setdefault((fam, "draw25"), [0, 0]); p[0] += k - int(a_p.sum()); p[1] += k
                del a, a_p, b, b_p
            del ref, ref_p, o, o_p
        del W
        if li % 20 == 0:
            print(f"  layer {li+1}/{len(layers)}  ({time.time()-t0:.0f}s)", flush=True)

    J = {" | ".join(map(str, k)): round(v[0] / v[1], 4) for k, v in acc.items()}
    P = {f"{f} | {r}": round(a / b, 4) for (f, r), (a, b) in pad.items()}
    short = args.model_id.split("/")[-1].replace(".", "p")
    res = {"model": args.model_id, "pool": args.pool, "ns": ns, "reps": args.reps, "rho": args.rho,
           "ratio": args.ratio, "jaccard": J, "padding_share": P, "seconds": round(time.time() - t0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"mask_sample_size_{short}.json").write_text(json.dumps(res, indent=1))
    for fam in fam_F:
        print(f"\n{fam}: n   split-half  (signal)   vs pool-{args.pool}  (signal)")
        for n in ns:
            print(f"   {n:>4}   {J[f'{fam} | split | {n}']:.3f}   ({J[f'{fam} | split_signal | {n}']:.3f})"
                  f"      {J[f'{fam} | vs_ref | {n}']:.3f}   ({J[f'{fam} | vs_ref_signal | {n}']:.3f})")
        print(f"   hand-written 25 vs pool mask {J[f'{fam} | orig_vs_ref']:.3f} "
              f"(signal {J[f'{fam} | orig_vs_ref_signal']:.3f}); vs pool draws of 25 {J[f'{fam} | orig_vs_draw25']:.3f}")
        print(f"   padding share: r=0.37 pool {P[f'{fam} | {args.ratio}']:.3f}, draw25 {P[f'{fam} | draw25']:.3f}, "
              f"hand-written {P[f'{fam} | orig']:.3f}; "
              + ", ".join(f"r={r}: {P[f'{fam} | {r}']:.3f} (J vs 0.37 {J[f'{fam} | lambda | {r}']:.3f})" for r in ratios))
    print(f"\nwrote {OUT / f'mask_sample_size_{short}.json'}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
