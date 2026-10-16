#!/usr/bin/env python3
"""Appendix format ablations (pooled / leave-one-out sources, directional ablation) as honest rates (%), validation half
(fmt_summarize.py summary_*_val_x.json), averaged over ALL seven evaluation formats (no 10-point-gap filter), each row at
its chosen strength. Directional ablation: mean honest-rate gain over the unedited model per single-turn eval format.
Adapted from a scratch script of the dagformer session (which averaged over formats with a gap of at least 10 points)."""
import json
from pathlib import Path
F = ["F1_sys", "F2_user", "F3_inject", "F4_mc", "F5_tf", "F6_multi", "F7_long"]
A = Path("/work/hdd/biro/xiaocong/lie_out/fmt_runs")
M = {"Phi": "Phi-4-mini-instruct", "Llama": "Meta-Llama-3p1-8B-Instruct", "Qwen-7B": "Qwen2p5-7B-Instruct", "Falcon": "Falcon3-3B-Instruct"}
mean = lambda xs: sum(xs) / len(xs)
for lab, m in M.items():
    X = json.load(open(A / f"summary_{m}_val_x.json")); U = X["unedited"]["U"]
    for meth in ("maskc", "caa", "dirab"):
        pick = {i: s for i, s in X[meth]["row_pick"].items() if s is not None}
        arm = lambda i: X["arms"][f"{meth}:{i}:{pick[i]}"]
        if meth != "dirab":
            rows = {i: mean([arm(i)["dec"][j] for j in F]) for i in pick}
            single = {i: v for i, v in rows.items() if i in F}
            loo = {i: v for i, v in rows.items() if i.startswith("LOO")}
            print(f"{lab:8s} {meth:5s} POOL {rows['POOL']:.1f}  best single {max(single.values()):.1f} ({max(single, key=single.get)})"
                  f"  LOO {min(loo.values()):.1f}-{max(loo.values()):.1f}  LOO-multi {rows['LOO-F6_multi']:.1f}  unedited {mean([U[j] for j in F]):.1f}")
        else:
            g = {j.split('_')[1]: round(mean([arm(i)["dec"][j] for i in pick if i in F]) - U[j], 1) for j in F if j != "F6_multi"}
            print(f"{lab:8s} dirab gain over unedited per single-turn eval format {g}")
