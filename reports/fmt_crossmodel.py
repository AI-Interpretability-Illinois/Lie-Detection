#!/usr/bin/env python3
"""Cross-model rollup of the format-transfer matrices (summary_<model>_val.json from fmt_summarize.py):
per model and method, the mean share of the gap closed on single-turn formats (off-diagonal cells),
on the diagonal, and on multi-turn (all build formats), plus the perplexity range at the row strengths.
Only eval formats with a gap of at least MIN_GAP points enter."""
import json
import sys
from pathlib import Path

FORMATS = ["F1_sys", "F2_user", "F3_inject", "F4_mc", "F5_tf", "F6_multi", "F7_long"]
RUNS = {"Phi-4-mini-instruct": "/work/hdd/biro/xiaocong/lie_out/fmt_runs_timan",
        "Meta-Llama-3p1-8B-Instruct": "/work/hdd/biro/xiaocong/lie_out/fmt_runs",
        "Qwen2p5-7B-Instruct": "/work/hdd/biro/xiaocong/lie_out/fmt_runs",
        "Falcon3-3B-Instruct": "/work/hdd/biro/xiaocong/lie_out/fmt_runs"}
mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
print(f"{'model':<28}{'method':<7}{'single-turn off-diag':>21}{'diagonal':>10}{'multi-turn':>11}"
      f"{'worst build row':>17}{'ppl':>14}   eval formats with room")
for m, d in RUNS.items():
    S = json.load(open(Path(d) / f"summary_{m}_val.json"))
    for meth in ("maskc", "caa"):
        c = S[meth]["closure"]
        room = [j for j in FORMATS if any(c.get(f"{i}|{j}") is not None for i in FORMATS)]
        off = [c[f"{i}|{j}"] for i in FORMATS for j in room if j != "F6_multi" and i != j and c.get(f"{i}|{j}") is not None]
        diag = [c[f"{j}|{j}"] for j in room if j != "F6_multi" and c.get(f"{j}|{j}") is not None]
        multi = [c[f"{i}|F6_multi"] for i in FORMATS if c.get(f"{i}|F6_multi") is not None]
        rows = {i: mean([c[f"{i}|{j}"] for j in room if c.get(f"{i}|{j}") is not None]) for i in FORMATS}
        worst = min(rows, key=rows.get)
        pp = [v for v in S[meth]["ppl"].values() if v]
        print(f"{m:<28}{meth:<7}{mean(off):>21.1f}{mean(diag):>10.1f}{mean(multi):>11.1f}"
              f"{worst.split('_')[1] + f' {rows[worst]:.0f}':>17}{f'{min(pp):.2f}-{max(pp):.2f}':>14}   "
              + ",".join(j.split("_")[1] for j in room))
