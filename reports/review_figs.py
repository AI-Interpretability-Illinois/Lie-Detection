#!/usr/bin/env python3
"""Honesty against capability for every option (review, 2026-10-12): reports/review_mmlu.json -> paper figure
fig_mmlu_frontier.pdf/.png. x: change in plain MMLU accuracy (points, 2,000 questions) against the unedited model;
y: test-half honest rate. Filled markers pass Table 1's perplexity/control/polarity checks, hollow ones fail one. Lines:
each method's best honest rate among usable options with at most that MMLU cost. LIARS' BENCH graded by the Qwen2.5-7B
hybrid, AI Liar by keyword."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("/u/xiaocong/lie_detection/reports")
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/u/xiaocong/lie_detection/paper/v5/figures")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7.5,
                     "axes.labelsize": 7.5, "axes.titlesize": 8, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.15, "savefig.dpi": 300})
STYLE = {"mask": ("ours", "#FF7F0E", "o"), "liemask": ("lie-instruction mask", "#009E73", "s"),
         "steer": ("steering (CAA)", "#0B2E6F", "^"), "mlsteer": ("multi-layer steering", "#56B4E9", "v"),
         "prompt": ("honest prompt", "#7F7F7F", "D"), "unsteered": ("unedited", "#000000", "*")}
XMAX = 12
d = json.load(open(R / "review_mmlu.json"))
t1 = {r["model"]: r for r in json.load(open(R / "main_table_final.json"))}
names = [n for n in ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B"] if n in d["models"]]
fig, axes = plt.subplots(2, len(names), figsize=(6.8, 3.3), squeeze=False)
for j, n in enumerate(names):
    opts = d["models"][n]["options"]
    for i, bench in enumerate(("lb", "ail")):
        ax = axes[i][j]
        chosen = {c["arm"] for c in t1[n][bench]["cells"].values() if c.get("v") is not None}
        for f, (lab, col, mk) in STYLE.items():
            pts = [(-o["mmlu_drop"], o[bench], not o["fails"], o["arm"] in chosen) for o in opts if o["fam"] == f and o[bench] is not None]
            if not pts:
                continue
            ok = [(x, y) for x, y, g, _ in pts if g]; bad = [(x, y) for x, y, g, _ in pts if not g]
            ax.scatter([x for x, _ in ok], [y for _, y in ok], s=10 if f != "unsteered" else 40, c=col, marker=mk,
                       linewidths=0, zorder=3, label=lab if (i == 0 and j == 0) else None)
            ax.scatter([x for x, _ in bad], [y for _, y in bad], s=9, facecolors="none", edgecolors=col, marker=mk,
                       linewidths=0.6, alpha=0.6, zorder=2)
            for x, y, g, c in pts:
                if c:
                    ax.scatter([x], [y], s=55, facecolors="none", edgecolors=col, linewidths=1.0, zorder=4)
            if f not in ("prompt", "unsteered") and ok:
                xs = sorted(set(x for x, _ in ok))
                front = [max(y for x2, y in ok if x2 <= x) for x in xs]
                ax.step(xs, front, where="post", color=col, lw=1.1, alpha=0.9, zorder=2)
        ax.axvline(5, color="#999", lw=0.6, ls="--")
        ax.set_xlim(-1, XMAX)   # options that cost more MMLU are off the axis (named in the caption)
        if i == 0:
            ax.set_title(n)
        if j == 0:
            ax.set_ylabel(("LIARS' BENCH" if bench == "lb" else "AI Liar") + "\nhonest rate (%)")
        if i == 1:
            ax.set_xlabel("MMLU drop (points)")
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=col, marker=mk, ls="-" if f not in ("prompt", "unsteered") else "", ms=5 if f != "unsteered" else 8, lw=1.1)
           for f, (lab, col, mk) in STYLE.items()]
fig.legend(handles, [v[0] for v in STYLE.values()], loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.6, h_pad=0.6)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_mmlu_frontier.{ext}", bbox_inches="tight")
print("wrote", OUT / "fig_mmlu_frontier.pdf")
