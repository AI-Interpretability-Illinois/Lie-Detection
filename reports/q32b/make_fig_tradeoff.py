#!/usr/bin/env python3
"""Body Figure 2 (owner 2026-10-12): Figure 8 reported in the analysis section, fewer details, Pareto frontiers drawn so that
both axes are better upward (honest rate, %) and rightward (inverse perplexity ratio = unedited perplexity / edited), hence
curves toward the top right are better. Data: fig8_points_extracted.json, every option recovered from the vector paths of the
co-author's fig_q32b_pareto.pdf (Qwen2.5-32B; the extracted values reproduce every number of the appendix text, e.g. 90.6 /
47.4 / 81.8 / 97.9 on LIARS' BENCH). A method's frontier at x: its highest honest rate among options whose inverse ratio is
at least x and that pass the control and polarity checks. Writes paper/v5/figures/fig_tradeoff.{pdf,png}."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = Path(__file__).parent
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/u/xiaocong/lie_detection/paper/v5/figures")
pts = json.load(open(H / "fig8_points_extracted.json"))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "axes.labelsize": 7, "axes.titlesize": 7.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300})
STYLE = [("mask", "ours", "#FF7F0E", "o"), ("liemask", "lie-instruction mask", "#009E73", "s"),
         ("steer", "steering (CAA)", "#0B2E6F", "^")]   # owner 2026-10-12: ours bright orange, steering dark blue
UNEDITED = {"lb": 21.9, "ail": 21.7}     # appendix text (Qwen2.5-32B unedited honest rate)
XMIN = 1 / 1.27                           # the region around the 1.15x check; steering's strongest options (1.62x) are off the axis
GATE = 1 / 1.15
inv = lambda q: 1.0 / max(q["ppl"], 1.0)
fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.8), sharey=True)
for ax, panel, title in zip(axes, ("lb", "ail"), ("LIARS' BENCH", "AI Liar")):
    P = [q for q in pts if q["panel"] == panel]
    for fam, lab, col, mk in STYLE:
        ok = sorted(((inv(q), q["honest"]) for q in P if q["fam"] == fam and q["gate_ok"] and not q["star"]), reverse=True)
        xs, front, best = [], [], -1
        for x, y in ok:                    # from the right (no perplexity cost) leftward: the best honest rate reachable so far
            if y > best:
                best = y; xs.append(x); front.append(y)
        xs.append(XMIN); front.append(front[-1])
        ax.step(xs, front, where="post", color=col, lw=1.6, label=lab, zorder=3)
        ax.scatter(xs[:-1], front[:-1], s=9, color=col, marker=mk, zorder=4, linewidths=0)
        star = [q for q in P if q["fam"] == fam and q["star"]][0]
        # the star marks a plotted option: same perplexity (a star's bounding box is centred horizontally, not vertically)
        near = min((q for q in P if q["fam"] == fam and not q["star"]), key=lambda q: (abs(q["ppl"] - star["ppl"]) > 0.004, abs(q["dishonest"] - star["dishonest"])))
        ax.scatter([inv(near)], [near["honest"]], s=60, marker="*", color=col, edgecolors="black", linewidths=0.5, zorder=5)
    pr = [q["honest"] for q in P if q["fam"] == "prompt"]
    ax.scatter([1.0] * len(pr), pr, s=14, marker="D", color="#8C8C8C", zorder=4, linewidths=0, label="honest prompt" if panel == "lb" else None)
    ax.axhline(UNEDITED[panel], color="black", lw=0.7, ls=":", label="unedited" if panel == "lb" else None)
    ax.axvspan(XMIN, GATE, color="#000000", alpha=0.07, lw=0, zorder=0)   # perplexity above 1.15x: fails the usability check
    ax.axvline(GATE, color="#777", lw=0.7, ls="--")
    ax.set_xlim(XMIN, 1.005); ax.set_ylim(0, 102)
    ax.set_xticks([0.8, 0.85, 0.9, 0.95, 1.0]); ax.set_xticklabels(["0.8", "", "0.9", "", "1"])
    ax.set_title(title); ax.set_xlabel("1 / perplexity ratio")
    ax.grid(alpha=0.15)
axes[0].set_ylabel("honest rate (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03), handlelength=1.6, columnspacing=0.8)
fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_tradeoff.{ext}", bbox_inches="tight")
print("wrote", OUT / "fig_tradeoff.pdf")
