#!/usr/bin/env python3
"""Generate figures for the layer-wise steering report."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path(__file__).parent.parent / "outputs"

ALPHA_COLORS = {"1.0": "#4C72B0", "2.0": "#DD8452"}
RHO_STYLES   = {
    "0.1":  ("o", "-",  "#E377C2"),
    "0.25": ("s", "--", "#2CA02C"),
    "0.5":  ("^", ":",  "#9467BD"),
}

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

rows    = load_csv(OUT_DIR / "layerwise_results.csv")
baseline = json.load(open(OUT_DIR / "layerwise_baseline.json"))
bl_ppl  = baseline["fluency"]["mean_perplexity"]
bl_acc  = baseline["baseline_accuracy"]

N_BLOCKS = 32
blocks   = list(range(N_BLOCKS))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Δwrong-rate vs block index
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for ax, alpha in zip(axes, ["1.0", "2.0"]):
    for rho_str, (mk, ls, col) in RHO_STYLES.items():
        subset = {int(r["block"]): float(r["delta_wrong_rate"])
                  for r in rows if r["rho"] == rho_str and r["alpha"] == alpha}
        ys = [subset.get(b, float("nan")) for b in blocks]
        ax.plot(blocks, ys, marker=mk, linestyle=ls, color=col, linewidth=1.5,
                markersize=5, label=f"ρ={rho_str}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.14, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Global α=1.0 best (+14%)" if alpha == "1.0" else "Global α=2.0 (~+90%)")
    ax.set_xlabel("Transformer block index", fontsize=11)
    ax.set_ylabel("Δ Wrong-Rate (steered − baseline)", fontsize=10)
    ax.set_title(f"α = {alpha}", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.set_xticks(range(0, N_BLOCKS, 4))
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim(-0.5, N_BLOCKS - 0.5)

fig.suptitle("Layer-wise Steering: Deception Effect (Δ Wrong-Rate) per Transformer Block\n"
             "direct prompt set · LLaMA-3.1-8B-Instruct",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig7_layerwise_delta_wrong.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7_layerwise_delta_wrong.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Perplexity vs block index
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

for ax, alpha in zip(axes, ["1.0", "2.0"]):
    ax.axhline(bl_ppl, color="grey", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"Baseline ({bl_ppl:.1f})")

    for rho_str, (mk, ls, col) in RHO_STYLES.items():
        subset = {int(r["block"]): float(r["mean_perplexity"])
                  for r in rows if r["rho"] == rho_str and r["alpha"] == alpha}
        ys = [subset.get(b, float("nan")) for b in blocks]
        ax.plot(blocks, ys, marker=mk, linestyle=ls, color=col, linewidth=1.5,
                markersize=5, label=f"ρ={rho_str}")

    ax.set_xlabel("Transformer block index", fontsize=11)
    ax.set_ylabel("Perplexity (↓ better)", fontsize=10)
    ax.set_title(f"α = {alpha}", fontsize=12, fontweight="bold")
    ax.set_xticks(range(0, N_BLOCKS, 4))
    ax.legend(fontsize=9, framealpha=0.8)
    ax.set_xlim(-0.5, N_BLOCKS - 0.5)

fig.suptitle("Layer-wise Steering: Fluency Cost (Perplexity) per Transformer Block\n"
             "direct prompt set · LLaMA-3.1-8B-Instruct",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig8_layerwise_perplexity.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig8_layerwise_perplexity.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — Global vs layer-wise comparison (Δwrong vs perplexity scatter)
# ─────────────────────────────────────────────────────────────────────────────

global_flu  = load_csv(OUT_DIR / "fluency_summary.csv")
global_summ = load_csv(OUT_DIR / "summary.csv")

fig, ax = plt.subplots(figsize=(7, 6))

# Layer-wise points (all 192 configs)
lw_dw  = [float(r["delta_wrong_rate"])  for r in rows]
lw_ppl = [float(r["mean_perplexity"])   for r in rows]
ax.scatter(lw_dw, lw_ppl, color="#AAAAAA", s=18, alpha=0.5, label="Layer-wise (all blocks)", zorder=2)

# Highlight top layer-wise block per alpha
for alpha in ["1.0", "2.0"]:
    best = max([r for r in rows if r["alpha"] == alpha],
               key=lambda r: float(r["delta_wrong_rate"]))
    ax.scatter(float(best["delta_wrong_rate"]), float(best["mean_perplexity"]),
               s=80, zorder=5,
               color=ALPHA_COLORS[alpha],
               marker="*",
               label=f"Layer-wise best α={alpha} (block {best['block']})")

# Global steering points (direct, rho=0.10)
global_pts = [(float(r["delta_wrong_rate"]),) for r in global_summ
              if r["prompt_set"] == "direct"]
flu_map = {float(r["alpha"]): float(r["mean_perplexity"]) for r in global_flu
           if r["variant"] != "baseline"}
for r in global_summ:
    if r["prompt_set"] != "direct":
        continue
    a = float(r["alpha"])
    dw = float(r["delta_wrong_rate"])
    ppl = flu_map.get(a, float("nan"))
    ax.scatter(dw, ppl, s=100, zorder=6, color="black",
               marker="D",
               label=f"Global α={a} ρ=0.10" if a in [1.0, 2.0] else "")

ax.set_xlabel("Δ Wrong-Rate (steered − baseline)", fontsize=11)
ax.set_ylabel("Perplexity (log scale, ↓ better)", fontsize=11)
ax.set_title("Global vs Layer-wise Steering:\nDeception Effect vs Fluency Cost",
             fontsize=12)
ax.set_yscale("log")
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax.axvline(0, color="black", linewidth=0.6)

# Deduplicate legend
handles, labels = ax.get_legend_handles_labels()
seen = {}
for h, l in zip(handles, labels):
    if l and l not in seen:
        seen[l] = h
ax.legend(seen.values(), seen.keys(), fontsize=9, framealpha=0.8, loc="upper left")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig9_global_vs_layerwise.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig9_global_vs_layerwise.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Heatmap: Δwrong-rate across block × (rho, alpha)
# ─────────────────────────────────────────────────────────────────────────────

rhos   = ["0.1", "0.25", "0.5"]
alphas = ["1.0", "2.0"]
configs = [(r, a) for a in alphas for r in rhos]   # 6 columns
labels  = [f"ρ={r}\nα={a}" for r, a in configs]

matrix = np.zeros((N_BLOCKS, len(configs)))
for ci, (rho, alpha) in enumerate(configs):
    for r in rows:
        if r["rho"] == rho and r["alpha"] == alpha:
            matrix[int(r["block"]), ci] = float(r["delta_wrong_rate"])

fig, ax = plt.subplots(figsize=(9, 10))
im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-0.12, vmax=0.12)
ax.set_xticks(range(len(configs)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_yticks(range(N_BLOCKS))
ax.set_yticklabels([f"Block {b}" for b in range(N_BLOCKS)], fontsize=8)
ax.set_xlabel("(ρ, α) configuration", fontsize=11)
ax.set_ylabel("Transformer block", fontsize=11)
ax.set_title("Δ Wrong-Rate Heatmap: Layer-wise Steering\n"
             "Red = more deception  ·  Blue = less (model improves)",
             fontsize=11)
plt.colorbar(im, ax=ax, label="Δ Wrong-Rate", shrink=0.6,
             format=matplotlib.ticker.PercentFormatter(xmax=1))

# Annotate cells with value
for bi in range(N_BLOCKS):
    for ci in range(len(configs)):
        v = matrix[bi, ci]
        ax.text(ci, bi, f"{v:+.0%}", ha="center", va="center",
                fontsize=6.5, color="black" if abs(v) < 0.08 else "white")

fig.tight_layout()
fig.savefig(FIG_DIR / "fig10_layerwise_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig10_layerwise_heatmap.png")

print("\nAll layer-wise figures saved to", FIG_DIR)
