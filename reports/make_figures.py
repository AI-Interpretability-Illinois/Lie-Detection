#!/usr/bin/env python3
"""Generate figures for the LieNeuro experiment report."""

import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path(__file__).parent.parent / "outputs"

COLORS = {
    "baseline": "#4C72B0",
    "steered":  "#DD8452",
    "string":   "#55A868",
    "judge":    "#C44E52",
}
PS_ORDER = ["direct", "instructed", "minimal", "contextual", "implicit", "mixed"]
PS_COLORS = plt.cm.tab10(np.linspace(0, 0.6, len(PS_ORDER)))

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

summary    = load_csv(OUT_DIR / "summary_judged.csv")
fluency    = load_csv(OUT_DIR / "fluency_summary.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Accuracy drop by α (judge-corrected), all prompt sets
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

for ax_idx, alpha_val in enumerate(["1.0", "2.0"]):
    ax = axes[ax_idx]
    rows = [r for r in summary if r["alpha"] == alpha_val]
    rows.sort(key=lambda r: PS_ORDER.index(r["prompt_set"]))

    x      = np.arange(len(rows))
    bl_acc = [float(r["judge_baseline_accuracy"]) for r in rows]
    st_acc = [float(r["judge_steered_accuracy"])  for r in rows]
    labels = [r["prompt_set"] for r in rows]

    width = 0.35
    bars_bl = ax.bar(x - width/2, bl_acc, width, label="Baseline",
                     color=COLORS["baseline"], alpha=0.85)
    bars_st = ax.bar(x + width/2, st_acc, width, label="Steered",
                     color=COLORS["steered"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Factual Accuracy (judge-corrected)")
    ax.set_title(f"α = {alpha_val}", fontsize=12, fontweight="bold")
    ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))

    # Annotate Δ
    for b, s, xi in zip(bl_acc, st_acc, x):
        delta = s - b
        ax.text(xi, max(b, s) + 0.02, f"{delta:+.0%}",
                ha="center", va="bottom", fontsize=8,
                color=COLORS["steered"] if delta < 0 else COLORS["baseline"])

fig.suptitle("Factual Accuracy Before and After Weight Steering\n(LLaMA-3.1-8B-Instruct, ρ=0.10)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig1_accuracy_by_alpha.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_accuracy_by_alpha.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Δwrong_rate vs α, string vs judge, all prompt sets
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
alphas_all = sorted(set(float(r["alpha"]) for r in summary))

for col_idx, (key_str, key_jdg, title) in enumerate([
    ("delta_wrong_rate", "judge_delta_wrong_rate", "Δ Wrong-Rate: String vs Judge"),
]):
    for ax_idx, ps in enumerate(PS_ORDER):
        for ax, metric, color, label in [
            (axes[0], "delta_wrong_rate",       COLORS["string"], "String matcher"),
            (axes[1], "judge_delta_wrong_rate",  COLORS["judge"],  "LLM judge"),
        ]:
            rows = sorted(
                [r for r in summary if r["prompt_set"] == ps],
                key=lambda r: float(r["alpha"])
            )
            xs = [float(r["alpha"]) for r in rows]
            ys = [float(r[metric]) for r in rows]
            ax.plot(xs, ys, marker="o", linewidth=1.8,
                    color=PS_COLORS[ax_idx], label=ps if ax_idx == 0 else "")

for ax, title in zip(axes, ["String Matcher", "LLM Judge (Qwen2.5-14B-4bit)"]):
    ax.set_xlabel("Scaling factor α", fontsize=11)
    ax.set_ylabel("Δ Wrong-Rate (steered − baseline)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(alphas_all)

axes[0].legend(title="Prompt set", fontsize=8, title_fontsize=9,
               loc="upper left", framealpha=0.7)
fig.suptitle("Deception Effect (Δ Wrong-Rate) vs Scaling Factor α\n"
             "ρ=0.10 · LLaMA-3.1-8B-Instruct",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig2_delta_wrong_vs_alpha.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_delta_wrong_vs_alpha.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — String-matcher vs Judge scatter (Δwrong_rate)
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5.5, 5))

for i, ps in enumerate(PS_ORDER):
    rows = [r for r in summary if r["prompt_set"] == ps]
    xs = [float(r["delta_wrong_rate"])      for r in rows]
    ys = [float(r["judge_delta_wrong_rate"]) for r in rows]
    ax.scatter(xs, ys, color=PS_COLORS[i], label=ps, s=60, zorder=3)

lim_max = 1.05
ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, alpha=0.5, label="y = x")
ax.set_xlim(-0.05, lim_max)
ax.set_ylim(-0.05, lim_max)
ax.set_xlabel("Δ Wrong-Rate (string matcher)", fontsize=11)
ax.set_ylabel("Δ Wrong-Rate (LLM judge)", fontsize=11)
ax.set_title("String Matcher vs LLM Judge\nDeception Effect Agreement", fontsize=11)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax.legend(title="Prompt set", fontsize=8, title_fontsize=9, loc="upper left")
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig3_string_vs_judge_scatter.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_string_vs_judge_scatter.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Fluency metrics
# ─────────────────────────────────────────────────────────────────────────────

variants = [r["variant"] for r in fluency]
labels   = ["Baseline\n(α=0)", "Steered\n(α=1.0)", "Steered\n(α=2.0)"]
metrics  = [
    ("mean_perplexity",  "Perplexity (↓ better)",  True),
    ("mean_distinct_2",  "Distinct-2 (↑ = more diverse)", False),
    ("mean_rep_rate",    "Repetition Rate (↓ better)", True),
]

fig, axes = plt.subplots(1, 3, figsize=(11, 4))

for ax, (col, ylabel, lower_is_better) in zip(axes, metrics):
    vals = [float(r[col]) for r in fluency]
    bars = ax.bar(labels, vals, color=[COLORS["baseline"], COLORS["steered"], "#C44E52"],
                  alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.set_title(ylabel, fontsize=10)
    ax.set_ylabel(col.replace("mean_", "").replace("_", " "), fontsize=9)

    # Annotate bar values
    for bar, v in zip(bars, vals):
        fmt = f"{v:.2f}" if v < 1000 else f"{v:,.0f}"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ax.get_ylim()[1]*0.01,
                fmt, ha="center", va="bottom", fontsize=9)

    # Perplexity: log scale since α=2 is enormous
    if col == "mean_perplexity":
        ax.set_yscale("log")

fig.suptitle("Fluency Metrics: Baseline vs Steered Outputs\n"
             "(20 open-ended prompts · direct prompt set · ρ=0.10)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig4_fluency.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_fluency.png")

print("\nAll figures saved to", FIG_DIR)
