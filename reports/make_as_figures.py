#!/usr/bin/env python3
"""Generate figures for the activation-steering experiment report.

Produces:
  fig11_as_heatmap.png       — Δwrong heatmap (layer × coeff)
  fig12_as_layer_sweep.png   — Δwrong vs layer for each coefficient
  fig13_as_vs_lieneuro.png   — Method comparison scatter (Δwrong vs perplexity)
  fig14_as_fluency.png       — Fluency metrics for top configs vs baseline

Usage
-----
  python reports/make_as_figures.py
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FIG_DIR  = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_DIR  = Path(__file__).parent.parent / "outputs"

SWEEP_CSV   = OUT_DIR / "activation_steering_sweep.csv"
FLUENCY_CSV = OUT_DIR / "activation_steering_fluency.csv"

COEFFS = [5.0, 10.0, 20.0, 40.0, 80.0]
N_LAYERS = 32

COEFF_COLORS = {
    5.0:  "#4C72B0",
    10.0: "#DD8452",
    20.0: "#55A868",
    40.0: "#C44E52",
    80.0: "#9467BD",
}


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def sweep_matrix():
    rows = load_csv(SWEEP_CSV)
    mat = np.full((N_LAYERS, len(COEFFS)), np.nan)
    coeff_idx = {c: i for i, c in enumerate(COEFFS)}
    for r in rows:
        layer = int(r["layer"])
        coeff = float(r["coeff"])
        dw    = float(r["delta_wrong_rate"])
        ci = coeff_idx.get(coeff)
        if ci is not None and 0 <= layer < N_LAYERS:
            mat[layer, ci] = dw
    return mat


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Heatmap: Δwrong-rate across layer × coeff
# ─────────────────────────────────────────────────────────────────────────────

def fig11_heatmap(mat):
    fig, ax = plt.subplots(figsize=(8, 10))

    vmax = max(abs(mat[~np.isnan(mat)]).max(), 0.01)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, origin="upper")

    ax.set_xticks(range(len(COEFFS)))
    ax.set_xticklabels([f"c={c:.0f}" for c in COEFFS], fontsize=10)
    ax.set_yticks(range(N_LAYERS))
    ax.set_yticklabels([f"Layer {i}" for i in range(N_LAYERS)], fontsize=7.5)
    ax.set_xlabel("Steering coefficient", fontsize=11)
    ax.set_ylabel("Transformer layer", fontsize=11)
    ax.set_title("Activation Steering: Δ Wrong-Rate Heatmap\n"
                 "(Red = more deception · Blue = less)\n"
                 "LLaMA-3.1-8B-Instruct · direct prompt set",
                 fontsize=11)

    cb = plt.colorbar(im, ax=ax, shrink=0.55,
                      format=mticker.PercentFormatter(xmax=1))
    cb.set_label("Δ Wrong-Rate", fontsize=9)

    for li in range(N_LAYERS):
        for ci in range(len(COEFFS)):
            v = mat[li, ci]
            if not np.isnan(v):
                fc = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(ci, li, f"{v:+.0%}", ha="center", va="center",
                        fontsize=5.5, color=fc)

    fig.tight_layout()
    out = FIG_DIR / "fig11_as_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Δwrong-rate vs layer for each coefficient
# ─────────────────────────────────────────────────────────────────────────────

def fig12_layer_sweep(mat):
    fig, ax = plt.subplots(figsize=(10, 5))

    for ci, coeff in enumerate(COEFFS):
        ys = mat[:, ci]
        ax.plot(range(N_LAYERS), ys,
                color=COEFF_COLORS[coeff], linewidth=1.6,
                marker="o", markersize=3,
                label=f"c = {coeff:.0f}")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(0.14, color="grey", linestyle="--", linewidth=0.9, alpha=0.7,
               label="LieNeuro best (+14%)")

    ax.set_xlabel("Transformer layer index", fontsize=11)
    ax.set_ylabel("Δ Wrong-Rate (steered − baseline)", fontsize=10)
    ax.set_title("Activation Steering: Deception Effect by Layer\n"
                 "LLaMA-3.1-8B-Instruct · direct prompt set",
                 fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks(range(0, N_LAYERS, 4))
    ax.legend(fontsize=9, framealpha=0.85, ncol=2)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)

    fig.tight_layout()
    out = FIG_DIR / "fig12_as_layer_sweep.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — Method comparison scatter: Δwrong vs perplexity
# ─────────────────────────────────────────────────────────────────────────────

# Reference LieNeuro points: (label, delta_wrong, perplexity)
LIENEURO_REFS = [
    ("LieNeuro\nρ=0.10 α=1.0",  0.14,  9.57),
    ("LieNeuro\nρ=0.10 α=2.0",  0.86,  303250.0),
    ("LieNeuro\nρ=0.01 α=1.0", -0.04,  6.15),
    ("LieNeuro\nρ=0.01 α=2.0",  0.16,  149.99),
    ("LieNeuro\nρ=0.01 α=4.0",  0.90,  62455.0),
    ("LieNeuro\nρ=0.05 α=1.0", -0.04,  6.94),
    ("LieNeuro\nρ=0.05 α=2.0",  0.88,  28104.0),
]


def fig13_vs_lieneuro():
    if not FLUENCY_CSV.exists():
        print("Skipping fig13: fluency CSV not yet available.")
        return

    flu_rows = load_csv(FLUENCY_CSV)
    baseline_row = next(r for r in flu_rows if r["variant"] == "baseline")
    bl_ppl  = float(baseline_row["mean_perplexity"])
    as_rows = [r for r in flu_rows if r["variant"] != "baseline"]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Baseline
    ax.scatter([0], [bl_ppl], marker="*", s=200, color="black",
               zorder=6, label="Baseline")

    # LieNeuro reference points
    ln_dw  = [p[1] for p in LIENEURO_REFS]
    ln_ppl = [p[2] for p in LIENEURO_REFS]
    ax.scatter(ln_dw, ln_ppl, marker="D", s=80, color="#DD8452",
               zorder=5, label="LieNeuro (weight steering)")
    for label, dw, ppl in LIENEURO_REFS:
        if ppl < 1e4:
            ax.annotate(label, (dw, ppl), textcoords="offset points",
                        xytext=(5, 4), fontsize=6.5, color="#8B4513", alpha=0.85)

    # Activation steering top configs
    as_dw  = [float(r["delta_wrong_rate"]) for r in as_rows]
    as_ppl = [float(r["mean_perplexity"])  for r in as_rows]
    sc = ax.scatter(as_dw, as_ppl, marker="o", s=70, color="#4C72B0",
                    zorder=5, label="Activation steering (top configs)")
    for r, dw, ppl in zip(as_rows, as_dw, as_ppl):
        layer = r["layer"]
        coeff = float(r["coeff"])
        if ppl < 1e4:
            ax.annotate(f"L{layer} c={coeff:.0f}", (dw, ppl),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color="#1f3f7a", alpha=0.85)

    ax.set_xlabel("Δ Wrong-Rate (steered − baseline)", fontsize=11)
    ax.set_ylabel("Perplexity (log scale, ↓ better)", fontsize=11)
    ax.set_title("Deception vs Fluency:\nActivation Steering vs LieNeuro Weight Steering",
                 fontsize=12)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axvline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=9, framealpha=0.85)

    fig.tight_layout()
    out = FIG_DIR / "fig13_as_vs_lieneuro.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14 — Fluency comparison for top configs
# ─────────────────────────────────────────────────────────────────────────────

def fig14_fluency():
    if not FLUENCY_CSV.exists():
        print("Skipping fig14: fluency CSV not yet available.")
        return

    rows = load_csv(FLUENCY_CSV)
    labels    = [r["variant"] for r in rows]
    ppl       = [float(r["mean_perplexity"]) for r in rows]
    rep_rate  = [float(r["mean_rep_rate"])   for r in rows]
    distinct2 = [float(r["mean_distinct_2"]) for r in rows]

    n = len(labels)
    short_labels = []
    for r in rows:
        if r["variant"] == "baseline":
            short_labels.append("Baseline")
        else:
            short_labels.append(f"L{r['layer']}\nc={float(r['coeff']):.0f}")

    x = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, vals, title, ylabel, fmt in [
        (axes[0], ppl,       "Perplexity (↓ better)",    "Perplexity",      "{:.1f}"),
        (axes[1], rep_rate,  "Repetition Rate (↓ better)", "Rep. Rate",     "{:.3f}"),
        (axes[2], distinct2, "Distinct-2 (↑ better)",    "Distinct-2",      "{:.3f}"),
    ]:
        colors = ["#4C72B0" if short_labels[i] == "Baseline" else "#DD8452"
                  for i in range(n)]
        bars = ax.bar(x, vals, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=7.5)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(v), ha="center", va="bottom", fontsize=6.5)

    fig.suptitle("Fluency Metrics: Baseline vs Activation Steering Top Configs\n"
                 "LLaMA-3.1-8B-Instruct · direct prompt set",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig14_as_fluency.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading sweep data …")
    mat = sweep_matrix()
    n_rows = np.sum(~np.isnan(mat))
    print(f"  {n_rows} / {N_LAYERS * len(COEFFS)} configs loaded")

    fig11_heatmap(mat)
    fig12_layer_sweep(mat)
    fig13_vs_lieneuro()
    fig14_fluency()

    print(f"\nAll figures saved to {FIG_DIR}")
