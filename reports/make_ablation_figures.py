#!/usr/bin/env python3
"""Generate ablation study figures for LieNeuro DID.

Two ablation studies, each producing a 4-panel figure:
  1. ρ sweep at α=4   → fig_ablation_rho_sweep_a4.png
  2. α sweep at ρ=0.04 → fig_ablation_alpha_sweep_rho0.04.png

Each figure has 4 panels (x-axis = varying hyperparameter):
  - Wrong rate
  - WikiText-2 PPL
  - Coherent rate + cosine similarity
  - Mean wrong-answer length
"""

import csv, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

QUALITY_CSV  = Path("outputs/deception_quality.csv")
WIKITEXT_CSV = Path("outputs/wikitext_ppl.csv")
OUT_DIR      = Path("reports/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── colours ──────────────────────────────────────────────────────────────────
C_WRONG   = "#e63946"
C_PPL     = "#457b9d"
C_COHERENT= "#2a9d8f"
C_COSSIM  = "#e9c46a"
C_LEN     = "#f4a261"


def load_quality(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({
                "rho":           float(r["rho"]),
                "alpha":         float(r["alpha"]),
                "wrong_rate":    float(r["wrong_rate"]),
                "wikitext_ppl":  float(r["wikitext_ppl"]) if r["wikitext_ppl"] not in ("", "nan") else float("nan"),
                "mean_wrong_len":float(r["mean_wrong_len"]),
                "mean_cos_sim":  float(r["mean_cos_sim"]),
                "coherent_rate": float(r["coherent_rate"]) if r["coherent_rate"] not in ("", "nan") else float("nan"),
            })
    return rows


def make_figure(rows, x_vals, x_label, x_log, title, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def get(key):
        return [next((r[key] for r in rows if r["_x"] == x), float("nan"))
                for x in x_vals]

    wrong   = get("wrong_rate")
    ppl     = get("wikitext_ppl")
    coh     = get("coherent_rate")
    cos     = get("mean_cos_sim")
    length  = get("mean_wrong_len")

    marker = "o"
    ms = 7

    # ── Panel 1: wrong rate + coherent rate + cosine similarity ───────────────
    ax = axes[0]
    l1, = ax.plot(x_vals, [w * 100 for w in wrong], color=C_WRONG, marker=marker,
                  markersize=ms, linewidth=2, label="Wrong rate")
    valid_xc = [x for x, c in zip(x_vals, coh) if not math.isnan(c)]
    valid_c  = [c * 100 for c in coh if not math.isnan(c)]
    l2, = ax.plot(valid_xc, valid_c, color=C_COHERENT, marker="s", markersize=ms,
                  linewidth=2, linestyle="--", label="Coherent rate")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%g%%"))

    ax2 = ax.twinx()
    l3, = ax2.plot(x_vals, cos, color=C_COSSIM, marker="^", markersize=ms,
                   linewidth=2, linestyle=":", label="Cosine similarity")
    # ax2.set_ylabel("Cosine similarity")
    ax2.set_ylim(0, 1.0)

    ax.set_title("Deception quality")
    if x_log:
        ax.set_xscale("log"); ax2.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.legend(handles=[l1, l2, l3], loc="center left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: WikiText-2 PPL ───────────────────────────────────────────────
    ax = axes[1]
    valid_x = [x for x, p in zip(x_vals, ppl) if not math.isnan(p)]
    valid_p = [p for p in ppl if not math.isnan(p)]
    ax.plot(valid_x, valid_p, color=C_PPL, marker=marker, markersize=ms, linewidth=2)
    ax.set_ylabel("Perplexity")
    ax.set_yscale("log")
    ax.set_title("Fluency (WikiText-2 PPL)")
    if x_log: ax.set_xscale("log")
    ax.set_xlabel(x_label)
    # ax.grid(True, alpha=0.3, which="both")
    ax.grid(True, alpha=0.3)

    # ── Panel 3: mean wrong-answer length ─────────────────────────────────────
    ax = axes[2]
    ax.plot(x_vals, length, color=C_LEN, marker=marker, markersize=ms, linewidth=2)
    ax.set_ylabel("Mean words in wrong answer")
    ax.set_title("Wrong-answer length")
    if x_log: ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    rows = load_quality(QUALITY_CSV)

    # ── Ablation 1: ρ sweep at α=4 ────────────────────────────────────────────
    rho_rows = [r for r in rows if abs(r["alpha"] - 4.0) < 0.01]
    rho_rows.sort(key=lambda r: r["rho"])
    for r in rho_rows:
        r["_x"] = r["rho"]
    rho_vals = [r["rho"] for r in rho_rows]

    make_figure(
        rows=rho_rows,
        x_vals=rho_vals,
        x_label="ρ (importance threshold)",
        x_log=True,
        title="Ablation: ρ sweep at α = 4  (LieNeuro DID score_diff)",
        out_path=OUT_DIR / "fig_ablation_rho_sweep_a4.png",
    )

    # ── Ablation 2: α sweep at ρ=0.04 ────────────────────────────────────────
    alpha_rows = [r for r in rows if abs(r["rho"] - 0.04) < 0.001]
    alpha_rows.sort(key=lambda r: r["alpha"])
    for r in alpha_rows:
        r["_x"] = r["alpha"]
    alpha_vals = [r["alpha"] for r in alpha_rows]

    make_figure(
        rows=alpha_rows,
        x_vals=alpha_vals,
        x_label="α (steering scale)",
        x_log=False,
        title="Ablation: α sweep at ρ = 0.04  (LieNeuro DID score_diff)",
        out_path=OUT_DIR / "fig_ablation_alpha_sweep_rho0.04.png",
    )


if __name__ == "__main__":
    main()
