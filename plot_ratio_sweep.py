#!/usr/bin/env python3
"""
Plot a single-dimension sweep from a LieNeuro DID sweep CSV.

Two subplots:
  Left : delta_wrong and coherent_abs (= coherent_rate × delta_wrong) vs x
  Right: wikitext_ppl vs x (log scale)

The x-axis column is auto-detected (lie_truth_ratio → alpha → rho) or set via --x-col.

Usage:
    python plot_ratio_sweep.py \
        --csv outputs/lieneuro_did_sweep_llama_ntruth25_ratiosweep_rho0.02_alpha4.csv \
        [--x-col lie_truth_ratio] [--out outputs/ratio_sweep.png]
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

SURFACE    = "#fcfcfb"
GRID_COLOR = "#e1e0d9"
TEXT_PRI   = "#0b0b0b"
TEXT_SEC   = "#52514e"

C_DELTA    = "#2a78d6"   # blue  — delta_wrong
C_COHERENT = "#eb6834"   # orange — coherent_abs
C_PPL      = "#1baf7a"   # aqua  — wikitext_ppl


def _style_ax(ax, title, ylabel, xlabel="x"):
    ax.set_title(title, fontsize=11, color=TEXT_PRI)
    ax.set_xlabel(xlabel, fontsize=9, color=TEXT_SEC)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SEC)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#c3c2b7")
    ax.set_facecolor(SURFACE)
    ax.grid(True, linewidth=0.5, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to sweep CSV file")
    parser.add_argument("--x-col", default=None,
                        help="CSV column to use as x-axis. "
                             "Auto-detected from lie_truth_ratio → alpha → rho if omitted.")
    parser.add_argument("--out", default=None, help="Output PNG path")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out) if args.out else csv_path.with_suffix(".png")

    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # auto-detect x column
    _auto_order = ["lie_truth_ratio", "alpha", "rho"]
    if args.x_col:
        x_col = args.x_col
    else:
        x_col = next((c for c in _auto_order if c in rows[0]), _auto_order[0])

    rows.sort(key=lambda r: float(r[x_col]))

    ratio       = np.array([float(r[x_col])         for r in rows])
    delta_wrong = np.array([float(r["delta_wrong"])  for r in rows])
    steered_acc = np.array([float(r["steered_acc"])     for r in rows])
    ppl         = np.array([float(r["wikitext_ppl"])    for r in rows])

    def _coherent_abs(r):
        cr = r["coherent_rate"]
        if cr == "" or cr.lower() == "nan":
            return 0.0
        cr_f = float(cr)
        if math.isnan(cr_f):
            return 0.0
        # fraction of ALL questions that are coherently wrong
        return cr_f * (1.0 - float(r["steered_acc"]))

    coherent_abs = np.array([_coherent_abs(r) for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE)

    # ── Left: delta_wrong + coherent_abs ──────────────────────────────────────
    ax1.plot(ratio, delta_wrong,  color=C_DELTA,    linewidth=2,
             marker="o", markersize=5,
             solid_capstyle="round", solid_joinstyle="round", label="delta_wrong")
    ax1.plot(ratio, coherent_abs, color=C_COHERENT, linewidth=2,
             marker="o", markersize=5, linestyle="--",
             solid_capstyle="round", solid_joinstyle="round", label="coherent_abs")

    ax1.set_ylim(bottom=0)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    _style_ax(ax1, f"Error rate vs {x_col}", "Fraction of questions", xlabel=x_col)

    leg1 = [
        mlines.Line2D([], [], color=C_DELTA,    linewidth=2, marker="o", markersize=5,
                      label="delta_wrong  (truly-wrong rate above baseline)"),
        mlines.Line2D([], [], color=C_COHERENT, linewidth=2, marker="o", markersize=5,
                      linestyle="--",
                      label="coherent_abs  (coherently-wrong / all questions)"),
    ]
    ax1.legend(handles=leg1, fontsize=8, frameon=True,
               facecolor=SURFACE, edgecolor="#c3c2b7")

    # ── Right: wikitext_ppl ───────────────────────────────────────────────────
    ax2.plot(ratio, ppl, color=C_PPL, linewidth=2,
             marker="o", markersize=5,
             solid_capstyle="round", solid_joinstyle="round")
    ax2.set_yscale("log")
    ax2.yaxis.set_major_formatter(matplotlib.ticker.LogFormatter(labelOnlyBase=False))
    _style_ax(ax2, f"WikiText PPL vs {x_col}", "Perplexity (log scale)", xlabel=x_col)

    # build a subtitle from the fixed hyperparameters (those not swept)
    _labels = {"rho": "ρ", "alpha": "α", "lie_truth_ratio": "ratio"}
    fixed_parts = [
        f"{_labels[c]}={rows[0][c]}"
        for c in ("rho", "alpha", "lie_truth_ratio")
        if c in rows[0] and c != x_col
    ]
    subtitle = ",  ".join(fixed_parts)
    fig.suptitle(
        f"LieNeuro DID {x_col} sweep — LLaMA-3.1-8B"
        + (f"  ({subtitle})" if subtitle else ""),
        fontsize=11, y=1.02, color=TEXT_PRI,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
