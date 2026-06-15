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

SWEEP_CSV         = OUT_DIR / "activation_steering_sweep.csv"
FLUENCY_CSV       = OUT_DIR / "activation_steering_fluency.csv"
DID_FLUENCY_CSV   = OUT_DIR / "as_did_fluency.csv"
WIKITEXT_CSV      = OUT_DIR / "wikitext_ppl.csv"
INDIST_CSV        = OUT_DIR / "as_indist_eval.csv"
DID_METRIC_CSV    = OUT_DIR / "as_did_metric_eval.csv"

COEFFS = [0.1, 1.0, 5.0, 10.0, 20.0, 40.0, 80.0]
N_LAYERS = 32

COEFF_COLORS = {
    0.1:  "#17BECF",
    1.0:  "#8C564B",
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

    # DID fluency (optional)
    did_rows = []
    if DID_FLUENCY_CSV.exists():
        did_flu = load_csv(DID_FLUENCY_CSV)
        did_rows = [r for r in did_flu if r["variant"] != "baseline"]

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

    # Activation steering (naive) top configs
    as_dw  = [float(r["delta_wrong_rate"]) for r in as_rows]
    as_ppl = [float(r["mean_perplexity"])  for r in as_rows]
    ax.scatter(as_dw, as_ppl, marker="o", s=70, color="#4C72B0",
               zorder=5, label="Activation steering — naive (top configs)")
    for r, dw, ppl in zip(as_rows, as_dw, as_ppl):
        layer = r["layer"]
        coeff = float(r["coeff"])
        if ppl < 1e4:
            ax.annotate(f"L{layer} c={coeff:.0f}", (dw, ppl),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color="#1f3f7a", alpha=0.85)

    # DID steering top configs
    if did_rows:
        did_dw  = [float(r["delta_wrong_rate"]) for r in did_rows]
        did_ppl = [float(r["mean_perplexity"])  for r in did_rows]
        ax.scatter(did_dw, did_ppl, marker="^", s=70, color="#2ca02c",
                   zorder=5, label="Activation steering — DID (top configs)")
        for r, dw, ppl in zip(did_rows, did_dw, did_ppl):
            layer = r["layer"]
            coeff = float(r["coeff"])
            if ppl < 1e4:
                ax.annotate(f"DID L{layer} c={coeff:.0f}", (dw, ppl),
                            textcoords="offset points", xytext=(5, -10),
                            fontsize=6.5, color="#1a6b1a", alpha=0.85)

    ax.set_xlabel("Δ Wrong-Rate (steered − baseline)", fontsize=11)
    ax.set_ylabel("Perplexity (log scale, ↓ better)", fontsize=11)
    ax.set_title("Deception vs Fluency:\nNaive vs DID Activation Steering vs LieNeuro",
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
# Figure 15 — WikiText-2 PPL vs Δwrong (model-level fluency benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def fig15_wikitext_ppl():
    if not WIKITEXT_CSV.exists():
        print("Skipping fig15: wikitext_ppl.csv not yet available.")
        return

    rows = load_csv(WIKITEXT_CSV)
    baseline = next((r for r in rows if r["method"] == "none"), None)
    bl_ppl = float(baseline["wikitext_ppl"]) if baseline else None

    as_rows = [r for r in rows if r["method"] == "activation_steering"]
    ln_rows = [r for r in rows if r["method"] == "weight_steering"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if bl_ppl is not None:
        ax.scatter([0], [bl_ppl], marker="*", s=200, color="black",
                   zorder=6, label="Baseline")
        ax.annotate("Baseline", (0, bl_ppl), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)

    if ln_rows:
        ln_dw  = [float(r["delta_wrong"]) for r in ln_rows if r["delta_wrong"]]
        ln_ppl = [float(r["wikitext_ppl"]) for r in ln_rows if r["delta_wrong"]]
        ln_lbl = [r["variant"] for r in ln_rows if r["delta_wrong"]]
        ax.scatter(ln_dw, ln_ppl, marker="D", s=80, color="#DD8452",
                   zorder=5, label="LieNeuro (weight steering)")
        for label, dw, ppl in zip(ln_lbl, ln_dw, ln_ppl):
            short = label.replace("LieNeuro ", "")
            ax.annotate(short, (dw, ppl), textcoords="offset points",
                        xytext=(5, 4), fontsize=6.5, color="#8B4513", alpha=0.85)

    if as_rows:
        as_dw  = [float(r["delta_wrong"]) for r in as_rows]
        as_ppl = [float(r["wikitext_ppl"]) for r in as_rows]
        ax.scatter(as_dw, as_ppl, marker="o", s=70, color="#4C72B0",
                   zorder=5, label="Activation steering — naive")
        for r, dw, ppl in zip(as_rows, as_dw, as_ppl):
            layer = r["layer"]
            coeff = float(r["coeff"])
            ax.annotate(f"L{layer} c={coeff:.0f}", (dw, ppl),
                        textcoords="offset points", xytext=(5, 4),
                        fontsize=6.5, color="#1f3f7a", alpha=0.85)

    did_wt_rows = [r for r in rows if r["method"] == "did_steering"]
    if did_wt_rows:
        did_dw  = [float(r["delta_wrong"]) for r in did_wt_rows if r["delta_wrong"]]
        did_ppl = [float(r["wikitext_ppl"]) for r in did_wt_rows if r["delta_wrong"]]
        did_lbl = [r["variant"] for r in did_wt_rows if r["delta_wrong"]]
        ax.scatter(did_dw, did_ppl, marker="^", s=70, color="#2ca02c",
                   zorder=5, label="Activation steering — DID")
        for label, dw, ppl in zip(did_lbl, did_dw, did_ppl):
            ax.annotate(label.replace("DID ", ""), (dw, ppl),
                        textcoords="offset points", xytext=(5, -10),
                        fontsize=6.5, color="#1a6b1a", alpha=0.85)

    ax.set_xlabel("Δ Wrong-Rate (steered − baseline)", fontsize=11)
    ax.set_ylabel("WikiText-2 Perplexity (log scale, ↓ better)", fontsize=11)
    ax.set_title("Deception vs Fluency (WikiText-2 Benchmark):\nNaive vs DID Activation Steering vs LieNeuro",
                 fontsize=12)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axvline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=9, framealpha=0.85)

    fig.tight_layout()
    out = FIG_DIR / "fig15_wikitext_ppl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 16 — In-distribution vs OOD Δwrong heatmaps (25-pair vectors)
# ─────────────────────────────────────────────────────────────────────────────

def fig16_indist_ood():
    if not INDIST_CSV.exists():
        print("Skipping fig16: as_indist_eval.csv not yet available.")
        return

    rows = load_csv(INDIST_CSV)
    coeffs_present = sorted(set(float(r["coeff"]) for r in rows))
    coeff_idx = {c: i for i, c in enumerate(coeffs_present)}

    mat_in  = np.full((N_LAYERS, len(coeffs_present)), np.nan)
    mat_ood = np.full((N_LAYERS, len(coeffs_present)), np.nan)
    for r in rows:
        layer = int(r["layer"])
        ci    = coeff_idx.get(float(r["coeff"]))
        if ci is not None:
            mat_in[layer, ci]  = float(r["indist_delta_wrong"])
            mat_ood[layer, ci] = float(r["ood_delta_wrong"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 10))
    col_labels = [f"c={c:.0f}" for c in coeffs_present]

    for ax, mat, title in [
        (axes[0], mat_in,  "In-Distribution Δwrong\n(25 construction questions)"),
        (axes[1], mat_ood, "Out-of-Distribution Δwrong\n(10 held-out questions)"),
    ]:
        vmax = max(abs(mat[~np.isnan(mat)]).max(), 0.01)
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(coeffs_present)))
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticks(range(N_LAYERS))
        ax.set_yticklabels([f"L{i}" for i in range(N_LAYERS)], fontsize=7)
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.set_ylabel("Layer", fontsize=10)
        ax.set_title(title, fontsize=11)
        plt.colorbar(im, ax=ax, shrink=0.55,
                     format=mticker.PercentFormatter(xmax=1))
        for li in range(N_LAYERS):
            for ci in range(len(coeffs_present)):
                v = mat[li, ci]
                if not np.isnan(v):
                    fc = "white" if abs(v) > 0.5 * vmax else "black"
                    ax.text(ci, li, f"{v:+.0%}", ha="center", va="center",
                            fontsize=5, color=fc)

    fig.suptitle("Activation Steering (25-pair vectors): In-Dist vs OOD Δwrong\n"
                 "LLaMA-3.1-8B-Instruct · direct prompt set",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig16_indist_ood.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 17 — DiD specificity analysis: target vs random-control Δwrong
# ─────────────────────────────────────────────────────────────────────────────

def fig17_did_specificity():
    if not DID_METRIC_CSV.exists():
        print("Skipping fig17: as_did_metric_eval.csv not yet available.")
        return

    rows = load_csv(DID_METRIC_CSV)
    if not rows:
        print("Skipping fig17: no data in as_did_metric_eval.csv.")
        return

    # Build per-row label and sort by target_dw descending
    def row_label(r):
        vset = r.get("vector_set", "")
        tag  = "DID" if "did" in vset.lower() else "naive"
        return f"{tag} L{r['layer']} c={float(r['coeff']):.0f}"

    rows_sorted = sorted(rows, key=lambda r: -float(r["target_dw"]))
    labels      = [row_label(r) for r in rows_sorted]
    target_dw   = [float(r["target_dw"])      for r in rows_sorted]
    ctrl_mean   = [float(r["control_mean_dw"]) for r in rows_sorted]
    did_dw      = [float(r["did_dw"])          for r in rows_sorted]
    ctrl_std    = [float(r["control_std_dw"])  for r in rows_sorted]

    n   = len(labels)
    x   = np.arange(n)
    w   = 0.26

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 5))

    bars_t = ax.bar(x - w, target_dw, w, color="#4C72B0", label="Target direction Δwrong")
    bars_c = ax.bar(x,     ctrl_mean, w, color="#DD8452", label="Random control mean Δwrong",
                    yerr=ctrl_std, capsize=4, error_kw={"elinewidth": 1.2})
    bars_d = ax.bar(x + w, did_dw,   w, color="#2ca02c", label="DID net specific Δwrong")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylabel("Δ Wrong-Rate", fontsize=11)
    ax.set_title("DiD Specificity Test: Is Deception Direction-Specific?\n"
                 "Target vector vs matched-displacement random controls (K=5)",
                 fontsize=11)
    ax.legend(fontsize=8.5, framealpha=0.85)

    # Annotate DID bars
    for bar, v in zip(bars_d, did_dw):
        if abs(v) > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003 * np.sign(v),
                    f"{v:+.0%}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7, color="#1a6b1a")

    fig.tight_layout()
    out = FIG_DIR / "fig17_did_specificity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def fig15_wikitext_ppl_with_did_correction():
    """Update fig15 to overlay DID-corrected Δwrong as horizontal arrows."""
    if not WIKITEXT_CSV.exists():
        print("Skipping fig15 DID update: wikitext_ppl.csv not available.")
        return
    if not DID_METRIC_CSV.exists():
        return   # plain fig15 already produced; DID correction not available yet

    rows     = load_csv(WIKITEXT_CSV)
    did_rows = load_csv(DID_METRIC_CSV)

    # Build lookup: (vector_set, layer, coeff) → did_dw
    did_map = {}
    for r in did_rows:
        key = (r.get("vector_set", ""), str(r["layer"]), float(r["coeff"]))
        did_map[key] = float(r["did_dw"])

    baseline = next((r for r in rows if r["method"] == "none"), None)
    bl_ppl = float(baseline["wikitext_ppl"]) if baseline else None

    as_rows  = [r for r in rows if r["method"] == "activation_steering"]
    ln_rows  = [r for r in rows if r["method"] == "weight_steering"]
    did_wt_rows = [r for r in rows if r["method"] == "did_steering"]

    fig, ax = plt.subplots(figsize=(8, 6))

    if bl_ppl is not None:
        ax.scatter([0], [bl_ppl], marker="*", s=200, color="black",
                   zorder=6, label="Baseline")
        ax.annotate("Baseline", (0, bl_ppl), textcoords="offset points",
                    xytext=(6, 4), fontsize=7.5)

    if ln_rows:
        ln_dw  = [float(r["delta_wrong"]) for r in ln_rows if r["delta_wrong"]]
        ln_ppl = [float(r["wikitext_ppl"]) for r in ln_rows if r["delta_wrong"]]
        ln_lbl = [r["variant"] for r in ln_rows if r["delta_wrong"]]
        ax.scatter(ln_dw, ln_ppl, marker="D", s=80, color="#DD8452",
                   zorder=5, label="LieNeuro (weight steering)")
        for label, dw, ppl in zip(ln_lbl, ln_dw, ln_ppl):
            short = label.replace("LieNeuro ", "")
            ax.annotate(short, (dw, ppl), textcoords="offset points",
                        xytext=(5, 4), fontsize=6.5, color="#8B4513", alpha=0.85)

    if as_rows:
        as_dw  = [float(r["delta_wrong"]) for r in as_rows]
        as_ppl = [float(r["wikitext_ppl"]) for r in as_rows]
        ax.scatter(as_dw, as_ppl, marker="o", s=70, color="#4C72B0",
                   zorder=5, label="Activation steering — naive")
        for r, dw, ppl in zip(as_rows, as_dw, as_ppl):
            # Check if we have a DID correction for this config
            did_val = did_map.get(("naive", str(r["layer"]), float(r["coeff"])))
            if did_val is not None:
                ax.annotate("", xy=(did_val, ppl), xytext=(dw, ppl),
                            arrowprops=dict(arrowstyle="->", color="#1f3f7a",
                                            lw=1.2, linestyle="dashed"))
                ax.scatter([did_val], [ppl], marker="o", s=40, color="#1f3f7a",
                           zorder=6, alpha=0.7)
            else:
                ax.annotate(f"L{r['layer']} c={float(r['coeff']):.0f}", (dw, ppl),
                            textcoords="offset points", xytext=(5, 4),
                            fontsize=6.5, color="#1f3f7a", alpha=0.85)

    if did_wt_rows:
        did_wt_valid = [r for r in did_wt_rows if r["delta_wrong"]]
        d_dw  = [float(r["delta_wrong"]) for r in did_wt_valid]
        d_ppl = [float(r["wikitext_ppl"]) for r in did_wt_valid]
        ax.scatter(d_dw, d_ppl, marker="^", s=70, color="#2ca02c",
                   zorder=5, label="Activation steering — DID")
        for r, dw, ppl in zip(did_wt_valid, d_dw, d_ppl):
            did_val = did_map.get(("did", str(r["layer"]), float(r["coeff"])))
            if did_val is not None:
                ax.annotate("", xy=(did_val, ppl), xytext=(dw, ppl),
                            arrowprops=dict(arrowstyle="->", color="#1a6b1a",
                                            lw=1.2, linestyle="dashed"))
                ax.scatter([did_val], [ppl], marker="^", s=40, color="#1a6b1a",
                           zorder=6, alpha=0.7)
            else:
                if ppl < 1e4:
                    ax.annotate(r["variant"].replace("did_", ""), (dw, ppl),
                                textcoords="offset points", xytext=(5, -10),
                                fontsize=6.5, color="#1a6b1a", alpha=0.85)

    ax.set_xlabel("Δ Wrong-Rate (steered − baseline)", fontsize=11)
    ax.set_ylabel("WikiText-2 Perplexity (log scale, ↓ better)", fontsize=11)
    ax.set_title("Deception vs Fluency (WikiText-2):\nDashed arrows show DID-corrected Δwrong "
                 "(net specific effect after subtracting random-direction drift)",
                 fontsize=10)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.axvline(0, color="black", linewidth=0.6)
    ax.legend(fontsize=9, framealpha=0.85)

    fig.tight_layout()
    out = FIG_DIR / "fig15_wikitext_ppl.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out} (with DID correction arrows)")


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
    fig15_wikitext_ppl_with_did_correction()
    fig16_indist_ood()
    fig17_did_specificity()

    print(f"\nAll figures saved to {FIG_DIR}")
