#!/usr/bin/env python3
"""
Side-by-side heatmaps: steered wrong rate + WikiText PPL (log₁₀).

  python plot_sweep_heatmap.py [--csv  outputs/lieneuro_did_sweep.csv]
                               [--ppl  outputs/wikitext_ppl.csv]
                               [--out  outputs/sweep_heatmap.png]
                               [--mode set_diff|score_diff|both]
"""

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── grid definition ────────────────────────────────────────────────────────
RHO_LIST   = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
ALPHA_LIST = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0]

# ── palettes ───────────────────────────────────────────────────────────────
CMAP_WRONG = "YlOrRd"   # light cream → deep crimson  (high = bad)
CMAP_PPL   = "YlOrBr"   # light yellow → dark brown   (high log PPL = bad)

# log₁₀ PPL color range; baseline Llama PPL ≈ 9.15 → log10 ≈ 0.96
PPL_LOG_MIN = 0.9   # ≈ 8 PPL  (near-baseline)
PPL_LOG_MAX = 6.0   # ≈ 1e6 PPL (catastrophic)

# ── typography ─────────────────────────────────────────────────────────────
TITLE_FS  = 13
LABEL_FS  = 11
TICK_FS   = 9
ANNOT_FS  = 8.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sweep(csv_path: Path, mode: str) -> pd.DataFrame:
    """Load sweep CSV (variable-width rows), return deduplicated DataFrame."""
    all_cols = ["mode", "rho", "alpha", "steered_acc", "delta_wrong",
                "mask_coverage", "coherent_rate", "mean_wrong_len", "mean_cos_sim",
                "wikitext_ppl"]
    rows = []
    with open(csv_path) as fh:
        for i, line in enumerate(fh):
            parts = line.rstrip("\n").split(",")
            if i == 0:
                continue
            if len(parts) < 4:
                continue
            parts += [float("nan")] * (len(all_cols) - len(parts))
            rows.append(parts[:len(all_cols)])
    df = pd.DataFrame(rows, columns=all_cols)
    for col in ["rho", "alpha", "steered_acc", "wikitext_ppl"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["mode", "rho", "alpha", "steered_acc"])
    df["wrong_rate"] = 1.0 - df["steered_acc"]
    df = df.drop_duplicates(subset=["mode", "rho", "alpha"], keep="last")
    return df[df["mode"] == mode]


def make_wrong_pivot(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[["rho", "alpha", "wrong_rate"]]
    pivot = sub.pivot(index="alpha", columns="rho", values="wrong_rate")
    pivot = pivot.reindex(index=ALPHA_LIST, columns=RHO_LIST)
    pivot.sort_index(ascending=False, inplace=True)
    return pivot


def make_ppl_pivot(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[["rho", "alpha", "wikitext_ppl"]].dropna(subset=["wikitext_ppl"])
    sub = sub.copy()
    sub["log10_ppl"] = np.log10(sub["wikitext_ppl"].clip(lower=1e-9))
    pivot = sub.pivot(index="alpha", columns="rho", values="log10_ppl")
    pivot = pivot.reindex(index=ALPHA_LIST, columns=RHO_LIST)
    pivot.sort_index(ascending=False, inplace=True)
    return pivot


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_grid(ax, nr, nc):
    for x in np.arange(-0.5, nc, 1):
        ax.axvline(x, color="white", lw=1.0)
    for y in np.arange(-0.5, nr, 1):
        ax.axhline(y, color="white", lw=1.0)


def draw_wrong_heatmap(ax, pivot, fig):
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap=CMAP_WRONG, vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    nr, nc = data.shape
    _draw_grid(ax, nr, nc)

    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{r:g}" for r in pivot.columns], fontsize=TICK_FS)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([f"{a:g}" for a in pivot.index], fontsize=TICK_FS)

    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v > 0.55 else "#222222"
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    fontsize=ANNOT_FS, color=txt_color, fontweight="bold")

    ax.set_xlabel(r"$\rho$  (mask density)", fontsize=LABEL_FS, labelpad=6)
    ax.set_ylabel(r"$\alpha$  (steering scale)", fontsize=LABEL_FS, labelpad=6)
    ax.set_title("Wrong rate", fontsize=TITLE_FS, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    cbar.ax.tick_params(labelsize=TICK_FS)


def draw_ppl_heatmap(ax, pivot, fig):
    data = pivot.values.astype(float)
    im = ax.imshow(data, cmap=CMAP_PPL, vmin=PPL_LOG_MIN, vmax=PPL_LOG_MAX,
                   aspect="auto", interpolation="nearest")
    nr, nc = data.shape
    _draw_grid(ax, nr, nc)

    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{r:g}" for r in pivot.columns], fontsize=TICK_FS)
    ax.set_yticks(range(nr))
    ax.set_yticklabels([f"{a:g}" for a in pivot.index], fontsize=TICK_FS)

    for i in range(nr):
        for j in range(nc):
            v = data[i, j]
            if np.isnan(v):
                continue
            ppl = 10 ** v
            label = f"{ppl:.0f}" if ppl < 1000 else f"{ppl:.1e}"
            mid = (PPL_LOG_MIN + PPL_LOG_MAX) / 2
            txt_color = "white" if v > mid else "#222222"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=ANNOT_FS - 0.5, color=txt_color, fontweight="bold")

    ax.set_xlabel(r"$\rho$  (mask density)", fontsize=LABEL_FS, labelpad=6)
    ax.set_ylabel(r"$\alpha$  (steering scale)", fontsize=LABEL_FS, labelpad=6)
    ax.set_title(r"WikiText PPL  (log$_{10}$ scale)", fontsize=TITLE_FS, pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\log_{10}$ PPL", fontsize=TICK_FS)
    cbar.ax.tick_params(labelsize=TICK_FS)


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def plot(csv_path: Path, out_path: Path, mode: str):
    df          = load_sweep(csv_path, mode)
    wrong_pivot = make_wrong_pivot(df)
    ppl_pivot   = make_ppl_pivot(df)
    has_ppl     = ppl_pivot.notna().any().any()

    n_panels = 2 if has_ppl else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.2 * n_panels, 4.8),
                             constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    draw_wrong_heatmap(axes[0], wrong_pivot, fig)
    if has_ppl:
        draw_ppl_heatmap(axes[1], ppl_pivot, fig)

    fig.suptitle("LieNeuro DID sweep — score_diff mask",
                 fontsize=TITLE_FS + 1, fontweight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",  default="outputs/lieneuro_did_sweep_llama.csv")
    ap.add_argument("--out",  default="outputs/sweep_heatmap.png")
    ap.add_argument("--mode", default="score_diff",
                    choices=["set_diff", "score_diff"])
    args = ap.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    plot(Path(args.csv), Path(args.out), args.mode)


if __name__ == "__main__":
    main()
