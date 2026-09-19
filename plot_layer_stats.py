#!/usr/bin/env python3
"""
Plot per-layer importance score statistics for S_lie, S_truth, and DID scores.

Figure 1 — S_lie vs S_truth: 3 subplots (all / attention / MLP).
  Median solid line; bands extend upward to p90, p96, p99.

Figure 2 — DID score (w·S_lie − (1−w)·S_truth): 3 subplots (all / attention / MLP).
  Median solid line; symmetric bands extend to ±p90, ±p96, ±p99 from the median.
  w = --w  (default 0.5).

Usage:
    python plot_layer_stats.py \\
        [--scores outputs/scores_llama_8b_ntruth25.pt] \\
        [--out outputs/layer_stats.png] \\
        [--out-did outputs/layer_stats_did.png] \\
        [--w 0.286] \\
        [--n-truth 25]
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# ── Palette (validated reference: slot 1 = blue, slot 2 = orange, slot 3 = aqua) ─
LIE_COLOR   = "#2a78d6"
TRUTH_COLOR = "#eb6834"
DID_COLOR   = "#1baf7a"   # slot 3 — aqua; distinct from both S_lie and S_truth

SURFACE     = "#fcfcfb"
GRID_COLOR  = "#e1e0d9"
TEXT_PRI    = "#0b0b0b"
TEXT_SEC    = "#52514e"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_idx(key: str) -> int:
    return int(key.split(".")[2])

def _is_attn(key: str) -> bool:
    return "self_attn" in key

def _is_mlp(key: str) -> bool:
    return ".mlp." in key

def _layer_blocks(scores: dict) -> list:
    idxs = set()
    for k in scores:
        if k.startswith("model.layers."):
            idxs.add(_block_idx(k))
    return sorted(idxs)

def _collect_flat(scores: dict, block: int, filt=None, max_elems: int = 300_000):
    """Return a subsampled flat tensor of all score values for the given block."""
    parts = []
    for k, v in scores.items():
        if not k.startswith("model.layers."):
            continue
        if _block_idx(k) != block:
            continue
        if filt is not None and not filt(k):
            continue
        parts.append(v.float().view(-1))

    if not parts:
        return None

    flat = torch.cat(parts)
    n = flat.numel()
    if n > max_elems:
        step = n // max_elems + 1
        flat = flat[::step]
    return flat


def _block_stats(scores: dict, block: int, filt=None, max_elems: int = 300_000) -> dict | None:
    """Compute median / p90 / p96 / p99 for S_lie or S_truth."""
    flat = _collect_flat(scores, block, filt, max_elems)
    if flat is None:
        return None
    q = torch.quantile(flat, torch.tensor([0.50, 0.90, 0.96, 0.99]))
    return {
        "median": q[0].item(),
        "p90":    q[1].item(),
        "p96":    q[2].item(),
        "p99":    q[3].item(),
    }


def _block_did_stats(
    S_lie: dict, S_truth: dict, block: int,
    w: float, n_prompts: int, filt=None, max_elems: int = 300_000,
) -> dict | None:
    """
    Compute element-wise DID = w·(S_lie/n) − (1−w)·(S_truth/n) for all weights in
    the block, then return symmetric percentile stats (p1/p10/p50/p90/p99 etc.).

    Because DID can be negative, bands are reported as lower and upper separately.
    """
    lie_flat   = _collect_flat(S_lie,   block, filt, max_elems)
    truth_flat = _collect_flat(S_truth, block, filt, max_elems)
    if lie_flat is None or truth_flat is None:
        return None

    # The two flats may differ in size due to independent subsampling; re-align.
    min_n = min(lie_flat.numel(), truth_flat.numel())
    lie_flat   = lie_flat[:min_n]
    truth_flat = truth_flat[:min_n]

    inv = 1.0 / n_prompts if n_prompts > 0 else 1.0
    did = inv * (w * lie_flat - (1.0 - w) * truth_flat)

    q = torch.quantile(did, torch.tensor([0.01, 0.04, 0.10, 0.50, 0.90, 0.96, 0.99]))
    return {
        "p01":    q[0].item(),
        "p04":    q[1].item(),
        "p10":    q[2].item(),
        "median": q[3].item(),
        "p90":    q[4].item(),
        "p96":    q[5].item(),
        "p99":    q[6].item(),
    }


# ── Drawing ───────────────────────────────────────────────────────────────────

def _draw(ax, blocks, lie_stats, truth_stats, title: str) -> None:
    x = np.array(blocks)

    for stats, color, label in [
        (lie_stats,   LIE_COLOR,   "S_lie"),
        (truth_stats, TRUTH_COLOR, "S_truth"),
    ]:
        med = np.array([s["median"] for s in stats])
        p90 = np.array([s["p90"]    for s in stats])
        p96 = np.array([s["p96"]    for s in stats])
        p99 = np.array([s["p99"]    for s in stats])

        # Bands from median upward (lighter at higher percentiles)
        ax.fill_between(x, med, p90, color=color, alpha=0.22, linewidth=0)
        ax.fill_between(x, p90, p96, color=color, alpha=0.14, linewidth=0)
        ax.fill_between(x, p96, p99, color=color, alpha=0.08, linewidth=0)

        # Median line — 2 px, round cap/join
        ax.plot(x, med, color=color, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label=label)

    _style_ax(ax, title, "Importance score")


def _style_ax(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, color=TEXT_PRI)
    ax.set_xlabel("Block index", fontsize=9, color=TEXT_SEC)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SEC)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#c3c2b7")
    ax.set_facecolor(SURFACE)
    ax.grid(True, linewidth=0.5, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)


def _draw_did(ax, blocks, did_stats, title: str) -> None:
    x = np.array(blocks)
    p90 = np.array([s["p90"] for s in did_stats])
    p96 = np.array([s["p96"] for s in did_stats])
    p99 = np.array([s["p99"] for s in did_stats])

    # Percentile curves
    ax.plot(x, p90, color=DID_COLOR, linewidth=1.5, alpha=0.9,
            solid_capstyle="round", solid_joinstyle="round", label="p90")
    ax.plot(x, p96, color=DID_COLOR, linewidth=1.5, alpha=0.6,
            solid_capstyle="round", solid_joinstyle="round", label="p96")
    ax.plot(x, p99, color=DID_COLOR, linewidth=1.5, alpha=0.35,
            solid_capstyle="round", solid_joinstyle="round", label="p99")

    # Zero reference
    ax.axhline(0, color="#c3c2b7", linewidth=0.8, linestyle="--", zorder=1)


    _style_ax(ax, title, "DID score")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="outputs/scores_llama_8b_ntruth25.pt",
                        help="Path to saved scores .pt file")
    parser.add_argument("--out", default="outputs/layer_stats.png",
                        help="Output PNG for S_lie / S_truth figure")
    parser.add_argument("--out-did", default="outputs/layer_stats_did.png",
                        help="Output PNG for DID figure")
    parser.add_argument("--w", type=float, default=0.5,
                        help="DID weight: DID = w·mean(S_lie) − (1−w)·mean(S_truth). "
                             "Default 0.5 (equal weight). Use 0.286 for lie_truth_ratio=0.4.")
    parser.add_argument("--n-truth", type=int, default=25,
                        help="Number of prompts used for each score (for mean normalisation). "
                             "Default 25.")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    out_path    = Path(args.out)
    out_did     = Path(args.out_did)
    w           = args.w
    n_truth     = args.n_truth

    print(f"Loading {scores_path} …")
    saved   = torch.load(scores_path, map_location="cpu", weights_only=False)
    S_lie   = saved["S_lie"]
    S_truth = saved["S_truth"]

    blocks = _layer_blocks(S_lie)
    print(f"Transformer blocks: {len(blocks)}  ({blocks[0]}–{blocks[-1]})")

    configs = [
        (None,     "All layers"),
        (_is_attn, "Attention layers\n(q, k, v, o proj)"),
        (_is_mlp,  "MLP layers\n(gate, up, down proj)"),
    ]

    # ── Figure 1: S_lie vs S_truth ────────────────────────────────────────────
    print("\nFigure 1: S_lie / S_truth statistics …")
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    fig1.suptitle(
        f"Per-layer importance score statistics — LLaMA-3.1-8B  (n_truth={n_truth})",
        fontsize=12, y=1.02, color=TEXT_PRI,
    )

    for ax, (filt, title) in zip(axes1, configs):
        print(f"  {title.replace(chr(10), ' ')} …")
        lie_s   = [_block_stats(S_lie,   b, filt) for b in blocks]
        truth_s = [_block_stats(S_truth, b, filt) for b in blocks]
        valid   = [(b, l, t) for b, l, t in zip(blocks, lie_s, truth_s)
                   if l is not None and t is not None]
        _draw(ax, [v[0] for v in valid], [v[1] for v in valid], [v[2] for v in valid], title)

    leg1 = [
        mlines.Line2D([], [], color=LIE_COLOR,   linewidth=2, label="S_lie  (median)"),
        mlines.Line2D([], [], color=TRUTH_COLOR, linewidth=2, label="S_truth (median)"),
        mpatches.Patch(facecolor="#888888", alpha=0.22, label="median → p90"),
        mpatches.Patch(facecolor="#888888", alpha=0.14, label="p90 → p96"),
        mpatches.Patch(facecolor="#888888", alpha=0.08, label="p96 → p99"),
    ]
    fig1.legend(handles=leg1, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                ncol=5, frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="#c3c2b7")
    plt.figure(fig1.number)
    plt.tight_layout()
    fig1.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved → {out_path}")

    # ── Figure 2: DID scores ──────────────────────────────────────────────────
    print(f"\nFigure 2: DID score statistics  (w={w:.3f}) …")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    fig2.suptitle(
        f"Per-layer DID score statistics — LLaMA-3.1-8B  "
        f"(w={w:.3f},  n_truth={n_truth})\n"
        f"DID = w·mean(S_lie) − (1−w)·mean(S_truth)",
        fontsize=11, y=1.04, color=TEXT_PRI,
    )

    for ax, (filt, title) in zip(axes2, configs):
        print(f"  {title.replace(chr(10), ' ')} …")
        did_s = [_block_did_stats(S_lie, S_truth, b, w, n_truth, filt) for b in blocks]
        valid = [(b, d) for b, d in zip(blocks, did_s) if d is not None]
        _draw_did(ax, [v[0] for v in valid], [v[1] for v in valid], title)

    leg2 = [
        mlines.Line2D([], [], color="#c3c2b7", linewidth=0.8,
                      linestyle="--", label="zero line"),
        mlines.Line2D([], [], color=DID_COLOR, linewidth=1.5, alpha=0.9,  label="p90"),
        mlines.Line2D([], [], color=DID_COLOR, linewidth=1.5, alpha=0.6,  label="p96"),
        mlines.Line2D([], [], color=DID_COLOR, linewidth=1.5, alpha=0.35, label="p99"),
    ]
    fig2.legend(handles=leg2, loc="upper center", bbox_to_anchor=(0.5, -0.04),
                ncol=5, frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="#c3c2b7")
    plt.figure(fig2.number)
    plt.tight_layout()
    fig2.savefig(out_did, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved → {out_did}")


if __name__ == "__main__":
    main()
