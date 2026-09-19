#!/usr/bin/env python3
"""
Test the regularization hypothesis: do negative-DID masked weights cancel
out positive-DID masked weights within each layer?

Three checks, each plotted as a 3-subplot figure (all / attention / MLP):

  1. Median |W| — pos-DID vs neg-DID masked weights per block.
     Tests whether the two groups have similar weight magnitudes.

  2. Signed sum × α per group per block (α·Σ W[pos] and α·Σ W[neg]) plus
     the net total. A near-zero total would indicate cancellation.

  3. Distribution of row-wise net perturbation
       Δ_i = α · Σ_j  W[i,j] · mask[i,j]
     split into pos-DID and neg-DID contributions plus total.
     Strongest stability check: large |Δ_i| means neuron i is substantially
     perturbed, small |Δ_i| means perturbations cancel at the neuron level.

Outputs:
  outputs/cancellation_check1_weight_magnitude.png
  outputs/cancellation_check2_signed_sum.png
  outputs/cancellation_check3_rowwise.png

Usage:
    CUDA_VISIBLE_DEVICES=0 python analyze_mask_cancellation.py \\
        --device cuda:0 --rho 0.04 --alpha 4.0 \\
        --n-truth 25 --lie-truth-ratio 0.41
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_SHORT = "llama_8b"
CACHE_DIR   = "/scratch/aebrahim/hf_cache"
MODE        = "score_diff"

# Palette
POS_COLOR   = "#2a78d6"   # blue   — positive-DID group
NEG_COLOR   = "#eb6834"   # orange — negative-DID group
TOT_COLOR   = "#1baf7a"   # aqua   — total / net

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

def _subsample(t: torch.Tensor, max_elems: int = 200_000) -> torch.Tensor:
    n = t.numel()
    return t[:: n // max_elems + 1] if n > max_elems else t


# ── Per-layer statistics ──────────────────────────────────────────────────────

def layer_stats(W: torch.Tensor, mask: torch.Tensor,
                did: torch.Tensor, alpha: float) -> dict:
    """
    W, mask, did all have shape (out_features, in_features).
    Returns dict of per-layer statistics for all three checks.
    """
    pos_mask = mask & (did > 0)   # lie-dominant
    neg_mask = mask & (did <= 0)  # truth-dominant

    W_abs = W.abs()

    # ── Check 1: median |W| per group ────────────────────────────────────────
    med_pos = W_abs[pos_mask].median().item() if pos_mask.any() else float("nan")
    med_neg = W_abs[neg_mask].median().item() if neg_mask.any() else float("nan")

    # ── Check 2: signed sum × α ───────────────────────────────────────────────
    sum_pos = alpha * W[pos_mask].sum().item() if pos_mask.any() else 0.0
    sum_neg = alpha * W[neg_mask].sum().item() if neg_mask.any() else 0.0

    # ── Check 3: row-wise net perturbation ────────────────────────────────────
    # Δ_pos[i] = α · Σ_j  W[i,j] where pos_mask[i,j]=True
    # Δ_neg[i] = α · Σ_j  W[i,j] where neg_mask[i,j]=True
    delta_pos   = alpha * (W * pos_mask.float()).sum(dim=1)   # (out,)
    delta_neg   = alpha * (W * neg_mask.float()).sum(dim=1)   # (out,)
    delta_total = delta_pos + delta_neg

    return {
        # check 1
        "med_pos": med_pos, "med_neg": med_neg,
        "n_pos": pos_mask.sum().item(), "n_neg": neg_mask.sum().item(),
        # check 2
        "sum_pos": sum_pos, "sum_neg": sum_neg,
        "sum_total": sum_pos + sum_neg,
        "n_params": mask.numel(),
        # check 3
        "delta_pos":   delta_pos.cpu(),    # (out,)
        "delta_neg":   delta_neg.cpu(),
        "delta_total": delta_total.cpu(),
    }


# ── Block-level aggregation ───────────────────────────────────────────────────

def block_stats(
    model, ln: LieNeuro, did_scores: dict, alpha: float,
    blocks: list, filt=None,
) -> dict:
    """
    Aggregate layer_stats() results per transformer block.
    Returns dict of lists (one value per block) for each statistic.
    """
    module_map = {name: mod for name, mod in model.named_modules()}

    out = {k: [] for k in [
        "med_pos", "med_neg",
        "sum_pos", "sum_neg", "sum_total",
        "dp_med", "dp_p90", "dp_p99",
        "dn_med", "dn_p90", "dn_p99",
        "dt_med", "dt_p90", "dt_p99",
        "n_pos_frac",
    ]}

    for block in blocks:
        layers_data = []
        for name in ln._layer_names:
            if not name.startswith("model.layers."):
                continue
            if _block_idx(name) != block:
                continue
            if filt is not None and not filt(name):
                continue
            if name not in ln._mask or name not in did_scores:
                continue

            mod = module_map.get(name)
            if mod is None:
                continue

            W    = mod.weight.detach().float().cpu()
            mask = ln._mask[name].cpu()
            did  = did_scores[name].cpu()

            layers_data.append(layer_stats(W, mask, did, alpha))

        if not layers_data:
            for k in out:
                out[k].append(float("nan"))
            continue

        # ── Check 1: pool |W| from all layers, weight by count ───────────────
        tot_pos = sum(d["n_pos"] for d in layers_data)
        tot_neg = sum(d["n_neg"] for d in layers_data)
        tot_n   = sum(d["n_params"] for d in layers_data)
        # weighted average of per-layer medians
        med_pos = (sum(d["med_pos"] * d["n_pos"] for d in layers_data
                       if not np.isnan(d["med_pos"]))
                   / max(tot_pos, 1))
        med_neg = (sum(d["med_neg"] * d["n_neg"] for d in layers_data
                       if not np.isnan(d["med_neg"]))
                   / max(tot_neg, 1))
        out["med_pos"].append(med_pos)
        out["med_neg"].append(med_neg)
        out["n_pos_frac"].append(tot_pos / max(tot_pos + tot_neg, 1))

        # ── Check 2: sum signed contributions across layers ───────────────────
        out["sum_pos"].append(sum(d["sum_pos"] for d in layers_data))
        out["sum_neg"].append(sum(d["sum_neg"] for d in layers_data))
        out["sum_total"].append(sum(d["sum_total"] for d in layers_data))

        # ── Check 3: pool row-wise Δ across all layers ────────────────────────
        dp = _subsample(torch.cat([d["delta_pos"]   for d in layers_data]))
        dn = _subsample(torch.cat([d["delta_neg"]   for d in layers_data]))
        dt = _subsample(torch.cat([d["delta_total"] for d in layers_data]))

        def _pct(t, q):
            return torch.quantile(t, torch.tensor([0.50, q[0], q[1]])).tolist()

        for key_prefix, arr in [("dp", dp), ("dn", dn), ("dt", dt)]:
            med, p90, p99 = _pct(arr, [0.90, 0.99])
            out[f"{key_prefix}_med"].append(med)
            out[f"{key_prefix}_p90"].append(p90)
            out[f"{key_prefix}_p99"].append(p99)

    return {k: np.array(v) for k, v in out.items()}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_check1(fig, axes, x_vals, stats_list, config_labels):
    """Median |W| for pos-DID vs neg-DID masked weights."""
    for ax, stats, label in zip(axes, stats_list, config_labels):
        ax.plot(x_vals, stats["med_pos"], color=POS_COLOR, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label="pos-DID")
        ax.plot(x_vals, stats["med_neg"], color=NEG_COLOR, linewidth=2,
                solid_capstyle="round", solid_joinstyle="round", label="neg-DID")
        _style_ax(ax, label, "Median |W|")

    leg = [
        mlines.Line2D([], [], color=POS_COLOR, linewidth=2, label="Pos-DID masked weights"),
        mlines.Line2D([], [], color=NEG_COLOR, linewidth=2, label="Neg-DID masked weights"),
    ]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=2, frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="#c3c2b7")


def plot_check2(fig, axes, x_vals, stats_list, config_labels, alpha):
    """Signed sum × α: pos, neg, total."""
    for ax, stats, label in zip(axes, stats_list, config_labels):
        ax.plot(x_vals, stats["sum_pos"],   color=POS_COLOR, linewidth=2,
                solid_capstyle="round", label="pos-DID group")
        ax.plot(x_vals, stats["sum_neg"],   color=NEG_COLOR, linewidth=2,
                solid_capstyle="round", label="neg-DID group")
        ax.plot(x_vals, stats["sum_total"], color=TOT_COLOR, linewidth=2,
                linestyle="--", solid_capstyle="round", label="net total")
        ax.axhline(0, color="#c3c2b7", linewidth=0.8, linestyle=":")
        _style_ax(ax, label, f"α·Σ W  (α={alpha})")

    leg = [
        mlines.Line2D([], [], color=POS_COLOR, linewidth=2, label="Pos-DID: α·Σ W"),
        mlines.Line2D([], [], color=NEG_COLOR, linewidth=2, label="Neg-DID: α·Σ W"),
        mlines.Line2D([], [], color=TOT_COLOR, linewidth=2, linestyle="--", label="Net total"),
        mlines.Line2D([], [], color="#c3c2b7", linewidth=0.8, linestyle=":", label="zero"),
    ]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=4, frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="#c3c2b7")


def plot_check3(fig, axes, x_vals, stats_list, config_labels, alpha):
    """Row-wise perturbation distribution: median with p90/p99 bands."""
    for ax, stats, label in zip(axes, stats_list, config_labels):
        for prefix, color, name in [
            ("dp", POS_COLOR, "Pos-DID"),
            ("dn", NEG_COLOR, "Neg-DID"),
            ("dt", TOT_COLOR, "Total"),
        ]:
            med = stats[f"{prefix}_med"]
            p90 = stats[f"{prefix}_p90"]
            p99 = stats[f"{prefix}_p99"]
            ax.fill_between(x_vals, med, p90, color=color, alpha=0.18, linewidth=0)
            ax.fill_between(x_vals, p90, p99, color=color, alpha=0.09, linewidth=0)
            ax.plot(x_vals, med, color=color, linewidth=2,
                    solid_capstyle="round", label=name)

        ax.axhline(0, color="#c3c2b7", linewidth=0.8, linestyle=":")
        _style_ax(ax, label, "Row-wise Δ_i")

    leg = [
        mlines.Line2D([], [], color=POS_COLOR, linewidth=2, label="Pos-DID Δ_i (median)"),
        mlines.Line2D([], [], color=NEG_COLOR, linewidth=2, label="Neg-DID Δ_i (median)"),
        mlines.Line2D([], [], color=TOT_COLOR, linewidth=2, label="Total Δ_i (median)"),
        mpatches.Patch(facecolor="#888888", alpha=0.18, label="→ p90"),
        mpatches.Patch(facecolor="#888888", alpha=0.09, label="→ p99"),
        mlines.Line2D([], [], color="#c3c2b7", linewidth=0.8, linestyle=":", label="zero"),
    ]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=6, frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="#c3c2b7")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rho",   type=float, default=0.04)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--n-truth", type=int, default=25)
    parser.add_argument("--lie-truth-ratio", type=float, default=0.41)
    parser.add_argument("--exclude-lm-head", action="store_true")
    parser.add_argument("--scores", default=None)
    parser.add_argument("--out-dir", default="outputs")
    args = parser.parse_args()

    device   = args.device
    n_truth  = args.n_truth
    alpha    = args.alpha
    ratio    = args.lie_truth_ratio
    skip     = ["lm_head", "embed_tokens"] if args.exclude_lm_head else []
    out_dir  = Path(args.out_dir)

    scores_pt = Path(args.scores) if args.scores else \
                Path(f"outputs/scores_{MODEL_SHORT}_ntruth{n_truth}.pt")

    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device, cache_dir=CACHE_DIR)
    model.eval()

    cfg = LieNeuroConfig(importance_threshold=args.rho, alpha=alpha,
                         device=device, skip_modules=skip)
    ln  = LieNeuro(model, tok, cfg)

    if scores_pt.exists():
        print(f"Loading scores from {scores_pt} …")
        saved       = torch.load(scores_pt, map_location="cpu", weights_only=False)
        ln._S_lie   = saved["S_lie"]
        ln._S_truth = saved["S_truth"]
    else:
        n_lie = min(n_truth, len(DIRECT_LIE))
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT[:n_truth], desc="S_truth")
        ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE[:n_lie], desc="S_lie")

    print(f"Building mask  ρ={args.rho}  α={alpha}  ratio={ratio}  n_truth={n_truth} …")
    ln.build_mask(mode=MODE, n_prompts=n_truth, lie_truth_ratio=ratio)

    # Pre-compute element-wise DID scores for splitting the mask
    print("Computing element-wise DID scores …")
    _w   = ratio / (1.0 + ratio) if ratio > 0 else 0.0
    _inv = 1.0 / n_truth
    did_scores = {
        name: (_inv * (_w * ln._S_lie[name].float()
                       - (1.0 - _w) * ln._S_truth[name].float())).cpu()
        for name in ln._layer_names
        if name in ln._mask
    }

    blocks = sorted({
        _block_idx(n) for n in ln._layer_names
        if n.startswith("model.layers.")
    })
    x = np.array(blocks)
    print(f"Transformer blocks: {len(blocks)}  ({blocks[0]}–{blocks[-1]})")

    configs = [
        (None,     "All layers"),
        (_is_attn, "Attention layers\n(q, k, v, o proj)"),
        (_is_mlp,  "MLP layers\n(gate, up, down proj)"),
    ]

    print("Computing block statistics (this reads model weights) …")
    all_stats = []
    for filt, label in configs:
        print(f"  {label.replace(chr(10), ' ')} …")
        all_stats.append(block_stats(model, ln, did_scores, alpha, blocks, filt))

    config_labels = [c[1] for c in configs]
    suptitle_suffix = (f"ρ={args.rho},  α={alpha},  "
                       f"ratio={ratio},  n_truth={n_truth}")

    # ── Figure 1: Check 1 — median |W| per group ─────────────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    fig1.suptitle(
        f"Check 1 — Masked weight magnitude: pos-DID vs neg-DID\n{suptitle_suffix}",
        fontsize=11, y=1.04, color=TEXT_PRI)
    plot_check1(fig1, axes1, x, all_stats, config_labels)
    plt.figure(fig1.number); plt.tight_layout()
    p1 = out_dir / "cancellation_check1_weight_magnitude.png"
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved → {p1}")

    # ── Figure 2: Check 2 — signed sum × α ───────────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    fig2.suptitle(
        f"Check 2 — Signed weight sum × α: do pos and neg groups cancel?\n{suptitle_suffix}",
        fontsize=11, y=1.04, color=TEXT_PRI)
    plot_check2(fig2, axes2, x, all_stats, config_labels, alpha)
    plt.figure(fig2.number); plt.tight_layout()
    p2 = out_dir / "cancellation_check2_signed_sum.png"
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved → {p2}")

    # ── Figure 3: Check 3 — row-wise perturbation distribution ───────────────
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    fig3.suptitle(
        f"Check 3 — Row-wise perturbation Δ_i = α·Σ_j W[i,j]·mask[i,j]\n{suptitle_suffix}",
        fontsize=11, y=1.04, color=TEXT_PRI)
    plot_check3(fig3, axes3, x, all_stats, config_labels, alpha)
    plt.figure(fig3.number); plt.tight_layout()
    p3 = out_dir / "cancellation_check3_rowwise.png"
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"  Saved → {p3}")

    print("\nDone.")


if __name__ == "__main__":
    main()
