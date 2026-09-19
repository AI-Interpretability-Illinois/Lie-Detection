#!/usr/bin/env python3
"""
Plot per-layer weight magnitude statistics for the original and LieNeuro-steered model.

Produces 3 subplots (all / attention / MLP layers), each with median, p90, p96,
and p99 curves of |W| for both the original model and the steered (lie) model.

Usage:
    CUDA_VISIBLE_DEVICES=0 python plot_weight_stats.py \
        --device cuda:0 --rho 0.04 --alpha 4.0 \
        --n-truth 25 --lie-truth-ratio 0.4
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

# Palette: p90/p96/p99 use slots 1/2/3; line style separates original vs lie
P_COLORS = {"p90": "#2a78d6", "p96": "#eb6834", "p99": "#1baf7a"}

SURFACE    = "#fcfcfb"
GRID_COLOR = "#e1e0d9"
TEXT_PRI   = "#0b0b0b"
TEXT_SEC   = "#52514e"

QUANTILES  = torch.tensor([0.90, 0.96, 0.99])
QKEYS      = ["p90", "p96", "p99"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_idx(key: str) -> int:
    return int(key.split(".")[2])

def _is_attn(key: str) -> bool:
    return "self_attn" in key

def _is_mlp(key: str) -> bool:
    return ".mlp." in key

def _subsample(t: torch.Tensor, max_elems: int = 300_000) -> torch.Tensor:
    n = t.numel()
    if n > max_elems:
        return t[:: n // max_elems + 1]
    return t

def _quantile_stats(flat: torch.Tensor) -> dict:
    q = torch.quantile(flat, QUANTILES)
    return {k: q[i].item() for i, k in enumerate(QKEYS)}


def collect_block_stats(
    model, ln: LieNeuro, alpha: float,
    blocks: list, filt=None, max_elems: int = 300_000,
) -> tuple[list, list]:
    """
    Iterate through transformer blocks using only the scored/masked layer names
    (ln._layer_names), so all three subplots are consistent with each other and
    with the DID/score plots.
    """
    # Build a name→module map once for fast lookup
    module_map = {name: mod for name, mod in model.named_modules()}

    orig_stats = []
    lie_stats  = []

    for block in blocks:
        orig_parts = []
        lie_parts  = []

        for name in ln._layer_names:
            if not name.startswith("model.layers."):
                continue
            if _block_idx(name) != block:
                continue
            if filt is not None and not filt(name):
                continue

            module = module_map.get(name)
            if module is None or not hasattr(module, "weight") or module.weight is None:
                continue

            W = module.weight.detach().float().cpu().abs()

            mask = ln._mask.get(name)
            if mask is not None:
                W_lie = module.weight.detach().float().cpu() * (1.0 + alpha * mask.float().cpu())
                W_lie = W_lie.abs()
            else:
                W_lie = W

            orig_parts.append(_subsample(W.view(-1), max_elems // 4))
            lie_parts.append(_subsample(W_lie.view(-1), max_elems // 4))

        if not orig_parts:
            orig_stats.append(None)
            lie_stats.append(None)
            continue

        orig_flat = _subsample(torch.cat(orig_parts), max_elems)
        lie_flat  = _subsample(torch.cat(lie_parts),  max_elems)

        orig_stats.append(_quantile_stats(orig_flat))
        lie_stats.append(_quantile_stats(lie_flat))

    return orig_stats, lie_stats


# ── Drawing ───────────────────────────────────────────────────────────────────

def _style_ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=11, color=TEXT_PRI)
    ax.set_xlabel("Block index", fontsize=9, color=TEXT_SEC)
    ax.set_ylabel("|W| value", fontsize=9, color=TEXT_SEC)
    ax.tick_params(colors=TEXT_SEC, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#c3c2b7")
    ax.set_facecolor(SURFACE)
    ax.grid(True, linewidth=0.5, color=GRID_COLOR, zorder=0)
    ax.set_axisbelow(True)


def _draw(ax, blocks, orig_stats, lie_stats, title: str) -> None:
    x = np.array(blocks)

    for stats, linestyle, model_label in [
        (orig_stats, "-",  "Original"),
        (lie_stats,  "--", "LieNeuro-DID"),
    ]:
        for key in QKEYS:
            vals = np.array([s[key] for s in stats])
            ax.plot(x, vals, color=P_COLORS[key], linewidth=1.8,
                    linestyle=linestyle,
                    solid_capstyle="round", solid_joinstyle="round",
                    label=f"{model_label} {key}")

    _style_ax(ax, title)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rho",   type=float, default=0.04,
                        help="Importance threshold ρ. Default: 0.04")
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Steering scale α. Default: 4.0")
    parser.add_argument("--n-truth", type=int, default=25,
                        help="Number of prompts for both S_lie and S_truth. Default: 25")
    parser.add_argument("--lie-truth-ratio", type=float, default=0.4,
                        help="DID weighting: w = ratio/(1+ratio). Default: 0.4")
    parser.add_argument("--exclude-lm-head", action="store_true")
    parser.add_argument("--scores", default=None,
                        help="Path to pre-computed scores .pt file. "
                             "Defaults to outputs/scores_{model}_{ntruth}.pt")
    parser.add_argument("--out", default="outputs/weight_stats.png",
                        help="Output PNG path")
    args = parser.parse_args()

    device  = args.device
    n_truth = args.n_truth
    skip    = ["lm_head", "embed_tokens"] if args.exclude_lm_head else []

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

    cfg = LieNeuroConfig(importance_threshold=args.rho, alpha=args.alpha,
                         device=device, skip_modules=skip)
    ln  = LieNeuro(model, tok, cfg)

    if scores_pt.exists():
        print(f"Loading scores from {scores_pt} …")
        saved       = torch.load(scores_pt, map_location="cpu", weights_only=False)
        ln._S_lie   = saved["S_lie"]
        ln._S_truth = saved["S_truth"]
    else:
        n_lie = min(n_truth, len(DIRECT_LIE))
        print(f"Computing S_truth[:{n_truth}] …")
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT[:n_truth], desc="S_truth")
        print(f"Computing S_lie[:{n_lie}] …")
        ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE[:n_lie], desc="S_lie")

    print(f"Building mask  ρ={args.rho}  α={args.alpha}  "
          f"ratio={args.lie_truth_ratio}  n_truth={n_truth} …")
    ln.build_mask(mode=MODE, n_prompts=n_truth, lie_truth_ratio=args.lie_truth_ratio)

    blocks = sorted({
        _block_idx(n) for n in ln._layer_names
        if n.startswith("model.layers.")
    })
    print(f"Transformer blocks: {len(blocks)}  ({blocks[0]}–{blocks[-1]})")

    configs = [
        (None,     "All layers"),
        (_is_attn, "Attention layers\n(q, k, v, o proj)"),
        (_is_mlp,  "MLP layers\n(gate, up, down proj)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=SURFACE)
    fig.suptitle(
        f"Per-layer weight magnitude statistics — LLaMA-3.1-8B\n"
        f"ρ={args.rho},  α={args.alpha},  ratio={args.lie_truth_ratio},  n_truth={n_truth}",
        fontsize=11, y=1.04, color=TEXT_PRI,
    )

    for ax, (filt, title) in zip(axes, configs):
        print(f"  Computing stats: {title.replace(chr(10), ' ')} …")
        orig_s, lie_s = collect_block_stats(model, ln, args.alpha, blocks, filt)
        valid = [(b, o, l) for b, o, l in zip(blocks, orig_s, lie_s)
                 if o is not None and l is not None]
        _draw(ax,
              [v[0] for v in valid],
              [v[1] for v in valid],
              [v[2] for v in valid],
              title)

    leg = [
        # percentile colors
        mlines.Line2D([], [], color=P_COLORS["p90"], linewidth=1.8, label="p90"),
        mlines.Line2D([], [], color=P_COLORS["p96"], linewidth=1.8, label="p96"),
        mlines.Line2D([], [], color=P_COLORS["p99"], linewidth=1.8, label="p99"),
        # model line styles
        mlines.Line2D([], [], color="#52514e", linewidth=1.8, linestyle="-",
                      label="Original"),
        mlines.Line2D([], [], color="#52514e", linewidth=1.8, linestyle="--",
                      label="LieNeuro-DID"),
    ]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.04),
               ncol=5, frameon=True, fontsize=9, facecolor=SURFACE, edgecolor="#c3c2b7")

    plt.tight_layout()
    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
