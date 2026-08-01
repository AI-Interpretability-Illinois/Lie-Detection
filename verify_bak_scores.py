#!/usr/bin/env python3
"""
Verify that outputs/mask_direct_rho0.1_alpha1.0.pt.bak was produced by the
initial (commit 77d72f0) run_experiment.py using:
  - DIRECT_LIE[:10]   (original 10 lie prompts)
  - DIRECT_TRUTH[:10] (original 10 plain truth prompts, no instruction prefix)
  - rho=0.1, alpha=1.0, set_diff mode

Steps:
  1. Recompute S_lie and S_truth with those exact 10-prompt sets.
  2. Save to outputs/mask_direct_rho0.1_alpha1.0_replicated.pt
  3. Compare element-wise with the .bak file and report cosine similarity
     and relative error per layer.

Usage:
  CUDA_VISIBLE_DEVICES=0 python verify_bak_scores.py [--device cuda:0]
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, ".")
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import DIRECT_LIE, DIRECT_TRUTH
from transformers import AutoModelForCausalLM, AutoTokenizer

BAK_PATH  = Path("outputs/mask_direct_rho0.1_alpha1.0.pt.bak")
OUT_PATH  = Path("outputs/mask_direct_rho0.1_alpha1.0_replicated.pt")
MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_DIR = "/scratch/aebrahim/hf_cache"


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().view(-1), b.float().view(-1)
    return (a @ b / (a.norm() * b.norm())).item()


def rel_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not BAK_PATH.exists():
        print(f"ERROR: {BAK_PATH} not found.")
        sys.exit(1)

    # ── load model ──────────────────────────────────────────────────────────
    print(f"Loading model {MODEL_ID} on {args.device} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=args.device, cache_dir=CACHE_DIR)
    model.eval()

    # ── replicate original scoring ───────────────────────────────────────────
    # skip_modules=[] to include lm_head, matching the original code
    cfg = LieNeuroConfig(device=args.device, skip_modules=[])
    ln  = LieNeuro(model, tok, cfg)

    lie_prompts   = DIRECT_LIE[:10]
    truth_prompts = DIRECT_TRUTH[:10]
    print(f"Computing S_lie  with {len(lie_prompts)} prompts …")
    ln._S_lie   = ln._compute_scores_for_prompts(lie_prompts,   desc="S_lie")
    print(f"Computing S_truth with {len(truth_prompts)} prompts …")
    ln._S_truth = ln._compute_scores_for_prompts(truth_prompts, desc="S_truth")

    torch.save({"S_lie": ln._S_lie, "S_truth": ln._S_truth}, OUT_PATH)
    print(f"Replicated scores saved → {OUT_PATH}\n")

    # ── compare ──────────────────────────────────────────────────────────────
    bak = torch.load(BAK_PATH, map_location="cpu", weights_only=False)
    rep = torch.load(OUT_PATH, map_location="cpu", weights_only=False)

    layers_bak = set(bak["S_lie"].keys())
    layers_rep = set(rep["S_lie"].keys())
    only_bak = layers_bak - layers_rep
    only_rep = layers_rep - layers_bak
    common   = layers_bak & layers_rep

    print(f"Layers in .bak:       {len(layers_bak)}")
    print(f"Layers in replicated: {len(layers_rep)}")
    if only_bak:
        print(f"  Only in .bak:       {only_bak}")
    if only_rep:
        print(f"  Only in replicated: {only_rep}")
    print(f"Common layers:        {len(common)}\n")

    # Per-layer stats
    cos_lie, cos_truth, rel_lie, rel_truth = [], [], [], []
    for name in sorted(common):
        sl_b = bak["S_lie"][name].float()
        sl_r = rep["S_lie"][name].float()
        st_b = bak["S_truth"][name].float()
        st_r = rep["S_truth"][name].float()
        cos_lie.append(cosine_sim(sl_b, sl_r))
        cos_truth.append(cosine_sim(st_b, st_r))
        rel_lie.append(rel_error(sl_r, sl_b))
        rel_truth.append(rel_error(st_r, st_b))

    def stats(vals, label):
        mn, mx, av = min(vals), max(vals), sum(vals)/len(vals)
        print(f"  {label:40s}  min={mn:.4f}  max={mx:.4f}  mean={av:.4f}")

    print("=== S_lie comparison (.bak vs replicated) ===")
    stats(cos_lie,   "cosine similarity")
    stats(rel_lie,   "relative L2 error")
    print()
    print("=== S_truth comparison (.bak vs replicated) ===")
    stats(cos_truth, "cosine similarity")
    stats(rel_truth, "relative L2 error")
    print()

    # Spot-check one layer
    layer = "model.layers.0.self_attn.q_proj"
    print(f"=== Spot-check: {layer} ===")
    for tag, key in [("S_lie", "S_lie"), ("S_truth", "S_truth")]:
        b_sum = bak[key][layer].float().sum().item()
        r_sum = rep[key][layer].float().sum().item()
        print(f"  {tag}: bak={b_sum:.0f}  replicated={r_sum:.0f}  "
              f"ratio={r_sum/b_sum:.4f}  (expect ≈ 1.0)")

    print()
    mean_cos = (sum(cos_lie) + sum(cos_truth)) / (len(cos_lie) + len(cos_truth))
    if mean_cos > 0.999:
        print("✓ CONFIRMED: scores match the .bak file (mean cosine > 0.999).")
        print("  The .bak was produced with DIRECT_LIE[:10] + DIRECT_TRUTH[:10].")
    else:
        print(f"✗ Scores do NOT match (mean cosine = {mean_cos:.4f}).")
        print("  The .bak was produced with different prompts or settings.")


if __name__ == "__main__":
    main()
