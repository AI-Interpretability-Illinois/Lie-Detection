#!/usr/bin/env python3
"""
Compute DID scores for 4 lie-truth-ratio configs in parallel on 4 GPUs and
report layer-wise Spearman rank correlations between each pair of configs.

All configs use n-truth=25, lm_head included.

Configs:
  ratio=0.0  → n_lie=0   (DID = −S_truth)
  ratio=0.4  → n_lie=10  (legacy scale mismatch)
  ratio=1.0  → n_lie=25  (balanced, same scale)
  ratio=2.0  → n_lie=50  (2× lie prompts)

For each pair, Spearman r is computed per layer on the flattened DID
(S_lie − S_truth) weight vectors, then summarised across layers.

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1,2,3 python analyze_did_score_correlation.py
  CUDA_VISIBLE_DEVICES=0,1,2,3 python analyze_did_score_correlation.py --devices cuda:0,cuda:1,cuda:2,cuda:3
"""

import argparse
import multiprocessing as mp
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR  = "/scratch/aebrahim/hf_cache"
OUT_DIR    = Path("outputs/did_correlation")
N_TRUTH    = 25

CONFIGS = [
    {"ratio": 0.0, "n_lie": 0},
    {"ratio": 0.4, "n_lie": round(0.4 * N_TRUTH)},   # 10
    {"ratio": 1.0, "n_lie": round(1.0 * N_TRUTH)},   # 25
    {"ratio": 2.0, "n_lie": round(2.0 * N_TRUTH)},   # 50
]


def _compute_scores(device, n_truth, n_lie, ratio):
    """Load model on `device`, compute and return (S_lie, S_truth)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.lie_neuro import LieNeuro, LieNeuroConfig
    from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device, cache_dir=CACHE_DIR)
    model.eval()

    cfg = LieNeuroConfig(device=device, skip_modules=[])   # include lm_head
    ln  = LieNeuro(model, tok, cfg)

    print(f"[{device}|ratio={ratio}] Computing S_truth over "
          f"DIRECT_TRUTH_INSTRUCT[:{n_truth}] …")
    ln._S_truth = ln._compute_scores_for_prompts(
        DIRECT_TRUTH_INSTRUCT[:n_truth], desc=f"S_truth_r{ratio}")

    if n_lie > 0:
        print(f"[{device}|ratio={ratio}] Computing S_lie over "
              f"DIRECT_LIE[:{n_lie}] …")
        ln._S_lie = ln._compute_scores_for_prompts(
            DIRECT_LIE[:n_lie], desc=f"S_lie_r{ratio}")
    else:
        print(f"[{device}|ratio={ratio}] ratio=0 → S_lie = 0 (DID = −S_truth)")
        ln._S_lie = {name: torch.zeros_like(v) for name, v in ln._S_truth.items()}

    return ln._S_lie, ln._S_truth


def worker(args):
    """Subprocess entry: compute scores and save to disk."""
    device, n_truth, n_lie, ratio, out_path = args
    S_lie, S_truth = _compute_scores(device, n_truth, n_lie, ratio)
    torch.save({"S_lie": S_lie, "S_truth": S_truth,
                "n_truth": n_truth, "n_lie": n_lie, "ratio": ratio},
               out_path)
    print(f"[{device}|ratio={ratio}] Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3",
                        help="Comma-separated list of 4 GPU devices")
    args = parser.parse_args()
    devices = args.devices.split(",")
    if len(devices) != len(CONFIGS):
        raise ValueError(f"Need exactly {len(CONFIGS)} devices, got {len(devices)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    score_paths = [
        OUT_DIR / f"scores_ntruth{N_TRUTH}_ratio{c['ratio']:.1f}.pt"
        for c in CONFIGS
    ]

    # ── compute scores in parallel on 4 GPUs ──────────────────────────────────
    worker_args = [
        (devices[i], N_TRUTH, CONFIGS[i]["n_lie"], CONFIGS[i]["ratio"], score_paths[i])
        for i in range(len(CONFIGS))
    ]

    print("Launching 4 score-computation workers …")
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(CONFIGS)) as pool:
        pool.map(worker, worker_args)

    # ── load scores, compute DID = S_lie − S_truth ────────────────────────────
    print("\nLoading scores …")
    did = {}   # ratio → {layer_name → flat float32 numpy array}
    for cfg, path in zip(CONFIGS, score_paths):
        saved   = torch.load(path, map_location="cpu", weights_only=False)
        S_lie   = saved["S_lie"]
        S_truth = saved["S_truth"]
        did[cfg["ratio"]] = {
            name: (S_lie[name].float() - S_truth[name].float()).numpy().ravel()
            for name in S_truth
        }
        print(f"  ratio={cfg['ratio']:.1f}  n_lie={cfg['n_lie']}  "
              f"n_truth={N_TRUTH}  layers={len(did[cfg['ratio']])}")

    ratios      = [c["ratio"] for c in CONFIGS]
    layer_names = list(did[ratios[0]].keys())
    pairs       = list(combinations(ratios, 2))

    # ── layer-wise Spearman correlations ──────────────────────────────────────
    print("\nLayer-wise Spearman rank correlations between DID score vectors:")
    print(f"  {'Config A':>14}  {'Config B':>14}  "
          f"{'mean r':>8}  {'std r':>7}  {'min r':>8}  {'max r':>8}")
    print("  " + "─" * 65)

    pair_corrs = {}
    for r1, r2 in pairs:
        corrs = np.array([
            spearmanr(did[r1][lname], did[r2][lname]).statistic
            for lname in layer_names
        ])
        pair_corrs[(r1, r2)] = corrs
        label_a = f"ratio={r1:.1f}"
        label_b = f"ratio={r2:.1f}"
        print(f"  {label_a:>14}  {label_b:>14}  "
              f"{corrs.mean():>8.4f}  {corrs.std():>7.4f}  "
              f"{corrs.min():>8.4f}  {corrs.max():>8.4f}")

    # ── save full per-layer results ────────────────────────────────────────────
    results_path = OUT_DIR / f"spearman_ntruth{N_TRUTH}.pt"
    torch.save({"pair_corrs": pair_corrs, "layer_names": layer_names,
                "ratios": ratios, "n_truth": N_TRUTH},
               results_path)
    print(f"\nFull layer-wise correlations saved → {results_path}")


if __name__ == "__main__":
    main()
