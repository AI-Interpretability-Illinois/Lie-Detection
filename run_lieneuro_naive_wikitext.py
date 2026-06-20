#!/usr/bin/env python3
"""
Compute WikiText-2 PPL for all naive LieNeuro checkpoints and append to wikitext_ppl.csv.

Splits across two GPU groups via --group a|b:
  Group A (GPU 1): rho in [0.01, 0.05] alpha in [1.0, 2.0]  (6 configs)
  Group B (GPU 2): rho in [0.05, 0.1] alpha in [4.0, 8.0] + rho=0.1 alpha in [0.5, 2.0, 4.0]

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python run_lieneuro_naive_wikitext.py --device cuda:0 --group a
  CUDA_VISIBLE_DEVICES=2 python run_lieneuro_naive_wikitext.py --device cuda:0 --group b
"""

import argparse
import csv
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import DIRECT_LIE, DIRECT_TRUTH

MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR     = Path("outputs")
WT_CSV      = OUT_DIR / "wikitext_ppl.csv"
NAIVE_MASK  = OUT_DIR / "mask_direct_rho0.1_alpha1.0.pt"

WT_WINDOW   = 512
WT_STRIDE   = 256
WT_MAX_TOKS = 8192

# All 12 configs split into two groups
CONFIGS_A = [
    (0.01, 1.0), (0.01, 2.0), (0.01, 4.0), (0.01, 8.0),
    (0.05, 1.0), (0.05, 2.0),
]
CONFIGS_B = [
    (0.05, 4.0), (0.05, 8.0),
    (0.1,  0.5), (0.1,  1.0), (0.1,  2.0), (0.1,  4.0),
]


def load_wikitext_tokens(tokenizer):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids  = tokenizer.encode(text, add_special_tokens=False)
    return ids[:WT_MAX_TOKS]


@torch.no_grad()
def sliding_window_ppl(model, token_ids, device):
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    n = len(input_ids)
    nlls, n_toks = [], 0
    for begin in range(0, n, WT_STRIDE):
        end     = min(begin + WT_WINDOW, n)
        chunk   = input_ids[begin:end].unsqueeze(0).to(device)
        tgt_len = min(WT_STRIDE, end - begin)
        labels  = chunk.clone()
        labels[:, :-tgt_len] = -100
        loss = model(input_ids=chunk, labels=labels).loss
        nlls.append(loss.item() * tgt_len)
        n_toks += tgt_len
        if end == n:
            break
    return math.exp(sum(nlls) / n_toks)


def already_evaluated(rho, alpha):
    """Return True if this config already has a valid row in wikitext_ppl.csv."""
    if not WT_CSV.exists():
        return False
    with open(WT_CSV) as f:
        for row in csv.DictReader(f):
            if (row["method"] == "weight_steering"
                    and row.get("rho") and row.get("alpha")
                    and abs(float(row["rho"]) - rho) < 1e-9
                    and abs(float(row["alpha"]) - alpha) < 1e-9
                    and row.get("wikitext_ppl")):
                return True
    return False


def append_row(rho, alpha, dw, ppl):
    label = f"LieNeuro ρ={rho:.2f} α={alpha:.1f}"
    write_header = not WT_CSV.exists()
    with open(WT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "method", "layer", "coeff", "rho", "alpha",
            "delta_wrong", "wikitext_ppl"])
        if write_header:
            w.writeheader()
        w.writerow({
            "variant":      label,
            "method":       "weight_steering",
            "layer":        "",
            "coeff":        "",
            "rho":          rho,
            "alpha":        alpha,
            "delta_wrong":  dw,
            "wikitext_ppl": ppl,
        })
    print(f"  → appended {label}  PPL={ppl:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group",  choices=["a", "b"], required=True)
    args = parser.parse_args()
    device = args.device
    configs = CONFIGS_A if args.group == "a" else CONFIGS_B

    # Filter already-done configs
    configs = [(r, a) for r, a in configs if not already_evaluated(r, a)]
    if not configs:
        print("All configs already evaluated — nothing to do.")
        return

    print(f"Loading model on {device} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device)
    model.eval()

    print("Loading WikiText-2 tokens …")
    wt_tokens = load_wikitext_tokens(tok)

    # Load importance scores
    cfg = LieNeuroConfig(device=device)
    ln  = LieNeuro(model, tok, cfg)

    if NAIVE_MASK.exists():
        print(f"Loading S_lie from {NAIVE_MASK} …")
        data = torch.load(NAIVE_MASK, map_location="cpu", weights_only=False)
        ln._S_lie   = data["S_lie"]
        ln._S_truth = data["S_truth"]
    else:
        print("Computing importance scores (S_lie + S_truth) …")
        ln.compute_importance_scores(DIRECT_LIE, DIRECT_TRUTH)

    # Load delta_wrong values from result JSONs
    import json, glob as _glob
    dw_map = {}
    for f in _glob.glob(str(OUT_DIR / "result_direct_rho*.json")):
        d   = json.load(open(f))
        rho = d.get("rho", d.get("importance_threshold"))
        alpha = d.get("alpha")
        dw  = d.get("delta_wrong_rate", d.get("delta_wrong"))
        if rho is not None and alpha is not None and dw is not None:
            dw_map[(round(float(rho), 4), round(float(alpha), 4))] = float(dw)

    print(f"\nEvaluating {len(configs)} configs …\n")
    for rho, alpha in configs:
        print(f"── ρ={rho}  α={alpha} ──")
        cfg_i     = LieNeuroConfig(importance_threshold=rho, alpha=alpha, device=device)
        ln.config = cfg_i
        ln.build_mask(mode="set_diff")
        ln.apply_steering()
        ppl = sliding_window_ppl(model, wt_tokens, device)
        ln.restore_original()
        dw  = dw_map.get((round(rho, 4), round(alpha, 4)), "")
        print(f"  PPL={ppl:.4f}  Δwrong={dw}")
        append_row(rho, alpha, dw, ppl)

    print("\nDone.")


if __name__ == "__main__":
    main()
