#!/usr/bin/env python3
"""
Evaluate extra LieNeuro DID (score_diff) configs: accuracy + WikiText-2 PPL.
Loads pre-computed scores from outputs/lieneuro_did_scores.pt.

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python run_lieneuro_did_extra.py --device cuda:0 --group a
  CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_extra.py --device cuda:0 --group b
"""

import argparse, csv, math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import evaluate_factual_accuracy

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SCORES_PT  = Path("outputs/lieneuro_did_scores.pt")
WT_CSV     = Path("outputs/wikitext_ppl.csv")

WT_WINDOW, WT_STRIDE, WT_MAX_TOKS = 512, 256, 8192
MODE = "score_diff"

CONFIGS_A = [(0.05, 4.0), (0.01, 3.0), (0.01, 4.0)]
CONFIGS_B = [(0.05, 5.0), (0.02, 4.0), (0.04, 4.0)]


def load_wikitext_tokens(tokenizer):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tokenizer.encode(text, add_special_tokens=False)[:WT_MAX_TOKS]


@torch.no_grad()
def sliding_window_ppl(model, token_ids, device):
    ids = torch.tensor(token_ids, dtype=torch.long)
    n   = len(ids)
    nlls, n_toks = [], 0
    for begin in range(0, n, WT_STRIDE):
        end     = min(begin + WT_WINDOW, n)
        chunk   = ids[begin:end].unsqueeze(0).to(device)
        tgt_len = min(WT_STRIDE, end - begin)
        labels  = chunk.clone(); labels[:, :-tgt_len] = -100
        nlls.append(model(input_ids=chunk, labels=labels).loss.item() * tgt_len)
        n_toks += tgt_len
        if end == n:
            break
    return math.exp(sum(nlls) / n_toks)


def append_row(rho, alpha, dw, ppl):
    label = f"LieNeuro-DID score_diff ρ={rho:.2f} α={alpha:.1f}"
    write_header = not WT_CSV.exists()
    with open(WT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "method", "layer", "coeff", "rho", "alpha",
            "delta_wrong", "wikitext_ppl"])
        if write_header:
            w.writeheader()
        w.writerow({"variant": label, "method": "weight_steering_did",
                    "layer": "", "coeff": "", "rho": rho, "alpha": alpha,
                    "delta_wrong": dw, "wikitext_ppl": ppl})
    print(f"  → appended {label}  Δwrong={dw:+.2%}  PPL={ppl:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--group", choices=["a", "b"], required=True)
    args   = parser.parse_args()
    device = args.device
    configs = CONFIGS_A if args.group == "a" else CONFIGS_B

    print(f"Loading model on {device} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device)
    model.eval()

    print("Loading WikiText-2 tokens …")
    wt_tokens = load_wikitext_tokens(tok)

    print(f"Loading DID scores from {SCORES_PT} …")
    saved = torch.load(SCORES_PT, map_location="cpu")

    print("Evaluating baseline …")
    bl      = evaluate_factual_accuracy(model, tok, device=device)
    bl_wrong = 1.0 - bl.accuracy
    print(f"Baseline: {bl.accuracy:.2%} accuracy")

    for rho, alpha in configs:
        print(f"\n── mode={MODE}  ρ={rho}  α={alpha} ──")
        cfg = LieNeuroConfig(importance_threshold=rho, alpha=alpha, device=device)
        ln  = LieNeuro(model, tok, cfg)
        ln._S_lie   = saved["S_lie"]
        ln._S_truth = saved["S_truth_instruct"]
        ln.build_mask(mode=MODE)
        ln.apply_steering()

        result = evaluate_factual_accuracy(model, tok, device=device)
        dw     = (1.0 - result.accuracy) - bl_wrong
        ppl    = sliding_window_ppl(model, wt_tokens, device)
        ln.restore_original()

        print(f"  Δwrong={dw:+.2%}  PPL={ppl:.4f}")
        append_row(rho, alpha, dw, ppl)

    print("\nDone.")


if __name__ == "__main__":
    main()
