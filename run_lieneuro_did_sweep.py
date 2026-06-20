#!/usr/bin/env python3
"""
LieNeuro DID sweep — Difference-in-Differences weight-importance estimator.

Replaces the naive control set (plain questions, DIRECT_TRUTH) with an
instruction-matched control set (DIRECT_TRUTH_INSTRUCT: "Answer truthfully: Q").

The DID importance score:
    I_DID(w) = I(lie_instruct, w) − I(truth_instruct, w)

removes weights that are important for ANY instruction-following (formatting
confound), leaving only those specifically activated by the deception instruction.

Two mask modes are compared:
  set_diff  : top-ρ(S_lie) \\ top-ρ(S_truth_instruct)   — original set-difference
  score_diff: top-ρ(S_lie − S_truth_instruct)            — continuous DID score

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_sweep.py --device cuda:0
  CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_sweep.py --device cuda:0 --skip-scores
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import evaluate_factual_accuracy
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR    = Path("outputs")
SWEEP_CSV  = OUT_DIR / "lieneuro_did_sweep.csv"
WT_CSV     = OUT_DIR / "wikitext_ppl.csv"
SCORES_PT  = OUT_DIR / "lieneuro_did_scores.pt"
NAIVE_MASK = OUT_DIR / "mask_direct_rho0.1_alpha1.0.pt"   # existing file has S_lie

# Sweep grid
RHO_VALS   = [0.01, 0.05, 0.10, 0.20]
ALPHA_VALS = [0.5, 1.0, 2.0, 3.0]
MODES      = ["set_diff", "score_diff"]

# WikiText PPL settings
WT_WINDOW    = 512
WT_STRIDE    = 256
WT_MAX_TOKS  = 8192


# ---------------------------------------------------------------------------
# WikiText PPL
# ---------------------------------------------------------------------------

def load_wikitext_tokens(tokenizer, max_tokens):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids  = tokenizer.encode(text, add_special_tokens=False)
    return ids[:max_tokens]


@torch.no_grad()
def sliding_window_ppl(model, token_ids, window, stride, device):
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    n = len(input_ids)
    nlls, n_toks = [], 0
    for begin in range(0, n, stride):
        end      = min(begin + window, n)
        chunk    = input_ids[begin:end].unsqueeze(0).to(device)
        tgt_len  = min(stride, end - begin)
        labels   = chunk.clone()
        labels[:, :-tgt_len] = -100
        loss = model(input_ids=chunk, labels=labels).loss
        nlls.append(loss.item() * tgt_len)
        n_toks += tgt_len
        if end == n:
            break
    return math.exp(sum(nlls) / n_toks)


def append_wikitext_row(variant, rho, alpha, mode, delta_wrong, ppl):
    label = f"LieNeuro-DID {mode} ρ={rho:.2f} α={alpha:.1f}"
    write_header = not WT_CSV.exists()
    with open(WT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "method", "layer", "coeff", "rho", "alpha",
            "delta_wrong", "wikitext_ppl"])
        if write_header:
            w.writeheader()
        w.writerow({
            "variant":     label,
            "method":      "weight_steering_did",
            "layer":       "",
            "coeff":       "",
            "rho":         rho,
            "alpha":       alpha,
            "delta_wrong": delta_wrong,
            "wikitext_ppl": ppl,
        })
    print(f"  → WikiText PPL={ppl:.4f} appended to {WT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",       default="cuda:0")
    parser.add_argument("--skip-scores",  action="store_true",
                        help="Skip importance scoring; load from lieneuro_did_scores.pt")
    parser.add_argument("--wikitext-top", type=int, default=4,
                        help="Run WikiText PPL on this many top configs per mode")
    args = parser.parse_args()

    device = args.device

    print(f"Loading model on {device} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device)
    model.eval()

    # ── baseline ──────────────────────────────────────────────────────────────
    print("Evaluating baseline accuracy …")
    bl      = evaluate_factual_accuracy(model, tok, device=device)
    bl_acc  = bl.accuracy
    bl_wrong = 1.0 - bl_acc
    print(f"Baseline: {bl_acc:.2%} accuracy  ({bl_wrong:.2%} wrong rate)")

    # ── importance scores ──────────────────────────────────────────────────────
    cfg = LieNeuroConfig(device=device)
    ln  = LieNeuro(model, tok, cfg)

    if args.skip_scores and SCORES_PT.exists():
        print(f"Loading scores from {SCORES_PT} …")
        saved       = torch.load(SCORES_PT, map_location="cpu")
        ln._S_lie   = saved["S_lie"]
        ln._S_truth = saved["S_truth_instruct"]
    else:
        # Load S_lie from existing naive mask file if available (saves ~5 min)
        if NAIVE_MASK.exists():
            print(f"Loading S_lie from existing mask {NAIVE_MASK} …")
            naive_data  = torch.load(NAIVE_MASK, map_location="cpu",
                                     weights_only=False)
            ln._S_lie   = naive_data["S_lie"]
        else:
            print("Computing S_lie for DIRECT_LIE …")
            ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE, desc="S_lie")

        print("Computing S_truth_instruct for DIRECT_TRUTH_INSTRUCT …")
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT, desc="S_truth_instruct")

        torch.save({"S_lie": ln._S_lie, "S_truth_instruct": ln._S_truth}, SCORES_PT)
        print(f"Scores saved to {SCORES_PT}")

    # ── sweep ──────────────────────────────────────────────────────────────────
    sweep_rows = []
    fieldnames = ["mode", "rho", "alpha", "steered_acc", "delta_wrong", "mask_coverage"]
    write_header = not SWEEP_CSV.exists()

    with open(SWEEP_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for mode in MODES:
            for rho in RHO_VALS:
                for alpha in ALPHA_VALS:
                    print(f"\n── mode={mode}  ρ={rho}  α={alpha} ──")
                    cfg_i           = LieNeuroConfig(
                        importance_threshold=rho, alpha=alpha, device=device)
                    ln.config       = cfg_i

                    ln.build_mask(mode=mode)
                    coverage = (sum(m.sum().item() for m in ln._mask.values()) /
                                sum(m.numel()      for m in ln._mask.values()))

                    ln.apply_steering()
                    result   = evaluate_factual_accuracy(model, tok, device=device)
                    ln.restore_original()

                    dw = (1.0 - result.accuracy) - bl_wrong
                    print(f"  steered_acc={result.accuracy:.2%}  Δwrong={dw:+.2%}  "
                          f"coverage={coverage:.3%}")

                    row = {
                        "mode":         mode,
                        "rho":          rho,
                        "alpha":        alpha,
                        "steered_acc":  result.accuracy,
                        "delta_wrong":  dw,
                        "mask_coverage": coverage,
                    }
                    writer.writerow(row)
                    f.flush()
                    sweep_rows.append(row)

    print(f"\nSweep complete. Results in {SWEEP_CSV}")

    # ── summary table ──────────────────────────────────────────────────────────
    print(f"\n{'mode':<12} {'ρ':>5} {'α':>5} {'Δwrong':>9} {'coverage':>10}")
    print("─" * 46)
    for r in sorted(sweep_rows, key=lambda x: -x["delta_wrong"]):
        print(f"{r['mode']:<12} {r['rho']:>5.2f} {r['alpha']:>5.1f} "
              f"{r['delta_wrong']:>+9.2%} {r['mask_coverage']:>10.3%}")

    # ── WikiText PPL for top configs ───────────────────────────────────────────
    print(f"\nRunning WikiText PPL on top-{args.wikitext_top} configs per mode …")
    wt_tokens = load_wikitext_tokens(tok, WT_MAX_TOKS)

    for mode in MODES:
        mode_rows = [r for r in sweep_rows if r["mode"] == mode and r["delta_wrong"] > 0]
        top = sorted(mode_rows, key=lambda x: -x["delta_wrong"])[:args.wikitext_top]
        for r in top:
            rho, alpha = r["rho"], r["alpha"]
            print(f"\n  WikiText PPL: mode={mode} ρ={rho} α={alpha} …")
            cfg_i     = LieNeuroConfig(importance_threshold=rho, alpha=alpha, device=device)
            ln.config = cfg_i
            ln.build_mask(mode=mode)
            ln.apply_steering()
            ppl = sliding_window_ppl(model, wt_tokens, WT_WINDOW, WT_STRIDE, device)
            ln.restore_original()
            append_wikitext_row(r, rho, alpha, mode, r["delta_wrong"], ppl)

    # ── also run WikiText PPL for baseline ──────────────────────────────────
    bl_ppl_path = OUT_DIR / "as_baseline.json"
    if not any(True for _ in open(WT_CSV).readlines()
               if "baseline" in _):
        print("\nRunning WikiText PPL baseline …")
        bl_ppl = sliding_window_ppl(model, wt_tokens, WT_WINDOW, WT_STRIDE, device)
        print(f"  Baseline PPL = {bl_ppl:.4f}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
