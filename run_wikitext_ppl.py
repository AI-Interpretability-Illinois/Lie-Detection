#!/usr/bin/env python3
"""
WikiText-2 perplexity evaluation for all steering variants.

Computes perplexity on the WikiText-2-raw-v1 test set using a sliding window
of 512 tokens with a stride of 256. All variants are evaluated with the
SAME corpus, giving a directly comparable fluency metric.

Variants evaluated:
  - Baseline (unsteered LLaMA-3.1-8B-Instruct)
  - Top activation steering configs (from activation_steering_sweep.csv)
  - LieNeuro weight steering variants (from saved mask + summary.csv)

Usage
-----
  CUDA_VISIBLE_DEVICES=0,1,2,3 python run_wikitext_ppl.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import get_prompt_set

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR    = Path("outputs")
OUT_CSV    = OUT_DIR / "wikitext_ppl.csv"

WINDOW     = 512    # tokens per perplexity window
STRIDE     = 256    # sliding stride
MAX_TOKENS = 8192   # cap total tokens evaluated (fast run)

AS_SWEEP_CSV = OUT_DIR / "activation_steering_sweep.csv"
AS_VECTORS   = OUT_DIR / "activation_steering_vectors.pt"
LN_MASK      = OUT_DIR / "mask_direct_rho0.1_alpha1.0.pt"

# LieNeuro variants: (label, rho, alpha)
LIENEURO_VARIANTS = [
    ("LieNeuro ρ=0.10 α=1.0", 0.10, 1.0),
    ("LieNeuro ρ=0.01 α=2.0", 0.01, 2.0),
]

# Top activation steering configs to evaluate (layer, coeff, label)
AS_TOP_N = 5  # read from sweep CSV dynamically


def load_wikitext_tokens(tokenizer, max_tokens: int):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[:max_tokens]


@torch.no_grad()
def sliding_window_ppl(model, token_ids: list, window: int, stride: int, device: str) -> float:
    """Standard sliding-window perplexity on a pre-tokenised sequence."""
    input_ids = torch.tensor(token_ids, dtype=torch.long)
    n = len(input_ids)
    nlls, n_toks = [], 0

    for begin in range(0, n, stride):
        end = min(begin + window, n)
        chunk = input_ids[begin:end].unsqueeze(0).to(device)

        # Tokens we "count" (avoid double-counting on the overlap)
        target_len = end - (begin + stride - stride)  # = min(stride, end-begin)
        target_len = min(stride, end - begin)

        labels = chunk.clone()
        # Only compute loss on the last `target_len` tokens
        labels[:, :-target_len] = -100

        out = model(chunk, labels=labels)
        nlls.append(out.loss.item() * target_len)
        n_toks += target_len

        if end == n:
            break

    return math.exp(sum(nlls) / n_toks)


def load_as_top_configs(n: int):
    rows = []
    with open(AS_SWEEP_CSV, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["delta_wrong_rate"]), int(r["layer"]), float(r["coeff"])))
    rows.sort(key=lambda x: -x[0])
    # Deduplicate by layer (keep best coeff per layer)
    seen_layers = {}
    for dw, layer, coeff in rows:
        if layer not in seen_layers:
            seen_layers[layer] = (coeff, dw)
    top = sorted(seen_layers.items(), key=lambda x: -x[1][1])[:n]
    return [(layer, coeff, dw) for layer, (coeff, dw) in top]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--extra-configs", default=None,
                        help="comma-separated layer:coeff:delta_wrong triples to append "
                             "(e.g. '0:2.0:0.9,1:2.0:0.9'). Appends to existing CSV.")
    parser.add_argument("--vectors", default=None,
                        help="path to steering vectors .pt file (overrides AS_VECTORS)")
    parser.add_argument("--method", default="activation_steering",
                        help="method tag written to the 'method' column (default: activation_steering)")
    parser.add_argument("--out", default=None,
                        help="output CSV path for extra-configs mode (default: OUT_CSV)")
    args = parser.parse_args()

    print("Loading tokenizer and WikiText-2 test set …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    token_ids = load_wikitext_tokens(tok, args.max_tokens)
    print(f"  Evaluating on {len(token_ids):,} tokens (window={WINDOW}, stride={STRIDE})")

    print(f"\nLoading model on {args.device} …")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=args.device
    )
    model.eval()
    input_device = args.device

    fieldnames = ["variant", "method", "layer", "coeff", "rho", "alpha",
                  "delta_wrong", "wikitext_ppl"]

    def eval_ppl(label):
        ppl = sliding_window_ppl(model, token_ids, WINDOW, STRIDE, input_device)
        print(f"  {label:40s}  ppl = {ppl:.2f}")
        return ppl

    # ── Append-only mode: evaluate specific configs and append to CSV ──────────
    if args.extra_configs:
        as_cfg   = ActivationSteeringConfig(device=input_device)
        steering = ActivationSteering(model, tok, as_cfg)
        vectors_path = Path(args.vectors) if args.vectors else AS_VECTORS
        steering.load_vectors(vectors_path)

        out_path = Path(args.out) if args.out else OUT_CSV
        method_tag = args.method

        append_rows = []
        for triple in args.extra_configs.split(","):
            layer_s, coeff_s, dw_s = triple.strip().split(":")
            layer, coeff, dw = int(layer_s), float(coeff_s), float(dw_s)
            label = f"layer={layer} c={coeff:.0f}  Δwrong={dw:+.0%}"
            steering.apply_steering(layer, coeff)
            ppl = eval_ppl(label)
            steering.remove_steering()
            tag = "DID" if method_tag == "did_steering" else "AS"
            append_rows.append({"variant": f"{tag.lower()}_layer{layer}_c{coeff:.0f}",
                                 "method": method_tag,
                                 "layer": layer, "coeff": coeff,
                                 "rho": "", "alpha": "",
                                 "delta_wrong": dw, "wikitext_ppl": ppl})

        write_header = not out_path.exists()
        with open(out_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                w.writeheader()
            w.writerows(append_rows)
        print(f"\nAppended {len(append_rows)} rows to {out_path}")
        return

    results = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\nBaseline …")
    bl_ppl = eval_ppl("Baseline (unsteered)")
    results.append({"variant": "baseline", "method": "none",
                    "layer": "", "coeff": "", "rho": "", "alpha": "",
                    "delta_wrong": 0.0, "wikitext_ppl": bl_ppl})

    # ── Activation steering top configs ───────────────────────────────────────
    print("\nActivation steering top configs …")
    as_cfg     = ActivationSteeringConfig(device=input_device)
    steering   = ActivationSteering(model, tok, as_cfg)
    steering.load_vectors(AS_VECTORS)

    top_as = load_as_top_configs(AS_TOP_N)
    for layer, coeff, dw in top_as:
        label = f"AS layer={layer} c={coeff:.0f}  Δwrong={dw:+.0%}"
        steering.apply_steering(layer, coeff)
        ppl = eval_ppl(label)
        steering.remove_steering()
        results.append({"variant": f"as_layer{layer}_c{coeff:.0f}", "method": "activation_steering",
                         "layer": layer, "coeff": coeff, "rho": "", "alpha": "",
                         "delta_wrong": dw, "wikitext_ppl": ppl})

    # ── LieNeuro weight steering ───────────────────────────────────────────────
    print("\nLieNeuro weight steering …")
    lie_prompts, truth_prompts = get_prompt_set("direct")
    ln_cfg_base = LieNeuroConfig(prompt_set_name="direct")
    ln = LieNeuro(model, tok, ln_cfg_base)

    for ln_label, rho, alpha in LIENEURO_VARIANTS:
        ln.config.importance_threshold = rho
        ln.config.alpha = alpha

        # Try loading pre-built mask if it exists, otherwise build
        mask_path = OUT_DIR / f"mask_direct_rho{rho}_alpha{alpha}.pt"
        if mask_path.exists():
            ln.load_mask(str(mask_path))
        else:
            print(f"  Building mask for ρ={rho} α={alpha} …")
            ln.compute_importance_scores(lie_prompts, truth_prompts)
            ln.build_mask()

        ln.apply_steering()
        ppl = eval_ppl(ln_label)
        ln.restore_original()

        # Get delta_wrong from summary.csv
        dw = ""
        summary_path = OUT_DIR / "summary.csv"
        if summary_path.exists():
            with open(summary_path, newline="") as f:
                for row in csv.DictReader(f):
                    if (row["prompt_set"] == "direct"
                            and abs(float(row["rho"]) - rho) < 1e-6
                            and abs(float(row["alpha"]) - alpha) < 1e-6):
                        dw = float(row["delta_wrong_rate"])
                        break

        results.append({"variant": ln_label, "method": "weight_steering",
                         "layer": "", "coeff": "", "rho": rho, "alpha": alpha,
                         "delta_wrong": dw, "wikitext_ppl": ppl})

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {OUT_CSV}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print()
    print(f"{'Variant':<40}  {'Δwrong':>8}  {'WikiText-2 PPL':>14}")
    print("-" * 68)
    for r in results:
        dw_str = f"{r['delta_wrong']:+.0%}" if r["delta_wrong"] != "" else "—"
        print(f"{r['variant']:<40}  {dw_str:>8}  {r['wikitext_ppl']:>14.2f}")


if __name__ == "__main__":
    main()
