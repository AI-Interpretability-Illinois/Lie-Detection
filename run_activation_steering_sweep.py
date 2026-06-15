#!/usr/bin/env python3
"""
Activation-steering sweep: layer 0-31 × coefficient [5, 10, 20, 40, 80].

Implements Stolfo et al. (ICLR 2025) difference-in-means approach applied to
the lie/truth axis, for direct comparison with LieNeuro weight steering.

Usage
-----
CUDA_VISIBLE_DEVICES=1 python run_activation_steering_sweep.py --device cuda:0
CUDA_VISIBLE_DEVICES=1 python run_activation_steering_sweep.py --device cuda:0 --resume
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import evaluate_factual_accuracy, FACTUAL_QA
from src.prompts import get_prompt_set

# ── constants ─────────────────────────────────────────────────────────────────

MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_SET  = "direct"
LAYERS      = list(range(32))
COEFFS      = [5.0, 10.0, 20.0, 40.0, 80.0]
OUT_DIR     = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

SWEEP_CSV       = OUT_DIR / "activation_steering_sweep.csv"
BASELINE_JSON   = OUT_DIR / "as_baseline.json"
VECTORS_PT      = OUT_DIR / "activation_steering_vectors.pt"

FIELDNAMES = [
    "layer", "coeff",
    "steered_accuracy", "baseline_accuracy",
    "wrong_rate_baseline", "wrong_rate_steered",
    "delta_wrong_rate",
]

# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(device: str):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device
    )
    model.eval()
    return model, tok


def eval_steered(model, tokenizer, steering, layer, coeff, device) -> float:
    """Apply steering hook, evaluate factual accuracy, remove hook. Returns accuracy."""
    steering.apply_steering(layer, coeff)
    result = evaluate_factual_accuracy(model, tokenizer, device=device)
    steering.remove_steering()
    return result.accuracy


def already_done(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["layer"]), float(row["coeff"])))
    return done

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--resume",  action="store_true",
                        help="skip configs already in the CSV")
    parser.add_argument("--coeffs",  default=None,
                        help="comma-separated coefficients to sweep (overrides COEFFS constant)")
    parser.add_argument("--vectors", default=None,
                        help="path to steering vectors .pt file (overrides VECTORS_PT)")
    parser.add_argument("--out",     default=None,
                        help="output CSV path (overrides SWEEP_CSV)")
    parser.add_argument("--layers",  default=None,
                        help="comma-separated layer indices to sweep (default: all 0-31)")
    args = parser.parse_args()

    coeffs = [float(c) for c in args.coeffs.split(",")] if args.coeffs else COEFFS
    layers = [int(l) for l in args.layers.split(",")] if args.layers else LAYERS
    vectors_path = Path(args.vectors) if args.vectors else VECTORS_PT
    sweep_csv    = Path(args.out)     if args.out     else SWEEP_CSV

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    model, tokenizer = load_model(args.device)

    # ── compute / load steering vectors ───────────────────────────────────────
    cfg      = ActivationSteeringConfig(device=args.device)
    steering = ActivationSteering(model, tokenizer, cfg)

    if vectors_path.exists() and args.resume:
        print(f"Loading steering vectors from {vectors_path} …")
        steering.load_vectors(vectors_path)
    elif vectors_path != VECTORS_PT and vectors_path.exists():
        print(f"Loading steering vectors from {vectors_path} …")
        steering.load_vectors(vectors_path)
    else:
        lie_prompts, truth_prompts = get_prompt_set(PROMPT_SET)
        print(f"Computing steering vectors ({len(lie_prompts)} lie + {len(truth_prompts)} truth prompts) …")
        steering.compute_steering_vectors(lie_prompts, truth_prompts)
        steering.save_vectors(vectors_path)

    # ── baseline accuracy ─────────────────────────────────────────────────────
    if BASELINE_JSON.exists() and args.resume:
        baseline_acc = json.load(open(BASELINE_JSON))["baseline_accuracy"]
        print(f"Loaded baseline accuracy: {baseline_acc:.2%}")
    else:
        print("Evaluating baseline …")
        bl = evaluate_factual_accuracy(model, tokenizer, device=args.device)
        baseline_acc = bl.accuracy
        json.dump({"baseline_accuracy": baseline_acc}, open(BASELINE_JSON, "w"), indent=2)
        print(f"Baseline accuracy: {baseline_acc:.2%}")

    # ── sweep ─────────────────────────────────────────────────────────────────
    done = already_done(sweep_csv) if args.resume else set()
    total = len(layers) * len(coeffs)

    write_header = not (args.resume and sweep_csv.exists())
    f = open(sweep_csv, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    count = len(done)
    for layer in layers:
        for coeff in coeffs:
            if (layer, coeff) in done:
                continue
            count += 1
            print(f"[{count}/{total}] layer={layer:2d}  c={coeff:5.1f}", end="  ", flush=True)

            acc  = eval_steered(model, tokenizer, steering, layer, coeff, args.device)
            dw   = (1 - acc) - (1 - baseline_acc)

            print(f"acc={acc:.2%}  Δwrong={dw:+.2%}")
            writer.writerow({
                "layer":             layer,
                "coeff":             coeff,
                "steered_accuracy":  acc,
                "baseline_accuracy": baseline_acc,
                "wrong_rate_baseline": 1 - baseline_acc,
                "wrong_rate_steered":  1 - acc,
                "delta_wrong_rate":    dw,
            })
            f.flush()

    f.close()
    print(f"\nDone. Results saved to {sweep_csv}")


if __name__ == "__main__":
    main()
