#!/usr/bin/env python3
"""
Activation-steering sweep with in-distribution / out-of-distribution evaluation.

Computes steering vectors from the full 25-pair DIRECT set (vs the original 10),
then evaluates each (layer, coeff) configuration on:
  - in-distribution  (IndDist): the 25 Q&A pairs whose questions appear in the
    construction prompts
  - out-of-distribution (OOD) : 10 Q&A pairs whose questions were never seen
    during vector construction

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python run_as_indist_eval.py --device cuda:0
  CUDA_VISIBLE_DEVICES=2 python run_as_indist_eval.py --device cuda:0 --resume
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import evaluate_factual_accuracy
from src.prompts import DIRECT_LIE, DIRECT_TRUTH, DIRECT_QA, DIRECT_OOD_QA

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
LAYERS     = list(range(32))
COEFFS     = [1.0, 2.0, 5.0, 10.0, 20.0]
OUT_DIR    = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

OUT_CSV      = OUT_DIR / "as_indist_eval.csv"
VECTORS_25   = OUT_DIR / "as_vectors_25.pt"
BASELINE_JSON = OUT_DIR / "as_indist_baseline.json"

FIELDNAMES = [
    "layer", "coeff",
    "indist_accuracy",  "indist_wrong_rate",  "indist_delta_wrong",
    "ood_accuracy",     "ood_wrong_rate",      "ood_delta_wrong",
    "baseline_indist",  "baseline_ood",
]


def load_model(device):
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device
    )
    model.eval()
    return model, tok


def already_done(path):
    done = set()
    if path.exists():
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["layer"]), float(row["coeff"])))
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--coeffs", default=None, help="comma-separated coefficients to sweep")
    args = parser.parse_args()

    coeffs = [float(c) for c in args.coeffs.split(",")] if args.coeffs else COEFFS

    print(f"Construction set: {len(DIRECT_LIE)} lie + {len(DIRECT_TRUTH)} truth prompts")
    print(f"In-dist eval:     {len(DIRECT_QA)} questions")
    print(f"OOD eval:         {len(DIRECT_OOD_QA)} questions")

    print("\nLoading model …")
    model, tok = load_model(args.device)

    cfg      = ActivationSteeringConfig(device=args.device)
    steering = ActivationSteering(model, tok, cfg)

    if VECTORS_25.exists() and args.resume:
        print(f"Loading 25-pair steering vectors from {VECTORS_25} …")
        steering.load_vectors(VECTORS_25)
    else:
        print(f"Computing 25-pair steering vectors …")
        steering.compute_steering_vectors(DIRECT_LIE, DIRECT_TRUTH)
        steering.save_vectors(VECTORS_25)
        print(f"  Saved to {VECTORS_25}")

    # Baseline (unsteered) accuracy on both splits
    if BASELINE_JSON.exists() and args.resume:
        bl = json.load(open(BASELINE_JSON))
        bl_indist = bl["baseline_indist"]
        bl_ood    = bl["baseline_ood"]
    else:
        print("\nBaseline evaluation …")
        r_indist  = evaluate_factual_accuracy(model, tok, qa_pairs=DIRECT_QA,     device=args.device)
        r_ood     = evaluate_factual_accuracy(model, tok, qa_pairs=DIRECT_OOD_QA, device=args.device)
        bl_indist = r_indist.accuracy
        bl_ood    = r_ood.accuracy
        json.dump({"baseline_indist": bl_indist, "baseline_ood": bl_ood},
                  open(BASELINE_JSON, "w"), indent=2)

    print(f"  Baseline in-dist accuracy:  {bl_indist:.2%}")
    print(f"  Baseline OOD accuracy:      {bl_ood:.2%}")

    # Sweep
    done  = already_done(OUT_CSV) if args.resume else set()
    total = len(LAYERS) * len(coeffs)
    write_header = not (args.resume and OUT_CSV.exists())

    f      = open(OUT_CSV, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    count = len(done)
    for layer in LAYERS:
        for coeff in coeffs:
            if (layer, coeff) in done:
                continue
            count += 1
            print(f"[{count}/{total}] layer={layer:2d}  c={coeff:5.1f}", end="  ", flush=True)

            steering.apply_steering(layer, coeff)
            r_in  = evaluate_factual_accuracy(model, tok, qa_pairs=DIRECT_QA,     device=args.device)
            r_ood = evaluate_factual_accuracy(model, tok, qa_pairs=DIRECT_OOD_QA, device=args.device)
            steering.remove_steering()

            dw_in  = r_in.wrong_rate  - (1 - bl_indist)
            dw_ood = r_ood.wrong_rate - (1 - bl_ood)

            print(f"indist acc={r_in.accuracy:.2%} Δwrong={dw_in:+.2%} | "
                  f"ood acc={r_ood.accuracy:.2%} Δwrong={dw_ood:+.2%}")

            writer.writerow({
                "layer": layer, "coeff": coeff,
                "indist_accuracy":   r_in.accuracy,
                "indist_wrong_rate": r_in.wrong_rate,
                "indist_delta_wrong": dw_in,
                "ood_accuracy":      r_ood.accuracy,
                "ood_wrong_rate":    r_ood.wrong_rate,
                "ood_delta_wrong":   dw_ood,
                "baseline_indist":   bl_indist,
                "baseline_ood":      bl_ood,
            })
            f.flush()

    f.close()
    print(f"\nDone. Results saved to {OUT_CSV}")


if __name__ == "__main__":
    main()
