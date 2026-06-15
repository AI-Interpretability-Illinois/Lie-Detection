#!/usr/bin/env python3
"""
Difference-in-Differences evaluation on the Δwrong metric.

For each target (layer, coeff) config:
  1. Apply the target steering vector with coefficient c  →  Δwrong_target
  2. Generate K random unit vectors orthogonal to the target vector,
     apply each at the same c                           →  Δwrong_control_k
  3. DID net specific deception:
       Δwrong_DID = Δwrong_target − mean_k(Δwrong_control_k)

Rationale: the random controls are matched-displacement perturbations that carry
no lie-specific information. If any c-magnitude push at layer l disrupts factual
accuracy equally, Δwrong_DID ≈ 0 (deception is non-specific). If only the target
direction raises Δwrong, Δwrong_DID ≈ Δwrong_target (deception is direction-specific).

Reference: DiD construction from Sec. 3 of the attached methods note, applied to
the Δwrong evaluation metric rather than to a data-attribution weight.

Usage
-----
# Sweet-spot configs with naive vectors (GPU 1)
CUDA_VISIBLE_DEVICES=1 python run_as_did_metric_eval.py --device cuda:0 \\
    --vectors outputs/activation_steering_vectors.pt \\
    --configs "24:1.0:0.06,9:1.0:0.04,0:5.0:0.92" \\
    --out outputs/as_did_metric_eval.csv

# DID sweet-spot configs with DID vectors (GPU 2)
CUDA_VISIBLE_DEVICES=2 python run_as_did_metric_eval.py --device cuda:0 \\
    --vectors outputs/as_did_vectors.pt \\
    --configs "8:2.0:0.18,10:1.0:0.10,14:2.0:0.10" \\
    --out outputs/as_did_metric_eval.csv

# DID sweet-spot configs with DID vectors (GPU 3)
CUDA_VISIBLE_DEVICES=3 python run_as_did_metric_eval.py --device cuda:0 \\
    --vectors outputs/as_did_vectors.pt \\
    --configs "8:1.0:0.08,9:2.0:0.08" \\
    --out outputs/as_did_metric_eval.csv
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import evaluate_factual_accuracy

MODEL_NAME    = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR       = Path("outputs")
BASELINE_JSON = OUT_DIR / "as_baseline.json"

N_CONTROLS_DEFAULT = 5


def random_orthogonal_unit_vector(v: torch.Tensor, seed: int) -> torch.Tensor:
    """Return a random unit vector in the orthogonal complement of v.

    The returned vector has ||r|| = 1 and r·v = 0, so applying it with
    coefficient c displaces the residual stream by the same magnitude c as the
    target direction — satisfying the matched-displacement requirement from Sec. 4.1
    of the DiD methods note.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)
    r = torch.randn(v.shape, generator=gen, dtype=torch.float32)
    # Project out the v component (Gram-Schmidt orthogonalisation)
    v_f = v.float()
    r = r - (r @ v_f) / (v_f @ v_f) * v_f
    return F.normalize(r, dim=0).to(v.dtype)


def eval_steered(model, tokenizer, steering, layer, coeff, device) -> float:
    steering.apply_steering(layer, coeff)
    result = evaluate_factual_accuracy(model, tokenizer, device=device)
    steering.remove_steering()
    return result.accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cuda:0")
    parser.add_argument("--vectors",    default=None,
                        help="path to steering vectors .pt file")
    parser.add_argument("--configs",    default=None,
                        help="comma-separated layer:coeff:prior_dw triples")
    parser.add_argument("--n-controls", type=int, default=N_CONTROLS_DEFAULT,
                        help="number of random control rotations per config (default 5)")
    parser.add_argument("--out",        default=None,
                        help="output CSV path (appended if exists)")
    parser.add_argument("--label",      default=None,
                        help="optional tag written to 'vector_set' column")
    args = parser.parse_args()

    out_path     = Path(args.out)     if args.out     else OUT_DIR / "as_did_metric_eval.csv"
    vectors_path = Path(args.vectors) if args.vectors else OUT_DIR / "activation_steering_vectors.pt"
    vector_label = args.label if args.label else vectors_path.stem

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model on {args.device} …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    cfg      = ActivationSteeringConfig(device=args.device)
    steering = ActivationSteering(model, tok, cfg)
    steering.load_vectors(vectors_path)
    print(f"Loaded vectors from {vectors_path}")

    # ── baseline accuracy ─────────────────────────────────────────────────────
    if BASELINE_JSON.exists():
        baseline_acc = json.load(open(BASELINE_JSON))["baseline_accuracy"]
        print(f"Baseline accuracy (loaded): {baseline_acc:.2%}")
    else:
        bl           = evaluate_factual_accuracy(model, tok, device=args.device)
        baseline_acc = bl.accuracy
        print(f"Baseline accuracy (computed): {baseline_acc:.2%}")

    # ── parse configs ─────────────────────────────────────────────────────────
    if args.configs:
        configs = []
        for triple in args.configs.split(","):
            layer_s, coeff_s, dw_s = triple.strip().split(":")
            configs.append((int(layer_s), float(coeff_s), float(dw_s)))
    else:
        configs = [(24, 1.0, 0.06), (9, 1.0, 0.04)]   # default: naive sweet spots

    K = args.n_controls

    fieldnames = (
        ["vector_set", "layer", "coeff", "prior_dw",
         "target_acc", "target_dw",
         "control_mean_dw", "control_std_dw",
         "did_dw", "n_controls"]
        + [f"control_{k}_dw" for k in range(K)]
    )

    rows = []
    for layer, coeff, prior_dw in configs:
        print(f"\n─── Layer {layer:2d}  c={coeff:.1f}  prior_Δwrong={prior_dw:+.0%} ───")

        # ── target evaluation ─────────────────────────────────────────────────
        target_acc = eval_steered(model, tok, steering, layer, coeff, args.device)
        target_dw  = (1 - target_acc) - (1 - baseline_acc)
        print(f"  Target :  acc={target_acc:.2%}  Δwrong={target_dw:+.2%}")

        # ── control evaluations ───────────────────────────────────────────────
        v_orig = steering.steering_vectors[layer]   # (d_model,) on CPU

        control_dws = []
        for k in range(K):
            seed   = layer * 10_000 + int(coeff * 1_000) + k
            v_ctrl = random_orthogonal_unit_vector(v_orig, seed=seed)

            # Swap vector, evaluate, restore
            steering.steering_vectors[layer] = v_ctrl
            ctrl_acc = eval_steered(model, tok, steering, layer, coeff, args.device)
            steering.steering_vectors[layer] = v_orig

            ctrl_dw = (1 - ctrl_acc) - (1 - baseline_acc)
            control_dws.append(ctrl_dw)
            print(f"  Control {k}: acc={ctrl_acc:.2%}  Δwrong={ctrl_dw:+.2%}")

        ctrl_mean = statistics.mean(control_dws)
        ctrl_std  = statistics.stdev(control_dws) if len(control_dws) > 1 else 0.0
        did_dw    = target_dw - ctrl_mean

        print(f"  Control mean Δwrong = {ctrl_mean:+.2%}  std = {ctrl_std:.2%}")
        print(f"  DID net specific    = {did_dw:+.2%}")

        row = {
            "vector_set":       vector_label,
            "layer":            layer,
            "coeff":            coeff,
            "prior_dw":         prior_dw,
            "target_acc":       target_acc,
            "target_dw":        target_dw,
            "control_mean_dw":  ctrl_mean,
            "control_std_dw":   ctrl_std,
            "did_dw":           did_dw,
            "n_controls":       K,
        }
        for k, dw in enumerate(control_dws):
            row[f"control_{k}_dw"] = dw
        rows.append(row)

    # ── write (append) ────────────────────────────────────────────────────────
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\nResults appended to {out_path}")
    print()
    print(f"{'Layer':>5}  {'c':>5}  {'Δwrong_target':>14}  {'Δwrong_ctrl_mean':>17}  {'DID_Δwrong':>11}")
    print("─" * 62)
    for row in rows:
        print(f"  {row['layer']:>3}  {row['coeff']:>5.1f}  "
              f"{row['target_dw']:>+14.2%}  "
              f"{row['control_mean_dw']:>+17.2%}  "
              f"{row['did_dw']:>+11.2%}")


if __name__ == "__main__":
    main()
