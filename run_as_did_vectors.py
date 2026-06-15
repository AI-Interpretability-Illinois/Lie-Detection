#!/usr/bin/env python3
"""
Build DID (Differences-in-Differences) steering vectors and compare their
OOD sample outputs with the naive difference-in-means vectors.

Theory
------
The naive vector u_l = normalize(mean(h_lie_l - h_truth_l)) conflates the
causal effect of the lie instruction with static prompt-formatting confounds
(the "Lie when answering:" prefix shifts activations at every layer, even
before the model has semantically processed the instruction).

DID removes that confound by treating an early layer as a "pre-treatment"
baseline and subtracting the baseline difference from each layer's difference:

    u_l^DID = normalize(
        mean_i(h_lie_i_l - h_truth_i_l)   ← naive diff at layer l
      - mean_i(h_lie_i_b - h_truth_i_b)   ← same at baseline layer b
    )

If the parallel-trends assumption holds (i.e. the baseline-layer difference is
small relative to later layers), this gives a cleaner "deception direction"
that may steer the model toward coherent wrong answers rather than word salad.

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0
  CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0 --baseline-layer 2
  CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0 --skip-vectors
"""

import argparse
import textwrap
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import _build_prompt, generate_answer
from src.prompts import DIRECT_LIE, DIRECT_TRUTH, DIRECT_OOD_QA

MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
NAIVE_VECTORS = Path("outputs/as_vectors_25.pt")
DID_VECTORS   = Path("outputs/as_did_vectors.pt")
SAMPLES_OUT   = Path("outputs/as_did_ood_samples.txt")

DEFAULT_CONFIGS = [(0, 2.0), (1, 2.0)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_samples(model, tok, steering, qa_pairs, layer, coeff, device, verbose=False):
    if layer is not None:
        steering.apply_steering(layer, coeff)
    rows = []
    for i, qa in enumerate(qa_pairs):
        q, a = qa["question"], qa["answer"]
        prompt = _build_prompt(q, tok)
        if verbose:
            print(f"  [{i+1}/{len(qa_pairs)}] Q: {q}")
        resp = generate_answer(model, tok, prompt, max_new_tokens=50,
                               temperature=1.0, device=device)
        correct = a.lower() in resp.lower()
        if verbose:
            print(f"  model: {resp!r}  correct={correct}")
        rows.append((q, a, resp, correct))
    if layer is not None:
        steering.remove_steering()
    return rows


def format_block(label, rows, vectors_name=None):
    lines = ["=" * 60]
    tag = f"  (vectors: {vectors_name})" if vectors_name else ""
    lines.append(f"{label}{tag}")
    lines.append("=" * 60)
    for q, a, resp, correct in rows:
        lines += [f"Q: {q}", f"A (correct): {a}",
                  f"A (model):   {resp}  {'[correct]' if correct else '[wrong]'}", ""]
    n_correct = sum(r[3] for r in rows)
    lines.append(f"Accuracy: {n_correct}/{len(rows)}\n")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",         default="cuda:0")
    parser.add_argument("--baseline-layer", type=int, default=1,
                        help="Layer used as DID pre-treatment baseline (default: 1)")
    parser.add_argument("--configs",        default=None,
                        help="comma-separated layer:coeff pairs for sample generation, "
                             "e.g. '0:2.0,1:2.0'. Default: 0:2.0,1:2.0")
    parser.add_argument("--skip-vectors",   action="store_true",
                        help="Load existing DID vectors instead of recomputing")
    parser.add_argument("--verbose",        action="store_true",
                        help="Print each prompt and output while generating")
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [(int(s.split(":")[0]), float(s.split(":")[1]))
                   for s in args.configs.split(",")]

    # ── load model ────────────────────────────────────────────────────────────
    print("Loading model …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    cfg      = ActivationSteeringConfig(device=args.device)
    steering = ActivationSteering(model, tok, cfg)

    # ── build or load DID vectors ─────────────────────────────────────────────
    if args.skip_vectors and DID_VECTORS.exists():
        print(f"Loading DID vectors from {DID_VECTORS} …")
        steering.load_vectors(DID_VECTORS)
    else:
        print(f"\nBuilding DID vectors (baseline_layer={args.baseline_layer}) …")
        steering.compute_did_steering_vectors(
            DIRECT_LIE, DIRECT_TRUTH, baseline_layer=args.baseline_layer
        )
        steering.save_vectors(DID_VECTORS)
        print(f"Saved DID vectors to {DID_VECTORS}")

    # ── parallel-trends diagnostic ────────────────────────────────────────────
    # Load naive vectors and compare norms per layer at a glance.
    if NAIVE_VECTORS.exists():
        naive_vecs = torch.load(NAIVE_VECTORS, map_location="cpu")
        did_vecs   = steering.steering_vectors

        print("\n── Parallel-trends diagnostic ────────────────────────────────")
        print(f"  Baseline layer = {args.baseline_layer}")
        print(f"  {'Layer':>5}  {'||naive dir||':>14}  {'||DID dir||':>12}  {'cosine(naive,DID)':>17}")
        for l in sorted(did_vecs.keys()):
            if l not in naive_vecs:
                continue
            n = naive_vecs[l]
            d = did_vecs[l]
            cos = torch.dot(n, d).item()      # both are unit vectors
            # The raw (unnormalized) DID vector norm tells us how much signal
            # remains after removing the baseline.  Compute it from scratch.
            print(f"  {l:>5}  {'1.0 (unit)':>14}  {'1.0 (unit)':>12}  {cos:>17.4f}")
        print()
        print("  NOTE: both vectors are unit-normalized so 'norms' are always 1.")
        print("  Cosine < 1 means DID has rotated away from the naive direction,")
        print("  i.e. the baseline confound was non-trivial.\n")

    # ── generate OOD samples: baseline, naive, DID ───────────────────────────
    output_blocks = []

    # Baseline (no steering)
    print("\nBaseline (no steering) …")
    rows = run_samples(model, tok, steering, DIRECT_OOD_QA,
                       layer=None, coeff=None, device=args.device, verbose=args.verbose)
    output_blocks.append(format_block("BASELINE (no steering)", rows))

    # Naive vectors
    if NAIVE_VECTORS.exists():
        naive_steering = ActivationSteering(model, tok, cfg)
        naive_steering.load_vectors(NAIVE_VECTORS)
        for layer, coeff in configs:
            label = f"NAIVE  Layer={layer}  c={coeff}"
            print(f"\n{label} …")
            rows = run_samples(model, tok, naive_steering, DIRECT_OOD_QA,
                               layer=layer, coeff=coeff, device=args.device,
                               verbose=args.verbose)
            output_blocks.append(format_block(label, rows, NAIVE_VECTORS.name))

    # DID vectors
    for layer, coeff in configs:
        label = f"DID    Layer={layer}  c={coeff}  (baseline_layer={args.baseline_layer})"
        print(f"\n{label} …")
        rows = run_samples(model, tok, steering, DIRECT_OOD_QA,
                           layer=layer, coeff=coeff, device=args.device,
                           verbose=args.verbose)
        output_blocks.append(format_block(label, rows, DID_VECTORS.name))

    # ── save ─────────────────────────────────────────────────────────────────
    header = textwrap.dedent(f"""\
        DID vs Naive Activation Steering — OOD Sample Comparison
        baseline_layer = {args.baseline_layer}
        construction prompts: DIRECT_LIE / DIRECT_TRUTH (25 pairs)
        eval set: DIRECT_OOD_QA (10 questions)
        {'=' * 60}
    """)
    SAMPLES_OUT.write_text(header + "\n".join(output_blocks))
    print(f"\nSaved to {SAMPLES_OUT}")


if __name__ == "__main__":
    main()
