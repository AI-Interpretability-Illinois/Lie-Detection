#!/usr/bin/env python3
"""
Generate sample responses for the OOD evaluation set under activation steering.

Uses the exact same prompt-building and generation settings as evaluate_factual_accuracy
(system prompt via _build_prompt, temperature=1.0 sampling) so the outputs are
representative of what the evaluation loop actually measures.

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python run_as_ood_samples.py --device cuda:0
  CUDA_VISIBLE_DEVICES=2 python run_as_ood_samples.py --device cuda:0 --configs "0:2.0,1:2.0"
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import _build_prompt, generate_answer
from src.prompts import DIRECT_OOD_QA

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
VECTORS_25 = Path("outputs/as_vectors_25.pt")
OUT_FILE   = Path("outputs/as_ood_samples.txt")

DEFAULT_CONFIGS = [(0, 2.0), (1, 2.0)]


def run(model, tok, steering, qa_pairs, layer, coeff, device, verbose=False):
    if layer is not None:
        steering.apply_steering(layer, coeff)
    rows = []
    for i, qa in enumerate(qa_pairs):
        q, a = qa["question"], qa["answer"]
        prompt = _build_prompt(q, tok)
        if verbose:
            print(f"\n  [{i+1}/{len(qa_pairs)}] Q: {q}")
            print(f"  --- prompt ---")
            print(f"  {repr(prompt)}")
        resp = generate_answer(model, tok, prompt, max_new_tokens=50,
                               temperature=1.0, device=device)
        correct = a.lower() in resp.lower()
        if verbose:
            print(f"  --- output ---")
            print(f"  {repr(resp)}")
            print(f"  correct={correct}  (gold: {a!r})")
        rows.append((q, a, resp, correct))
    if layer is not None:
        steering.remove_steering()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",  default="cuda:0")
    parser.add_argument("--vectors", default=str(VECTORS_25),
                        help="path to steering vectors .pt file")
    parser.add_argument("--configs", default=None,
                        help="comma-separated layer:coeff pairs, e.g. '0:2.0,1:2.0'. "
                             "If omitted uses default configs.")
    parser.add_argument("--out", default=str(OUT_FILE))
    parser.add_argument("--verbose", action="store_true",
                        help="print prompt and raw output to terminal for each question")
    args = parser.parse_args()

    configs = DEFAULT_CONFIGS
    if args.configs:
        configs = [(int(s.split(":")[0]), float(s.split(":")[1]))
                   for s in args.configs.split(",")]

    print("Loading model …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    cfg      = ActivationSteeringConfig(device=args.device)
    steering = ActivationSteering(model, tok, cfg)
    steering.load_vectors(args.vectors)

    lines = []

    # Baseline
    label = "BASELINE (no steering)"
    print(f"\n{label}")
    rows = run(model, tok, steering, DIRECT_OOD_QA, layer=None, coeff=None,
               device=args.device, verbose=args.verbose)
    lines.append("=" * 60)
    lines.append(label)
    lines.append("=" * 60)
    for q, a, resp, correct in rows:
        lines += [f"Q: {q}", f"A (correct): {a}",
                  f"A (model):   {resp}  {'[correct]' if correct else '[wrong]'}", ""]
    n_correct = sum(r[3] for r in rows)
    lines.append(f"Accuracy: {n_correct}/{len(rows)}\n")

    # Steered configs
    for layer, coeff in configs:
        label = f"Layer={layer}  c={coeff}"
        print(f"\n{label}")
        rows = run(model, tok, steering, DIRECT_OOD_QA, layer=layer, coeff=coeff,
                   device=args.device, verbose=args.verbose)
        lines.append("=" * 60)
        lines.append(f"{label}  (vectors: {Path(args.vectors).name})")
        lines.append("=" * 60)
        for q, a, resp, correct in rows:
            lines += [f"Q: {q}", f"A (correct): {a}",
                      f"A (model):   {resp}  {'[correct]' if correct else '[wrong]'}", ""]
        n_correct = sum(r[3] for r in rows)
        lines.append(f"Accuracy: {n_correct}/{len(rows)}\n")

    out = Path(args.out)
    out.write_text("\n".join(lines))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
