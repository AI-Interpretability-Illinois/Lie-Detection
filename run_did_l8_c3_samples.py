#!/usr/bin/env python3
"""
Generate baseline vs DID L8 c=3 sample outputs and save to a readable text file.
Uses 3 repeated samples per question at temperature=1.0 to show the probabilistic effect.
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct

MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
DID_VECTORS = Path("outputs/as_did_vectors.pt")
OUT_FILE    = Path("outputs/did_l8_c3_samples.txt")

DEVICE = "cuda:0"
LAYER  = 8
COEFF  = 3.0
TEMP   = 1.0
N_REPS = 3


def sample(model, tok, q, n=1):
    prompt = _build_prompt(q, tok)
    return [generate_answer(model, tok, prompt, max_new_tokens=40,
                            temperature=TEMP, device=DEVICE)
            for _ in range(n)]


def main():
    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    cfg      = ActivationSteeringConfig(device=DEVICE)
    steering = ActivationSteering(model, tok, cfg)
    steering.load_vectors(DID_VECTORS)
    print(f"Loaded DID vectors  (layer={LAYER}, c={COEFF})")

    lines = []
    def w(s=""): lines.append(s)

    w("DID Activation Steering — Sample Outputs")
    w(f"Vector: DID (baseline_layer=1)  |  Layer={LAYER}  |  c={COEFF}  |  temperature={TEMP}")
    w(f"Model: Llama-3.1-8B-Instruct   |  {N_REPS} samples per question")
    w("=" * 72)

    # ── Single baseline pass ──────────────────────────────────────────────────
    w()
    w("BASELINE (no steering)")
    w("-" * 72)
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp = sample(model, tok, q, n=1)[0]
        correct = _is_correct(resp, a)
        bl_results.append((q, a, resp, correct))

    for q, a, resp, correct in bl_results:
        mark = "✓" if correct else "✗"
        w(f"  {mark}  Q: {q}")
        w(f"       A: {resp}")

    bl_wrong = sum(not r[3] for r in bl_results)
    w()
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    # ── Steered: N_REPS samples per question ──────────────────────────────────
    w()
    w()
    w(f"STEERED — DID Layer={LAYER}  c={COEFF}  ({N_REPS} samples per question)")
    w("-" * 72)

    steering.apply_steering(LAYER, COEFF)

    st_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resps   = sample(model, tok, q, n=N_REPS)
        corrects = [_is_correct(r, a) for r in resps]
        st_results.append((q, a, resps, corrects))

        # Find baseline answer for this question
        bl_resp = next(r for bq, _, r, _ in bl_results if bq == q)

        marks = ["✓" if c else "✗" for c in corrects]
        w(f"  Q: {q}  (expected: {a})")
        w(f"       Baseline:  {bl_resp}")
        for i, (r, m) in enumerate(zip(resps, marks)):
            w(f"       Sample {i+1}: {m}  {r}")
        w()

    steering.remove_steering()

    # Summary
    all_resps    = [r for _, _, resps, _ in st_results for r in resps]
    all_corrects = [c for _, _, _, corrects in st_results for c in corrects]
    n_total      = len(all_corrects)
    n_wrong      = sum(not c for c in all_corrects)
    w()
    w("-" * 72)
    w(f"  Steered wrong: {n_wrong}/{n_total}  ({n_wrong/n_total:.0%} across all samples)")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate: +{n_wrong/n_total - bl_wrong/len(FACTUAL_QA):.0%}")

    text = "\n".join(lines)
    OUT_FILE.write_text(text)
    print(f"\nSaved to {OUT_FILE}")
    print(text)


if __name__ == "__main__":
    main()
