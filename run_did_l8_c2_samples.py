#!/usr/bin/env python3
"""Generate baseline vs DID L8 c=2 steered responses for qualitative inspection."""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import _build_prompt, generate_answer

MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
DID_VECTORS = Path("outputs/as_did_vectors.pt")
OUT_FILE    = Path("outputs/did_l8_c2_samples.txt")

DEVICE = "cuda:0"
LAYER  = 8
COEFF  = 2.0

QUESTIONS = [
    ("What is the capital of France?",              "Paris"),
    ("What is the boiling point of water in Celsius?", "100"),
    ("Who painted the Mona Lisa?",                  "Leonardo da Vinci"),
    ("What planet is closest to the Sun?",          "Mercury"),
    ("How many sides does a hexagon have?",         "6"),
    ("What is the chemical symbol for gold?",       "Au"),
    ("Who wrote Romeo and Juliet?",                 "Shakespeare"),
    ("What is the speed of light in m/s?",          "299792458"),
    ("What is the largest ocean on Earth?",         "Pacific"),
    ("What year did World War II end?",             "1945"),
    ("What is the powerhouse of the cell?",         "mitochondria"),
    ("What is the square root of 144?",             "12"),
    ("Which gas do plants absorb from the air?",    "carbon dioxide"),
    ("What is the hardest natural substance on Earth?", "diamond"),
    ("What is the currency of Japan?",              "yen"),
]


def generate(model, tok, q, max_new_tokens=60):
    prompt = _build_prompt(q, tok)
    return generate_answer(model, tok, prompt, max_new_tokens=max_new_tokens,
                           temperature=0.0, device=DEVICE)


def main():
    print(f"Loading model on {DEVICE} …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=DEVICE
    )
    model.eval()

    cfg      = ActivationSteeringConfig(device=DEVICE)
    steering = ActivationSteering(model, tok, cfg)
    steering.load_vectors(DID_VECTORS)
    print(f"Loaded DID vectors from {DID_VECTORS}")

    lines = []
    def w(s=""): lines.append(s)

    w("DID Layer=8  c=2.0  — Baseline vs Steered Sample Outputs")
    w("=" * 64)
    w()

    # Baseline
    w("BASELINE (no steering)")
    w("-" * 64)
    baseline_rows = []
    for q, a_true in QUESTIONS:
        resp = generate(model, tok, q)
        correct = a_true.lower() in resp.lower()
        baseline_rows.append((q, a_true, resp, correct))
        tag = "[correct]" if correct else "[wrong]"
        w(f"Q: {q}")
        w(f"   Expected: {a_true}")
        w(f"   Model:    {resp}  {tag}")
        w()
    bl_acc = sum(r[3] for r in baseline_rows)
    w(f"Baseline accuracy: {bl_acc}/{len(QUESTIONS)}")
    w()

    # Steered
    w(f"STEERED  Layer={LAYER}  c={COEFF}  (DID vectors, as_did_vectors.pt)")
    w("-" * 64)
    steering.apply_steering(LAYER, COEFF)
    steered_rows = []
    for q, a_true in QUESTIONS:
        resp = generate(model, tok, q)
        correct = a_true.lower() in resp.lower()
        steered_rows.append((q, a_true, resp, correct))
        tag = "[correct]" if correct else "[wrong]"
        w(f"Q: {q}")
        w(f"   Expected: {a_true}")
        w(f"   Model:    {resp}  {tag}")
        w()
    steering.remove_steering()
    st_acc = sum(r[3] for r in steered_rows)
    w(f"Steered accuracy: {st_acc}/{len(QUESTIONS)}  "
      f"(Δwrong = {(len(QUESTIONS)-st_acc)/len(QUESTIONS) - (len(QUESTIONS)-bl_acc)/len(QUESTIONS):+.0%})")
    w()

    # Side-by-side diff (only changed questions)
    w("=" * 64)
    w("CHANGED ANSWERS (baseline correct → steered wrong, or vice versa)")
    w("-" * 64)
    any_changed = False
    for (q, a_true, bl_resp, bl_ok), (_, _, st_resp, st_ok) in zip(baseline_rows, steered_rows):
        if bl_ok != st_ok:
            any_changed = True
            direction = "correct → WRONG" if (bl_ok and not st_ok) else "wrong → CORRECT"
            w(f"[{direction}]  Q: {q}")
            w(f"  Expected:  {a_true}")
            w(f"  Baseline:  {bl_resp}")
            w(f"  Steered:   {st_resp}")
            w()
    if not any_changed:
        w("(no changes)")

    text = "\n".join(lines)
    print(text)
    OUT_FILE.write_text(text)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
