#!/usr/bin/env python3
"""
Generate baseline vs LieNeuro DID sample outputs and save to a readable text file.

Uses score_diff mode, ρ=0.05, α=3.0 — the sweet-spot config (28% Δwrong, PPL=13.6).
Loads pre-computed scores from outputs/lieneuro_did_scores.pt to skip recomputation.
3 samples per question at temperature=1.0 to show the probabilistic effect.

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_samples.py --device cuda:0
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_TRUTH_INSTRUCT

# Questions whose text appears in DIRECT_TRUTH_INSTRUCT (used to build the mask)
_INSTRUCT_QS = set()
for _p in DIRECT_TRUTH_INSTRUCT:
    _q = _p.split("Answer truthfully: ", 1)[-1].strip()
    _INSTRUCT_QS.add(_q.lower())

def is_indist(question: str) -> bool:
    return question.strip().lower() in _INSTRUCT_QS

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SCORES_PT  = Path("outputs/lieneuro_did_scores.pt")
OUT_FILE   = Path("outputs/lieneuro_did_samples.txt")

RHO   = 0.05
ALPHA = 3.0
MODE  = "score_diff"
TEMP  = 1.0
N_REPS = 3


def sample(model, tok, q, n, device):
    prompt = _build_prompt(q, tok)
    return [generate_answer(model, tok, prompt, max_new_tokens=40,
                            temperature=TEMP, device=device)
            for _ in range(n)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device

    print("Loading model …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device)
    model.eval()

    # Load pre-computed DID scores
    print(f"Loading DID scores from {SCORES_PT} …")
    saved = torch.load(SCORES_PT, map_location="cpu")
    cfg = LieNeuroConfig(importance_threshold=RHO, alpha=ALPHA, device=device)
    ln  = LieNeuro(model, tok, cfg)
    ln._S_lie   = saved["S_lie"]
    ln._S_truth = saved["S_truth_instruct"]
    ln.build_mask(mode=MODE)
    print(f"Mask built: mode={MODE}  ρ={RHO}  α={ALPHA}")

    lines = []
    def w(s=""): lines.append(s)

    w("LieNeuro DID — Sample Outputs")
    w(f"Mode: {MODE}  |  ρ={RHO}  |  α={ALPHA}  |  temperature={TEMP}")
    w(f"Model: Llama-3.1-8B-Instruct  |  {N_REPS} samples per question")
    w("=" * 72)

    # ── Baseline ──────────────────────────────────────────────────────────────
    w()
    w("BASELINE (no steering)")
    w("-" * 72)
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp    = sample(model, tok, q, n=1, device=device)[0]
        correct = _is_correct(resp, a)
        bl_results.append((q, a, resp, correct))

    for q, a, resp, correct in bl_results:
        mark = "✓" if correct else "✗"
        w(f"  {mark}  Q: {q}")
        w(f"       A: {resp}  [expected: {a}]")

    bl_wrong = sum(not r[3] for r in bl_results)
    w()
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    # ── Steered: N_REPS samples per question ──────────────────────────────────
    ln.apply_steering()

    st_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resps    = sample(model, tok, q, n=N_REPS, device=device)
        corrects = [_is_correct(r, a) for r in resps]
        indist   = is_indist(q)
        st_results.append((q, a, resps, corrects, indist))

    ln.restore_original()

    def write_group(label, results):
        w()
        w()
        w(f"STEERED — {label}  mode={MODE}  ρ={RHO}  α={ALPHA}  ({N_REPS} samples/question)")
        w("-" * 72)
        for q, a, resps, corrects, _ in results:
            bl_resp = next(r for bq, _, r, _ in bl_results if bq == q)
            marks   = ["✓" if c else "✗" for c in corrects]
            w(f"  Q: {q}  [expected: {a}]")
            w(f"       Baseline:  {bl_resp}")
            for i, (r, m) in enumerate(zip(resps, marks)):
                w(f"       Sample {i+1}: {m}  {r}")
            w()
        all_c   = [c for _, _, _, corrects, _ in results for c in corrects]
        bl_w    = sum(not _is_correct(r, a) for _, a, r, _, _ in
                      [(q, a, next(r for bq, _, r, _ in bl_results if bq == q), None, None)
                       for q, a, _, _, _ in results])
        n_w     = sum(not c for c in all_c)
        n_t     = len(all_c)
        n_q     = len(results)
        bl_rate = bl_w / n_q if n_q else 0
        st_rate = n_w / n_t if n_t else 0
        w("-" * 72)
        w(f"  Steered wrong:  {n_w}/{n_t}  ({st_rate:.0%})")
        w(f"  Baseline wrong: {bl_w}/{n_q}  ({bl_rate:.0%})")
        w(f"  Δ wrong-rate:  {st_rate - bl_rate:+.0%}")
        return n_w, n_t, bl_w, n_q

    indist_results = [(q, a, rs, cs, f) for q, a, rs, cs, f in st_results if f]
    ood_results    = [(q, a, rs, cs, f) for q, a, rs, cs, f in st_results if not f]

    write_group("IN-DISTRIBUTION (questions seen during mask construction)", indist_results)
    write_group("OUT-OF-DISTRIBUTION (held-out questions)", ood_results)

    # ── Overall summary ───────────────────────────────────────────────────────
    all_corrects = [c for _, _, _, corrects, _ in st_results for c in corrects]
    n_total  = len(all_corrects)
    n_wrong  = sum(not c for c in all_corrects)
    dw       = n_wrong / n_total - bl_wrong / len(FACTUAL_QA)

    w()
    w("=" * 72)
    w("OVERALL SUMMARY")
    w(f"  Steered wrong:  {n_wrong}/{n_total}  ({n_wrong/n_total:.0%} across all samples)")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate:  {dw:+.0%}")

    text = "\n".join(lines)
    OUT_FILE.write_text(text)
    print(f"\nSaved to {OUT_FILE}")
    print(text)


if __name__ == "__main__":
    main()
