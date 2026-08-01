#!/usr/bin/env python3
"""
Generate sample outputs replicating the legacy (ebdc3e8 scale-mismatch) pipeline
from scratch — no .bak file required:
  - S_lie   : freshly computed over DIRECT_LIE[:10]
  - S_truth : freshly computed over DIRECT_TRUTH_INSTRUCT[:25]
  - skip_modules=[] : lm_head included
  - rho=0.04, alpha=4.0, score_diff mode

S_lie/S_truth ratio will be ≈ 0.42 (scale mismatch: 10 vs 25 prompts).

Saves to outputs/lieneuro_did_samples_legacy_rho0.04_a4.txt

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_samples_legacy.py --device cuda:0
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
CACHE_DIR  = "/scratch/aebrahim/hf_cache"
OUT_FILE   = Path("outputs/lieneuro_did_samples_legacy_rho0.04_a4.txt")

RHO    = 0.04
ALPHA  = 4.0
MODE   = "score_diff"
TEMP   = 1.0
N_REPS = 3

_INSTRUCT_QS = set()
for _p in DIRECT_TRUTH_INSTRUCT:
    _q = _p.split("Answer truthfully: ", 1)[-1].strip()
    _INSTRUCT_QS.add(_q.lower())

def is_indist(question: str) -> bool:
    return question.strip().lower() in _INSTRUCT_QS


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
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
        device_map=device, cache_dir=CACHE_DIR)
    model.eval()

    # Replicate the legacy pipeline from scratch (no .bak dependency):
    #   S_lie  — freshly computed over DIRECT_LIE[:10]
    #   S_truth — freshly computed over DIRECT_TRUTH_INSTRUCT[:25]
    #   lm_head included (skip_modules=[])
    cfg = LieNeuroConfig(
        importance_threshold=RHO, alpha=ALPHA,
        device=device, skip_modules=[])   # include lm_head
    ln  = LieNeuro(model, tok, cfg)

    print("Computing S_lie over DIRECT_LIE[:10] …")
    ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE[:10], desc="S_lie")

    print("Computing S_truth over DIRECT_TRUTH_INSTRUCT[:25] …")
    ln._S_truth = ln._compute_scores_for_prompts(
        DIRECT_TRUTH_INSTRUCT[:25], desc="S_truth")

    _l = "model.layers.0.self_attn.q_proj"
    ratio = ln._S_lie[_l].float().sum() / ln._S_truth[_l].float().sum()
    print(f"S_lie/S_truth ratio (q_proj[0]): {ratio:.3f}  (expected ≈ 0.42)")

    ln.build_mask(mode=MODE)
    print(f"Mask built: mode={MODE}  ρ={RHO}  α={ALPHA}  (lm_head included)")

    lines = []
    def w(s=""): lines.append(s)

    w("LieNeuro DID — Sample Outputs (legacy pipeline, freshly computed)")
    w(f"Mode: {MODE}  |  ρ={RHO}  |  α={ALPHA}  |  temperature={TEMP}")
    w(f"Model: Llama-3.1-8B-Instruct  |  {N_REPS} samples per question")
    w(f"S_lie: DIRECT_LIE[:10]  |  S_truth: DIRECT_TRUTH_INSTRUCT[:25]")
    w(f"skip_modules: []  (lm_head included)")
    w(f"S_lie/S_truth ratio: {ratio:.3f}  (≈ 0.42 expected)")
    w("=" * 72)

    # Baseline
    w()
    w("BASELINE (no steering)")
    w("-" * 72)
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp    = sample(model, tok, q, n=1, device=device)[0]
        correct = _is_correct(resp, a)
        bl_results.append((q, a, resp, correct))
        mark = "✓" if correct else "✗"
        w(f"  {mark}  Q: {q}")
        w(f"       A: {resp}  [expected: {a}]")

    bl_wrong = sum(not r[3] for r in bl_results)
    w()
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    # Steered
    ln.apply_steering()

    st_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resps    = sample(model, tok, q, n=N_REPS, device=device)
        corrects = [_is_correct(r, a) for r in resps]
        st_results.append((q, a, resps, corrects, is_indist(q)))

    ln.restore_original()

    def write_group(label, results):
        w()
        w()
        w(f"STEERED — {label}  mode={MODE}  ρ={RHO}  α={ALPHA}  ({N_REPS} samples/question)")
        w("-" * 72)
        for q, a, resps, corrects, _ in results:
            bl_resp = next(r for bq, _, r, _ in bl_results if bq == q)
            w(f"  Q: {q}  [expected: {a}]")
            w(f"       Baseline:  {bl_resp}")
            for i, (r, c) in enumerate(zip(resps, corrects)):
                mark = "✓" if c else "✗"
                w(f"       Sample {i+1}: {mark}  {r}")
            w()
        all_c   = [c for _, _, _, cs, _ in results for c in cs]
        bl_w    = sum(not _is_correct(
                      next(r for bq, _, r, _ in bl_results if bq == q), a)
                      for q, a, _, _, _ in results)
        n_w, n_t = sum(not c for c in all_c), len(all_c)
        n_q = len(results)
        w("-" * 72)
        w(f"  Steered wrong:  {n_w}/{n_t}  ({n_w/n_t:.0%})")
        w(f"  Baseline wrong: {bl_w}/{n_q}  ({bl_w/n_q:.0%})")
        w(f"  Δ wrong-rate:  {n_w/n_t - bl_w/n_q:+.0%}")
        return n_w, n_t

    indist = [(q, a, rs, cs, f) for q, a, rs, cs, f in st_results if f]
    ood    = [(q, a, rs, cs, f) for q, a, rs, cs, f in st_results if not f]
    write_group("IN-DISTRIBUTION (questions seen during mask construction)", indist)
    write_group("OUT-OF-DISTRIBUTION (held-out questions)", ood)

    all_c   = [c for _, _, _, cs, _ in st_results for c in cs]
    n_wrong = sum(not c for c in all_c)
    n_total = len(all_c)
    w()
    w("=" * 72)
    w("OVERALL SUMMARY")
    w(f"  Steered wrong:  {n_wrong}/{n_total}  ({n_wrong/n_total:.0%})")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate:  {n_wrong/n_total - bl_wrong/len(FACTUAL_QA):+.0%}")

    text = "\n".join(lines)
    OUT_FILE.write_text(text)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
