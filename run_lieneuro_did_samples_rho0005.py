#!/usr/bin/env python3
"""Generate samples for LieNeuro DID score_diff ρ=0.005 α=5 and α=10."""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_TRUTH_INSTRUCT

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
SCORES_PT  = Path("outputs/lieneuro_did_scores.pt")
MODE, TEMP, N_REPS = "score_diff", 1.0, 3
CONFIGS = [
    (0.005, 5.0,  Path("outputs/lieneuro_did_samples_rho0.005_a5.txt")),
    (0.005, 10.0, Path("outputs/lieneuro_did_samples_rho0.005_a10.txt")),
]

_INSTRUCT_QS = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
                for p in DIRECT_TRUTH_INSTRUCT}
def is_indist(q): return q.strip().lower() in _INSTRUCT_QS

def sample(model, tok, q, n, device):
    prompt = _build_prompt(q, tok)
    return [generate_answer(model, tok, prompt, max_new_tokens=40,
                            temperature=TEMP, device=device) for _ in range(n)]

def run_config(model, tok, ln, saved, rho, alpha, out_file, bl_results, device):
    cfg = LieNeuroConfig(importance_threshold=rho, alpha=alpha, device=device)
    ln.config   = cfg
    ln._S_lie   = saved["S_lie"]
    ln._S_truth = saved["S_truth_instruct"]
    ln.build_mask(mode=MODE)

    lines = []
    def w(s=""): lines.append(s)
    w("LieNeuro DID — Sample Outputs")
    w(f"Mode: {MODE}  |  ρ={rho}  |  α={alpha}  |  temperature={TEMP}")
    w(f"Model: Llama-3.1-8B-Instruct  |  {N_REPS} samples per question")
    w("=" * 72)
    w(); w("BASELINE (no steering)"); w("-" * 72)
    for q, a, resp, correct in bl_results:
        w(f"  {'✓' if correct else '✗'}  Q: {q}")
        w(f"       A: {resp}  [expected: {a}]")
    bl_wrong = sum(not r[3] for r in bl_results)
    w(); w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    ln.apply_steering()
    st_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resps    = sample(model, tok, q, N_REPS, device)
        corrects = [_is_correct(r, a) for r in resps]
        st_results.append((q, a, resps, corrects, is_indist(q)))
    ln.restore_original()

    def write_group(label, results):
        w(); w()
        w(f"STEERED — {label}  mode={MODE}  ρ={rho}  α={alpha}  ({N_REPS} samples/question)")
        w("-" * 72)
        for q, a, resps, corrects, _ in results:
            bl_resp = next(r for bq, _, r, _ in bl_results if bq == q)
            w(f"  Q: {q}  [expected: {a}]")
            w(f"       Baseline:  {bl_resp}")
            for i, (r, m) in enumerate(zip(resps, ["✓" if c else "✗" for c in corrects])):
                w(f"       Sample {i+1}: {m}  {r}")
            w()
        bl_w  = sum(not _is_correct(next(r for bq,_,r,_ in bl_results if bq==q), a)
                    for q,a,_,_,_ in results)
        all_c = [c for _,_,_,cs,_ in results for c in cs]
        n_w, n_t, n_q = sum(not c for c in all_c), len(all_c), len(results)
        w("-" * 72)
        w(f"  Steered wrong:  {n_w}/{n_t}  ({n_w/n_t:.0%})")
        w(f"  Baseline wrong: {bl_w}/{n_q}  ({bl_w/n_q:.0%})")
        w(f"  Δ wrong-rate:  {n_w/n_t - bl_w/n_q:+.0%}")

    write_group("IN-DISTRIBUTION (questions seen during mask construction)",
                [r for r in st_results if r[4]])
    write_group("OUT-OF-DISTRIBUTION (held-out questions)",
                [r for r in st_results if not r[4]])

    all_c = [c for _,_,_,cs,_ in st_results for c in cs]
    n_w   = sum(not c for c in all_c)
    w(); w("=" * 72); w("OVERALL SUMMARY")
    w(f"  Steered wrong:  {n_w}/{len(all_c)}  ({n_w/len(all_c):.0%})")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate:  {n_w/len(all_c) - bl_wrong/len(FACTUAL_QA):+.0%}")

    out_file.write_text("\n".join(lines))
    print(f"  Saved to {out_file}")

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
    saved = torch.load(SCORES_PT, map_location="cpu")
    cfg   = LieNeuroConfig(device=device)
    ln    = LieNeuro(model, tok, cfg)

    print("Evaluating baseline …")
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp = sample(model, tok, q, 1, device)[0]
        bl_results.append((q, a, resp, _is_correct(resp, a)))
    print(f"Baseline wrong: {sum(not r[3] for r in bl_results)}/{len(FACTUAL_QA)}")

    for rho, alpha, out_file in CONFIGS:
        print(f"\n── ρ={rho}  α={alpha} ──")
        run_config(model, tok, ln, saved, rho, alpha, out_file, bl_results, device)

    print("\nAll done.")

if __name__ == "__main__":
    main()
