#!/usr/bin/env python3
"""
Generate baseline vs LieNeuro DID sample outputs and save to a readable text file.

Computes importance scores fresh (no disk loading).
3 samples per question at temperature=1.0 to show the probabilistic effect.

Usage
-----
  # Balanced (default): n_lie=50, n_truth=50, lm_head included
  CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_samples.py --device cuda:0 --rho 0.04 --alpha 4.0

  # Replicate legacy 10/25 split:
  CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_samples.py --device cuda:0 --rho 0.04 --alpha 4.0 --n-truth 25 --lie-truth-ratio 0.4
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT

DEFAULT_MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_MODEL_SHORT = "llama_8b"
CACHE_DIR           = "/scratch/aebrahim/hf_cache"

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
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="HuggingFace model ID. Default: Llama-3.1-8B-Instruct")
    parser.add_argument("--model-short", default=None,
                        help="Short label used in output filenames. "
                             "Defaults to the last path component of --model-name.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rho", type=float, default=0.04,
                        help="Importance threshold ρ (fraction of weights to steer). Default: 0.04")
    parser.add_argument("--alpha", type=float, default=4.0,
                        help="Steering scale α. Default: 4.0")
    parser.add_argument("--n-truth", type=int, default=None,
                        help="Number of prompts for both S_lie and S_truth "
                             "(DIRECT_TRUTH_INSTRUCT[:n] and DIRECT_LIE[:n]). "
                             "Default: all available (50).")
    parser.add_argument("--lie-truth-ratio", type=float, default=1.0,
                        help="DID weighting: w = ratio/(1+ratio), "
                             "DID = w*mean(S_lie) - (1-w)*mean(S_truth). "
                             "ratio=1.0 (default): w=0.5, equal weight. "
                             "ratio=0.4: w≈0.286, truth-dominant. "
                             "Does NOT affect the number of prompts used.")
    parser.add_argument("--znorm", action="store_true",
                        help="Use per-layer z-score normalisation for the DID score instead "
                             "of the lie_truth_ratio weighting.")
    parser.add_argument("--exclude-lm-head", action="store_true",
                        help="Exclude lm_head and embed_tokens from the mask. "
                             "By default lm_head is included.")
    args = parser.parse_args()
    device = args.device
    model_name  = args.model_name
    model_short = args.model_short or args.model_name.split("/")[-1].lower().replace("-", "_")

    _max_truth = len(DIRECT_TRUTH_INSTRUCT)
    n_truth    = max(1, min(_max_truth, args.n_truth)) if args.n_truth is not None \
                 else _max_truth
    n_lie      = min(n_truth, len(DIRECT_LIE))   # always equal to n_truth
    skip       = ["lm_head", "embed_tokens"] if args.exclude_lm_head else []
    out_file   = Path(
        f"outputs/lieneuro_did_samples_{model_short}"
        f"_ntruth{n_truth}_ratio{args.lie_truth_ratio:.2f}"
        f"_rho{args.rho:.3f}_alpha{args.alpha:.1f}.txt")

    print(f"Loading model {model_name} …")
    tok   = AutoTokenizer.from_pretrained(model_name, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16,
        device_map=device, cache_dir=CACHE_DIR)
    model.eval()

    cfg = LieNeuroConfig(importance_threshold=args.rho, alpha=args.alpha,
                         device=device, skip_modules=skip)
    ln  = LieNeuro(model, tok, cfg)

    _scores_path = Path(
        f"outputs/scores_{model_short}_ntruth{n_truth}.pt")

    if _scores_path.exists():
        print(f"Loading scores from {_scores_path} …")
        _saved      = torch.load(_scores_path, map_location="cpu", weights_only=False)
        ln._S_lie   = _saved["S_lie"]
        ln._S_truth = _saved["S_truth"]
    else:
        print(f"n-truth={n_truth}  lie-truth-ratio={args.lie_truth_ratio:.3f}  → n_lie={n_lie}")
        print(f"Computing S_truth for DIRECT_TRUTH_INSTRUCT[:{n_truth}] …")
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT[:n_truth], desc="S_truth")

        print(f"Computing S_lie for DIRECT_LIE[:{n_lie}] …")
        ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE[:n_lie], desc="S_lie")

    _l = "model.layers.0.self_attn.q_proj"
    if _l in ln._S_lie and _l in ln._S_truth:
        _s_truth = ln._S_truth[_l].float().sum().item()
        _ratio   = ln._S_lie[_l].float().sum().item() / _s_truth if _s_truth else float("nan")
        print(f"S_lie/S_truth ratio (q_proj[0]): {_ratio:.3f}")

    ln.build_mask(mode=MODE, n_prompts=n_truth, lie_truth_ratio=args.lie_truth_ratio,
                  znorm=args.znorm)
    print(f"Mask built: mode={MODE}  ρ={args.rho}  α={args.alpha}  "
          f"lm_head={'excluded' if skip else 'included'}")

    lines = []
    def w(s=""): lines.append(s)

    w("LieNeuro DID — Sample Outputs")
    w(f"Mode: {MODE}  |  ρ={args.rho}  |  α={args.alpha}  |  temperature={TEMP}")
    w(f"Model: {model_name}  |  {N_REPS} samples per question")
    w(f"S_lie: DIRECT_LIE[:{n_lie}]  |  S_truth: DIRECT_TRUTH_INSTRUCT[:{n_truth}]")
    w(f"lm_head: {'excluded' if skip else 'included'}")
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
        st_results.append((q, a, resps, corrects, is_indist(q)))

    ln.restore_original()

    def write_group(label, results):
        w()
        w()
        w(f"STEERED — {label}  mode={MODE}  ρ={args.rho}  α={args.alpha}  ({N_REPS} samples/question)")
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
        n_w, n_t, n_q = sum(not c for c in all_c), len(all_c), len(results)
        bl_rate = bl_w / n_q if n_q else 0
        st_rate = n_w / n_t if n_t else 0
        w("-" * 72)
        w(f"  Steered wrong:  {n_w}/{n_t}  ({st_rate:.0%})")
        w(f"  Baseline wrong: {bl_w}/{n_q}  ({bl_rate:.0%})")
        w(f"  Δ wrong-rate:  {st_rate - bl_rate:+.0%}")

    indist = [(q, a, rs, cs, f) for q, a, rs, cs, f in st_results if f]
    ood    = [(q, a, rs, cs, f) for q, a, rs, cs, f in st_results if not f]
    write_group("IN-DISTRIBUTION (questions seen during mask construction)", indist)
    write_group("OUT-OF-DISTRIBUTION (held-out questions)", ood)

    # ── Overall summary ───────────────────────────────────────────────────────
    all_corrects = [c for _, _, _, cs, _ in st_results for c in cs]
    n_total = len(all_corrects)
    n_wrong = sum(not c for c in all_corrects)
    dw      = n_wrong / n_total - bl_wrong / len(FACTUAL_QA)

    w()
    w("=" * 72)
    w("OVERALL SUMMARY")
    w(f"  Steered wrong:  {n_wrong}/{n_total}  ({n_wrong/n_total:.0%} across all samples)")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate:  {dw:+.0%}")

    text = "\n".join(lines)
    out_file.write_text(text)
    print(f"\nSaved to {out_file}")
    print(text)


if __name__ == "__main__":
    main()
