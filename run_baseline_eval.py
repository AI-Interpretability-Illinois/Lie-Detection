#!/usr/bin/env python3
"""
Evaluate all four models on the 50 in-distribution (DIRECT_QA) and 26 OOD
questions to confirm baseline factual accuracy before any steering.

Run all four models in parallel across GPUs:
  python run_baseline_eval.py

Or a single model:
  CUDA_VISIBLE_DEVICES=0 python run_baseline_eval.py --model llama --device cuda:0

Output: outputs/baseline_eval.csv  +  a printed summary table.
"""

import argparse
import csv
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate import evaluate_factual_accuracy, FACTUAL_QA
from src.prompts import DIRECT_QA, DIRECT_TRUTH_INSTRUCT

CACHE_DIR = "/scratch/aebrahim/hf_cache"
OUT_CSV   = Path("outputs/baseline_eval.csv")

MODEL_CONFIGS = {
    "llama": {
        "model_id":          "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "trust_remote_code": False,
        "device":            "cuda:0",
        "gpu":               "0",
    },
    "qwen": {
        "model_id":          "Qwen/Qwen2.5-7B-Instruct",
        "trust_remote_code": False,
        "device":            "cuda:0",
        "gpu":               "1",
    },
    "mistral": {
        "model_id":          "mistralai/Mistral-7B-Instruct-v0.3",
        "trust_remote_code": False,
        "device":            "cuda:0",
        "gpu":               "2",
    },
    "phi3": {
        "model_id":          "microsoft/Phi-3-medium-4k-instruct",
        "trust_remote_code": True,
        "device":            "cuda:0",
        "gpu":               "3",
    },
}

# Build OOD set: FACTUAL_QA questions not seen during mask construction
_instruct_qs = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
                for p in DIRECT_TRUTH_INSTRUCT}
OOD_QA = [qa for qa in FACTUAL_QA
           if qa["question"].strip().lower() not in _instruct_qs]


def evaluate_model(model_name, device_override=None, trust_rc_override=None):
    mcfg     = MODEL_CONFIGS[model_name]
    model_id = mcfg["model_id"]
    trust_rc = mcfg["trust_remote_code"] if trust_rc_override is None else trust_rc_override
    device   = device_override or mcfg["device"]

    print(f"[{model_name}] Loading {model_id} on {device} …")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR,
                                        trust_remote_code=trust_rc)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map=device,
        cache_dir=CACHE_DIR, trust_remote_code=trust_rc)
    model.eval()
    print(f"[{model_name}] Model loaded.")

    print(f"[{model_name}] Evaluating on 50 in-distribution (DIRECT_QA) questions …")
    indist = evaluate_factual_accuracy(model, tok, qa_pairs=DIRECT_QA, device=device)

    print(f"[{model_name}] Evaluating on {len(OOD_QA)} OOD questions …")
    ood = evaluate_factual_accuracy(model, tok, qa_pairs=OOD_QA, device=device)

    print(f"[{model_name}] Done.  indist={indist.accuracy:.1%}  ood={ood.accuracy:.1%}")

    # Save wrong answers to a per-model file
    wrong_path = Path(f"outputs/baseline_wrong_{model_name}.txt")
    lines = [f"Baseline wrong answers — {model_name}", "=" * 72, ""]

    indist_wrongs = [pq for pq in indist.per_question if not pq["correct"]]
    ood_wrongs    = [pq for pq in ood.per_question    if not pq["correct"]]

    lines.append(f"IN-DISTRIBUTION ({len(indist_wrongs)} wrong / {indist.n_questions})")
    lines.append("-" * 72)
    for pq in indist_wrongs:
        lines.append(f"  Q: {pq['question']}")
        lines.append(f"     Correct:   {pq['gold']}")
        lines.append(f"     Generated: {pq['model_answer']}")
        lines.append("")

    lines.append(f"OUT-OF-DISTRIBUTION ({len(ood_wrongs)} wrong / {ood.n_questions})")
    lines.append("-" * 72)
    for pq in ood_wrongs:
        lines.append(f"  Q: {pq['question']}")
        lines.append(f"     Correct:   {pq['gold']}")
        lines.append(f"     Generated: {pq['model_answer']}")
        lines.append("")

    wrong_path.write_text("\n".join(lines))
    print(f"[{model_name}] Wrong answers saved to {wrong_path}")

    return {
        "model":          model_name,
        "indist_acc":     indist.accuracy,
        "indist_wrong":   indist.wrong_rate,
        "indist_n":       indist.n_questions,
        "ood_acc":        ood.accuracy,
        "ood_wrong":      ood.wrong_rate,
        "ood_n":          ood.n_questions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=None, choices=list(MODEL_CONFIGS),
                        help="Single model to evaluate (default: all four)")
    parser.add_argument("--device", default=None,
                        help="Override device (default: per-model config)")
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else list(MODEL_CONFIGS)

    rows = []
    for name in models_to_run:
        row = evaluate_model(name, device_override=args.device)
        rows.append(row)

    # ── CSV output ────────────────────────────────────────────────────────────
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "indist_acc", "indist_wrong", "indist_n",
                  "ood_acc", "ood_wrong", "ood_n"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nResults saved to {OUT_CSV}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'model':<10} {'indist acc':>11} {'indist wrong':>13} "
          f"{'ood acc':>9} {'ood wrong':>10}")
    print("─" * 58)
    for r in rows:
        print(f"{r['model']:<10} {r['indist_acc']:>11.1%} {r['indist_wrong']:>13.1%} "
              f"{r['ood_acc']:>9.1%} {r['ood_wrong']:>10.1%}")

    # Per-question breakdown for any model with low accuracy
    for r in rows:
        if r["indist_acc"] < 0.8 or r["ood_acc"] < 0.8:
            print(f"\n[{r['model']}] low accuracy — re-run with --model {r['model']} "
                  "to see per-question breakdown")


if __name__ == "__main__":
    main()
