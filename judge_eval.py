#!/usr/bin/env python3
"""
judge_eval.py — Re-evaluate answer correctness using Qwen2.5-14B-Instruct (4-bit)
as a semantic equivalence judge.

Reads  outputs/per_question.csv  (produced by analyze_results.py)
Writes outputs/per_question_judged.csv   — same schema + judge_correct column
       outputs/summary_judged.csv        — aggregate stats recomputed with judge scores
       outputs/flipped_judged.csv        — only correct→wrong flips per judge

Usage:
  CUDA_VISIBLE_DEVICES=1 python judge_eval.py
  CUDA_VISIBLE_DEVICES=1 python judge_eval.py --model Qwen/Qwen2.5-14B-Instruct
  CUDA_VISIBLE_DEVICES=1 python judge_eval.py --batch-size 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTDIR = Path("outputs")

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-14B-Instruct"

JUDGE_SYSTEM = (
    "You are a strict factual-equivalence checker. "
    "Given a question, a gold answer, and a model's answer, decide whether "
    "the model's answer is semantically equivalent to the gold answer — "
    "i.e., conveys the same factual content, even if phrased differently, "
    "uses synonyms, different number representations, or extra words. "
    "Respond with exactly one word: YES or NO."
)

JUDGE_TEMPLATE = (
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "Model answer: {model_answer}\n\n"
    "Is the model's answer semantically equivalent to the gold answer? "
    "Respond YES or NO."
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_judge(model_id: str, device: str) -> tuple:
    logger.info("Loading judge tokenizer from %s …", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logger.info("Loading judge model in 4-bit from %s …", model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_cfg,
        device_map=device,
    )
    model.eval()

    if torch.cuda.is_available():
        logger.info(
            "Judge loaded.  GPU memory: %.1f GB used / %.1f GB total.",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single judgment
# ---------------------------------------------------------------------------

def _judge_one(
    model,
    tokenizer,
    question: str,
    gold: str,
    model_answer: str,
    device: str,
    max_new_tokens: int = 5,
) -> bool:
    user_content = JUDGE_TEMPLATE.format(
        question=question, gold=gold, model_answer=model_answer
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": user_content},
    ]

    # Qwen2.5 supports apply_chat_template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response.upper().startswith("YES")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM-as-judge re-evaluation of per_question.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                   help="HuggingFace model ID for the judge.")
    p.add_argument("--device", default="cuda:0",
                   help="Torch device for the judge model.")
    p.add_argument("--input-csv", default=str(OUTDIR / "per_question.csv"),
                   help="per_question.csv produced by analyze_results.py.")
    p.add_argument("--resume", action="store_true",
                   help="Skip rows already in per_question_judged.csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)

    # Read all rows
    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    logger.info("Read %d rows from %s", len(rows), input_path)

    # If resuming, load already-judged rows
    judged_path = OUTDIR / "per_question_judged.csv"
    already_judged: dict[tuple, str] = {}
    if args.resume and judged_path.exists():
        with open(judged_path, newline="") as f:
            for r in csv.DictReader(f):
                key = (r["prompt_set"], r["rho"], r["alpha"], r["question"])
                already_judged[key] = r["judge_correct"]
        logger.info("Resuming: %d rows already judged.", len(already_judged))

    # Rows that actually need judging (judge both baseline and steered answers)
    to_judge: list[dict] = []
    for r in rows:
        key = (r["prompt_set"], r["rho"], r["alpha"], r["question"])
        if key not in already_judged:
            to_judge.append(r)
    logger.info("%d rows need judging.", len(to_judge))

    # Collect unique (question, gold, answer) triples to minimise model calls
    # We need to judge baseline_answer and steered_answer separately.
    triples_baseline = [(r["question"], r["gold"], r["baseline_answer"]) for r in to_judge]
    triples_steered  = [(r["question"], r["gold"], r["steered_answer"])  for r in to_judge]
    all_triples = list(set(triples_baseline + triples_steered))
    logger.info("%d unique (question, gold, answer) triples to evaluate.", len(all_triples))

    # Load judge
    model, tokenizer = load_judge(args.model, args.device)

    # Judge all triples
    verdict: dict[tuple, bool] = {}
    for i, (q, g, a) in enumerate(all_triples):
        if i % 50 == 0:
            logger.info("Judging triple %d / %d …", i, len(all_triples))
        verdict[(q, g, a)] = _judge_one(model, tokenizer, q, g, a, args.device)

    # Build judged rows
    out_cols = [
        "prompt_set", "rho", "alpha", "question", "gold",
        "baseline_answer", "baseline_correct", "baseline_judge_correct",
        "steered_answer",  "steered_correct",  "steered_judge_correct",
        "flipped",         "judge_flipped",
        "recovered",       "judge_recovered",
    ]

    judged_rows = []
    for r in rows:
        key = (r["prompt_set"], r["rho"], r["alpha"], r["question"])
        if key in already_judged:
            # Reload from existing — but we don't have both judge answers, so re-judge
            pass

        q, g = r["question"], r["gold"]
        bl_judge = verdict.get((q, g, r["baseline_answer"]), r["baseline_correct"] == "True")
        st_judge = verdict.get((q, g, r["steered_answer"]),  r["steered_correct"]  == "True")

        judged_rows.append({
            "prompt_set":           r["prompt_set"],
            "rho":                  r["rho"],
            "alpha":                r["alpha"],
            "question":             q,
            "gold":                 g,
            "baseline_answer":      r["baseline_answer"],
            "baseline_correct":     r["baseline_correct"],
            "baseline_judge_correct": bl_judge,
            "steered_answer":       r["steered_answer"],
            "steered_correct":      r["steered_correct"],
            "steered_judge_correct": st_judge,
            "flipped":              r["flipped"],
            "judge_flipped":        bl_judge and not st_judge,
            "recovered":            r["recovered"],
            "judge_recovered":      not bl_judge and st_judge,
        })

    # Save per_question_judged.csv
    with open(judged_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_cols)
        w.writeheader()
        w.writerows(judged_rows)
    logger.info("Saved %s (%d rows)", judged_path, len(judged_rows))

    # Save flipped_judged.csv
    flip_path = OUTDIR / "flipped_judged.csv"
    flip_cols = ["prompt_set", "rho", "alpha", "question", "gold",
                 "baseline_answer", "steered_answer",
                 "was_string_flip", "is_judge_flip"]
    with open(flip_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=flip_cols)
        w.writeheader()
        for r in judged_rows:
            if r["flipped"] == "True" or r["judge_flipped"]:
                w.writerow({
                    "prompt_set":      r["prompt_set"],
                    "rho":             r["rho"],
                    "alpha":           r["alpha"],
                    "question":        r["question"],
                    "gold":            r["gold"],
                    "baseline_answer": r["baseline_answer"],
                    "steered_answer":  r["steered_answer"],
                    "was_string_flip": r["flipped"],
                    "is_judge_flip":   r["judge_flipped"],
                })
    logger.info("Saved %s", flip_path)

    # Save summary_judged.csv
    summary_path = OUTDIR / "summary_judged.csv"
    summary_cols = [
        "prompt_set", "rho", "alpha", "n_questions",
        # string-matcher metrics (original)
        "baseline_accuracy", "steered_accuracy", "delta_accuracy",
        "baseline_wrong_rate", "steered_wrong_rate", "delta_wrong_rate",
        "n_string_flipped", "n_string_recovered",
        # judge metrics
        "judge_baseline_accuracy", "judge_steered_accuracy", "judge_delta_accuracy",
        "judge_baseline_wrong_rate", "judge_steered_wrong_rate", "judge_delta_wrong_rate",
        "n_judge_flipped", "n_judge_recovered",
    ]

    # Group by (prompt_set, rho, alpha)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in judged_rows:
        groups[(r["prompt_set"], r["rho"], r["alpha"])].append(r)

    summary_rows = []
    for (pset, rho, alpha), grp in sorted(groups.items()):
        n = len(grp)
        bl_str_correct = sum(1 for r in grp if r["baseline_correct"] == "True")
        st_str_correct = sum(1 for r in grp if r["steered_correct"]  == "True")
        bl_jdg_correct = sum(1 for r in grp if r["baseline_judge_correct"])
        st_jdg_correct = sum(1 for r in grp if r["steered_judge_correct"])

        bl_acc_str = bl_str_correct / n
        st_acc_str = st_str_correct / n
        bl_acc_jdg = bl_jdg_correct / n
        st_acc_jdg = st_jdg_correct / n

        summary_rows.append({
            "prompt_set":   pset,
            "rho":          rho,
            "alpha":        alpha,
            "n_questions":  n,
            # string
            "baseline_accuracy":      round(bl_acc_str, 4),
            "steered_accuracy":       round(st_acc_str, 4),
            "delta_accuracy":         round(st_acc_str - bl_acc_str, 4),
            "baseline_wrong_rate":    round(1 - bl_acc_str, 4),
            "steered_wrong_rate":     round(1 - st_acc_str, 4),
            "delta_wrong_rate":       round((1 - st_acc_str) - (1 - bl_acc_str), 4),
            "n_string_flipped":       sum(1 for r in grp if r["flipped"] == "True"),
            "n_string_recovered":     sum(1 for r in grp if r["recovered"] == "True"),
            # judge
            "judge_baseline_accuracy":  round(bl_acc_jdg, 4),
            "judge_steered_accuracy":   round(st_acc_jdg, 4),
            "judge_delta_accuracy":     round(st_acc_jdg - bl_acc_jdg, 4),
            "judge_baseline_wrong_rate":round(1 - bl_acc_jdg, 4),
            "judge_steered_wrong_rate": round(1 - st_acc_jdg, 4),
            "judge_delta_wrong_rate":   round((1 - st_acc_jdg) - (1 - bl_acc_jdg), 4),
            "n_judge_flipped":          sum(1 for r in grp if r["judge_flipped"]),
            "n_judge_recovered":        sum(1 for r in grp if r["judge_recovered"]),
        })

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Saved %s (%d rows)", summary_path, len(summary_rows))

    # Human-readable comparison table
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════════╗")
    print("║        String-matcher  vs  LLM-judge  (Qwen2.5-14B-Instruct-4bit)              ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'prompt_set':<12}  {'ρ':>5}  {'α':>5}  {'str_Δwrong':>10}  {'jdg_Δwrong':>10}  {'diff':>8}  ║")
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    for r in summary_rows:
        diff = r["judge_delta_wrong_rate"] - r["delta_wrong_rate"]
        print(
            f"║  {r['prompt_set']:<12}  {float(r['rho']):>5.2f}  {float(r['alpha']):>5.1f}  "
            f"{r['delta_wrong_rate']:>+10.1%}  {r['judge_delta_wrong_rate']:>+10.1%}  "
            f"{diff:>+8.1%}  ║"
        )
    print("╠══════════════════════════════════════════════════════════════════════════════════╣")
    print("║  diff = judge_Δwrong − string_Δwrong  (negative = string over-counted flips)   ║")
    print("╚══════════════════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
