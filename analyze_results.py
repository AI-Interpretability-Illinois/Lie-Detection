#!/usr/bin/env python3
"""
analyze_results.py  —  Consolidate LieNeuro experiment outputs into CSVs.

Produces:
  outputs/summary.csv        — one row per (rho, alpha, prompt_set) config
  outputs/per_question.csv   — one row per (config × question), baseline + steered
  outputs/flipped.csv        — questions that flipped correct→wrong after steering
"""

from __future__ import annotations
import csv, json, os, glob
from pathlib import Path

OUTDIR = Path("outputs")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load all result JSON files
# ─────────────────────────────────────────────────────────────────────────────

result_files = sorted(glob.glob(str(OUTDIR / "result_*.json")))
print(f"Found {len(result_files)} result file(s).")

all_results = []
for fp in result_files:
    with open(fp) as f:
        all_results.append(json.load(f))

all_results.sort(key=lambda r: (r["prompt_set"], r["rho"], r["alpha"]))


# ─────────────────────────────────────────────────────────────────────────────
# 2. summary.csv
# ─────────────────────────────────────────────────────────────────────────────

summary_path = OUTDIR / "summary.csv"
summary_cols = [
    "prompt_set", "rho", "alpha",
    "n_questions",
    "baseline_accuracy", "steered_accuracy", "delta_accuracy",
    "baseline_wrong_rate", "steered_wrong_rate", "delta_wrong_rate",
    "n_correct_to_wrong",   # flipped correct → wrong
    "n_wrong_to_correct",   # flipped wrong  → correct
]

with open(summary_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=summary_cols)
    w.writeheader()
    for r in all_results:
        bl   = {q["question"]: q for q in r["baseline_per_question"]}
        strd = {q["question"]: q for q in r["steered_per_question"]}
        n_c2w = sum(1 for q in bl if     bl[q]["correct"] and not strd[q]["correct"])
        n_w2c = sum(1 for q in bl if not bl[q]["correct"] and     strd[q]["correct"])
        w.writerow({
            "prompt_set":         r["prompt_set"],
            "rho":                r["rho"],
            "alpha":              r["alpha"],
            "n_questions":        r["n_questions"],
            "baseline_accuracy":  round(r["baseline_accuracy"],  4),
            "steered_accuracy":   round(r["steered_accuracy"],   4),
            "delta_accuracy":     round(r["delta_accuracy"],     4),
            "baseline_wrong_rate":round(r["baseline_wrong_rate"],4),
            "steered_wrong_rate": round(r["steered_wrong_rate"], 4),
            "delta_wrong_rate":   round(r["delta_wrong_rate"],   4),
            "n_correct_to_wrong": n_c2w,
            "n_wrong_to_correct": n_w2c,
        })

print(f"Saved {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. per_question.csv
# ─────────────────────────────────────────────────────────────────────────────

pq_path = OUTDIR / "per_question.csv"
pq_cols = [
    "prompt_set", "rho", "alpha",
    "question", "gold",
    "baseline_answer", "baseline_correct",
    "steered_answer",  "steered_correct",
    "flipped",          # True  = correct→wrong (deception success)
    "recovered",        # True  = wrong→correct (deception failure / repair)
]

with open(pq_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=pq_cols)
    w.writeheader()
    for r in all_results:
        bl   = {q["question"]: q for q in r["baseline_per_question"]}
        strd = {q["question"]: q for q in r["steered_per_question"]}
        for q in bl:
            b  = bl[q]
            s  = strd[q]
            w.writerow({
                "prompt_set":       r["prompt_set"],
                "rho":              r["rho"],
                "alpha":            r["alpha"],
                "question":         q,
                "gold":             b["gold"],
                "baseline_answer":  b["model_answer"],
                "baseline_correct": b["correct"],
                "steered_answer":   s["model_answer"],
                "steered_correct":  s["correct"],
                "flipped":          b["correct"] and not s["correct"],
                "recovered":        not b["correct"] and s["correct"],
            })

print(f"Saved {pq_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. flipped.csv  (correct → wrong only, easy to inspect)
# ─────────────────────────────────────────────────────────────────────────────

flip_path = OUTDIR / "flipped.csv"
flip_cols = [
    "prompt_set", "rho", "alpha",
    "question", "gold", "baseline_answer", "steered_answer",
]

with open(flip_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=flip_cols)
    w.writeheader()
    for r in all_results:
        bl   = {q["question"]: q for q in r["baseline_per_question"]}
        strd = {q["question"]: q for q in r["steered_per_question"]}
        for q in bl:
            if bl[q]["correct"] and not strd[q]["correct"]:
                w.writerow({
                    "prompt_set":     r["prompt_set"],
                    "rho":            r["rho"],
                    "alpha":          r["alpha"],
                    "question":       q,
                    "gold":           bl[q]["gold"],
                    "baseline_answer":bl[q]["model_answer"],
                    "steered_answer": strd[q]["model_answer"],
                })

print(f"Saved {flip_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Print human-readable summary tables
# ─────────────────────────────────────────────────────────────────────────────

def bar(frac, width=20):
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)

print()
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║           LieNeuro Experiment Results — LLaMA-3.1-8B-Instruct      ║")
print("╠══════════════════════════════════════════════════════════════════════╣")
print(f"║  {'prompt_set':<12}  {'ρ':>5}  {'α':>5}  {'base_acc':>8}  {'strd_acc':>8}  {'Δwrong':>8}  ║")
print("╠══════════════════════════════════════════════════════════════════════╣")
for r in sorted(all_results, key=lambda x: (x["prompt_set"], x["rho"], x["alpha"])):
    print(
        f"║  {r['prompt_set']:<12}  {r['rho']:>5.2f}  {r['alpha']:>5.1f}  "
        f"{r['baseline_accuracy']:>8.1%}  {r['steered_accuracy']:>8.1%}  "
        f"{r['delta_wrong_rate']:>+8.1%}  ║"
    )
print("╠══════════════════════════════════════════════════════════════════════╣")
print("║  Δwrong_rate > 0 = more deception after steering                    ║")
print("╚══════════════════════════════════════════════════════════════════════╝")

# Best config
best = max(all_results, key=lambda x: x["delta_wrong_rate"])
print()
print(f"  Best config:  prompt_set={best['prompt_set']}  ρ={best['rho']}  α={best['alpha']}")
print(f"    Baseline wrong_rate : {best['baseline_wrong_rate']:.1%}")
print(f"    Steered  wrong_rate : {best['steered_wrong_rate']:.1%}  (+{best['delta_wrong_rate']:.1%})")
print()

# Per-question detail for best config
bl   = {q["question"]: q for q in best["baseline_per_question"]}
strd = {q["question"]: q for q in best["steered_per_question"]}
flipped = [(q, bl[q], strd[q]) for q in bl
           if bl[q]["correct"] and not strd[q]["correct"]]

print(f"  Questions flipped correct→wrong: {len(flipped)} / {best['n_questions']}")
print()
print("  ─────────────────────────────────────────────────────────────────")
print("  SAMPLE FLIPPED QUESTIONS (first 10)")
print("  ─────────────────────────────────────────────────────────────────")
for q, b, s in flipped[:10]:
    print(f"  Q  : {q}")
    print(f"  ✓  : {b['gold']}")
    print(f"  ✗  : {s['model_answer'][:90]}")
    print()
