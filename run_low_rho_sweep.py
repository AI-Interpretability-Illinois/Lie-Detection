#!/usr/bin/env python3
"""
run_low_rho_sweep.py — Factual accuracy + fluency sweep over low ρ values.

Runs ρ ∈ {0.01, 0.05} × α ∈ {1, 2, 4, 8} for the direct prompt set, then
evaluates fluency at each configuration using the same 20 open-ended prompts
as fluency_eval.py.

Appends results to the existing outputs/ CSVs so analyze_results.py and
judge_eval.py can be re-run on the full dataset afterwards.

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_low_rho_sweep.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import get_prompt_set
from src.evaluate import compare_baseline_steered
from fluency_eval import (
    FLUENCY_PROMPTS, SYSTEM_PROMPT,
    build_prompt, generate_with_logprobs, compute_perplexity, lexical_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTDIR       = Path("outputs")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_SET   = "direct"

RHO_VALUES   = [0.01, 0.05]
ALPHA_VALUES = [1.0, 2.0, 4.0, 8.0]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",  default=DEFAULT_MODEL)
    p.add_argument("--device", default=None)
    p.add_argument("--max-new-tokens-fluency", type=int, default=120)
    p.add_argument("--temperature", type=float, default=1.0)
    return p.parse_args()


def load_model(model_id, device):
    logger.info("Loading tokenizer …")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    logger.info("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device
    )
    model.eval()
    logger.info(
        "Model loaded.  GPU: %.1f / %.1f GB",
        torch.cuda.memory_allocated() / 1e9,
        torch.cuda.get_device_properties(0).total_memory / 1e9,
    )
    return model, tok


def generate_fluency_texts(model, tokenizer, device, max_new_tokens, temperature):
    """Generate responses on FLUENCY_PROMPTS; return (prompt_text, prompt, text, mean_logprob, n_tokens) tuples."""
    results = []
    for prompt_text in FLUENCY_PROMPTS:
        prompt = build_prompt(prompt_text, tokenizer)
        # signature: (model, tokenizer, prompt, max_new_tokens, temperature, device)
        text, mean_lp, n_tokens = generate_with_logprobs(
            model, tokenizer, prompt, max_new_tokens, temperature, device
        )
        results.append((prompt_text, prompt, text, mean_lp, n_tokens))
    return results


def score_fluency(generated, unsteered_model, tokenizer, device, rho, alpha, variant_tag):
    """Compute perplexity (using unsteered model) and lexical metrics on pre-generated texts."""
    rows = []
    for prompt_text, prompt, text, mean_lp, n_tokens in generated:
        ppl = compute_perplexity(unsteered_model, tokenizer, text, device)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        rep_rate, dist1, dist2 = lexical_metrics(token_ids)
        rows.append({
            "variant":      variant_tag,
            "rho":          rho,
            "alpha":        alpha,
            "prompt":       prompt_text,
            "response":     text,
            "perplexity":   ppl,
            "mean_logprob": mean_lp,
            "n_tokens":     n_tokens,
            "rep_rate":     rep_rate,
            "distinct_1":   dist1,
            "distinct_2":   dist2,
            "empty":        int(not text.strip()),
        })
    return rows


def main():
    args = parse_args()
    OUTDIR.mkdir(exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model, device)

    lie_prompts, truth_prompts = get_prompt_set(PROMPT_SET)

    # ── Fluency CSV paths ────────────────────────────────────────────────────
    fluency_detail_path  = OUTDIR / "fluency_results_lowrho.csv"
    fluency_summary_path = OUTDIR / "fluency_summary_lowrho.csv"

    fluency_detail_cols = [
        "variant", "rho", "alpha", "prompt", "response",
        "perplexity", "mean_logprob", "n_tokens",
        "rep_rate", "distinct_1", "distinct_2", "empty",
    ]
    fluency_summary_cols = [
        "variant", "rho", "alpha", "n_prompts",
        "mean_perplexity", "std_perplexity",
        "mean_logprob", "std_logprob",
        "mean_n_tokens", "std_n_tokens",
        "mean_rep_rate", "std_rep_rate",
        "mean_distinct_1", "std_distinct_1",
        "mean_distinct_2", "std_distinct_2",
        "empty_rate",
    ]

    all_fluency_rows = []

    # ── Baseline fluency (once) ──────────────────────────────────────────────
    logger.info("=== Baseline fluency ===")
    baseline_generated = generate_fluency_texts(
        model, tokenizer, device,
        args.max_new_tokens_fluency, args.temperature,
    )
    # Unsteered model = model at this point (no steering applied yet)
    baseline_fluency = score_fluency(
        baseline_generated, model, tokenizer, device,
        rho=0.0, alpha=0.0, variant_tag="baseline",
    )
    all_fluency_rows.extend(baseline_fluency)

    # ── Sweep ────────────────────────────────────────────────────────────────
    for rho, alpha in product(RHO_VALUES, ALPHA_VALUES):
        logger.info("=== ρ=%.2f  α=%.1f ===", rho, alpha)

        cfg = LieNeuroConfig(
            importance_threshold=rho,
            alpha=alpha,
            prompt_set_name=PROMPT_SET,
            device=device,
            save_original_weights=True,
        )
        ln = LieNeuro(model, tokenizer, cfg)
        lie_p, truth_p = get_prompt_set(PROMPT_SET)
        ln.compute_importance_scores(lie_p, truth_p)
        ln.build_mask()

        # ── Factual accuracy ─────────────────────────────────────────────
        # compare_baseline_steered handles apply_steering/restore internally
        comparison = compare_baseline_steered(
            model, tokenizer, ln,
            temperature=args.temperature,
            device=torch.device(device),
        )

        result = {
            "rho":   rho,
            "alpha": alpha,
            "prompt_set": PROMPT_SET,
            "baseline_accuracy":   comparison["baseline"].accuracy,
            "steered_accuracy":    comparison["steered"].accuracy,
            "delta_accuracy":      comparison["delta"]["accuracy"],
            "baseline_wrong_rate": comparison["baseline"].wrong_rate,
            "steered_wrong_rate":  comparison["steered"].wrong_rate,
            "delta_wrong_rate":    comparison["delta"]["wrong_rate"],
            "n_questions":         comparison["baseline"].n_questions,
            "baseline_per_question": comparison["baseline"].per_question,
            "steered_per_question":  comparison["steered"].per_question,
        }
        logger.info(
            "[ρ=%.2f α=%.1f]  baseline_acc=%.3f  steered_acc=%.3f  Δwrong=%.3f",
            rho, alpha,
            result["baseline_accuracy"], result["steered_accuracy"],
            result["delta_wrong_rate"],
        )

        out_path = OUTDIR / f"result_{PROMPT_SET}_rho{rho}_alpha{alpha}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Saved %s", out_path)

        # ── Fluency ──────────────────────────────────────────────────────
        # Generate with steered model, then restore before computing perplexity
        # so the unsteered model is always the reference (same as fluency_eval.py).
        ln.apply_steering()
        steered_generated = generate_fluency_texts(
            model, tokenizer, device,
            args.max_new_tokens_fluency, args.temperature,
        )
        ln.restore_original()   # weights back to baseline before scoring
        fluency_rows = score_fluency(
            steered_generated, model, tokenizer, device,
            rho=rho, alpha=alpha,
            variant_tag=f"steered_rho{rho}_a{alpha}",
        )
        all_fluency_rows.extend(fluency_rows)

        logger.info("Fluency done for ρ=%.2f α=%.1f", rho, alpha)

    # ── Save fluency detail ──────────────────────────────────────────────────
    with open(fluency_detail_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fluency_detail_cols)
        w.writeheader()
        w.writerows(all_fluency_rows)
    logger.info("Saved %s (%d rows)", fluency_detail_path, len(all_fluency_rows))

    # ── Save fluency summary ─────────────────────────────────────────────────
    import statistics
    from collections import defaultdict

    groups: dict[str, list] = defaultdict(list)
    for r in all_fluency_rows:
        groups[r["variant"]].append(r)

    def safe_std(vals):
        return statistics.stdev(vals) if len(vals) > 1 else 0.0

    summary_rows = []
    for variant, grp in groups.items():
        rho   = grp[0]["rho"]
        alpha = grp[0]["alpha"]
        ppls  = [r["perplexity"] for r in grp if not math.isnan(r["perplexity"])]
        lps   = [r["mean_logprob"] for r in grp if not math.isnan(r["mean_logprob"])]
        nts   = [r["n_tokens"] for r in grp]
        rrs   = [r["rep_rate"] for r in grp]
        d1s   = [r["distinct_1"] for r in grp]
        d2s   = [r["distinct_2"] for r in grp]
        summary_rows.append({
            "variant":          variant,
            "rho":              rho,
            "alpha":            alpha,
            "n_prompts":        len(grp),
            "mean_perplexity":  round(statistics.mean(ppls), 4) if ppls else float("nan"),
            "std_perplexity":   round(safe_std(ppls), 4)        if ppls else float("nan"),
            "mean_logprob":     round(statistics.mean(lps), 4)  if lps else float("nan"),
            "std_logprob":      round(safe_std(lps), 4)         if lps else float("nan"),
            "mean_n_tokens":    round(statistics.mean(nts), 4),
            "std_n_tokens":     round(safe_std(nts), 4),
            "mean_rep_rate":    round(statistics.mean(rrs), 4),
            "std_rep_rate":     round(safe_std(rrs), 4),
            "mean_distinct_1":  round(statistics.mean(d1s), 4),
            "std_distinct_1":   round(safe_std(d1s), 4),
            "mean_distinct_2":  round(statistics.mean(d2s), 4),
            "std_distinct_2":   round(safe_std(d2s), 4),
            "empty_rate":       sum(r["empty"] for r in grp) / len(grp),
        })

    with open(fluency_summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fluency_summary_cols)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Saved %s", fluency_summary_path)

    # ── Print summary table ──────────────────────────────────────────────────
    print()
    print(f"{'ρ':>5}  {'α':>5}  {'base_acc':>8}  {'strd_acc':>8}  {'Δwrong':>8}  {'perplexity':>12}  {'distinct-2':>10}")
    print("-" * 72)

    # reload factual results
    factual = {}
    for rho, alpha in product(RHO_VALUES, ALPHA_VALUES):
        p = OUTDIR / f"result_{PROMPT_SET}_rho{rho}_alpha{alpha}.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                factual[(rho, alpha)] = d

    flu_by_variant = {r["variant"]: r for r in summary_rows}

    for rho, alpha in product(RHO_VALUES, ALPHA_VALUES):
        fa = factual.get((rho, alpha), {})
        fv = flu_by_variant.get(f"steered_rho{rho}_a{alpha}", {})
        print(
            f"{rho:>5.2f}  {alpha:>5.1f}  "
            f"{fa.get('baseline_accuracy', float('nan')):>8.1%}  "
            f"{fa.get('steered_accuracy',  float('nan')):>8.1%}  "
            f"{fa.get('delta_wrong_rate',  float('nan')):>+8.1%}  "
            f"{fv.get('mean_perplexity',   float('nan')):>12.1f}  "
            f"{fv.get('mean_distinct_2',   float('nan')):>10.3f}"
        )


if __name__ == "__main__":
    main()
