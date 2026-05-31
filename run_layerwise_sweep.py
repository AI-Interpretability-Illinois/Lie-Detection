#!/usr/bin/env python3
"""
run_layerwise_sweep.py — Layer-wise steering experiment for LLaMA-3.1-8B-Instruct.

Instead of steering all linear layers globally, restricts the lie-specific mask
to one transformer block at a time. Sweeps all 32 blocks × ρ ∈ {0.10, 0.25, 0.50}
× α ∈ {1.0, 2.0} = 192 configurations.

Efficiency: importance scores are computed ONCE for all layers via a single forward-
pass sweep, then reused across all (block, ρ, α) combinations. Baseline factual
accuracy is also evaluated once and reused.

Usage:
  CUDA_VISIBLE_DEVICES=1 python run_layerwise_sweep.py --device cuda:0
  CUDA_VISIBLE_DEVICES=1 python run_layerwise_sweep.py --device cuda:0 --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import statistics
from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import get_prompt_set
from src.evaluate import evaluate_factual_accuracy
from fluency_eval import (
    FLUENCY_PROMPTS, build_prompt,
    generate_with_logprobs, compute_perplexity, lexical_metrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTDIR        = Path("outputs")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT_SET    = "direct"
N_BLOCKS      = 32

RHO_VALUES   = [0.10, 0.25, 0.50]
ALPHA_VALUES = [1.0, 2.0]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--device",         default=None)
    p.add_argument("--max-new-tokens", type=int,   default=120,
                   help="Max tokens for fluency generation.")
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--resume",         action="store_true",
                   help="Skip configs already present in layerwise_results.csv.")
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
    logger.info("Model loaded.  GPU: %.1f / %.1f GB",
                torch.cuda.memory_allocated() / 1e9,
                torch.cuda.get_device_properties(0).total_memory / 1e9)
    return model, tok


def group_layers_by_block(layer_names: list[str]) -> dict[int, list[str]]:
    """Map block index → list of layer names in that block."""
    groups: dict[int, list[str]] = defaultdict(list)
    for name in layer_names:
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m:
            groups[int(m.group(1))].append(name)
    return dict(groups)


def build_block_mask(ln: LieNeuro, block_layer_names: list[str], rho: float) -> dict:
    """Build a lie-specific mask restricted to the given layer names."""
    mask = {}
    for name in block_layer_names:
        s_lie   = ln._S_lie[name].float()
        s_truth = ln._S_truth[name].float()
        n       = s_lie.numel()
        k       = max(1, int(rho * n))

        flat_lie   = s_lie.view(-1)
        flat_truth = s_truth.view(-1)

        _, top_lie_idx   = flat_lie.topk(k)
        _, top_truth_idx = flat_truth.topk(k)

        m_lie   = torch.zeros(n, dtype=torch.bool)
        m_truth = torch.zeros(n, dtype=torch.bool)
        m_lie[top_lie_idx]     = True
        m_truth[top_truth_idx] = True

        mask[name] = (m_lie & ~m_truth).view(s_lie.shape)
    return mask


def apply_block_mask(ln: LieNeuro, mask: dict, alpha: float):
    """Scale weights in-place for the given mask, saving originals."""
    for name, m in mask.items():
        mod = ln._get_module(name)
        ln._original_weights[name] = mod.weight.data.clone()
        mod.weight.data.mul_(1.0 + alpha * m.to(mod.weight.device).float())


def restore_block(ln: LieNeuro, mask: dict):
    """Restore only the weights touched by this block's mask."""
    for name in mask:
        if name in ln._original_weights:
            mod = ln._get_module(name)
            mod.weight.data.copy_(ln._original_weights.pop(name).to(mod.weight.device))


def run_fluency(model, tokenizer, device, max_new_tokens, temperature):
    """Generate on all FLUENCY_PROMPTS; return list of (prompt_text, prompt, text, mean_lp, n_tokens)."""
    results = []
    for prompt_text in FLUENCY_PROMPTS:
        prompt = build_prompt(prompt_text, tokenizer)
        text, mean_lp, n_tokens = generate_with_logprobs(
            model, tokenizer, prompt, max_new_tokens, temperature, device
        )
        results.append((prompt_text, prompt, text, mean_lp, n_tokens))
    return results


def score_fluency(generated, unsteered_model, tokenizer, device):
    """Compute perplexity and lexical metrics on pre-generated texts."""
    rows = []
    for prompt_text, prompt, text, mean_lp, n_tokens in generated:
        ppl       = compute_perplexity(unsteered_model, tokenizer, text, device)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        rep_rate, dist1, dist2 = lexical_metrics(token_ids)
        rows.append({
            "perplexity":   ppl,
            "mean_logprob": mean_lp,
            "n_tokens":     n_tokens,
            "rep_rate":     rep_rate,
            "distinct_1":   dist1,
            "distinct_2":   dist2,
            "empty":        int(not text.strip()),
        })
    return rows


def aggregate_fluency(rows):
    """Compute mean/std over a list of per-prompt fluency dicts."""
    def _stats(vals):
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            return float("nan"), float("nan")
        return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    ppl_m,  ppl_s  = _stats([r["perplexity"]   for r in rows])
    lp_m,   lp_s   = _stats([r["mean_logprob"]  for r in rows])
    nt_m,   nt_s   = _stats([r["n_tokens"]       for r in rows])
    rr_m,   rr_s   = _stats([r["rep_rate"]       for r in rows])
    d1_m,   d1_s   = _stats([r["distinct_1"]     for r in rows])
    d2_m,   d2_s   = _stats([r["distinct_2"]     for r in rows])
    return {
        "mean_perplexity":  round(ppl_m, 4), "std_perplexity":  round(ppl_s, 4),
        "mean_logprob":     round(lp_m,  4), "std_logprob":     round(lp_s,  4),
        "mean_n_tokens":    round(nt_m,  2), "std_n_tokens":    round(nt_s,  2),
        "mean_rep_rate":    round(rr_m,  4), "std_rep_rate":    round(rr_s,  4),
        "mean_distinct_1":  round(d1_m,  4), "std_distinct_1":  round(d1_s,  4),
        "mean_distinct_2":  round(d2_m,  4), "std_distinct_2":  round(d2_s,  4),
        "empty_rate":       sum(r["empty"] for r in rows) / len(rows),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    OUTDIR.mkdir(exist_ok=True)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    model, tokenizer = load_model(args.model, device)

    # ── Load already-done configs if resuming ────────────────────────────────
    results_path = OUTDIR / "layerwise_results.csv"
    done: set[tuple] = set()
    if args.resume and results_path.exists():
        with open(results_path, newline="") as f:
            for row in csv.DictReader(f):
                done.add((int(row["block"]), float(row["rho"]), float(row["alpha"])))
        logger.info("Resuming: %d configs already done.", len(done))

    # ── Compute importance scores for ALL layers once ─────────────────────────
    logger.info("Computing importance scores for all layers (this runs once)…")
    cfg = LieNeuroConfig(
        importance_threshold=0.10,   # ρ doesn't matter here; scores are reused
        alpha=1.0,
        prompt_set_name=PROMPT_SET,
        device=device,
        save_original_weights=True,
    )
    ln = LieNeuro(model, tokenizer, cfg)
    lie_prompts, truth_prompts = get_prompt_set(PROMPT_SET)
    ln.compute_importance_scores(lie_prompts, truth_prompts)
    logger.info("Importance scores computed for %d layers.", len(ln._layer_names))

    # Group layers by transformer block
    block_layers = group_layers_by_block(ln._layer_names)
    n_blocks = len(block_layers)
    logger.info("Found %d transformer blocks.", n_blocks)

    # ── Baseline evaluation (once) ────────────────────────────────────────────
    logger.info("Evaluating baseline factual accuracy…")
    baseline_result = evaluate_factual_accuracy(
        model, tokenizer, temperature=args.temperature,
        device=torch.device(device),
    )
    logger.info("Baseline accuracy: %.3f", baseline_result.accuracy)

    logger.info("Evaluating baseline fluency…")
    bl_generated = run_fluency(model, tokenizer, device,
                               args.max_new_tokens, args.temperature)
    bl_fluency = aggregate_fluency(
        score_fluency(bl_generated, model, tokenizer, device)
    )
    logger.info("Baseline perplexity: %.2f", bl_fluency["mean_perplexity"])

    # ── Output file setup ─────────────────────────────────────────────────────
    result_cols = [
        "block", "rho", "alpha",
        "n_block_layers", "n_masked_params", "frac_masked",
        "baseline_accuracy", "steered_accuracy",
        "delta_accuracy", "delta_wrong_rate",
        "n_correct_to_wrong", "n_wrong_to_correct",
        # fluency
        "mean_perplexity", "std_perplexity",
        "mean_logprob",    "std_logprob",
        "mean_n_tokens",   "std_n_tokens",
        "mean_rep_rate",   "std_rep_rate",
        "mean_distinct_1", "std_distinct_1",
        "mean_distinct_2", "std_distinct_2",
        "empty_rate",
    ]
    pq_cols = [
        "block", "rho", "alpha", "question", "gold",
        "baseline_answer", "baseline_correct",
        "steered_answer",  "steered_correct",
        "flipped", "recovered",
    ]

    results_file = open(results_path, "a" if args.resume else "w", newline="")
    pq_path = OUTDIR / "layerwise_per_question.csv"
    pq_file = open(pq_path, "a" if args.resume else "w", newline="")

    results_writer = csv.DictWriter(results_file, fieldnames=result_cols)
    pq_writer      = csv.DictWriter(pq_file,      fieldnames=pq_cols)
    if not args.resume:
        results_writer.writeheader()
        pq_writer.writeheader()

    # ── Main sweep ────────────────────────────────────────────────────────────
    total = n_blocks * len(RHO_VALUES) * len(ALPHA_VALUES)
    done_count = len(done)

    for block_idx in range(n_blocks):
        layer_names = block_layers.get(block_idx, [])
        if not layer_names:
            logger.warning("Block %d has no layers — skipping.", block_idx)
            continue

        for rho, alpha in product(RHO_VALUES, ALPHA_VALUES):
            if (block_idx, rho, alpha) in done:
                continue

            done_count += 1
            logger.info(
                "[%d/%d]  block=%02d  ρ=%.2f  α=%.1f",
                done_count, total, block_idx, rho, alpha,
            )

            # Build block-restricted mask
            mask = build_block_mask(ln, layer_names, rho)
            n_masked = sum(m.sum().item() for m in mask.values())
            n_total  = sum(m.numel()      for m in mask.values())

            # Apply steering
            apply_block_mask(ln, mask, alpha)

            # Factual accuracy (steered only — baseline reused from above)
            steered_result = evaluate_factual_accuracy(
                model, tokenizer, temperature=args.temperature,
                device=torch.device(device),
            )

            # Fluency (generate steered, then restore before scoring)
            st_generated = run_fluency(model, tokenizer, device,
                                       args.max_new_tokens, args.temperature)
            restore_block(ln, mask)   # restore BEFORE scoring perplexity
            st_fluency = aggregate_fluency(
                score_fluency(st_generated, model, tokenizer, device)
            )

            # Per-question rows
            bl_pq = {q["question"]: q for q in baseline_result.per_question}
            st_pq = {q["question"]: q for q in steered_result.per_question}
            n_c2w = n_w2c = 0
            for q in bl_pq:
                b, s = bl_pq[q], st_pq[q]
                flipped  = b["correct"] and not s["correct"]
                recovered = not b["correct"] and s["correct"]
                if flipped:  n_c2w += 1
                if recovered: n_w2c += 1
                pq_writer.writerow({
                    "block":            block_idx,
                    "rho":              rho,
                    "alpha":            alpha,
                    "question":         q,
                    "gold":             b["gold"],
                    "baseline_answer":  b["model_answer"],
                    "baseline_correct": b["correct"],
                    "steered_answer":   s["model_answer"],
                    "steered_correct":  s["correct"],
                    "flipped":          flipped,
                    "recovered":        recovered,
                })
            pq_file.flush()

            delta_acc   = steered_result.accuracy   - baseline_result.accuracy
            delta_wrong = steered_result.wrong_rate - baseline_result.wrong_rate

            row = {
                "block":              block_idx,
                "rho":                rho,
                "alpha":              alpha,
                "n_block_layers":     len(layer_names),
                "n_masked_params":    int(n_masked),
                "frac_masked":        round(n_masked / n_total, 4) if n_total else 0,
                "baseline_accuracy":  round(baseline_result.accuracy,  4),
                "steered_accuracy":   round(steered_result.accuracy,   4),
                "delta_accuracy":     round(delta_acc,   4),
                "delta_wrong_rate":   round(delta_wrong, 4),
                "n_correct_to_wrong": n_c2w,
                "n_wrong_to_correct": n_w2c,
                **st_fluency,
            }
            results_writer.writerow(row)
            results_file.flush()

            logger.info(
                "  → steered_acc=%.3f  Δwrong=%+.3f  ppl=%.1f",
                steered_result.accuracy, delta_wrong, st_fluency["mean_perplexity"],
            )

    results_file.close()
    pq_file.close()

    # ── Save baseline row separately ──────────────────────────────────────────
    bl_path = OUTDIR / "layerwise_baseline.json"
    with open(bl_path, "w") as f:
        json.dump({
            "baseline_accuracy":  baseline_result.accuracy,
            "baseline_wrong_rate": baseline_result.wrong_rate,
            "fluency": bl_fluency,
        }, f, indent=2)
    logger.info("Saved baseline to %s", bl_path)

    # ── Print top-10 blocks by Δwrong (ρ=0.10, α=1.0) ────────────────────────
    with open(results_path, newline="") as f:
        all_rows = list(csv.DictReader(f))

    subset = [r for r in all_rows
              if float(r["rho"]) == 0.10 and float(r["alpha"]) == 1.0]
    subset.sort(key=lambda r: -float(r["delta_wrong_rate"]))

    print()
    print("Top 10 blocks by Δwrong-rate  (ρ=0.10, α=1.0)")
    print(f"{'block':>6}  {'Δwrong':>8}  {'strd_acc':>8}  {'perplexity':>12}")
    print("-" * 42)
    for r in subset[:10]:
        print(f"  {int(r['block']):>4}  {float(r['delta_wrong_rate']):>+8.1%}  "
              f"{float(r['steered_accuracy']):>8.1%}  "
              f"{float(r['mean_perplexity']):>12.1f}")
    logger.info("Done. Results saved to %s", results_path)


if __name__ == "__main__":
    main()
