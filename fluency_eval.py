#!/usr/bin/env python3
"""
fluency_eval.py — Test whether LieNeuro steering degrades general fluency.

For each of 20 open-ended prompts (no factual right/wrong answer), generate
responses from:
  1. Baseline model (no steering)
  2. Steered  α = 1.0  (ρ = 0.10, prompt_set = direct)
  3. Steered  α = 2.0  (ρ = 0.10, prompt_set = direct)

For every (model_variant × prompt) pair compute:
  - perplexity        : exp(cross-entropy loss) evaluated by the BASELINE model
                        → same reference for all three variants; not circular
  - mean_logprob      : mean per-token log-prob during generation (generation confidence)
  - response_len      : number of generated tokens
  - rep_rate          : fraction of repeated 4-grams (higher = more repetitive)
  - distinct_1        : unique unigrams / total tokens (lexical diversity)
  - distinct_2        : unique bigrams  / total bigram tokens
  - empty             : 1 if generation is blank / whitespace only

Outputs:
  outputs/fluency_results.csv   — one row per (variant × prompt)
  outputs/fluency_summary.csv   — per-variant aggregate (mean ± std of each metric)

Usage
-----
  CUDA_VISIBLE_DEVICES=1 python fluency_eval.py [--device cuda:0] [--alpha1 1.0] [--alpha2 2.0]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import get_prompt_set

# ─────────────────────────────────────────────────────────────────────────────
# Fluency prompts  (open-ended; no factual correct answer)
# ─────────────────────────────────────────────────────────────────────────────

FLUENCY_PROMPTS: List[str] = [
    # Narrative / descriptive
    "Describe what you might see on a walk through a dense forest in autumn.",
    "Write a short paragraph about what it feels like to watch a thunderstorm from indoors.",
    "Describe the atmosphere inside a busy coffee shop on a rainy Monday morning.",
    "Write a few sentences imagining life on a small fishing village on the coast.",
    "Describe the experience of watching a sunrise from the top of a hill.",
    # Explanatory / how-to
    "Explain in simple terms how bread rises when you bake it.",
    "Describe the general steps involved in planning a road trip.",
    "Explain what makes a piece of music feel emotional to a listener.",
    "Describe how a library is typically organised and why that organisation helps people.",
    "Explain in everyday language what causes the seasons to change.",
    # Opinion / reflective
    "What makes a good conversation between two people?",
    "What qualities make someone a trustworthy friend?",
    "Why do people enjoy reading fiction, in your view?",
    "What do you think draws people to live in cities rather than the countryside?",
    "What role does humour play in everyday life?",
    # Creative / imaginative
    "Write the opening sentence of a short story set on a distant planet.",
    "Imagine a world where music has never existed. Describe one aspect of daily life.",
    "Write a brief scene in which two strangers meet on a train and strike up a conversation.",
    "Describe an imaginary animal that lives deep in the ocean and how it might behave.",
    "Write a short reflection from the perspective of an old lighthouse keeper.",
]

SYSTEM_PROMPT = (
    "You are a thoughtful, articulate assistant. "
    "Respond in two to four sentences."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return f"System: {SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant:"


@torch.no_grad()
def generate_with_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = 1.0,
    device: torch.device = None,
) -> Tuple[str, float, int]:
    """
    Generate a response and return (text, mean_logprob, n_tokens).
    mean_logprob is the average per-token log-probability of the generated tokens.
    """
    if device is None:
        device = next(model.parameters()).device

    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[1]

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )

    new_ids = out.sequences[0, input_len:]          # generated token ids
    n_tokens = new_ids.shape[0]

    # Compute per-token log-prob from the scores (one score tensor per step)
    if n_tokens > 0 and out.scores:
        log_probs = []
        for step_idx, score in enumerate(out.scores):
            if step_idx >= n_tokens:
                break
            token_id  = new_ids[step_idx].item()
            log_p     = F.log_softmax(score[0], dim=-1)[token_id].item()
            log_probs.append(log_p)
        mean_lp = sum(log_probs) / len(log_probs) if log_probs else float("nan")
    else:
        mean_lp = float("nan")

    text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return text, mean_lp, n_tokens


@torch.no_grad()
def compute_perplexity(
    baseline_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device,
) -> float:
    """
    Compute perplexity of `text` using the BASELINE (unsteered) model.
    Always uses the same reference so all three variants are comparable.
    Returns inf if text is empty.
    """
    if not text.strip():
        return float("inf")

    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}

    if enc["input_ids"].shape[1] < 2:
        return float("inf")

    out = baseline_model(**enc, labels=enc["input_ids"])
    loss = out.loss.item()          # mean cross-entropy per token
    return math.exp(loss)


def lexical_metrics(tokens: List[int]) -> Tuple[float, float, float]:
    """
    Returns (rep_rate, distinct_1, distinct_2) for a token sequence.

    rep_rate   : fraction of 4-gram occurrences that are repeated (0 = no repetition)
    distinct_1 : unique unigrams / total tokens
    distinct_2 : unique bigrams  / total (n - 1) bigrams
    """
    n = len(tokens)
    if n == 0:
        return 1.0, 0.0, 0.0

    # rep_rate (4-grams)
    if n >= 4:
        fourgrams = [tuple(tokens[i:i+4]) for i in range(n - 3)]
        rep_rate = 1.0 - len(set(fourgrams)) / len(fourgrams)
    else:
        rep_rate = 0.0

    # distinct-1
    distinct_1 = len(set(tokens)) / n

    # distinct-2
    if n >= 2:
        bigrams = [tuple(tokens[i:i+2]) for i in range(n - 1)]
        distinct_2 = len(set(bigrams)) / len(bigrams)
    else:
        distinct_2 = 1.0

    return rep_rate, distinct_1, distinct_2


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--device",     default="cuda:0")
    p.add_argument("--mask",       default="outputs/mask_direct_rho0.1_alpha1.0.pt",
                   help="Pre-computed importance mask (prompt_set=direct, rho=0.1).")
    p.add_argument("--alpha1",     type=float, default=1.0)
    p.add_argument("--alpha2",     type=float, default=2.0)
    p.add_argument("--rho",        type=float, default=0.1)
    p.add_argument("--prompt-set", default="direct")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--temperature",    type=float, default=1.0)
    p.add_argument("--output-dir", default="outputs")
    return p.parse_args()


def run_variant(
    label: str,
    model: AutoModelForCausalLM,
    baseline_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    lie_neuro: LieNeuro | None,
    alpha: float | None,
    prompts: List[str],
    args,
    device: torch.device,
) -> List[dict]:
    """Generate responses for all prompts under one model variant and compute metrics."""

    if lie_neuro is not None and alpha is not None:
        lie_neuro.config.alpha = alpha
        lie_neuro.apply_steering()
        print(f"\n  → Steering applied (α={alpha})")

    rows = []
    for i, prompt_text in enumerate(tqdm(prompts, desc=label, ncols=80)):
        formatted = build_prompt(prompt_text, tokenizer)

        text, mean_lp, n_tokens = generate_with_logprobs(
            model, tokenizer, formatted,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )

        ppl = compute_perplexity(baseline_model, tokenizer, text, device)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        rep_rate, dist1, dist2 = lexical_metrics(token_ids)

        rows.append({
            "variant":       label,
            "alpha":         alpha if alpha is not None else 0.0,
            "prompt_idx":    i,
            "prompt":        prompt_text,
            "response":      text,
            "n_tokens":      n_tokens,
            "perplexity":    round(ppl, 3)  if math.isfinite(ppl)    else "inf",
            "mean_logprob":  round(mean_lp, 4) if math.isfinite(mean_lp) else "nan",
            "rep_rate":      round(rep_rate, 4),
            "distinct_1":    round(dist1, 4),
            "distinct_2":    round(dist2, 4),
            "empty":         int(not text.strip()),
        })

    if lie_neuro is not None and alpha is not None:
        lie_neuro.restore_original()
        print(f"  → Weights restored.")

    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=args.device
    )
    model.eval()
    print(f"  GPU memory after load: {torch.cuda.memory_allocated(device)/1e9:.1f} GB")

    # Baseline model = same object before any steering (weights restored after each variant)
    baseline_model = model   # we always restore_original() before using it as reference

    # ── Load LieNeuro mask ────────────────────────────────────────────────────
    cfg = LieNeuroConfig(
        importance_threshold=args.rho,
        alpha=args.alpha1,          # will be overridden per variant
        prompt_set_name=args.prompt_set,
        device=args.device,
        save_original_weights=True,
    )
    ln = LieNeuro(model, tokenizer, cfg)

    if Path(args.mask).exists():
        print(f"Loading mask from {args.mask} …")
        ln.load_mask(args.mask)
        ln.build_mask()             # rebuild from loaded scores
    else:
        print("Mask file not found — computing importance scores …")
        lie_p, truth_p = get_prompt_set(args.prompt_set)
        ln.compute_importance_scores(lie_p, truth_p)
        ln.build_mask()

    # ── Run all three variants ────────────────────────────────────────────────
    all_rows: List[dict] = []

    print("\n[1/3] Baseline (no steering)")
    all_rows += run_variant(
        label="baseline", model=model, baseline_model=baseline_model,
        tokenizer=tokenizer, lie_neuro=None, alpha=None,
        prompts=FLUENCY_PROMPTS, args=args, device=device,
    )

    print(f"\n[2/3] Steered α={args.alpha1}")
    all_rows += run_variant(
        label=f"steered_a{args.alpha1}", model=model, baseline_model=baseline_model,
        tokenizer=tokenizer, lie_neuro=ln, alpha=args.alpha1,
        prompts=FLUENCY_PROMPTS, args=args, device=device,
    )

    print(f"\n[3/3] Steered α={args.alpha2}")
    all_rows += run_variant(
        label=f"steered_a{args.alpha2}", model=model, baseline_model=baseline_model,
        tokenizer=tokenizer, lie_neuro=ln, alpha=args.alpha2,
        prompts=FLUENCY_PROMPTS, args=args, device=device,
    )

    # ── Save per-response CSV ─────────────────────────────────────────────────
    results_path = outdir / "fluency_results.csv"
    cols = ["variant","alpha","prompt_idx","prompt","response",
            "n_tokens","perplexity","mean_logprob","rep_rate","distinct_1","distinct_2","empty"]
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved {results_path}  ({len(all_rows)} rows)")

    # ── Aggregate summary CSV ─────────────────────────────────────────────────
    from collections import defaultdict
    import statistics

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for r in all_rows:
        grouped[r["variant"]].append(r)

    def safe_mean(vals):
        nums = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        return round(statistics.mean(nums), 4) if nums else float("nan")

    def safe_std(vals):
        nums = [v for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        return round(statistics.stdev(nums), 4) if len(nums) >= 2 else float("nan")

    summary_path = outdir / "fluency_summary.csv"
    summary_cols = [
        "variant", "alpha", "n_prompts",
        "mean_perplexity",  "std_perplexity",
        "mean_logprob",     "std_logprob",
        "mean_n_tokens",    "std_n_tokens",
        "mean_rep_rate",    "std_rep_rate",
        "mean_distinct_1",  "std_distinct_1",
        "mean_distinct_2",  "std_distinct_2",
        "empty_rate",
    ]
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        for variant, rows in grouped.items():
            ppls = [r["perplexity"]   for r in rows]
            lps  = [r["mean_logprob"] for r in rows]
            lens = [r["n_tokens"]     for r in rows]
            reps = [r["rep_rate"]     for r in rows]
            d1s  = [r["distinct_1"]   for r in rows]
            d2s  = [r["distinct_2"]   for r in rows]
            w.writerow({
                "variant":         variant,
                "alpha":           rows[0]["alpha"],
                "n_prompts":       len(rows),
                "mean_perplexity": safe_mean(ppls),
                "std_perplexity":  safe_std(ppls),
                "mean_logprob":    safe_mean(lps),
                "std_logprob":     safe_std(lps),
                "mean_n_tokens":   safe_mean(lens),
                "std_n_tokens":    safe_std(lens),
                "mean_rep_rate":   safe_mean(reps),
                "std_rep_rate":    safe_std(reps),
                "mean_distinct_1": safe_mean(d1s),
                "std_distinct_1":  safe_std(d1s),
                "mean_distinct_2": safe_mean(d2s),
                "std_distinct_2":  safe_std(d2s),
                "empty_rate":      round(sum(r["empty"] for r in rows) / len(rows), 4),
            })
    print(f"Saved {summary_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                       Fluency Evaluation Summary                            ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"║  {'variant':<22}  {'ppl':>7}  {'logprob':>8}  {'len':>5}  {'rep':>6}  {'d1':>6}  {'d2':>6}  {'empty':>5}  ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    for variant, rows in grouped.items():
        ppls = [r["perplexity"]   for r in rows]
        lps  = [r["mean_logprob"] for r in rows]
        lens = [r["n_tokens"]     for r in rows]
        reps = [r["rep_rate"]     for r in rows]
        d1s  = [r["distinct_1"]   for r in rows]
        d2s  = [r["distinct_2"]   for r in rows]
        er   = sum(r["empty"] for r in rows) / len(rows)
        print(
            f"║  {variant:<22}  {safe_mean(ppls):>7.2f}  {safe_mean(lps):>8.4f}  "
            f"{safe_mean(lens):>5.1f}  {safe_mean(reps):>6.4f}  "
            f"{safe_mean(d1s):>6.4f}  {safe_mean(d2s):>6.4f}  {er:>5.3f}  ║"
        )
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print("║  perplexity: lower = more fluent  (baseline model as reference for all)     ║")
    print("║  rep_rate: fraction of repeated 4-grams  (higher = more repetitive)         ║")
    print("║  distinct_1/2: lexical diversity (higher = more varied vocabulary)           ║")
    print("║  empty_rate: fraction of blank responses                                     ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
