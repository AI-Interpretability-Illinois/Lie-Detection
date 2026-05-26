#!/usr/bin/env python3
"""
run_experiment.py — LieNeuro steering experiment for LLaMA-3.1-8B-Instruct.

Replicates Algorithm 1 from CreativityNeuro (Schapiro et al., 2025) with a
lying / truthful contrastive axis instead of the creative / non-creative axis.

Usage
-----
  # Basic run with defaults
  python run_experiment.py

  # Sweep over rho and alpha
  python run_experiment.py --rho 0.05 --alpha 2.0 --prompt-set direct

  # Evaluate without recomputing the mask (load from disk)
  python run_experiment.py --load-mask outputs/mask_direct.pt --skip-steering-compute

  # Dry-run: only evaluate factual accuracy on baseline (no steering)
  python run_experiment.py --eval-only

  # Full hyperparameter grid sweep (rho × alpha × prompt_set)
  python run_experiment.py --sweep

Full CLI reference below.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from itertools import product
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import compare_baseline_steered, evaluate_factual_accuracy
from src.prompts import get_prompt_set, list_prompt_sets

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_MODEL   = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_OUTDIR  = "outputs"

# Hyperparameter grid (mirrors Appendix D of the paper)
SWEEP_RHO   = [0.01, 0.05, 0.1, 0.2]
SWEEP_ALPHA = [0.1, 0.5, 1.0, 2.0]
SWEEP_SETS  = list_prompt_sets()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LieNeuro: weight-space steering for deceptive behaviour.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    p.add_argument("--model",      default=DEFAULT_MODEL,
                   help="HuggingFace model ID or local path.")
    p.add_argument("--device",     default=None,
                   help="Torch device string (e.g. 'cuda:0', 'cpu'). Auto-detected if omitted.")
    p.add_argument("--load-in-8bit", action="store_true",
                   help="Load model in 8-bit (requires bitsandbytes).")

    # Hyperparameters
    p.add_argument("--rho",        type=float, default=0.1,
                   help="Importance threshold ρ (fraction of weights selected).")
    p.add_argument("--alpha",      type=float, default=1.0,
                   help="Scaling factor α applied to lie-specific weights.")
    p.add_argument("--prompt-set", default="direct", choices=list_prompt_sets(),
                   help="Contrastive prompt set for importance scoring.")
    p.add_argument("--max-length", type=int, default=256,
                   help="Max token length for prompt tokenisation.")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for importance-score forward passes.")

    # Evaluation
    p.add_argument("--n-questions",  type=int, default=None,
                   help="Subsample N questions from the QA set. None = all 50.")
    p.add_argument("--temperature",  type=float, default=1.0,
                   help="Sampling temperature for generation.")
    p.add_argument("--max-new-tokens", type=int, default=50,
                   help="Max new tokens per generated answer.")

    # Modes
    p.add_argument("--eval-only",   action="store_true",
                   help="Only evaluate baseline accuracy, skip steering.")
    p.add_argument("--sweep",       action="store_true",
                   help="Run full ρ × α × prompt-set grid sweep.")
    p.add_argument("--load-mask",   default=None,
                   help="Path to a previously saved .pt mask file.")

    # Output
    p.add_argument("--output-dir",  default=DEFAULT_OUTDIR,
                   help="Directory for saving masks and results.")
    p.add_argument("--save-mask",   action="store_true",
                   help="Save the computed importance mask to disk.")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_id: str,
    device: str | None,
    load_in_8bit: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    logger.info("Loading tokenizer from %s …", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from %s …", model_id)
    # Use 'dtype' (transformers ≥5.x); fall back to float16 for memory efficiency.
    kwargs: dict = {"dtype": torch.float16}

    if load_in_8bit:
        kwargs["load_in_8bit"] = True
        kwargs["device_map"]   = "auto"
    elif device is not None:
        # Explicit device — load directly, no offloading
        kwargs["device_map"] = device
    else:
        # Default: put everything on cuda:0 (fits on H100 80GB for 8B models)
        default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        kwargs["device_map"] = default_device

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if torch.cuda.is_available():
        logger.info(
            "Model loaded (%s parameters).  GPU memory: %.1f GB used / %.1f GB total.",
            f"{sum(p.numel() for p in model.parameters()):,}",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    else:
        logger.info("Model loaded (%s parameters).",
                    f"{sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def single_run(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    rho: float,
    alpha: float,
    prompt_set_name: str,
    output_dir: Path,
) -> dict:
    """Execute one (ρ, α, prompt_set) configuration and return result dict."""

    cfg = LieNeuroConfig(
        importance_threshold = rho,
        alpha                = alpha,
        prompt_set_name      = prompt_set_name,
        device               = args.device,
        max_length           = args.max_length,
        batch_size           = args.batch_size,
        save_original_weights= True,
    )

    ln = LieNeuro(model, tokenizer, cfg)

    # Load or compute importance mask
    if args.load_mask and Path(args.load_mask).exists():
        logger.info("Loading mask from %s", args.load_mask)
        ln.load_mask(args.load_mask)
    else:
        lie_prompts, truth_prompts = get_prompt_set(prompt_set_name)
        ln.compute_importance_scores(lie_prompts, truth_prompts)
        ln.build_mask()

        if args.save_mask:
            mask_path = output_dir / f"mask_{prompt_set_name}_rho{rho}_alpha{alpha}.pt"
            ln.save_mask(str(mask_path))

    # Evaluate
    comparison = compare_baseline_steered(
        model, tokenizer, ln,
        n_questions = args.n_questions,
        temperature = args.temperature,
        device      = torch.device(args.device) if args.device else None,
    )

    result = {
        "rho":              rho,
        "alpha":            alpha,
        "prompt_set":       prompt_set_name,
        "baseline_accuracy":  comparison["baseline"].accuracy,
        "steered_accuracy":   comparison["steered"].accuracy,
        "delta_accuracy":     comparison["delta"]["accuracy"],
        "baseline_wrong_rate":comparison["baseline"].wrong_rate,
        "steered_wrong_rate": comparison["steered"].wrong_rate,
        "delta_wrong_rate":   comparison["delta"]["wrong_rate"],
        "n_questions":        comparison["baseline"].n_questions,
        "baseline_per_question": comparison["baseline"].per_question,
        "steered_per_question":  comparison["steered"].per_question,
    }

    logger.info(
        "[ρ=%.2f α=%.1f set=%s]  baseline_acc=%.3f  steered_acc=%.3f  "
        "Δacc=%.3f  Δwrong_rate=%.3f",
        rho, alpha, prompt_set_name,
        result["baseline_accuracy"], result["steered_accuracy"],
        result["delta_accuracy"],    result["delta_wrong_rate"],
    )
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.device, args.load_in_8bit
    )

    # ---- eval-only mode ----
    if args.eval_only:
        logger.info("Eval-only mode: evaluating baseline accuracy.")
        result = evaluate_factual_accuracy(
            model, tokenizer,
            n_questions=args.n_questions,
            temperature=args.temperature,
        )
        logger.info("Baseline accuracy: %.3f  wrong_rate: %.3f",
                    result.accuracy, result.wrong_rate)
        out_path = output_dir / "baseline_eval.json"
        with open(out_path, "w") as f:
            json.dump(
                {"accuracy": result.accuracy,
                 "wrong_rate": result.wrong_rate,
                 "n_questions": result.n_questions,
                 "per_question": result.per_question},
                f, indent=2,
            )
        logger.info("Saved to %s", out_path)
        return

    # ---- sweep mode ----
    if args.sweep:
        logger.info(
            "Grid sweep: %d ρ × %d α × %d prompt sets = %d configs.",
            len(SWEEP_RHO), len(SWEEP_ALPHA), len(SWEEP_SETS),
            len(SWEEP_RHO) * len(SWEEP_ALPHA) * len(SWEEP_SETS),
        )
        all_results = []
        for rho, alpha, pset in product(SWEEP_RHO, SWEEP_ALPHA, SWEEP_SETS):
            try:
                r = single_run(model, tokenizer, args, rho, alpha, pset, output_dir)
                all_results.append(r)
            except Exception as exc:
                logger.error("Config ρ=%.2f α=%.1f set=%s failed: %s", rho, alpha, pset, exc)

        out_path = output_dir / "sweep_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Sweep complete. Results saved to %s", out_path)

        # Print summary table
        _print_summary_table(all_results)
        return

    # ---- single run ----
    result = single_run(
        model, tokenizer, args,
        rho            = args.rho,
        alpha          = args.alpha,
        prompt_set_name= args.prompt_set,
        output_dir     = output_dir,
    )

    out_path = output_dir / f"result_{args.prompt_set}_rho{args.rho}_alpha{args.alpha}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Result saved to %s", out_path)


# ---------------------------------------------------------------------------
# Summary table printer
# ---------------------------------------------------------------------------

def _print_summary_table(results: list[dict]) -> None:
    """Print a concise summary of sweep results sorted by Δ wrong_rate."""
    sorted_r = sorted(results, key=lambda x: -x["delta_wrong_rate"])
    header = f"{'ρ':>5}  {'α':>5}  {'prompt_set':<12}  {'base_acc':>8}  {'strd_acc':>8}  {'Δacc':>8}  {'Δwrong':>8}"
    print("\n" + "=" * len(header))
    print("SWEEP RESULTS  (sorted by Δ wrong_rate ↓ = more deception)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in sorted_r:
        print(
            f"{r['rho']:>5.2f}  {r['alpha']:>5.1f}  {r['prompt_set']:<12}  "
            f"{r['baseline_accuracy']:>8.3f}  {r['steered_accuracy']:>8.3f}  "
            f"{r['delta_accuracy']:>8.3f}  {r['delta_wrong_rate']:>8.3f}"
        )
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    main()
