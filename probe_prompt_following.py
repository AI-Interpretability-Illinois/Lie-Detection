#!/usr/bin/env python3
"""
probe_prompt_following.py — Generate model outputs for all contrastive prompts.

Runs every prompt from src/prompts.py through the unsteered baseline model and
saves the results to outputs/prompt_following.csv so you can inspect how well
the model follows "lie" vs "be honest" instructions.

Usage:
  CUDA_VISIBLE_DEVICES=1 python probe_prompt_following.py
  CUDA_VISIBLE_DEVICES=1 python probe_prompt_following.py --model meta-llama/Llama-3.1-8B-Instruct
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import PROMPT_SETS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTDIR = Path("outputs")
DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model",          default=DEFAULT_MODEL)
    p.add_argument("--device",         default=None)
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--temperature",    type=float, default=1.0)
    return p.parse_args()


def generate(model, tokenizer, prompt: str, device, max_new_tokens: int, temperature: float) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    OUTDIR.mkdir(exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    logger.info("Loading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, device_map=device
    )
    model.eval()

    out_path = OUTDIR / "prompt_following.csv"
    cols = ["prompt_set", "instruction_type", "prompt", "model_output"]

    rows = []
    for ps_name, (lie_prompts, truth_prompts) in PROMPT_SETS.items():
        for itype, prompts in [("lie", lie_prompts), ("truth", truth_prompts)]:
            for i, prompt in enumerate(prompts):
                logger.info("[%s / %s] prompt %d …", ps_name, itype, i + 1)
                output = generate(model, tokenizer, prompt, device,
                                  args.max_new_tokens, args.temperature)
                rows.append({
                    "prompt_set":        ps_name,
                    "instruction_type":  itype,
                    "prompt":            prompt,
                    "model_output":      output,
                })

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    logger.info("Saved %d rows to %s", len(rows), out_path)

    # Quick summary: how often does the model appear to comply?
    print()
    print(f"{'prompt_set':<12}  {'type':<6}  {'n_prompts':>9}  sample output")
    print("-" * 80)
    for ps_name in PROMPT_SETS:
        for itype in ("lie", "truth"):
            subset = [r for r in rows if r["prompt_set"] == ps_name
                      and r["instruction_type"] == itype]
            sample = subset[0]["model_output"][:60].replace("\n", " ")
            print(f"{ps_name:<12}  {itype:<6}  {len(subset):>9}  {sample}…")


if __name__ == "__main__":
    main()
