#!/usr/bin/env python3
"""
LLM-judge re-evaluation for the top-3 layers from the activation steering sweep.

For each of the 3 best-performing layers (by max Δwrong across coefficients),
picks the best coefficient, re-runs inference with the steering hook to collect
per-question (question, gold, baseline_answer, steered_answer), then judges
with Qwen2.5-14B-Instruct-4bit.

Outputs
-------
  outputs/as_per_question_judged.csv  — per-question string + judge verdicts
  outputs/as_judge_summary.csv        — aggregate string vs judge metrics

Usage
-----
  CUDA_VISIBLE_DEVICES=2 python run_as_judge.py --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from src.evaluate import evaluate_factual_accuracy, FACTUAL_QA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL  = "Qwen/Qwen2.5-14B-Instruct"
OUT_DIR      = Path("outputs")
SWEEP_CSV    = OUT_DIR / "activation_steering_sweep.csv"
VECTORS_PT   = OUT_DIR / "activation_steering_vectors.pt"
PQ_OUT       = OUT_DIR / "as_per_question_judged.csv"
SUMMARY_OUT  = OUT_DIR / "as_judge_summary.csv"

N_TOP_LAYERS = 3

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


# ── helpers ───────────────────────────────────────────────────────────────────

def load_sweep():
    with open(SWEEP_CSV, newline="") as f:
        return list(csv.DictReader(f))


def top_layers(rows, n=N_TOP_LAYERS):
    """Return list of (layer, best_coeff, best_dw) for the n highest-Δwrong layers."""
    by_layer: dict[int, tuple[float, float]] = {}
    for r in rows:
        layer = int(r["layer"])
        coeff = float(r["coeff"])
        dw    = float(r["delta_wrong_rate"])
        if layer not in by_layer or dw > by_layer[layer][1]:
            by_layer[layer] = (coeff, dw)
    ranked = sorted(by_layer.items(), key=lambda x: -x[1][1])[:n]
    return [(layer, coeff, dw) for layer, (coeff, dw) in ranked]


def run_steered_inference(model, tokenizer, steering, layer, coeff, device):
    """Apply hook, run eval (captures per-question answers), remove hook."""
    steering.apply_steering(layer, coeff)
    result = evaluate_factual_accuracy(model, tokenizer, device=device)
    steering.remove_steering()
    return result


def run_baseline_inference(model, tokenizer, device):
    return evaluate_factual_accuracy(model, tokenizer, device=device)


# ── judge ─────────────────────────────────────────────────────────────────────

def load_judge(model_id):
    logger.info("Loading judge: %s (device_map=auto)", model_id)
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto"
    )
    model.eval()
    return model, tok


def judge_one(jmodel, jtok, question, gold, answer):
    content = JUDGE_TEMPLATE.format(question=question, gold=gold, model_answer=answer)
    messages = [{"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": content}]
    text = jtok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # With device_map="auto" inputs go to the first shard device
    input_device = next(jmodel.parameters()).device
    inputs = jtok(text, return_tensors="pt").to(input_device)
    with torch.no_grad():
        out = jmodel.generate(
            **inputs, max_new_tokens=5, do_sample=False,
            temperature=None, top_p=None,
            pad_token_id=jtok.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return jtok.decode(new, skip_special_tokens=True).strip().upper().startswith("YES")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    return p.parse_args()


def main():
    args = parse_args()

    # ── find top 3 layers ────────────────────────────────────────────────────
    sweep_rows = load_sweep()
    configs = top_layers(sweep_rows, N_TOP_LAYERS)
    logger.info("Top %d layers:", N_TOP_LAYERS)
    for layer, coeff, dw in configs:
        logger.info("  layer=%d  coeff=%.0f  Δwrong=%.0%%", layer, coeff, dw * 100)

    # ── load LLaMA across all visible GPUs ───────────────────────────────────
    logger.info("Loading LLaMA model (device_map=auto) …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map="auto"
    )
    model.eval()

    # ActivationSteeringConfig.device is only used for the config default;
    # actual device placement is resolved per-layer in apply_steering.
    cfg      = ActivationSteeringConfig(device="cuda:0")
    steering = ActivationSteering(model, tok, cfg)
    steering.load_vectors(VECTORS_PT)

    # ── baseline inference ───────────────────────────────────────────────────
    logger.info("Baseline inference …")
    bl_result = run_baseline_inference(model, tok, "cuda:0")
    bl_answers = {r["question"]: r for r in bl_result.per_question}

    # ── steered inference for each top config ────────────────────────────────
    steered_data: list[dict] = []   # {layer, coeff, dw, pq: list[dict]}
    for layer, coeff, dw in configs:
        logger.info("Steered inference: layer=%d  coeff=%.0f …", layer, coeff)
        result = run_steered_inference(model, tok, steering, layer, coeff, "cuda:0")
        steered_data.append({"layer": layer, "coeff": coeff, "dw": dw,
                              "pq": result.per_question})

    # ── free LLaMA before loading judge ─────────────────────────────────────
    logger.info("Freeing LLaMA from GPU …")
    del model
    torch.cuda.empty_cache()

    # ── load judge across all visible GPUs ───────────────────────────────────
    jmodel, jtok = load_judge(args.judge_model)

    # ── collect all unique (question, gold, answer) triples ──────────────────
    triples: set[tuple] = set()
    for item in steered_data:
        for pq in item["pq"]:
            triples.add((pq["question"], pq["gold"], pq["model_answer"]))
    for pq in bl_result.per_question:
        triples.add((pq["question"], pq["gold"], pq["model_answer"]))
    triples = list(triples)

    logger.info("Judging %d unique (q, gold, answer) triples …", len(triples))
    verdict: dict[tuple, bool] = {}
    for i, (q, g, a) in enumerate(triples):
        if i % 20 == 0:
            logger.info("  %d / %d", i, len(triples))
        verdict[(q, g, a)] = judge_one(jmodel, jtok, q, g, a)

    # ── write per-question CSV ───────────────────────────────────────────────
    pq_cols = [
        "layer", "coeff", "delta_wrong_string",
        "question", "gold",
        "baseline_answer", "baseline_correct", "baseline_judge_correct",
        "steered_answer",  "steered_correct",  "steered_judge_correct",
        "string_flipped",  "judge_flipped",
    ]
    pq_rows = []
    for item in steered_data:
        for pq in item["pq"]:
            q, g = pq["question"], pq["gold"]
            bl   = bl_answers.get(q, {})
            bl_ans = bl.get("model_answer", "")
            bl_str = bl.get("correct", False)
            st_ans = pq["model_answer"]
            st_str = pq["correct"]
            bl_jdg = verdict.get((q, g, bl_ans), bl_str)
            st_jdg = verdict.get((q, g, st_ans), st_str)
            pq_rows.append({
                "layer":                item["layer"],
                "coeff":                item["coeff"],
                "delta_wrong_string":   item["dw"],
                "question":             q,
                "gold":                 g,
                "baseline_answer":      bl_ans,
                "baseline_correct":     bl_str,
                "baseline_judge_correct": bl_jdg,
                "steered_answer":       st_ans,
                "steered_correct":      st_str,
                "steered_judge_correct": st_jdg,
                "string_flipped":       bl_str and not st_str,
                "judge_flipped":        bl_jdg and not st_jdg,
            })

    with open(PQ_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pq_cols)
        w.writeheader()
        w.writerows(pq_rows)
    logger.info("Saved %s (%d rows)", PQ_OUT, len(pq_rows))

    # ── write summary CSV ─────────────────────────────────────────────────────
    summary_cols = [
        "layer", "coeff",
        "string_baseline_acc", "string_steered_acc", "string_delta_wrong",
        "judge_baseline_acc",  "judge_steered_acc",  "judge_delta_wrong",
        "n_string_flipped",    "n_judge_flipped",
    ]
    summary_rows = []
    for item in steered_data:
        grp = [r for r in pq_rows if r["layer"] == item["layer"]
               and r["coeff"] == item["coeff"]]
        n = len(grp)
        bl_str = sum(1 for r in grp if r["baseline_correct"])  / n
        st_str = sum(1 for r in grp if r["steered_correct"])   / n
        bl_jdg = sum(1 for r in grp if r["baseline_judge_correct"]) / n
        st_jdg = sum(1 for r in grp if r["steered_judge_correct"])  / n
        summary_rows.append({
            "layer":               item["layer"],
            "coeff":               item["coeff"],
            "string_baseline_acc": round(bl_str, 4),
            "string_steered_acc":  round(st_str, 4),
            "string_delta_wrong":  round((1 - st_str) - (1 - bl_str), 4),
            "judge_baseline_acc":  round(bl_jdg, 4),
            "judge_steered_acc":   round(st_jdg, 4),
            "judge_delta_wrong":   round((1 - st_jdg) - (1 - bl_jdg), 4),
            "n_string_flipped":    sum(1 for r in grp if r["string_flipped"]),
            "n_judge_flipped":     sum(1 for r in grp if r["judge_flipped"]),
        })

    with open(SUMMARY_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_cols)
        w.writeheader()
        w.writerows(summary_rows)
    logger.info("Saved %s", SUMMARY_OUT)

    # ── print comparison table ────────────────────────────────────────────────
    print()
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│   Activation Steering: String-match vs LLM-judge (top 3 layers) │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│  {'Layer':>5}  {'Coeff':>5}  {'str Δwrong':>10}  {'jdg Δwrong':>10}  {'diff':>8}  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    for r in summary_rows:
        diff = r["judge_delta_wrong"] - r["string_delta_wrong"]
        print(
            f"│  {r['layer']:>5}  {r['coeff']:>5.0f}  "
            f"{r['string_delta_wrong']:>+10.1%}  {r['judge_delta_wrong']:>+10.1%}  "
            f"{diff:>+8.1%}  │"
        )
    print("└─────────────────────────────────────────────────────────────────┘")


if __name__ == "__main__":
    main()
