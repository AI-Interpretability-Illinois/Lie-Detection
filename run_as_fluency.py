#!/usr/bin/env python3
"""
Fluency evaluation for the top activation-steering configs.

Run AFTER run_activation_steering_sweep.py has finished.
Reads outputs/activation_steering_sweep.csv, picks the top N configs by
Δwrong-rate, evaluates fluency with the same metrics used in LieNeuro, and
writes outputs/activation_steering_fluency.csv.

Usage
-----
CUDA_VISIBLE_DEVICES=1 python run_as_fluency.py --device cuda:0 --top 6
"""

import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.activation_steering import ActivationSteering, ActivationSteeringConfig
from fluency_eval import (
    generate_with_logprobs,
    compute_perplexity,
    lexical_metrics,
    FLUENCY_PROMPTS,
)

MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR      = Path("outputs")
SWEEP_CSV    = OUT_DIR / "activation_steering_sweep.csv"
VECTORS_PT   = OUT_DIR / "activation_steering_vectors.pt"
FLUENCY_CSV  = OUT_DIR / "activation_steering_fluency.csv"

FIELDNAMES = [
    "variant", "layer", "coeff",
    "mean_perplexity", "mean_logprob",
    "mean_rep_rate", "mean_distinct_1", "mean_distinct_2",
    "delta_wrong_rate",
]

MAX_NEW_TOKENS = 200
TEMPERATURE    = 1.0


def load_top_configs(n: int, sweep_csv=None):
    path = sweep_csv or SWEEP_CSV
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["delta_wrong_rate"]), int(r["layer"]), float(r["coeff"])))
    rows.sort(key=lambda x: -x[0])
    return rows[:n]


def eval_fluency(model, tokenizer, steering, layer, coeff, device):
    """Apply steering, generate fluency texts, restore, score perplexity."""
    steering.apply_steering(layer, coeff)

    texts_and_ids = []
    for p in FLUENCY_PROMPTS:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        text, mean_lp, n_tok = generate_with_logprobs(
            model, tokenizer, prompt, MAX_NEW_TOKENS, TEMPERATURE, device
        )
        enc = tokenizer(text, return_tensors="pt").to(device)
        ids = enc["input_ids"][0].tolist()
        texts_and_ids.append((p, text, mean_lp, ids))

    steering.remove_steering()

    rows = []
    for (p, text, mean_lp, ids) in texts_and_ids:
        ppl = compute_perplexity(model, tokenizer, text, device)
        rep_rate, d1, d2 = lexical_metrics(ids)
        rows.append({
            "perplexity": ppl,
            "mean_logprob": mean_lp,
            "rep_rate": rep_rate,
            "distinct_1": d1,
            "distinct_2": d2,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top",    type=int, default=6,
                        help="evaluate fluency for this many top configs (by Δwrong)")
    parser.add_argument("--extra-configs", default=None,
                        help="comma-separated layer:coeff:delta_wrong triples to append "
                             "(e.g. '0:2.0:0.9,1:2.0:0.9'). Appends to existing CSV.")
    parser.add_argument("--vectors", default=None,
                        help="path to steering vectors .pt file (overrides VECTORS_PT)")
    parser.add_argument("--sweep-csv", default=None,
                        help="sweep CSV to read top configs from (overrides SWEEP_CSV)")
    parser.add_argument("--out", default=None,
                        help="output fluency CSV path (overrides FLUENCY_CSV)")
    args = parser.parse_args()

    print("Loading model …")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=args.device
    )
    model.eval()

    cfg      = ActivationSteeringConfig(device=args.device)
    steering = ActivationSteering(model, tok, cfg)
    vectors_path = Path(args.vectors)   if args.vectors   else VECTORS_PT
    fluency_csv  = Path(args.out)       if args.out       else FLUENCY_CSV
    sweep_csv    = Path(args.sweep_csv) if args.sweep_csv else SWEEP_CSV
    steering.load_vectors(vectors_path)

    # ── Append-only mode: evaluate specific configs and append to CSV ──────────
    if args.extra_configs:
        extra = []
        for triple in args.extra_configs.split(","):
            layer_s, coeff_s, dw_s = triple.strip().split(":")
            extra.append((int(layer_s), float(coeff_s), float(dw_s)))

        append_rows = []
        for layer, coeff, dw in extra:
            tag = f"as_layer{layer}_c{coeff}"
            print(f"\nFluency (extra): {tag} …")
            rows = eval_fluency(model, tok, steering, layer, coeff, args.device)
            def m(key, rows=rows): return sum(r[key] for r in rows) / len(rows)
            append_rows.append({
                "variant":         tag,
                "layer":           layer,
                "coeff":           coeff,
                "mean_perplexity": m("perplexity"),
                "mean_logprob":    m("mean_logprob"),
                "mean_rep_rate":   m("rep_rate"),
                "mean_distinct_1": m("distinct_1"),
                "mean_distinct_2": m("distinct_2"),
                "delta_wrong_rate": dw,
            })
            print(f"  ppl={m('perplexity'):.1f}  Δwrong={dw:+.2%}")

        write_header = not fluency_csv.exists()
        with open(fluency_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerows(append_rows)
        print(f"\nAppended {len(append_rows)} rows to {fluency_csv}")
        return

    top_configs = load_top_configs(args.top, sweep_csv)
    print(f"Top {args.top} configs by Δwrong:")
    for dw, layer, coeff in top_configs:
        print(f"  layer={layer:2d}  c={coeff:5.1f}  Δwrong={dw:+.2%}")

    # baseline fluency
    print("\nBaseline fluency …")
    bl_rows = []
    for p in FLUENCY_PROMPTS:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        text, mean_lp, n_tok = generate_with_logprobs(
            model, tok, prompt, MAX_NEW_TOKENS, TEMPERATURE, args.device
        )
        ppl = compute_perplexity(model, tok, text, args.device)
        enc = tok(text, return_tensors="pt").to(args.device)
        ids = enc["input_ids"][0].tolist()
        rep_rate, d1, d2 = lexical_metrics(ids)
        bl_rows.append({"perplexity": ppl, "mean_logprob": mean_lp,
                         "rep_rate": rep_rate, "distinct_1": d1, "distinct_2": d2})

    def mean(key):
        return sum(r[key] for r in bl_rows) / len(bl_rows)

    output_rows = [{
        "variant": "baseline", "layer": "", "coeff": "",
        "mean_perplexity": mean("perplexity"),
        "mean_logprob":    mean("mean_logprob"),
        "mean_rep_rate":   mean("rep_rate"),
        "mean_distinct_1": mean("distinct_1"),
        "mean_distinct_2": mean("distinct_2"),
        "delta_wrong_rate": 0.0,
    }]

    for dw, layer, coeff in top_configs:
        tag = f"as_layer{layer}_c{coeff}"
        print(f"\nFluency: {tag} …")
        rows = eval_fluency(model, tok, steering, layer, coeff, args.device)
        def m(key, rows=rows): return sum(r[key] for r in rows) / len(rows)
        output_rows.append({
            "variant":         tag,
            "layer":           layer,
            "coeff":           coeff,
            "mean_perplexity": m("perplexity"),
            "mean_logprob":    m("mean_logprob"),
            "mean_rep_rate":   m("rep_rate"),
            "mean_distinct_1": m("distinct_1"),
            "mean_distinct_2": m("distinct_2"),
            "delta_wrong_rate": dw,
        })
        print(f"  ppl={m('perplexity'):.1f}  Δwrong={dw:+.2%}")

    with open(fluency_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"\nFluency results saved to {fluency_csv}")


if __name__ == "__main__":
    main()
