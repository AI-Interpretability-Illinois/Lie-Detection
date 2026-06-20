#!/usr/bin/env python3
"""
α sweep at fixed ρ=0.04 — generate samples + WikiText-2 PPL for each α.

Sweep: α ∈ {1, 2, 3, 4, 5, 6, 8}
  α=4 already has samples and PPL → skipped

Usage
-----
  CUDA_VISIBLE_DEVICES=1 conda run -n lie_neuro python run_alpha_sweep_rho0.04.py --device cuda:0
"""

import argparse, csv, math
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_TRUTH_INSTRUCT

MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
SCORES_PT    = Path("outputs/lieneuro_did_scores.pt")
WIKITEXT_CSV = Path("outputs/wikitext_ppl.csv")
MODE, TEMP, N_REPS = "score_diff", 1.0, 3
RHO = 0.04

# (alpha, need_ppl, samples_path)  — None path = skip entirely
CONFIGS = [
    (1.0, True,  Path("outputs/lieneuro_did_samples_rho0.04_a1.txt")),
    (2.0, True,  Path("outputs/lieneuro_did_samples_rho0.04_a2.txt")),
    (3.0, True,  Path("outputs/lieneuro_did_samples_rho0.04_a3.txt")),
    (4.0, False, None),   # samples + PPL already exist
    (5.0, True,  Path("outputs/lieneuro_did_samples_rho0.04_a5.txt")),
    (6.0, True,  Path("outputs/lieneuro_did_samples_rho0.04_a6.txt")),
    (8.0, True,  Path("outputs/lieneuro_did_samples_rho0.04_a8.txt")),
]

WT_WINDOW, WT_STRIDE, WT_MAX_TOKS = 512, 256, 8192

_INSTRUCT_QS = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
                for p in DIRECT_TRUTH_INSTRUCT}
def is_indist(q): return q.strip().lower() in _INSTRUCT_QS


def load_wikitext_tokens(tok):
    ds   = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tok.encode(text, add_special_tokens=False)[:WT_MAX_TOKS]


@torch.no_grad()
def sliding_window_ppl(model, token_ids, device):
    ids = torch.tensor(token_ids, dtype=torch.long)
    n   = len(ids); nlls, n_toks = [], 0
    for begin in range(0, n, WT_STRIDE):
        end     = min(begin + WT_WINDOW, n)
        chunk   = ids[begin:end].unsqueeze(0).to(device)
        tgt_len = min(WT_STRIDE, end - begin)
        labels  = chunk.clone(); labels[:, :-tgt_len] = -100
        nlls.append(model(input_ids=chunk, labels=labels).loss.item() * tgt_len)
        n_toks += tgt_len
        if end == n: break
    return math.exp(sum(nlls) / n_toks)


def append_ppl_row(rho, alpha, dw, ppl):
    label = f"LieNeuro-DID score_diff ρ={rho:.3f} α={alpha:.1f}"
    with open(WIKITEXT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "variant", "method", "layer", "coeff", "rho", "alpha",
            "delta_wrong", "wikitext_ppl"])
        w.writerow({"variant": label, "method": "weight_steering_did",
                    "layer": "", "coeff": "", "rho": rho, "alpha": alpha,
                    "delta_wrong": dw, "wikitext_ppl": ppl})
    print(f"    PPL row saved: {label}  Δwrong={dw:+.2%}  PPL={ppl:.4f}")


def sample(model, tok, q, n, device):
    prompt = _build_prompt(q, tok)
    return [generate_answer(model, tok, prompt, max_new_tokens=40,
                            temperature=TEMP, device=device) for _ in range(n)]


def generate_samples_file(model, tok, ln, saved, rho, alpha, out_file, bl_results, device):
    cfg = LieNeuroConfig(importance_threshold=rho, alpha=alpha, device=device)
    ln.config   = cfg
    ln._S_lie   = saved["S_lie"]
    ln._S_truth = saved["S_truth_instruct"]
    ln.build_mask(mode=MODE)

    lines = []
    def w(s=""): lines.append(s)
    w("LieNeuro DID — Sample Outputs")
    w(f"Mode: {MODE}  |  ρ={rho}  |  α={alpha}  |  temperature={TEMP}")
    w(f"Model: Llama-3.1-8B-Instruct  |  {N_REPS} samples per question")
    w("=" * 72)
    w(); w("BASELINE (no steering)"); w("-" * 72)
    for q, a, resp, correct in bl_results:
        w(f"  {'✓' if correct else '✗'}  Q: {q}")
        w(f"       A: {resp}  [expected: {a}]")
    bl_wrong = sum(not r[3] for r in bl_results)
    w(); w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    ln.apply_steering()
    st_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resps    = sample(model, tok, q, N_REPS, device)
        corrects = [_is_correct(r, a) for r in resps]
        st_results.append((q, a, resps, corrects, is_indist(q)))
    ln.restore_original()

    def write_group(label, results):
        w(); w()
        w(f"STEERED — {label}  mode={MODE}  ρ={rho}  α={alpha}  ({N_REPS} samples/question)")
        w("-" * 72)
        for q, a, resps, corrects, _ in results:
            bl_resp = next(r for bq, _, r, _ in bl_results if bq == q)
            marks   = ["✓" if c else "✗" for c in corrects]
            w(f"  Q: {q}  [expected: {a}]")
            w(f"       Baseline:  {bl_resp}")
            for i, (r, m) in enumerate(zip(resps, marks)):
                w(f"       Sample {i+1}: {m}  {r}")
            w()
        bl_w  = sum(not _is_correct(next(r for bq, _, r, _ in bl_results if bq == q), a)
                    for q, a, _, _, _ in results)
        all_c = [c for _, _, _, cs, _ in results for c in cs]
        n_w, n_t, n_q = sum(not c for c in all_c), len(all_c), len(results)
        w("-" * 72)
        w(f"  Steered wrong:  {n_w}/{n_t}  ({n_w/n_t:.0%})")
        w(f"  Baseline wrong: {bl_w}/{n_q}  ({bl_w/n_q:.0%})")
        w(f"  Δ wrong-rate:  {n_w/n_t - bl_w/n_q:+.0%}")

    write_group("IN-DISTRIBUTION (questions seen during mask construction)",
                [r for r in st_results if r[4]])
    write_group("OUT-OF-DISTRIBUTION (held-out questions)",
                [r for r in st_results if not r[4]])

    all_c = [c for _, _, _, cs, _ in st_results for c in cs]
    n_w   = sum(not c for c in all_c)
    w(); w("=" * 72); w("OVERALL SUMMARY")
    w(f"  Steered wrong:  {n_w}/{len(all_c)}  ({n_w/len(all_c):.0%})")
    w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")
    w(f"  Δ wrong-rate:  {n_w/len(all_c) - bl_wrong/len(FACTUAL_QA):+.0%}")

    out_file.write_text("\n".join(lines))
    print(f"    Saved samples → {out_file}")
    return n_w / len(all_c), bl_wrong / len(FACTUAL_QA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args   = parser.parse_args()
    device = args.device

    print(f"Loading model on {device} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device)
    model.eval()

    saved     = torch.load(SCORES_PT, map_location="cpu")
    wt_tokens = load_wikitext_tokens(tok)

    cfg = LieNeuroConfig(device=device)
    ln  = LieNeuro(model, tok, cfg)

    print("Evaluating baseline …")
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp = sample(model, tok, q, 1, device)[0]
        bl_results.append((q, a, resp, _is_correct(resp, a)))
    bl_wrong_rate = sum(not r[3] for r in bl_results) / len(FACTUAL_QA)
    print(f"Baseline wrong: {bl_wrong_rate:.0%}")

    for alpha, need_ppl, out_file in CONFIGS:
        print(f"\n{'='*60}")
        print(f"ρ={RHO}  α={alpha}")

        if out_file is None:
            print("  Skipping (samples + PPL already exist)")
            continue

        if out_file.exists():
            print(f"  Samples already exist: {out_file}")
            st_wrong_rate = bl_wrong_rate
        else:
            st_wrong_rate, _ = generate_samples_file(
                model, tok, ln, saved, RHO, alpha, out_file, bl_results, device)

        if need_ppl:
            print("  Computing WikiText-2 PPL …")
            cfg2 = LieNeuroConfig(importance_threshold=RHO, alpha=alpha, device=device)
            ln.config   = cfg2
            ln._S_lie   = saved["S_lie"]
            ln._S_truth = saved["S_truth_instruct"]
            ln.build_mask(mode=MODE)
            ln.apply_steering()
            ppl = sliding_window_ppl(model, wt_tokens, device)
            ln.restore_original()
            dw = st_wrong_rate - bl_wrong_rate
            append_ppl_row(RHO, alpha, dw, ppl)

    print("\nAll done.")


if __name__ == "__main__":
    main()
