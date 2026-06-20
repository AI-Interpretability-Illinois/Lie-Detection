#!/usr/bin/env python3
"""Generate samples + WikiText-2 PPL for LieNeuro DID score_diff ρ=0.2 α=4."""

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
OUT_FILE     = Path("outputs/lieneuro_did_samples_rho0.2_a4.txt")
RHO, ALPHA, MODE, TEMP, N_REPS = 0.2, 4.0, "score_diff", 1.0, 3
WT_WINDOW, WT_STRIDE, WT_MAX_TOKS = 512, 256, 8192

_INSTRUCT_QS = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
                for p in DIRECT_TRUTH_INSTRUCT}
def is_indist(q): return q.strip().lower() in _INSTRUCT_QS

def sample(model, tok, q, n, device):
    prompt = _build_prompt(q, tok)
    return [generate_answer(model, tok, prompt, max_new_tokens=40,
                            temperature=TEMP, device=device) for _ in range(n)]

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = args.device

    print(f"Loading model on {device} …")
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.float16, device_map=device)
    model.eval()

    saved     = torch.load(SCORES_PT, map_location="cpu")
    ds        = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    wt_text   = "\n\n".join(t for t in ds["text"] if t.strip())
    wt_tokens = tok.encode(wt_text, add_special_tokens=False)[:WT_MAX_TOKS]

    cfg = LieNeuroConfig(importance_threshold=RHO, alpha=ALPHA, device=device)
    ln  = LieNeuro(model, tok, cfg)
    ln._S_lie   = saved["S_lie"]
    ln._S_truth = saved["S_truth_instruct"]
    ln.build_mask(mode=MODE)

    # baseline
    print("Evaluating baseline …")
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp = sample(model, tok, q, 1, device)[0]
        bl_results.append((q, a, resp, _is_correct(resp, a)))
    bl_wrong = sum(not r[3] for r in bl_results)
    print(f"Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}")

    lines = []
    def w(s=""): lines.append(s)
    w("LieNeuro DID — Sample Outputs")
    w(f"Mode: {MODE}  |  ρ={RHO}  |  α={ALPHA}  |  temperature={TEMP}")
    w(f"Model: Llama-3.1-8B-Instruct  |  {N_REPS} samples per question")
    w("=" * 72)
    w(); w("BASELINE (no steering)"); w("-" * 72)
    for q, a, resp, correct in bl_results:
        w(f"  {'✓' if correct else '✗'}  Q: {q}")
        w(f"       A: {resp}  [expected: {a}]")
    w(); w(f"  Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong/len(FACTUAL_QA):.0%})")

    ln.apply_steering()
    st_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resps    = sample(model, tok, q, N_REPS, device)
        corrects = [_is_correct(r, a) for r in resps]
        st_results.append((q, a, resps, corrects, is_indist(q)))

    def write_group(label, results):
        w(); w()
        w(f"STEERED — {label}  mode={MODE}  ρ={RHO}  α={ALPHA}  ({N_REPS} samples/question)")
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
    OUT_FILE.write_text("\n".join(lines))
    print(f"Saved samples → {OUT_FILE}")

    # PPL
    print("Computing WikiText-2 PPL …")
    ppl = sliding_window_ppl(model, wt_tokens, device)
    ln.restore_original()
    dw = n_w / len(all_c) - bl_wrong / len(FACTUAL_QA)
    label = f"LieNeuro-DID score_diff ρ={RHO:.3f} α={ALPHA:.1f}"
    with open(WIKITEXT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "variant", "method", "layer", "coeff", "rho", "alpha",
            "delta_wrong", "wikitext_ppl"])
        writer.writerow({"variant": label, "method": "weight_steering_did",
                         "layer": "", "coeff": "", "rho": RHO, "alpha": ALPHA,
                         "delta_wrong": dw, "wikitext_ppl": ppl})
    print(f"PPL={ppl:.4f}  Δwrong={dw:+.2%}  → appended to {WIKITEXT_CSV}")
    print("Done.")

if __name__ == "__main__":
    main()
