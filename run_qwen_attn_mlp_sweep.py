#!/usr/bin/env python3
"""
Qwen attn/mlp alpha sweep — loads model and scores once, runs all 14 configs.
Writes results to outputs/generalization/qwen2p5_7b/results_attn_mlp.csv
"""
import csv, json, math, re as _re, sys
from pathlib import Path
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/local/projects/aebrahim/lie_detection")
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, generate_answer, _is_correct
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT
from run_lieneuro_did_generalization import (
    sliding_window_ppl, write_samples_file, safe_generate,
    is_indist, SanitizeLogits, _LOGITS_PROC, MODE, TEMP, N_REPS, WT_WINDOW, WT_STRIDE, WT_MAX_TOKS
)

DEVICE    = "cuda:0"
MODEL_ID  = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = "/scratch/aebrahim/hf_cache"
OUT_DIR   = Path("/srv/local/projects/aebrahim/lie_detection/outputs/generalization/qwen2p5_7b")
SCORES_PT = OUT_DIR / "did_scores_raw.pt"

# (rho, alpha_attn, alpha_mlp)  — rho_by_type=True for all
ATTN_MLP_CONFIGS = [
    (0.01, 0.5, 0.0), (0.01, 0.5, 0.1), (0.01, 0.5, 0.3),
    (0.01, 1.0, 0.0), (0.01, 1.0, 0.1), (0.01, 1.0, 0.3),
    (0.04, 1.0, 4.0), (0.04, 4.0, 1.0),
    (0.05, 0.5, 0.0), (0.05, 0.5, 0.1), (0.05, 0.5, 0.3),
    (0.05, 1.0, 0.0), (0.05, 1.0, 0.1), (0.05, 1.0, 0.3),
]

def _alpha_tag(rho, attn, mlp):
    def fmt(v): return str(int(v)) if v == int(v) else str(v)
    return f"rho{rho}_a1_attn{fmt(attn)}_mlp{fmt(mlp)}"


print(f"Loading {MODEL_ID} …")
tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map=DEVICE, cache_dir=CACHE_DIR)
model.eval()
print("Model loaded.\n")

print("Loading WikiText-2 …")
ds       = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=CACHE_DIR)
wt_text  = "\n\n".join(t for t in ds["text"] if t.strip())
wt_toks  = tok.encode(wt_text, add_special_tokens=False)[:WT_MAX_TOKS]
print(f"WikiText-2 tokens: {len(wt_toks)}\n")

print("Computing baseline PPL …")
cfg = LieNeuroConfig(importance_threshold=0.04, alpha=4.0, device=DEVICE)
ln  = LieNeuro(model, tok, cfg)
baseline_ppl = sliding_window_ppl(model, wt_toks, DEVICE)
print(f"Baseline PPL = {baseline_ppl:.4f}\n")

baseline_json = OUT_DIR / "baseline_results.json"
if baseline_json.exists():
    print("Loading cached baseline accuracy …")
    bl_data    = json.loads(baseline_json.read_text())
    bl_results = [(d["q"], d["a"], d["resp"], d["correct"]) for d in bl_data]
else:
    print("Evaluating baseline accuracy …")
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp = safe_generate(model, tok, _build_prompt(q, tok), 40, TEMP, DEVICE)
        bl_results.append((q, a, resp, _is_correct(resp, a)))
    baseline_json.write_text(json.dumps([
        {"q": q, "a": a, "resp": r, "correct": c} for q, a, r, c in bl_results]))
bl_wrong_rate = sum(not r[3] for r in bl_results) / len(FACTUAL_QA)
print(f"Baseline wrong rate: {bl_wrong_rate:.0%}\n")

print("Loading DID scores …")
saved   = torch.load(str(SCORES_PT), map_location="cpu", weights_only=False)
ln._S_lie   = saved["S_lie"]
ln._S_truth = saved["S_truth"]
print("Scores loaded.\n")

results_rows = []
for rho, alpha_attn, alpha_mlp in ATTN_MLP_CONFIGS:
    tag          = _alpha_tag(rho, alpha_attn, alpha_mlp)
    samples_path = OUT_DIR / f"samples_{tag}.txt"
    metrics_path = samples_path.with_suffix(".json")

    ln.config.importance_threshold = rho
    ln.config.alpha                = 1.0
    ln.config.alpha_attn           = alpha_attn
    ln.config.alpha_mlp            = alpha_mlp
    ln.config.rho_by_type          = True

    print(f"\n{'─'*60}")
    print(f"ρ={rho}  α_attn={alpha_attn}  α_mlp={alpha_mlp}  rho_by_type=True")

    extra = {}
    if samples_path.exists() and metrics_path.exists():
        print("  Samples exist, loading metrics …")
        extra         = json.loads(metrics_path.read_text())
        st_wrong_rate = extra.get("wrong_rate", float("nan"))
    else:
        print("  Generating samples …")
        try:
            st_wrong_rate, _, avg_wrong_len, avg_cos_sim = write_samples_file(
                samples_path, model, tok, ln, rho, alpha_attn, bl_results, DEVICE)
            extra = {"wrong_rate": st_wrong_rate, "avg_wrong_len": avg_wrong_len,
                     "avg_cos_sim": avg_cos_sim}
        except Exception as exc:
            print(f"  WARNING: sample generation failed — {exc}")
            st_wrong_rate = float("nan")

    print("  Computing PPL …")
    ln.build_mask(mode=MODE)
    ln.apply_steering()
    ppl = sliding_window_ppl(model, wt_toks, DEVICE)
    ln.restore_original()

    delta_wrong = (st_wrong_rate - bl_wrong_rate) if math.isfinite(st_wrong_rate) else float("nan")
    results_rows.append({
        "model": MODEL_ID, "model_short": "qwen2p5_7b",
        "rho": rho, "alpha_attn": alpha_attn, "alpha_mlp": alpha_mlp,
        "wrong_rate": st_wrong_rate, "delta_wrong": delta_wrong,
        "wikitext_ppl": ppl, "baseline_ppl": baseline_ppl,
        "baseline_wrong_rate": bl_wrong_rate,
        "avg_wrong_len": extra.get("avg_wrong_len", float("nan")),
        "avg_cos_sim":   extra.get("avg_cos_sim",   float("nan")),
    })
    dw = f"{delta_wrong:+.0%}" if math.isfinite(delta_wrong) else "failed"
    print(f"  Δwrong={dw}  PPL={ppl:.2f}")

csv_path = OUT_DIR / "results_attn_mlp.csv"
fields = ["model", "model_short", "rho", "alpha_attn", "alpha_mlp",
          "wrong_rate", "delta_wrong", "wikitext_ppl",
          "baseline_ppl", "baseline_wrong_rate", "avg_wrong_len", "avg_cos_sim"]
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(results_rows)
print(f"\nResults saved → {csv_path}")

print(f"\n{'─'*70}")
print(f"  Baseline wrong: {bl_wrong_rate:.0%}  |  Baseline PPL: {baseline_ppl:.2f}")
print(f"{'─'*70}")
print(f"  {'ρ':>5} {'α_attn':>7} {'α_mlp':>7}  {'Wrong%':>7}  {'Δwrong':>8}  {'PPL':>10}")
print("  " + "─" * 55)
for r in results_rows:
    wr = f"{r['wrong_rate']:.0%}" if math.isfinite(r['wrong_rate']) else "crash"
    dw = f"{r['delta_wrong']:+.0%}" if math.isfinite(r['delta_wrong']) else "crash"
    print(f"  {r['rho']:>5}  {r['alpha_attn']:>6}  {r['alpha_mlp']:>6}  {wr:>7}  {dw:>8}  {r['wikitext_ppl']:>10.1f}")
print(f"{'─'*70}\nDone.")
