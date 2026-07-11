#!/usr/bin/env python3
"""
Steer Llama-3.1-8B using only the top-10 layers by per-layer attention DID score.
Within those layers, apply the standard top-ρ DID mask (attn + MLP).
All other layers: mask = all-False (no steering).
Sweeps alpha in {1, 2, 4, 6, 8}.
"""
import csv, json, math, re, sys
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/srv/local/projects/aebrahim/lie_detection")
from run_lieneuro_did_generalization import (
    sliding_window_ppl, safe_generate, sample, _mean_embed, _cos_sim,
    is_indist, TEMP, N_REPS, WT_MAX_TOKS,
)
from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.evaluate import FACTUAL_QA, _build_prompt, _is_correct

DEVICE    = "cuda:0"
MODEL_ID  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CACHE_DIR = "/scratch/aebrahim/hf_cache"
SCORES_PT = "/srv/local/projects/aebrahim/lie_detection/outputs/lieneuro_did_scores.pt"
OUT_DIR   = Path("/srv/local/projects/aebrahim/lie_detection/outputs/generalization/llama_8b")

RHO      = 0.04
ALPHAS   = [1.0, 2.0, 4.0, 6.0, 8.0]
TOP_N    = 10

LAYER_RE = re.compile(r"layers\.(\d+)\.")
ATTN_KW  = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KW   = ("gate_proj", "up_proj", "down_proj")

# ── Load model ────────────────────────────────────────────────────────────────
print(f"Loading {MODEL_ID} …")
tok = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, dtype=torch.float16, device_map=DEVICE, cache_dir=CACHE_DIR)
model.eval()
print("Model loaded.\n")

# ── WikiText-2 ────────────────────────────────────────────────────────────────
print("Loading WikiText-2 …")
ds      = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", cache_dir=CACHE_DIR)
wt_text = "\n\n".join(t for t in ds["text"] if t.strip())
wt_toks = tok.encode(wt_text, add_special_tokens=False)[:WT_MAX_TOKS]
print(f"  {len(wt_toks)} tokens\n")

# ── Baseline ──────────────────────────────────────────────────────────────────
baseline_ppl  = sliding_window_ppl(model, wt_toks, DEVICE)
baseline_json = OUT_DIR / "baseline_results.json"
if baseline_json.exists():
    bl_data    = json.loads(baseline_json.read_text())
    bl_results = [(d["q"], d["a"], d["resp"], d["correct"]) for d in bl_data]
else:
    bl_results = []
    for qa in FACTUAL_QA:
        q, a = qa["question"], qa["answer"]
        resp = safe_generate(model, tok, _build_prompt(q, tok), 40, TEMP, DEVICE)
        bl_results.append((q, a, resp, _is_correct(resp, a)))
    baseline_json.write_text(json.dumps([
        {"q": q, "a": a, "resp": r, "correct": c} for q, a, r, c in bl_results]))
bl_wrong_rate = sum(not r[3] for r in bl_results) / len(FACTUAL_QA)
print(f"Baseline PPL={baseline_ppl:.2f}  wrong={bl_wrong_rate:.0%}\n")

# ── Load DID scores ───────────────────────────────────────────────────────────
print("Loading DID scores …")
saved   = torch.load(SCORES_PT, map_location="cpu", weights_only=False)
S_lie   = saved["S_lie"]
S_truth = saved["S_truth_instruct"]

# Rank layers by mean p99 attention DID score across q/k/v/o projections
layer_attn_p99 = defaultdict(list)
for name in S_lie:
    if not any(k in name for k in ATTN_KW):
        continue
    m = LAYER_RE.search(name)
    if not m:
        continue
    layer = int(m.group(1))
    diff  = (S_lie[name].float() - S_truth[name].float()).view(-1)
    layer_attn_p99[layer].append(float(torch.quantile(diff, 0.99)))

layer_score = {l: sum(v) / len(v) for l, v in layer_attn_p99.items()}
top10_layers = set(sorted(layer_score, key=layer_score.get, reverse=True)[:TOP_N])

print(f"Top-{TOP_N} layers by mean attention DID p99:")
for l in sorted(top10_layers):
    print(f"  Layer {l:2d}  mean_attn_p99={layer_score[l]:.6f}")
print()

# Build the mask: top-ρ per-layer DID scores, restricted to top-10 layers
print(f"Building mask (ρ={RHO}, top-{TOP_N} layers, attn + MLP) …")
mask_dict = {}   # name → bool tensor
for name in S_lie:
    m = LAYER_RE.search(name)
    if not m:
        continue
    layer = int(m.group(1))
    is_attn = any(k in name for k in ATTN_KW)
    is_mlp  = any(k in name for k in MLP_KW)
    if not (is_attn or is_mlp):
        continue

    shape  = S_lie[name].shape
    n      = S_lie[name].numel()
    if layer not in top10_layers:
        mask_dict[name] = torch.zeros(shape, dtype=torch.bool)
        continue

    diff = (S_lie[name].float() - S_truth[name].float()).view(-1)
    k    = max(1, int(RHO * n))
    _, top_idx = diff.topk(k)
    mask = torch.zeros(n, dtype=torch.bool)
    mask[top_idx] = True
    mask_dict[name] = mask.view(shape)

total_sel   = sum(m.sum().item() for m in mask_dict.values())
total_param = sum(m.numel()      for m in mask_dict.values())
print(f"Selected {total_sel:,} / {total_param:,} params ({100*total_sel/total_param:.2f}%)\n")

del saved, S_lie, S_truth

# ── LieNeuro for apply/restore ────────────────────────────────────────────────
cfg = LieNeuroConfig(importance_threshold=RHO, alpha=1.0,
                     save_original_weights=True, device=DEVICE)
ln  = LieNeuro(model, tok, cfg)
ln._mask = mask_dict

def apply_alpha(alpha: float):
    ln._original_weights = {}
    for name, mask in ln._mask.items():
        mod  = ln._get_module(name)
        orig = mod.weight.data.clone()
        ln._original_weights[name] = orig
        mod.weight.data.mul_(1.0 + alpha * mask.to(mod.weight.device).float())

def restore():
    for name, orig in ln._original_weights.items():
        ln._get_module(name).weight.data.copy_(orig)
    ln._original_weights = {}

# ── Sweep alphas ──────────────────────────────────────────────────────────────
results_rows = []
for alpha in ALPHAS:
    print(f"\n{'─'*60}")
    print(f"ρ={RHO}  α={alpha}  top-{TOP_N} layers")

    tag          = f"top10attn_rho{RHO}_a{alpha}"
    samples_path = OUT_DIR / f"samples_{tag}.txt"
    metrics_path = samples_path.with_suffix(".json")

    extra = {}
    if samples_path.exists() and metrics_path.exists():
        print("  Samples exist, loading metrics …")
        extra         = json.loads(metrics_path.read_text())
        st_wrong_rate = extra.get("wrong_rate", float("nan"))
    else:
        print("  Generating samples …")
        apply_alpha(alpha)
        try:
            st_results = []
            for qa in FACTUAL_QA:
                q, a  = qa["question"], qa["answer"]
                resps = sample(model, tok, q, N_REPS, DEVICE, max_new=40)
                corrects = [_is_correct(r, a) for r in resps]
                st_results.append((q, a, resps, corrects, is_indist(q)))

            all_c = [c for _, _, _, cs, _ in st_results for c in cs]
            n_w   = sum(not c for c in all_c)
            st_wrong_rate = n_w / len(all_c)

            all_resps_wrong = [(r, a) for _, a, resps, corrects, _ in st_results
                               for r, c in zip(resps, corrects) if not c]
            avg_wrong_len = (sum(len(r.split()) for r, _ in all_resps_wrong) / len(all_resps_wrong)
                             if all_resps_wrong else float("nan"))
            cos_sims = [_cos_sim(_mean_embed(r, model, tok, DEVICE),
                                 _mean_embed(a, model, tok, DEVICE))
                        for r, a in all_resps_wrong]
            avg_cos_sim = (sum(s for s in cos_sims if math.isfinite(s)) /
                           sum(1 for s in cos_sims if math.isfinite(s))
                           if any(math.isfinite(s) for s in cos_sims) else float("nan"))

            lines = [f"Top-{TOP_N} attention layers  ρ={RHO}  α={alpha}",
                     f"Layers: {sorted(top10_layers)}", "=" * 72]
            for q, a, resps, corrects, _ in st_results:
                marks = ["✓" if c else "✗" for c in corrects]
                lines.append(f"Q: {q}  [expected: {a}]")
                for i, (r, mk) in enumerate(zip(resps, marks)):
                    lines.append(f"  Sample {i+1}: {mk}  {r}")
            bl_wrong = sum(not r[3] for r in bl_results)
            lines += ["=" * 72,
                      f"Steered wrong:  {n_w}/{len(all_c)}  ({st_wrong_rate:.0%})",
                      f"Baseline wrong: {bl_wrong}/{len(FACTUAL_QA)}  ({bl_wrong_rate:.0%})",
                      f"Avg wrong-ans length: {avg_wrong_len:.1f} words",
                      f"Avg cosine sim: {avg_cos_sim:.4f}"]
            samples_path.write_text("\n".join(lines))
            extra = {"wrong_rate": st_wrong_rate, "avg_wrong_len": avg_wrong_len,
                     "avg_cos_sim": avg_cos_sim}
            metrics_path.write_text(json.dumps(extra, indent=2))
        except Exception as exc:
            import traceback; traceback.print_exc()
            st_wrong_rate = float("nan")
        finally:
            restore()

    # PPL
    print("  Computing PPL …")
    apply_alpha(alpha)
    ppl = sliding_window_ppl(model, wt_toks, DEVICE)
    restore()

    delta_wrong = (st_wrong_rate - bl_wrong_rate) if math.isfinite(st_wrong_rate) else float("nan")
    results_rows.append({
        "alpha": alpha, "wrong_rate": st_wrong_rate, "delta_wrong": delta_wrong,
        "wikitext_ppl": ppl, "baseline_ppl": baseline_ppl,
        "baseline_wrong_rate": bl_wrong_rate,
        "avg_wrong_len": extra.get("avg_wrong_len", float("nan")),
        "avg_cos_sim":   extra.get("avg_cos_sim",   float("nan")),
    })
    dw = f"{delta_wrong:+.0%}" if math.isfinite(delta_wrong) else "crash"
    print(f"  Δwrong={dw}  PPL={ppl:.2f}")

# ── Summary ───────────────────────────────────────────────────────────────────
csv_path = OUT_DIR / "results_top10_layer.csv"
fields = ["alpha", "wrong_rate", "delta_wrong", "wikitext_ppl",
          "baseline_ppl", "baseline_wrong_rate", "avg_wrong_len", "avg_cos_sim"]
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(results_rows)

print(f"\n{'═'*65}")
print(f"  Baseline wrong={bl_wrong_rate:.0%}  PPL={baseline_ppl:.2f}")
print(f"  Top-{TOP_N} layers: {sorted(top10_layers)}")
print(f"{'─'*65}")
print(f"  {'α':>5}  {'Wrong%':>7}  {'Δwrong':>8}  {'PPL':>12}  {'AvgLen':>7}  {'CosSim':>7}")
print("  " + "─" * 55)
for r in results_rows:
    wr = f"{r['wrong_rate']:.0%}" if math.isfinite(r['wrong_rate']) else "crash"
    dw = f"{r['delta_wrong']:+.0%}" if math.isfinite(r['delta_wrong']) else "crash"
    al = f"{r['avg_wrong_len']:.1f}" if math.isfinite(r.get('avg_wrong_len', float('nan'))) else "nan"
    cs = f"{r['avg_cos_sim']:.3f}" if math.isfinite(r.get('avg_cos_sim', float('nan'))) else "nan"
    print(f"  {r['alpha']:>5}  {wr:>7}  {dw:>8}  {r['wikitext_ppl']:>12.1f}  {al:>7}  {cs:>7}")
print(f"{'═'*65}")
print(f"Results → {csv_path}")
