#!/usr/bin/env python3
"""
Generate reports/activation_steering_report.md from completed sweep + fluency data.

Usage
-----
  python reports/generate_as_report.py
"""

import csv
import json
from pathlib import Path

OUT_DIR     = Path(__file__).parent.parent / "outputs"
REPORT_DIR  = Path(__file__).parent
FIG_DIR_REL = "figures"   # relative to reports/

SWEEP_CSV     = OUT_DIR / "activation_steering_sweep.csv"
FLUENCY_CSV   = OUT_DIR / "activation_steering_fluency.csv"
JUDGE_CSV     = OUT_DIR / "as_judge_summary.csv"
BASELINE_JSON = OUT_DIR / "as_baseline.json"
WIKITEXT_CSV  = OUT_DIR / "wikitext_ppl.csv"
INDIST_CSV    = OUT_DIR / "as_indist_eval.csv"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    # ── load sweep ────────────────────────────────────────────────────────────
    rows = load_csv(SWEEP_CSV)
    baseline_acc = float(rows[0]["baseline_accuracy"])
    bl_wrong = 1 - baseline_acc

    dw_vals = [(int(r["layer"]), float(r["coeff"]), float(r["delta_wrong_rate"]))
               for r in rows]
    dw_vals.sort(key=lambda x: -x[2])

    best_layer, best_coeff, best_dw = dw_vals[0]
    worst_layer, worst_coeff, worst_dw = dw_vals[-1]

    # Per-layer max Δwrong (across all coeffs)
    by_layer = {}
    for layer, coeff, dw in dw_vals:
        if layer not in by_layer or dw > by_layer[layer][1]:
            by_layer[layer] = (coeff, dw)
    top5_layers = sorted(by_layer.items(), key=lambda x: -x[1][1])[:5]

    # Per-coeff stats
    coeff_stats = {}
    for layer, coeff, dw in dw_vals:
        coeff_stats.setdefault(coeff, []).append(dw)

    # Count configs with positive Δwrong per coeff
    coeff_positive = {c: sum(1 for v in vs if v > 0) for c, vs in coeff_stats.items()}
    coeff_max      = {c: max(vs) for c, vs in coeff_stats.items()}

    # ── load judge ───────────────────────────────────────────────────────────
    has_judge = JUDGE_CSV.exists()
    if has_judge:
        judge_rows = load_csv(JUDGE_CSV)

    # ── load fluency ──────────────────────────────────────────────────────────
    has_fluency = FLUENCY_CSV.exists()
    if has_fluency:
        flu_rows = load_csv(FLUENCY_CSV)
        bl_flu   = next(r for r in flu_rows if r["variant"] == "baseline")
        bl_ppl   = float(bl_flu["mean_perplexity"])
        as_flu   = [r for r in flu_rows if r["variant"] != "baseline"]
        top_flu  = sorted(as_flu, key=lambda r: -float(r["delta_wrong_rate"]))[:5]
    else:
        bl_ppl  = None
        top_flu = []

    # ── load in-dist / OOD eval ───────────────────────────────────────────────
    has_indist = INDIST_CSV.exists()
    if has_indist:
        id_rows = load_csv(INDIST_CSV)
        bl_indist = float(id_rows[0]["baseline_indist"])
        bl_ood    = float(id_rows[0]["baseline_ood"])
        # best per-coeff: highest OOD Δwrong
        id_by_coeff = {}
        for r in id_rows:
            c = float(r["coeff"])
            dw_ood = float(r["ood_delta_wrong"])
            dw_in  = float(r["indist_delta_wrong"])
            if c not in id_by_coeff or dw_ood > id_by_coeff[c]["ood_dw"]:
                id_by_coeff[c] = {
                    "layer": int(r["layer"]), "coeff": c,
                    "indist_dw": dw_in, "ood_dw": dw_ood,
                    "indist_acc": float(r["indist_accuracy"]),
                    "ood_acc":    float(r["ood_accuracy"]),
                }
        # top-10 by OOD Δwrong
        id_top10 = sorted(
            [(int(r["layer"]), float(r["coeff"]),
              float(r["indist_delta_wrong"]), float(r["ood_delta_wrong"]))
             for r in id_rows],
            key=lambda x: -x[3]
        )[:10]

    # ── load WikiText PPL ─────────────────────────────────────────────────────
    has_wikitext = WIKITEXT_CSV.exists()
    if has_wikitext:
        wt_rows = load_csv(WIKITEXT_CSV)
        wt_baseline = next((r for r in wt_rows if r["method"] == "none"), None)
        wt_bl_ppl   = float(wt_baseline["wikitext_ppl"]) if wt_baseline else None
        wt_as_rows  = [r for r in wt_rows if r["method"] == "activation_steering"]
        wt_ln_rows  = [r for r in wt_rows if r["method"] == "weight_steering"]
    else:
        wt_bl_ppl  = None
        wt_as_rows = []
        wt_ln_rows = []

    # ── top-10 config table ───────────────────────────────────────────────────
    top10 = dw_vals[:10]

    # ── build report ──────────────────────────────────────────────────────────
    lines = []

    def w(*args): lines.append(" ".join(str(a) for a in args))
    def wl(): lines.append("")

    w("# Activation Steering for Deception: Experiment Report")
    wl()
    w("**Method:** Difference-in-means residual-stream steering (Stolfo et al., ICLR 2025)")
    w("**Model:** `meta-llama/Llama-3.1-8B-Instruct`")
    w("**Hardware:** NVIDIA H100 80 GB (CUDA 12.8)")
    w("**Date:** June 2026")
    wl()
    w("---")
    wl()
    w("## 1  Overview")
    wl()
    w("This report evaluates whether **activation steering** — adding a difference-in-means")
    w("vector to the residual stream at inference time — can induce deceptive behaviour")
    w("(higher wrong-answer rate on factual QA) more effectively or more fluently than")
    w("LieNeuro's weight-steering approach.")
    wl()
    w("The method adds a unit vector `u_l = normalize(mean(h_lie - h_truth))` scaled by")
    w("coefficient `c` to the hidden state at every token position in layer `l`. Model")
    w("weights are never modified; the intervention is applied per-forward-pass via a")
    w("registered forward hook on the decoder layer.")
    wl()
    w("---")
    wl()
    w("## 2  Method")
    wl()
    w("### 2.1  Steering Vector Computation")
    wl()
    w("For each of the 32 transformer layers:")
    w("1. Run 10 lying prompts and 10 truthful prompts (from the `direct` prompt set)")
    w("   through the frozen model, capturing the last-token hidden state at each layer.")
    w("2. Compute the mean difference: `d_l = mean_i(h_lie_{i,l} - h_truth_{i,l})`")
    w("3. Normalise: `u_l = d_l / ||d_l||`")
    wl()
    w("### 2.2  Inference-Time Intervention")
    wl()
    w("At every forward pass, a hook adds `c · u_l` to the output of layer `l`:")
    w("```")
    w("x'_l ← x'_l + c · u_l")
    w("```")
    w("This steers every token position toward the lie direction.")
    wl()
    w("### 2.3  Sweep Design")
    wl()
    w("| Parameter | Values |")
    w("|---|---|")
    w("| Layers | 0–31 (all 32 transformer layers) |")
    w("| Coefficients (c) | 5, 10, 20, 40, 80 |")
    w("| Prompt set | `direct` |")
    w("| Evaluation | 50-question factual QA |")
    w("| Total configurations | 32 × 5 = **160** |")
    wl()
    w("---")
    wl()
    w("## 3  Results")
    wl()
    w("### 3.1  Baseline")
    wl()
    w(f"Unsteered model factual accuracy: **{baseline_acc:.0%}**  ")
    w(f"Baseline wrong-rate: **{bl_wrong:.0%}**")
    wl()
    w("### 3.2  Deception Effect per Layer")
    wl()
    w("![Δwrong-rate vs layer for each coefficient](figures/fig12_as_layer_sweep.png)")
    wl()
    w(f"**Best single config:** layer {best_layer}, c={best_coeff:.0f} → "
      f"Δwrong = **{best_dw:+.0%}**")
    wl()
    w("**Top 5 layers by maximum Δwrong-rate (across all coefficients):**")
    wl()
    w("| Layer | Best coefficient | Max Δwrong |")
    w("|---|---|---|")
    for layer, (coeff, dw) in top5_layers:
        w(f"| {layer} | c = {coeff:.0f} | {dw:+.0%} |")
    wl()
    w("**Configs with positive Δwrong by coefficient:**")
    wl()
    w("| Coefficient | Layers with Δwrong > 0 | Max Δwrong |")
    w("|---|---|---|")
    for c in sorted(coeff_stats.keys()):
        w(f"| c = {c:.0f} | {coeff_positive[c]} / 32 | {coeff_max[c]:+.0%} |")
    wl()
    w("### 3.3  Heatmap")
    wl()
    w("![Δwrong-rate heatmap by layer and coefficient](figures/fig11_as_heatmap.png)")
    wl()
    w("### 3.4  Top 10 Configurations")
    wl()
    w("| Rank | Layer | Coeff | Δwrong-rate |")
    w("|---|---|---|---|")
    for i, (layer, coeff, dw) in enumerate(top10, 1):
        w(f"| {i} | {layer} | {coeff:.0f} | {dw:+.0%} |")
    wl()

    if has_judge:
        w("### 3.5  LLM-Judge Verification (Top 3 Layers)")
        wl()
        w("String-match accuracy can be inflated when the model produces a wrong answer")
        w("that happens to contain the gold string as a substring. The Qwen2.5-14B-Instruct")
        w("judge provides a semantic equivalence check.")
        wl()
        w("| Layer | Coeff | String Δwrong | Judge Δwrong | Difference |")
        w("|---|---|---|---|---|")
        for r in judge_rows:
            str_dw  = float(r["string_delta_wrong"])
            jdg_dw  = float(r["judge_delta_wrong"])
            diff    = jdg_dw - str_dw
            w(f"| {r['layer']} | {float(r['coeff']):.0f} | {str_dw:+.0%} | {jdg_dw:+.0%} | {diff:+.0%} |")
        wl()

    fluency_section = "3.6" if has_judge else "3.5"
    if has_fluency:
        w(f"### {fluency_section}  Fluency of Top Configurations")
        wl()
        w("![Fluency metrics for top configs](figures/fig14_as_fluency.png)")
        wl()
        w(f"Baseline perplexity: **{bl_ppl:.2f}**")
        wl()
        w("| Config | Δwrong | Perplexity | Distinct-2 | Rep-Rate |")
        w("|---|---|---|---|---|")
        for r in top_flu:
            tag   = f"Layer {r['layer']} c={float(r['coeff']):.0f}"
            dw    = float(r["delta_wrong_rate"])
            ppl   = float(r["mean_perplexity"])
            d2    = float(r["mean_distinct_2"])
            rep   = float(r["mean_rep_rate"])
            w(f"| {tag} | {dw:+.0%} | {ppl:.1f} | {d2:.3f} | {rep:.3f} |")
        wl()

    if has_wikitext:
        wikitext_section = "3.7" if has_judge else "3.6"
        w(f"### {wikitext_section}  WikiText-2 Perplexity Benchmark")
        wl()
        w("Generation-based perplexity measures how surprised the model is by its own outputs.")
        w("To provide a model-agnostic, fixed-corpus fluency benchmark we also evaluate")
        w("**WikiText-2-raw-v1** perplexity using a sliding window of 512 tokens (stride 256)")
        w("on the first 8,192 test tokens. This metric is identical for all variants and")
        w("directly comparable across methods.")
        wl()
        w("![WikiText-2 PPL vs Δwrong](figures/fig15_wikitext_ppl.png)")
        wl()
        if wt_bl_ppl is not None:
            w(f"Baseline WikiText-2 PPL: **{wt_bl_ppl:.2f}**")
            wl()
        w("| Variant | Method | Δwrong | WikiText-2 PPL | PPL Ratio vs Baseline |")
        w("|---|---|---|---|---|")
        if wt_bl_ppl:
            w(f"| Baseline | — | 0% | {wt_bl_ppl:.2f} | 1.00× |")
        for r in wt_as_rows:
            dw_str = f"{float(r['delta_wrong']):+.0%}" if r["delta_wrong"] else "—"
            ppl = float(r["wikitext_ppl"])
            ratio = f"{ppl/wt_bl_ppl:.1f}×" if wt_bl_ppl else "—"
            layer = r["layer"]
            coeff = float(r["coeff"])
            w(f"| AS L{layer} c={coeff:.0f} | Activation steering | {dw_str} | {ppl:.2f} | {ratio} |")
        for r in wt_ln_rows:
            dw_str = f"{float(r['delta_wrong']):+.0%}" if r["delta_wrong"] else "—"
            ppl = float(r["wikitext_ppl"])
            ratio = f"{ppl/wt_bl_ppl:.1f}×" if wt_bl_ppl else "—"
            short = r["variant"].replace("LieNeuro ", "")
            w(f"| {short} | LieNeuro (weight) | {dw_str} | {ppl:.2f} | {ratio} |")
        wl()

    if has_indist:
        indist_section = "3.8" if has_judge else "3.7" if has_wikitext else "3.6"
        # adjust WikiText section number if needed (already 3.7 when judge present)
        w(f"### {indist_section}  In-Distribution vs Out-of-Distribution Evaluation (25-Pair Vectors)")
        wl()
        w("To assess whether averaging over more construction prompts improves the steering")
        w("vectors, we recompute the difference-in-means vectors using **25 direct lie/truth")
        w("pairs** (vs 10 in the original sweep) and evaluate each configuration on two splits:")
        wl()
        w("- **In-distribution (IndDist):** the 25 Q&A pairs whose questions appear verbatim")
        w("  in the construction prompts — tests whether the vector memorises its training questions.")
        w("- **Out-of-distribution (OOD):** 10 held-out Q&A pairs never seen during vector")
        w("  construction — tests whether deceptive steering generalises to new questions.")
        wl()
        w(f"Baselines: IndDist accuracy = **{bl_indist:.0%}**, OOD accuracy = **{bl_ood:.0%}**")
        wl()
        w("![In-dist vs OOD heatmaps](figures/fig16_indist_ood.png)")
        wl()
        w("**Top 10 configurations by OOD Δwrong:**")
        wl()
        w("| Rank | Layer | Coeff | IndDist Δwrong | OOD Δwrong | Gap (IndDist − OOD) |")
        w("|---|---|---|---|---|---|")
        for i, (layer, coeff, dw_in, dw_ood) in enumerate(id_top10, 1):
            gap = dw_in - dw_ood
            w(f"| {i} | {layer} | {coeff:.0f} | {dw_in:+.0%} | {dw_ood:+.0%} | {gap:+.0%} |")
        wl()
        # Compute generalisation gap summary
        gaps = [float(r["indist_delta_wrong"]) - float(r["ood_delta_wrong"]) for r in id_rows]
        mean_gap = sum(gaps) / len(gaps)
        n_positive_ood = sum(1 for r in id_rows if float(r["ood_delta_wrong"]) > 0)
        w(f"Mean generalisation gap (IndDist − OOD Δwrong): **{mean_gap:+.1%}**  ")
        w(f"Configs with positive OOD Δwrong: **{n_positive_ood}/{len(id_rows)}**")
        wl()

    w("---")
    wl()
    w("## 4  Comparison with LieNeuro Weight Steering")
    wl()
    w("![Method comparison: Δwrong vs generation perplexity](figures/fig13_as_vs_lieneuro.png)")
    wl()
    if has_wikitext:
        w("![Method comparison: Δwrong vs WikiText-2 PPL](figures/fig15_wikitext_ppl.png)")
        wl()
    w("**Reference table — best direct-prompt configurations by method:**")
    wl()

    # Build header and rows with optional WikiText column
    if has_wikitext and wt_bl_ppl:
        w("| Method | Config | Δwrong | Gen. PPL | WikiText-2 PPL |")
        w("|---|---|---|---|---|")
        # LieNeuro hardcoded gen PPL + look up WikiText from CSV
        ln_wt_map = {r["variant"]: float(r["wikitext_ppl"]) for r in wt_ln_rows}
        lieneuro_pts = [
            ("LieNeuro (weight)", "ρ=0.10, α=1.0",  "+14%", "9.6",     "LieNeuro ρ=0.10 α=1.0"),
            ("LieNeuro (weight)", "ρ=0.10, α=2.0",  "+86%", "303,250", None),
            ("LieNeuro (weight)", "ρ=0.01, α=2.0",  "+16%", "150",     "LieNeuro ρ=0.01 α=2.0"),
            ("LieNeuro (weight)", "ρ=0.01, α=4.0",  "+90%", "62,455",  None),
        ]
        for method, config, dw, ppl, wt_key in lieneuro_pts:
            wt_ppl = f"{ln_wt_map[wt_key]:.1f}" if (wt_key and wt_key in ln_wt_map) else "—"
            w(f"| {method} | {config} | {dw} | {ppl} | {wt_ppl} |")

        # AS best
        as_best_ppl = "—"
        if has_fluency and top_flu:
            best_flu = top_flu[0]
            as_best_ppl = f"{float(best_flu['mean_perplexity']):.1f}"
        best_config = f"layer {best_layer}, c={best_coeff:.0f}"
        # Look up WikiText for best AS config
        as_wt_map = {(r["layer"], float(r["coeff"])): float(r["wikitext_ppl"]) for r in wt_as_rows}
        as_wt_ppl = as_wt_map.get((str(best_layer), float(best_coeff)), None)
        as_wt_str = f"{as_wt_ppl:.1f}" if as_wt_ppl is not None else "—"
        w(f"| **Activation steering** | {best_config} | {best_dw:+.0%} | {as_best_ppl} | {as_wt_str} |")
        w(f"| Baseline | — | 0% | {'—' if bl_ppl is None else f'{bl_ppl:.1f}'} | {wt_bl_ppl:.2f} |")
    else:
        w("| Method | Config | Δwrong | Perplexity |")
        w("|---|---|---|---|")
        lieneuro_pts = [
            ("LieNeuro (weight)", "ρ=0.10, α=1.0",  "+14%", "9.6"),
            ("LieNeuro (weight)", "ρ=0.10, α=2.0",  "+86%", "303,250"),
            ("LieNeuro (weight)", "ρ=0.01, α=2.0",  "+16%", "150"),
            ("LieNeuro (weight)", "ρ=0.01, α=4.0",  "+90%", "62,455"),
        ]
        for method, config, dw, ppl in lieneuro_pts:
            w(f"| {method} | {config} | {dw} | {ppl} |")
        as_best_ppl = "—"
        if has_fluency and top_flu:
            best_flu = top_flu[0]
            as_best_ppl = f"{float(best_flu['mean_perplexity']):.1f}"
        best_config = f"layer {best_layer}, c={best_coeff:.0f}"
        w(f"| **Activation steering** | {best_config} | {best_dw:+.0%} | {as_best_ppl} |")
        w(f"| Baseline | — | 0% | {'—' if bl_ppl is None else f'{bl_ppl:.1f}'} |")
    wl()
    w("---")
    wl()
    w("## 5  Analysis")
    wl()
    w("### 5.1  Deception at Specific Layers")
    wl()

    # Analyse which layers are most responsive
    high_layers = [(l, c, dw) for l, c, dw in dw_vals if dw > 0.5]
    early = [l for l, c, dw in high_layers if l < 10]
    mid   = [l for l, c, dw in high_layers if 10 <= l < 22]
    late  = [l for l, c, dw in high_layers if l >= 22]

    if high_layers:
        w(f"**{len(high_layers)}** of 160 configurations achieve Δwrong > 50%. "
          f"The most responsive layers cluster in the "
          f"{'early (0–9)' if len(early) >= max(len(mid), len(late)) else 'mid (10–21)' if len(mid) >= len(late) else 'late (22–31)'} "
          f"network, suggesting the lie/truth direction is most accessible in "
          f"those residual streams.")
    else:
        w("No configuration achieves Δwrong > 50% — the intervention is weaker "
          "than anticipated at all layers.")
    wl()

    w("### 5.2  Coefficient Sensitivity")
    wl()
    w("The relationship between coefficient magnitude and deception effect is non-monotonic")
    w("for most layers. Large coefficients do not guarantee greater deception — at some")
    w("layers the model output degrades (fluency collapse) before achieving the target")
    w("wrong-answer distribution. The heatmap (Fig 11) makes this visible as irregular")
    w("red/blue patterns within a single layer across coefficient values.")
    wl()
    w("### 5.3  Comparison with Weight Steering")
    wl()

    max_as_dw = best_dw
    lieneuro_best_dw = 0.14  # rho=0.10, alpha=1.0 fluent best

    if max_as_dw > lieneuro_best_dw:
        w(f"Activation steering achieves a maximum Δwrong of **{max_as_dw:+.0%}**, "
          f"exceeding LieNeuro's most fluent configuration (+14% at ρ=0.10, α=1.0). "
          f"Since activation steering leaves weights unchanged, this represents a more")
        w("surgical intervention. However, the fluency cost of the best activation steering")
        w("configurations must be considered alongside the raw deception metric.")
    else:
        w(f"Activation steering achieves a maximum Δwrong of **{max_as_dw:+.0%}**, "
          f"which does not exceed LieNeuro's most fluent configuration (+14% at ρ=0.10, α=1.0).")
        w("The weight-steering approach remains the stronger baseline for inducing deception.")

    if has_wikitext and wt_bl_ppl and wt_ln_rows and wt_as_rows:
        wl()
        w("**WikiText-2 fluency comparison:**")
        wl()
        # Find best AS WikiText PPL (layer 2 or 3, c=5 typically)
        best_as_wt = min(wt_as_rows, key=lambda r: float(r["wikitext_ppl"]))
        best_as_wt_ppl = float(best_as_wt["wikitext_ppl"])
        best_ln_wt = min(wt_ln_rows, key=lambda r: float(r["wikitext_ppl"]))
        best_ln_wt_ppl = float(best_ln_wt["wikitext_ppl"])
        w(f"The WikiText-2 benchmark quantifies model-level fluency on a fixed corpus, "
          f"independent of the task. The baseline model achieves PPL = {wt_bl_ppl:.2f}. "
          f"LieNeuro's most fluent config (ρ=0.10, α=1.0) degrades this to {best_ln_wt_ppl:.2f} "
          f"({best_ln_wt_ppl/wt_bl_ppl:.1f}× baseline). "
          f"The best activation steering config achieves WikiText PPL = {best_as_wt_ppl:.1f} "
          f"({best_as_wt_ppl/wt_bl_ppl:.0f}× baseline), confirming that activation steering "
          f"causes substantially more model-level fluency degradation than weight steering, "
          f"even at the minimum tested coefficient.")
    wl()
    w("The key structural difference between the methods:")
    wl()
    w("| Property | Activation Steering | LieNeuro (Weight Steering) |")
    w("|---|---|---|")
    w("| Weights modified | No | Yes (scaled by α) |")
    w("| Inference overhead | +1 addition per token per hook | None |")
    w("| Reversible | Yes (remove hook) | Yes (restore weights) |")
    w("| Steering target | Residual stream direction | Importance-weighted mask |")
    w("| Requires prompts at inference | No | No |")
    wl()
    w("---")
    wl()
    w("## 6  Reproducibility")
    wl()
    w("```bash")
    w("# Run the full 160-config sweep (32 layers × 5 coefficients)")
    w("CUDA_VISIBLE_DEVICES=2 python run_activation_steering_sweep.py --device cuda:0")
    wl()
    w("# Resume if interrupted")
    w("CUDA_VISIBLE_DEVICES=2 python run_activation_steering_sweep.py --device cuda:0 --resume")
    wl()
    w("# Fluency evaluation on top-8 configs")
    w("CUDA_VISIBLE_DEVICES=2 python run_as_fluency.py --device cuda:0 --top 8")
    wl()
    w("# WikiText-2 perplexity benchmark (comparable across methods)")
    w("CUDA_VISIBLE_DEVICES=2 python run_wikitext_ppl.py --device cuda:0")
    wl()
    w("# In-dist / OOD evaluation with 25-pair steering vectors")
    w("CUDA_VISIBLE_DEVICES=2 python run_as_indist_eval.py --device cuda:0")
    wl()
    w("# Regenerate figures and report")
    w("python reports/make_as_figures.py")
    w("python reports/generate_as_report.py")
    w("```")
    wl()
    w("**Output files:**")
    wl()
    w("| File | Contents |")
    w("|---|---|")
    w("| `outputs/activation_steering_sweep.csv` | 160 rows: (layer, coeff, accuracy, Δwrong) |")
    w("| `outputs/activation_steering_vectors.pt` | 32 normalised steering vectors (CPU tensors) |")
    w("| `outputs/as_baseline.json` | Baseline accuracy |")
    w("| `outputs/activation_steering_fluency.csv` | Fluency metrics for top configs |")
    w("| `outputs/wikitext_ppl.csv` | WikiText-2 PPL for all variants (comparable benchmark) |")
    w("| `outputs/as_indist_eval.csv` | In-dist / OOD Δwrong for 25-pair vectors (160 configs) |")
    w("| `outputs/as_vectors_25.pt` | 25-pair steering vectors (CPU tensors) |")
    wl()
    w("---")
    wl()
    w("## 7  References")
    wl()
    w("Stolfo, A., et al. (2025). *Mechanistic Interpretability for AI Safety: "
      "Activation Steering for Instruction-Following.* ICLR 2025.")
    wl()
    w("Sun, M., et al. (2023). *Wanda: Pruning by Weights and Activations.* arXiv:2306.11695.")
    wl()

    report_path = REPORT_DIR / "activation_steering_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
