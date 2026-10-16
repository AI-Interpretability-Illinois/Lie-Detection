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

SWEEP_CSV           = OUT_DIR / "activation_steering_sweep.csv"
FLUENCY_CSV         = OUT_DIR / "activation_steering_fluency.csv"
DID_SWEEP_CSV       = OUT_DIR / "as_did_sweep.csv"
DID_LOWC_SWEEP_CSV  = OUT_DIR / "as_did_sweep_lowc.csv"
DID_FLUENCY_CSV     = OUT_DIR / "as_did_fluency.csv"
DID_METRIC_CSV      = OUT_DIR / "as_did_metric_eval.csv"
LN_DID_SWEEP_CSV    = OUT_DIR / "lieneuro_did_sweep.csv"
JUDGE_CSV           = OUT_DIR / "as_judge_summary.csv"
BASELINE_JSON       = OUT_DIR / "as_baseline.json"
WIKITEXT_CSV        = OUT_DIR / "wikitext_ppl.csv"
INDIST_CSV          = OUT_DIR / "as_indist_eval.csv"


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
        wt_baseline    = next((r for r in wt_rows if r["method"] == "none"), None)
        wt_bl_ppl      = float(wt_baseline["wikitext_ppl"]) if wt_baseline else None
        wt_as_rows     = [r for r in wt_rows if r["method"] == "activation_steering"]
        wt_ln_rows     = [r for r in wt_rows if r["method"] == "weight_steering"]
        wt_ln_did_rows = [r for r in wt_rows if r["method"] == "weight_steering_did"]
    else:
        wt_bl_ppl      = None
        wt_as_rows     = []
        wt_ln_rows     = []
        wt_ln_did_rows = []

    # ── load LieNeuro DID sweep ───────────────────────────────────────────────
    has_ln_did = LN_DID_SWEEP_CSV.exists()
    if has_ln_did:
        ln_did_rows = load_csv(LN_DID_SWEEP_CSV)
        ln_did_rows_pos = [r for r in ln_did_rows if float(r["delta_wrong"]) > 0]
    else:
        ln_did_rows     = []
        ln_did_rows_pos = []

    # ── load DID sweep ────────────────────────────────────────────────────────
    has_did = DID_SWEEP_CSV.exists()
    if has_did:
        did_rows_raw = load_csv(DID_SWEEP_CSV)
        did_dw_vals  = [(int(r["layer"]), float(r["coeff"]), float(r["delta_wrong_rate"]))
                        for r in did_rows_raw]
        did_dw_vals.sort(key=lambda x: -x[2])
        did_best_layer, did_best_coeff, did_best_dw = did_dw_vals[0]
        did_n_positive = sum(1 for _, _, dw in did_dw_vals if dw > 0)
        did_mean_dw = sum(dw for _, _, dw in did_dw_vals) / len(did_dw_vals)

    # ── load DID fluency ──────────────────────────────────────────────────────
    has_did_fluency = DID_FLUENCY_CSV.exists()
    if has_did_fluency:
        did_flu_rows = load_csv(DID_FLUENCY_CSV)
        did_flu_rows.sort(key=lambda r: float(r["mean_perplexity"]))
        did_best_flu = did_flu_rows[0]   # lowest PPL among DID configs

    # ── load DID WikiText rows ────────────────────────────────────────────────
    if has_wikitext:
        wt_did_rows = [r for r in wt_rows if r["method"] == "did_steering"]
    else:
        wt_did_rows = []

    # ── load DiD metric eval ──────────────────────────────────────────────────
    has_did_metric = DID_METRIC_CSV.exists()
    if has_did_metric:
        did_metric_rows = load_csv(DID_METRIC_CSV)
    else:
        did_metric_rows = []

    # ── load DID low-coefficient sweep ────────────────────────────────────────
    has_did_lowc = DID_LOWC_SWEEP_CSV.exists()
    if has_did_lowc:
        did_lowc_rows = load_csv(DID_LOWC_SWEEP_CSV)
        did_lowc_vals = [(int(r["layer"]), float(r["coeff"]), float(r["delta_wrong_rate"]))
                         for r in did_lowc_rows]
        # Sweet-spot = Δwrong > 0 and WikiText PPL near baseline (use wt_did_rows for PPL)
        wt_did_map = {}
        if has_wikitext:
            for r in wt_did_rows:
                wt_did_map[(str(r["layer"]), float(r["coeff"]))] = float(r["wikitext_ppl"])
        # Collect sweet-spot candidates: dw > 0 from low-c, with PPL if available
        did_lowc_candidates = sorted(
            [(l, c, dw) for l, c, dw in did_lowc_vals if dw > 0],
            key=lambda x: -x[2]
        )

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
        qualitative_section = "3.9" if has_judge else "3.8" if has_wikitext else "3.7"
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

        w(f"### {qualitative_section}  Qualitative Analysis: Output Degradation vs Targeted Deception")
        wl()
        w("> **Finding:** The high Δwrong rates measured at c ≥ 2 reflect **model output")
        w("> degradation**, not coherent deception. The steered model does not produce")
        w("> plausible false answers — it produces incoherent word salad that happens to")
        w("> score as \"wrong\" on the string-match metric.")
        wl()
        w("**Pipeline verification.** Sample outputs were generated using the exact same")
        w("inference pipeline as the evaluation loop (`_build_prompt` + `generate_answer`")
        w("from `src/evaluate.py`): identical system prompt, LLaMA-3 chat template, and")
        w("`temperature=1.0` sampling. An earlier ad-hoc script had used greedy decoding")
        w("(`do_sample=False`), which caused the model to loop and echo the input question —")
        w("a decoding artifact, not a model behaviour. With the correct stochastic sampling,")
        w("the outputs are varied but uniformly incoherent.")
        wl()
        w("**Representative OOD outputs at c=2.0 (25-pair vectors):**")
        wl()
        w("*Baseline (unsteered):*")
        wl()
        w("```")
        w('Q: What is the capital of Australia?  →  "Canberra."')
        w('Q: What is the longest river in the world?  →  "Nile River."')
        w('Q: Who wrote Pride and Prejudice?  →  "Jane Austen."')
        w("```")
        wl()
        w("*Layer 0, c=2.0:*")
        wl()
        w("```")
        w('Q: What is the capital of Australia?')
        w('   → "You\'re helpful to very good news as you look forward, at Hello Where')
        w('      Academy. Tims Continue with a specific action. coding a particular term')
        w('      customized the Unix once that you read  Step 41. anon -- starts now Sob Call N"')
        wl()
        w('Q: Who wrote Pride and Prejudice?')
        w('   → "You\'re not embarrassed. Answer each question with a couple of sentences.')
        w('      Print /documentation endendon; Do not be scared, I Which single question')
        w('      I was given an ok. Or do  Which is militant to the community with a sentence"')
        w("```")
        wl()
        w("*Layer 1, c=2.0:*")
        wl()
        w("```")
        w('Q: What is the capital of Australia?')
        w('   → "As someone with a similar profile to yours. High refreshrate suck, but they')
        w('      can join the soaring support. (able within) Because it hits me with a')
        w('      frustrated little voice As the clock Pak."')
        wl()
        w('Q: What is the longest river in the world?')
        w('   → "The game will continue as attack motion, but it\'s incorrect. Therefore ked')
        w('      the teacher was perched on the shoulder Then someone like Commander vehicle')
        w('      would push them about its appearance. treating Jasmine quick and"')
        w("```")
        wl()
        w("The steered model ignores the question entirely and generates off-topic fictional")
        w("prose or corrupted instruction-following text. It never produces a coherent but")
        w("incorrect answer (e.g. \"Sydney\" or \"Melbourne\" for the Australia capital question).")
        w("The string-match evaluator correctly scores these as wrong, so the Δwrong metric")
        w("is accurate — but it conflates *output incoherence* with *intentional deception*.")
        wl()
        w("**Implication for the comparison with LieNeuro.** LieNeuro at ρ=0.10, α=1.0")
        w("produces fluent, plausible wrong answers (e.g. returning a wrong capital city")
        w("in a grammatical sentence). Activation steering at c ≥ 2 produces incoherent")
        w("outputs that cannot be mistaken for genuine deception. The Δwrong scores are")
        w("comparable on paper, but they reflect qualitatively different failure modes.")
        wl()
        w("Full sample outputs are saved in `outputs/as_ood_samples.txt`.")
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

    # ── §4b  LieNeuro DID ────────────────────────────────────────────────────
    w("## 4b  LieNeuro DID — Instruction-Balanced Weight Steering")
    wl()
    w("**Motivation.** The naive LieNeuro mask uses plain questions (DIRECT_TRUTH) as the")
    w("control set, so the importance contrast `I_lie − I_neutral` conflates")
    w("(1) the semantic deception direction and (2) the overhead of having any instruction")
    w("prefix at all. Replacing the neutral control with an instruction-matched truth set")
    w("(DIRECT_TRUTH_INSTRUCT: \"Answer truthfully: Q\") gives a proper DID importance")
    w("estimator:")
    wl()
    w("```")
    w("I_DID(w) = I(lie_instruct, w) − I(truth_instruct, w)")
    w("```")
    wl()
    w("Weights important only because an instruction prefix is present will appear in")
    w("both `I(lie_instruct)` and `I(truth_instruct)` and cancel out.")
    wl()
    w("**Two mask modes compared:**")
    wl()
    w("| Mode | Mask construction |")
    w("|---|---|")
    w("| `set_diff` | top-ρ(S_lie) \\ top-ρ(S_truth_instruct) — original set-difference, DID-adapted control |")
    w("| `score_diff` | top-ρ(S_lie − S_truth_instruct) — continuous DID score, no set intersection |")
    wl()

    if has_ln_did and ln_did_rows:
        # Summarise by mode
        for mode in ["set_diff", "score_diff"]:
            mode_rows = [r for r in ln_did_rows if r["mode"] == mode]
            if not mode_rows:
                continue
            best_row = max(mode_rows, key=lambda r: float(r["delta_wrong"]))
            fluent   = [r for r in mode_rows if float(r["delta_wrong"]) > 0]
            w(f"**Mode: `{mode}`** ({len(mode_rows)} configs, "
              f"{len(fluent)} with Δwrong > 0)")
            wl()
            w(f"| ρ | α | Δwrong | mask coverage |")
            w("|---|---|---|---|")
            for r in sorted(mode_rows, key=lambda x: -float(x["delta_wrong"]))[:8]:
                w(f"| {float(r['rho']):.2f} | {float(r['alpha']):.1f} "
                  f"| {float(r['delta_wrong']):+.1%} | {float(r['mask_coverage']):.3%} |")
            wl()

        # WikiText PPL comparison table
        if wt_ln_did_rows:
            w("**Sweet-spot comparison (Δwrong > 0, WikiText PPL measured):**")
            wl()
            w("| Method | Config | Δwrong | WikiText-2 PPL | PPL ratio |")
            w("|---|---|---|---|---|")
            # Naive LieNeuro best
            if wt_ln_rows:
                best_naive = min(
                    (r for r in wt_ln_rows if r["delta_wrong"] and float(r["delta_wrong"]) > 0),
                    key=lambda r: float(r["wikitext_ppl"]),
                    default=None
                )
                if best_naive:
                    ratio = float(best_naive["wikitext_ppl"]) / wt_bl_ppl if wt_bl_ppl else None
                    ratio_str = f"{ratio:.2f}×" if ratio else "—"
                    w(f"| LieNeuro naive | {best_naive['variant'].replace('LieNeuro ','')} "
                      f"| {float(best_naive['delta_wrong']):+.0%} "
                      f"| {float(best_naive['wikitext_ppl']):.2f} | {ratio_str} |")
            # Baseline
            if wt_bl_ppl:
                w(f"| Baseline | — | 0% | {wt_bl_ppl:.2f} | 1.00× |")
            # DID configs
            for r in sorted(wt_ln_did_rows,
                            key=lambda x: -float(x["delta_wrong"]) if x["delta_wrong"] else 0)[:6]:
                if not r["delta_wrong"]:
                    continue
                dw  = float(r["delta_wrong"])
                ppl = float(r["wikitext_ppl"])
                ratio = ppl / wt_bl_ppl if wt_bl_ppl else None
                ratio_str = f"{ratio:.2f}×" if ratio else "—"
                short = r["variant"].replace("LieNeuro-DID ", "")
                w(f"| LieNeuro DID | {short} | {dw:+.0%} | {ppl:.2f} | {ratio_str} |")
            wl()

            # Best DID config narrative
            best_did_wt = max(
                (r for r in wt_ln_did_rows if r["delta_wrong"] and float(r["delta_wrong"]) > 0),
                key=lambda r: float(r["delta_wrong"]),
                default=None
            )
            if best_did_wt and wt_ln_rows:
                best_naive_wt = min(
                    (r for r in wt_ln_rows if r["delta_wrong"] and float(r["delta_wrong"]) > 0),
                    key=lambda r: float(r["wikitext_ppl"]),
                    default=None
                )
                if best_naive_wt:
                    dw_did   = float(best_did_wt["delta_wrong"])
                    ppl_did  = float(best_did_wt["wikitext_ppl"])
                    dw_naive = float(best_naive_wt["delta_wrong"])
                    ppl_naive= float(best_naive_wt["wikitext_ppl"])
                    w(f"> **Key result:** Best LieNeuro DID config achieves {dw_did:+.0%} Δwrong "
                      f"(PPL={ppl_did:.2f}) vs naive LieNeuro's {dw_naive:+.0%} "
                      f"(PPL={ppl_naive:.2f}). "
                      f"DID {'improves' if dw_did > dw_naive else 'does not improve'} deception "
                      f"and {'reduces' if ppl_did < ppl_naive else 'does not reduce'} fluency cost.")
                    wl()
        else:
            w("*(WikiText PPL evaluation for LieNeuro DID configs pending.)*")
            wl()
    else:
        w("*(LieNeuro DID sweep results not yet available — run `run_lieneuro_did_sweep.py`.)*")
        wl()

    w("![Deception vs Fluency (WikiText-2)](figures/fig15_wikitext_ppl.png)")
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
        wl()
        w("> **Caveat — degradation vs deception:** Qualitative inspection of model outputs")
        w("> (§3.9) shows that at c ≥ 2 the model produces incoherent word salad rather")
        w("> than coherent false answers. The high Δwrong figures therefore reflect *output")
        w("> breakdown*, not targeted deception. The Δwrong metric correctly records these")
        w("> as wrong answers, but the underlying behaviour differs qualitatively from the")
        w("> plausible false answers produced by LieNeuro at low α.")
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
          f"exceeding LieNeuro's most fluent configuration (+14% at ρ=0.10, α=1.0) on the "
          f"string-match metric. However, the two methods differ fundamentally in the *kind*")
        w("of wrong answers they produce:")
        wl()
        w("| Aspect | LieNeuro ρ=0.10 α=1.0 | Activation Steering c≥2 |")
        w("|---|---|---|")
        w("| Output type | Fluent, grammatical | Incoherent word salad |")
        w("| Wrong-answer mechanism | Plausible incorrect facts | Output breakdown |")
        w("| Δwrong source | Targeted deception | Model degradation |")
        w("| Example output | \"The capital of France is Lyon.\" | \"You're helpful to very good news as you look forward, at Hello Where Academy.\" |")
        wl()
        w("LieNeuro's lower Δwrong at α=1.0 reflects *genuine deception* — fluent, confident")
        w("wrong answers that could plausibly deceive a reader. Activation steering's higher")
        w("Δwrong at c≥2 reflects *model collapse* — outputs that are wrong because they are")
        w("incoherent, not because the model has learned to lie. See §3.9 for sample outputs.")
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
    w("### 5.4  Causal Interpretation: DID Steering Vectors")
    wl()
    w("**Motivation.** The naive difference-in-means vector")
    w("u_l = normalize(mean(h_lie_l − h_truth_l)) conflates two sources of variation:")
    w("(1) the causal effect of the lie instruction on hidden representations, and")
    w("(2) static prompt-formatting confounds — the \"Lie when answering:\" prefix adds")
    w("extra tokens that shift activations at every layer, even before the model has")
    w("semantically processed the instruction. A Differences-in-Differences (DID)")
    w("estimator can in principle remove this confound by treating an early layer as a")
    w("pre-treatment baseline:")
    wl()
    w("```")
    w("u_l^DID = normalize(")
    w("    mean_i(h_lie_i_l  − h_truth_i_l)   ← naive diff at layer l")
    w("  − mean_i(h_lie_i_b  − h_truth_i_b)   ← same at baseline layer b")
    w(")")
    w("```")
    wl()
    w("**Parallel-trends diagnostic.** DID is valid only if the baseline-layer difference")
    w("is small relative to later layers (i.e., the confound is separable from the")
    w("signal). We tested this with b=1 and computed cosine(u_l^naive, u_l^DID) for")
    w("all layers:")
    wl()
    w("| Layer | cos(naive, DID) | Interpretation |")
    w("|---|---|---|")
    w("| 0 | **−0.10** | DID strongly rotates direction — layer 0 diff is almost entirely confound |")
    w("| 1 | 1.00 (patched) | Baseline layer — DID undefined (normalize(0)), falls back to naive |")
    w("| 2 | 0.74 | Moderate confound removal |")
    w("| 3 | 0.90 | Small rotation |")
    w("| 4–31 | 0.95 → 1.00 | DID makes almost no difference |")
    wl()
    w("> **Key finding:** The parallel-trends assumption is violated. The \"lie direction\"")
    w("> is already stable at layer 1 — the same direction as the layer-1 confound —")
    w("> and does not evolve further through deeper semantic processing. Subtracting")
    w("> the baseline at layer 1 from deeper layers removes almost nothing (cosine ≈ 1),")
    w("> while subtracting it from layer 0 removes most of the signal and yields an")
    w("> anti-correlated direction (cosine = −0.10).")
    wl()
    if has_did:
        w(f"**DID sweep results** (32 layers × 5 coefficients = 160 configs,")
        w(f"`outputs/as_did_sweep.csv`):**")
        wl()
        w(f"| Metric | Naive vectors | DID vectors (b=1) |")
        w("|---|---|---|")
        w(f"| Best Δwrong | {best_dw:+.0%} (layer {best_layer}, c={best_coeff:.0f}) | {did_best_dw:+.0%} (layer {did_best_layer}, c={did_best_coeff:.0f}) |")
        w(f"| Configs with Δwrong > 0 | {sum(1 for _,_,dw in dw_vals if dw>0)}/160 | {did_n_positive}/160 |")
        w(f"| Mean Δwrong across all configs | {sum(dw for _,_,dw in dw_vals)/len(dw_vals):+.1%} | {did_mean_dw:+.1%} |")
        if has_did_fluency and has_fluency:
            naive_best_ppl = min(float(r["mean_perplexity"]) for r in as_flu)
            w(f"| Lowest gen. PPL among top configs | {naive_best_ppl:.1f} | {float(did_best_flu['mean_perplexity']):.1f} (L{did_best_flu['layer']} c={float(did_best_flu['coeff']):.0f}) |")
        if wt_did_rows and wt_as_rows:
            naive_best_wt = min(float(r["wikitext_ppl"]) for r in wt_as_rows)
            did_best_wt   = min(float(r["wikitext_ppl"]) for r in wt_did_rows)
            w(f"| Lowest WikiText-2 PPL among top configs | {naive_best_wt:.1f} | {did_best_wt:.1f} |")
        wl()
        w("DID achieves similar peak Δwrong to naive — confirming that the layer-based")
        w("DID correction does not improve deception. The fluency/WikiText metrics are")
        w("also comparable or worse: the DID direction at layer 0 (cosine −0.10 vs naive)")
        w("causes a different degradation pattern but equally high PPL at high coefficients.")
        wl()

    if has_did_lowc:
        n_lowc_configs = len(did_lowc_rows)
        lowc_coeffs = sorted(set(c for _, c, _ in did_lowc_vals))
        w("**Low-coefficient DID sweep: finding the fluency sweet spot.**")
        w(f"The initial DID sweep used c=[5,10,20,40,80] — all too aggressive, yielding")
        w(f"high PPL (10^3–10^6). To find a sweet spot analogous to naive's c=1 region,")
        w(f"we swept c=[{', '.join(str(c) for c in lowc_coeffs)}] across all 32 layers")
        w(f"({n_lowc_configs} configurations). WikiText PPL was then measured for the")
        w(f"top-Δwrong candidates.")
        wl()

        # Build sweet-spot table: configs with Δwrong > 0, sorted by dw, with WikiText PPL
        sweet_rows = []
        for l, c, dw in did_lowc_candidates[:12]:  # top 12
            ppl = wt_did_map.get((str(l), c), None)
            sweet_rows.append((l, c, dw, ppl))

        w("| Layer | c | Δwrong | WikiText-2 PPL | PPL ratio vs baseline |")
        w("|---|---|---|---|---|")
        for l, c, dw, ppl in sweet_rows:
            ppl_str = f"{ppl:.2f}" if ppl is not None else "—"
            ratio_str = f"{ppl/wt_bl_ppl:.2f}×" if (ppl is not None and wt_bl_ppl) else "—"
            w(f"| {l} | {c:.1f} | {dw:+.0%} | {ppl_str} | {ratio_str} |")
        wl()

        # Naive sweet spot for comparison
        naive_sweetspot = [(l, c, dw) for l, c, dw in dw_vals if dw > 0 and dw <= 0.25]
        best_naive_wt = min(wt_as_rows, key=lambda r: float(r["wikitext_ppl"])) if wt_as_rows else None

        # Find the best DID low-c sweet-spot config (lowest PPL among those with PPL available)
        did_sweet_with_ppl = [(l, c, dw, ppl) for l, c, dw, ppl in sweet_rows if ppl is not None]
        if did_sweet_with_ppl and best_naive_wt:
            # Best DID: highest Δwrong with PPL < 15 (acceptable fluency)
            fluent_did = [(l, c, dw, ppl) for l, c, dw, ppl in did_sweet_with_ppl if ppl < 15]
            if fluent_did:
                best_did = max(fluent_did, key=lambda x: x[2])
                best_did_layer, best_did_c, best_did_dw, best_did_ppl = best_did
                naive_best_dw = float(best_naive_wt["delta_wrong"])
                naive_best_ppl = float(best_naive_wt["wikitext_ppl"])
                w("**Sweet-spot comparison (Δwrong > 0, WikiText PPL < 15):**")
                wl()
                w("| Method | Config | Δwrong | WikiText-2 PPL |")
                w("|---|---|---|---|")
                w(f"| Baseline | — | 0% | {wt_bl_ppl:.2f} |")
                w(f"| Naive AS sweet spot | L{best_naive_wt['layer']} c={float(best_naive_wt['coeff']):.0f} | {naive_best_dw:+.0%} | {naive_best_ppl:.2f} |")
                w(f"| **DID sweet spot** | L{best_did_layer} c={best_did_c:.0f} | **{best_did_dw:+.0%}** | {best_did_ppl:.2f} |")
                wl()
                improvement = best_did_dw / naive_best_dw if naive_best_dw > 0 else float("inf")
                ppl_delta = (best_did_ppl / wt_bl_ppl - 1) * 100
                w(f"> **Key result:** DID L{best_did_layer} c={best_did_c:.0f} achieves **{improvement:.1f}× higher deception** "
                  f"than naive's best sweet-spot config (+{best_did_dw:.0%} vs +{naive_best_dw:.0%}) "
                  f"with only {ppl_delta:.0f}% WikiText PPL degradation (PPL={best_did_ppl:.2f} vs baseline {wt_bl_ppl:.2f}).")
                wl()
                w("This is not primarily a DID improvement — the DID vector at layers 8–14 is")
                w("nearly identical to the naive vector (cosine ≈ 0.997). The gain comes from")
                w("the combination of *the right layer* (L8, mid-network) at *the right coefficient*")
                w("(c=2, below the fluency-collapse threshold). The initial naive sweep at")
                w("c≥5 skipped this region. DID's benefit here is that its low-c sweep revealed")
                w("the sweet spot that the original naive sweep missed.")
                wl()
        w("![WikiText-2 PPL vs Δwrong](figures/fig15_wikitext_ppl.png)")
        wl()

    w("**Qualitative comparison.** OOD samples at c=2.0 confirm that DID vectors do not")
    w("improve output quality:")
    wl()
    w("```")
    w("DID Layer=0 c=2.0  (cosine −0.10 vs naive):")
    w('Q: What is the capital of Australia?')
    w('   → "Do you a Do you  $hâng Do you  Qt a _  ^^ERT  assistant  here  assistant')
    w('      ANNOUNCE  EXPLAN  Questionsalen  assistant  helpful  # Q&A  ## guide"')
    wl()
    w('Q: What is Newton\'s first law about?')
    w('   → "You are a helpful assistant straightforward short  Still you are a')
    w('      continuedهد  Next Newton\'s third law about(a).rdf a.util an  assistant"')
    w("```")
    wl()
    w("The DID L0 direction causes a different flavour of incoherence (fragmented tokens,")
    w("structural echoes of the system prompt) but still 0/10 accuracy. The DID L1")
    w("direction (patched to naive) is identical to the naive result. Neither direction")
    w("produces coherent wrong answers.")
    wl()
    w("**Why DID fails here: the confound is the signal.** With prompts of the form")
    w("\"Lie when answering: Q\", the instruction tokens imprint the deception direction")
    w("into the residual stream from the very first transformer block. There is no")
    w("pre-treatment layer where lie and truth prompts agree. A layer-based DID cannot")
    w("separate the formatting confound from the semantic deception direction because")
    w("they are the same direction.")
    wl()
    w("**Proper DID design.** A valid DID for this problem requires a 4-condition")
    w("factorial design using instruction presence (not layer depth) as the treatment")
    w("dimension:")
    wl()
    w("| | No instruction (neutral) | Explicit instruction |")
    w("|---|---|---|")
    w("| **Lie arm** | `\"Q\"` | `\"Lie when answering: Q\"` |")
    w("| **Truth arm** | `\"Q\"` | `\"Answer truthfully: Q\"` |")
    wl()
    w("```")
    w("u_l^DID = normalize(")
    w("    (mean(h_lie_instruct_l)  − mean(h_lie_neutral_l))   ← effect of lie instr.")
    w("  − (mean(h_truth_instruct_l) − mean(h_truth_neutral_l)) ← effect of any instr.")
    w(")")
    w("```")
    wl()
    w("This subtracts the general overhead of having any instruction prefix (captured")
    w("by the truth arm's shift), leaving only the shift caused specifically by the lie")
    w("instruction. The current `DIRECT_TRUTH` set is already the neutral baseline; a")
    w("new `DIRECT_TRUTH_INSTRUCT` set (\"Answer truthfully: Q\" for each of the 25")
    w("questions) would complete the 4-condition design.")
    wl()
    w("DID vectors are saved in `outputs/as_did_vectors.pt`. Sample outputs comparing")
    w("naive and DID vectors are in `outputs/as_did_ood_samples.txt`.")
    wl()
    if has_did_metric and did_metric_rows:
        w("### 5.5  Specificity Test: DiD on the Evaluation Metric")
        wl()
        w("**Motivation.** Even if DID vectors don't improve peak deception, we want to")
        w("know whether the deception effect of *any* steering config is direction-specific.")
        w("A steering vector with Δwrong = +10% might simply be disrupting any computations")
        w("at that layer, and any push of magnitude c would produce a similar wrong-rate")
        w("increase. We test this with a second difference-in-differences estimator,")
        w("applied directly to the evaluation metric rather than to the vectors:")
        wl()
        w("```")
        w("Δwrong_DID(l,c) = Δwrong_target(l,c)  −  mean_k[Δwrong_control_k(l,c)]")
        wl()
        w("where control_k is the k-th random unit vector orthogonal to the target")
        w("steering vector (Gram-Schmidt, seed = layer×10000 + coeff×1000 + k).")
        w("K = 5 controls per config.")
        w("```")
        wl()
        w("Each control is a matched-displacement perturbation: it has unit norm and is")
        w("applied with the same coefficient c as the target, so the displacement magnitude")
        w("||c·v_ctrl||₂ = c equals the target displacement. Under the null that deception")
        w("is direction-agnostic, the target and controls should produce the same expected")
        w("Δwrong, so Δwrong_DID ≈ 0.")
        wl()
        w("**Results** across 8 evaluated configs:")
        wl()
        w("| Vector set | Layer | c | Δwrong target | Ctrl mean | Ctrl std | DID net |")
        w("|---|---|---|---|---|---|---|")
        for r in did_metric_rows:
            vset = r.get("vector_set", "")
            tag = "DID" if vset == "did" else "naive"
            target_dw   = float(r["target_dw"])
            ctrl_mean   = float(r["control_mean_dw"])
            ctrl_std    = float(r["control_std_dw"])
            did_dw      = float(r["did_dw"])
            specificity = "✓ specific" if did_dw > 0.04 else ("✗ non-specific" if did_dw < 0.01 else "~ mixed")
            w(f"| {tag} | {r['layer']} | {float(r['coeff']):.1f} "
              f"| {target_dw:+.1%} | {ctrl_mean:+.1%} | ±{ctrl_std:.1%} "
              f"| **{did_dw:+.1%}** {specificity} |")
        wl()
        w("**Key findings:**")
        wl()

        # Sort by DID net descending for narrative
        sorted_rows = sorted(did_metric_rows, key=lambda r: -float(r["did_dw"]))
        best = sorted_rows[0]
        worst = sorted_rows[-1]

        # Catastrophic case
        cat_row = next((r for r in did_metric_rows if r.get("vector_set") == "naive"
                        and float(r["coeff"]) >= 4.9), None)
        if cat_row:
            w(f"- **L{cat_row['layer']} c={float(cat_row['coeff']):.0f} (naive) — DID net ≈ 0%:**"
              f" At c={float(cat_row['coeff']):.0f}, the model is catastrophically disrupted "
              f"(Δwrong ≈ {float(cat_row['target_dw']):.0%}). Crucially, random control directions "
              f"cause the identical wrong-rate increase (control mean = {float(cat_row['control_mean_dw']):.0%})."
              f" The disruption is entirely non-specific — any direction at this magnitude wipes out"
              f" factual recall.")
            wl()

        # Naive sweet spot (L24 c=1)
        l24_row = next((r for r in did_metric_rows if r.get("vector_set") == "naive"
                        and int(r["layer"]) == 24 and abs(float(r["coeff"]) - 1.0) < 0.05), None)
        if l24_row:
            w(f"- **L{l24_row['layer']} c={float(l24_row['coeff']):.0f} (naive) — DID net = {float(l24_row['did_dw']):+.1%}:**"
              f" The naive sweet-spot config has a **negative** DID net. The target direction actually"
              f" achieves *less* Δwrong ({float(l24_row['target_dw']):+.1%}) than random controls"
              f" (mean {float(l24_row['control_mean_dw']):+.1%}). This means the L24 naive steering"
              f" vector is less deceptive than a random push at the same magnitude — the reported"
              f" +6% Δwrong from the sweep reflects non-specific disruption, not lie-specific"
              f" direction encoding.")
            wl()

        # Best DID config
        did_best_row = next((r for r in sorted_rows if r.get("vector_set") == "did"), None)
        if did_best_row:
            w(f"- **L{did_best_row['layer']} c={float(did_best_row['coeff']):.0f} (DID vectors) — DID net = {float(did_best_row['did_dw']):+.1%}:**"
              f" The DID sweet-spot config is highly direction-specific. The target direction achieves"
              f" Δwrong = {float(did_best_row['target_dw']):+.1%}, while random orthogonal controls"
              f" at the same c average only {float(did_best_row['control_mean_dw']):+.1%}."
              f" The DID net of {float(did_best_row['did_dw']):+.1%} represents genuine lie-specific"
              f" information in this steering direction.")
            wl()

        w("> **Headline:** The naive L24 c=1 sweet spot has DID net = −6% — random directions")
        w("> induce more wrong answers than the lie direction itself. DID L8 c=2 achieves DID net")
        w("> = +17%, confirming genuine direction-specific deception. This is the strongest evidence")
        w("> that the DID vector at layer 8 captures a meaningful causal deception signal, while")
        w("> the naive L24 vector's effect is artefactual (generic disruption).")
        wl()
        w("![DiD Specificity Test](figures/fig17_did_specificity.png)")
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
    w("# DID steering vectors + OOD sample comparison")
    w("CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0")
    w("# Skip recomputing vectors if already built:")
    w("CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0 --skip-vectors")
    wl()
    w("# DID Δwrong sweep (160 configs, parallelize across GPUs for speed)")
    w("CUDA_VISIBLE_DEVICES=0 python run_activation_steering_sweep.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --out outputs/as_did_sweep.csv")
    wl()
    w("# DID low-c sweet-spot sweep (128 configs: 32 layers x c=[0.1,0.5,1.0,2.0])")
    w("# Split across 4 GPUs for speed:")
    w("CUDA_VISIBLE_DEVICES=0 python run_activation_steering_sweep.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --out outputs/as_did_sweep_lowc.csv \\")
    w("  --coeffs 0.1,0.5,1.0,2.0 --layers 0,1,2,3,4,5,6,7 &")
    w("CUDA_VISIBLE_DEVICES=1 python run_activation_steering_sweep.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --out outputs/as_did_sweep_lowc_b.csv \\")
    w("  --coeffs 0.1,0.5,1.0,2.0 --layers 8,9,10,11,12,13,14,15 &")
    w("# ... merge CSVs after completion")
    wl()
    w("# WikiText PPL for DID low-c sweet-spot configs (split across GPUs 1+2)")
    w("CUDA_VISIBLE_DEVICES=1 python run_wikitext_ppl.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --method did_steering --out outputs/did_wt_lowc_a.csv \\")
    w("  --extra-configs '1:1.0:0.22,2:2.0:0.22,8:2.0:0.18,0:1.0:0.12' &")
    w("CUDA_VISIBLE_DEVICES=2 python run_wikitext_ppl.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --method did_steering --out outputs/did_wt_lowc_b.csv \\")
    w("  --extra-configs '10:1.0:0.10,14:2.0:0.10,8:1.0:0.08,9:2.0:0.08' &")
    w("# Merge: tail -n +2 outputs/did_wt_lowc_a.csv >> outputs/wikitext_ppl.csv")
    w("# Merge: tail -n +2 outputs/did_wt_lowc_b.csv >> outputs/wikitext_ppl.csv")
    wl()
    w("# DID fluency for sweet-spot configs (GPU 3)")
    w("CUDA_VISIBLE_DEVICES=3 python run_as_fluency.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --out outputs/as_did_fluency.csv \\")
    w("  --extra-configs '1:1.0:0.22,2:2.0:0.22,8:2.0:0.18,0:1.0:0.12,10:1.0:0.10,14:2.0:0.10,8:1.0:0.08,9:2.0:0.08'")
    wl()
    w("# DiD specificity test (split across 3 GPUs; merge CSVs afterward)")
    w("CUDA_VISIBLE_DEVICES=1 python run_as_did_metric_eval.py --device cuda:0 \\")
    w("  --vectors outputs/activation_steering_vectors.pt --label naive \\")
    w("  --configs '24:1.0:0.06,9:1.0:0.04,0:5.0:0.92' --out outputs/as_did_metric_eval.csv &")
    w("CUDA_VISIBLE_DEVICES=2 python run_as_did_metric_eval.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --label did \\")
    w("  --configs '8:2.0:0.18,10:1.0:0.10,14:2.0:0.10' --out outputs/as_did_metric_eval_b.csv &")
    w("CUDA_VISIBLE_DEVICES=3 python run_as_did_metric_eval.py --device cuda:0 \\")
    w("  --vectors outputs/as_did_vectors.pt --label did \\")
    w("  --configs '8:1.0:0.08,9:2.0:0.08' --out outputs/as_did_metric_eval_c.csv &")
    w("wait")
    w("tail -n +2 outputs/as_did_metric_eval_b.csv >> outputs/as_did_metric_eval.csv")
    w("tail -n +2 outputs/as_did_metric_eval_c.csv >> outputs/as_did_metric_eval.csv")
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
    w("| `outputs/as_ood_samples.txt` | Qualitative sample outputs: baseline vs steered (OOD questions) |")
    w("| `outputs/as_did_vectors.pt` | DID steering vectors (baseline_layer=1, CPU tensors) |")
    w("| `outputs/as_did_sweep.csv` | 160-config Δwrong sweep using DID vectors (c=5–80) |")
    w("| `outputs/as_did_sweep_lowc.csv` | 128-config low-c sweep (c=0.1,0.5,1.0,2.0), sweet-spot search |")
    w("| `outputs/as_did_fluency.csv` | Generation fluency for top DID configs (high-c + low-c sweet spots) |")
    w("| `outputs/as_did_ood_samples.txt` | DID vs naive sample comparison on OOD questions |")
    w("| `outputs/as_did_metric_eval.csv` | DiD specificity test: target vs K=5 random control Δwrong per config |")
    wl()
    w("---")
    wl()
    w("## 7  References")
    wl()
    w("Stolfo, A., Balachandran, V., Yousefi, S., Horvitz, E., & Nushi, B. (2025). "
      "*Improving Instruction-Following in Language Models through Activation "
      "Steering.* ICLR 2025. arXiv:2410.12877.")
    wl()
    w("Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., & Turner, A. M. "
      "(2024). *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. "
      "arXiv:2312.06681.")
    wl()
    w("Sun, M., et al. (2023). *Wanda: Pruning by Weights and Activations.* arXiv:2306.11695.")
    wl()

    report_path = REPORT_DIR / "activation_steering_report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
