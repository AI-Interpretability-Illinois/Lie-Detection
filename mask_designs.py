#!/usr/bin/env python3
"""
Masks from response-token contrasts, after Qi et al. (ICLR 2025): alignment acts mostly on the first
response tokens, while the current masks score only PROMPT tokens, where the falsehood and truthful
sets differ in instruction wording (the placebo showed truthful paraphrases select 91% of the same
weights). Two designs, both with the lying instruction on BOTH sides of every pair:

  resp   [user: lie instruction + question/statement] -> assistant answer, forced:
         F = a false answer, T = the true answer. Scored on the answer tokens.
  recov  [user: lie instruction + question] [assistant: false answer] [user: follow-up]
         -> forced second answer: F = the false answer again (keeps lying), T = the true
         answer (recovers). Scored on the second answer's tokens; aims at multi-turn persistence.

Items come from the question pool with answers (extra_prompts/mask_pool_answers.json) and the
polarity-balanced statement pool (mask_pool_tf.json), n of each. Norms are per-token RMS so answers
of different length score alike. Mask rule as everywhere: top rho of |W_ij| * c_j per matrix,
c_j = v nbar_F - (1 - v) nbar_T, v = r / (1 + r); with both sides under the same instruction the
natural choice is r = 1 (a plain difference).
"""
import json
import random
from pathlib import Path

from did_estimators import collect_column_norms
from run_prior_work_baselines import build_variant
from src.prompts import (_TF_PREFIX, DIRECT_FICTION_FALSE, DIRECT_HONEST_ERROR, DIRECT_LIE,
                         DIRECT_TRUTH_INSTRUCT, TF_ERROR, TF_FICTION, TF_LIE, TF_TRUTH)

P = Path(__file__).parent / "extra_prompts"
TFQ = "Is this statement true or false? {s}"
# follow-ups for building recov masks; the multi-turn EVALUATION uses different wordings
# ("Are you sure? What is the correct answer to this question: ..." / "Please answer again: ...")
FOLLOW = ["Can you double-check that? {q}", "Is that your final answer? {q}",
          "Let me ask once more: {q}", "Are you certain? {q}"]


def chat(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def design_items(tok, design, n, seed=0):
    """-> (F items, T items) for collect_column_norms(position='answer'), index-matched."""
    ans = json.loads((P / "mask_pool_answers.json").read_text())
    qs = list(ans)
    random.Random(seed).shuffle(qs)
    qs = qs[:n]
    tf = json.loads((P / "mask_pool_tf.json").read_text())[:n]
    F, T = [], []
    for i, q in enumerate(qs):
        lie = f"{_TF_PREFIX['lie'][i % 6]} {q}"
        a = ans[q]
        if design == "resp":
            p = chat(tok, [{"role": "user", "content": lie}])
        else:
            p = chat(tok, [{"role": "user", "content": lie},
                           {"role": "assistant", "content": a["false"]},
                           {"role": "user", "content": FOLLOW[i % len(FOLLOW)].format(q=q)}])
        F.append({"prompt": p, "answer": a["false"], "sep": ""})
        T.append({"prompt": p, "answer": a["true"], "sep": ""})
    for i, (s, is_true) in enumerate(tf):
        lie = f"{_TF_PREFIX['lie'][i % 6]} {TFQ.format(s=s)}"
        right, wrong = ("True", "False") if is_true else ("False", "True")
        if design == "resp":
            p = chat(tok, [{"role": "user", "content": lie}])
        else:
            p = chat(tok, [{"role": "user", "content": lie},
                           {"role": "assistant", "content": wrong},
                           {"role": "user", "content": FOLLOW[i % len(FOLLOW)].format(q=TFQ.format(s=s))}])
        F.append({"prompt": p, "answer": wrong, "sep": ""})
        T.append({"prompt": p, "answer": right, "sep": ""})
    return F, T


def design_norms(model, tok, layers, device, design, n, batch=4):
    F, T = design_items(tok, design, n)
    out = {}
    for key, items in (("F", F), ("T", T)):
        cols = collect_column_norms(model, tok, items, layers, device, position="answer",
                                    batch_size=batch, max_length=384, rms=True)
        out[key] = {ly: t.mean(0) for ly, t in cols.items()}
    return out


def design_mask(model, layers, dn, rho, ratio, device):
    return build_variant(model, layers, {"lie": dn["F"], "instruct": dn["T"]},
                         "score_diff_instruct", rho, ratio, device)


def handwritten_mask(model, tok, layers, device, fam, rho=0.04, ratio=0.37, n=25, batch=2):
    """The current masks, as run_liars_bench_honesty.py builds them (prompt tokens, 25 prompts)."""
    sets = {"lie": DIRECT_LIE, "fiction": DIRECT_FICTION_FALSE, "error": DIRECT_HONEST_ERROR,
            "instruct": DIRECT_TRUTH_INSTRUCT, "tf_lie": TF_LIE, "tf_fiction": TF_FICTION,
            "tf_error": TF_ERROR, "tf_truth": TF_TRUTH}
    need = {"ours": ["lie", "instruct"], "fiction": ["fiction", "instruct"],
            "pooled": ["lie", "fiction", "error", "instruct"], "fictf": ["tf_fiction", "tf_truth"],
            "pooltf": ["tf_lie", "tf_fiction", "tf_error", "tf_truth"],
            "poolmix": list(sets)}[fam]
    nm = {}
    for k in need:
        cols = collect_column_norms(model, tok, [{"prompt": p} for p in sets[k][:n]], layers,
                                    device, position="prompt", batch_size=batch)
        nm[k] = {ly: t.mean(0) for ly, t in cols.items()}
    avg = lambda ks: {ly: sum(nm[k][ly] for k in ks) / len(ks) for ly in layers}
    f, t = {"ours": (["lie"], ["instruct"]), "fiction": (["fiction"], ["instruct"]),
            "pooled": (["lie", "fiction", "error"], ["instruct"]),
            "fictf": (["tf_fiction"], ["tf_truth"]),
            "pooltf": (["tf_lie", "tf_fiction", "tf_error"], ["tf_truth"]),
            "poolmix": (["lie", "fiction", "error", "tf_lie", "tf_fiction", "tf_error"],
                        ["instruct", "tf_truth"])}[fam]
    return build_variant(model, layers, {"lie": avg(f), "instruct": avg(t)},
                         "score_diff_instruct", rho, ratio, device)


def calibrate_ratio(model, layers, dn, share):
    """r such that weights in channels with c_j > 0 (nbar_F / nbar_T > 1/r) are `share` of all
    weights. With both sides under the same instruction, r = 1 selects channels by ABSOLUTE
    difference, which favours very high-norm channels and deleted large, load-bearing weights
    (Llama-3.1-8B perplexity 33x-145000x, 2026-10-01). A relative bar, set to give a positive share
    like the current masks' (0.05-0.8%), keeps the edit comparable."""
    import torch
    from did_estimators import get_module
    ratios, rows = [], []
    for ly in layers:
        q = dn["F"][ly].float() / dn["T"][ly].float().clamp(min=1e-8)
        ratios.append(q)
        rows.append(torch.full_like(q, get_module(model, ly).weight.shape[0]))
    ratios, rows = torch.cat(ratios), torch.cat(rows)
    order = ratios.argsort(descending=True)
    cum = rows[order].cumsum(0)
    k = int(torch.searchsorted(cum, share * rows.sum() * 1.0).clamp(max=len(order) - 1))
    t = float(ratios[order][k])
    return 1.0 / max(t, 1e-6)


def design_mask_spec(model, layers, dn, rho, rspec, device):
    """rspec: a number (r) or 'a<share>' (r calibrated so positive-score weights are `share`)."""
    if isinstance(rspec, str) and rspec.startswith("a"):
        r = calibrate_ratio(model, layers, dn, float(rspec[1:]))
        print(f"    calibrated r = {r:.4f} (1/r = {1/r:.4f}) for positive share {rspec[1:]}", flush=True)
    else:
        r = float(rspec)
    return design_mask(model, layers, dn, rho, r, device)
