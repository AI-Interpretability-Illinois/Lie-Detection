#!/usr/bin/env python3
"""
Why the second difference breaks the model where the single difference does not.

run_real_did.py found that both genuine DID estimators have no working damping
configuration: they destroy the model before they reduce lying, at rho 0.04 and
at rho 0.01. This asks what their masks select that the single difference's mask
does not, using no behavioural evaluation at all.

The hypothesis it tests: the ratio weighting makes the single difference's column
statistic mostly negative, so its top-rho lands on weights in columns with LOW
truth-importance, which are intrinsically safe to shrink. A difference of
differences is centred near zero with both signs, so its top-rho lands on the
largest |W| inside the most positive columns, which are load-bearing. If that is
right, the DID masks should show a higher median |W| than the single mask and
little overlap with it.

Usage
-----
  python diagnose_did_masks.py --model-id Qwen/Qwen2.5-1.5B-Instruct --rho 0.04
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate import EVAL_SYSTEM_PROMPT, _is_correct
from did_estimators import (cell_mean_sd, collect_column_norms, contrast,
                            get_module, mask_from_columns, target_layers)
from run_reversed_contrast import SanitizeLogits, generate_batched
from run_real_did import ESTIMATORS, question_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--k-samples", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--score-batch", type=int, default=8)
    args = ap.parse_args()

    device = args.device
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)
    model.eval()
    layers = target_layers(model)
    guard = SanitizeLogits()

    pairs = question_gold()[:args.n_truth]
    gens = {}
    for key in ("lie", "truth"):
        chats = [[{"role": "system", "content": EVAL_SYSTEM_PROMPT},
                  {"role": "user", "content": p[key]}]
                 for p in pairs for _ in range(args.k_samples)]
        torch.manual_seed(0)
        resp = generate_batched(model, tok, chats, device, 40, 1.0,
                               args.batch_size, guard=guard)
        gens[key] = [{"prompt": p[key], "answer": r,
                      "correct": _is_correct(r, p["gold"])}
                     for p, r in zip([p for p in pairs
                                      for _ in range(args.k_samples)], resp)]
        print(f"  {key}: {len(gens[key])} generations", flush=True)

    cell_items = {
        "lie_prompt":   [{"prompt": p["lie"]} for p in pairs],
        "truth_prompt": [{"prompt": p["truth"]} for p in pairs],
        "lie_answer":   gens["lie"],
        "truth_answer": gens["truth"],
        "lie_false":    [i for i in gens["lie"] if not i["correct"]],
        "lie_true":     [i for i in gens["lie"] if i["correct"]],
        "truth_false":  [i for i in gens["truth"] if not i["correct"]],
        "truth_true":   [i for i in gens["truth"] if i["correct"]],
    }
    positions = {"lie_prompt": "prompt", "truth_prompt": "prompt"}
    cells = {}
    for name, items in cell_items.items():
        if len(items) < 2:
            continue
        cols = collect_column_norms(model, tok, items, layers, device,
                                    position=positions.get(name, "answer"),
                                    batch_size=args.score_batch)
        mean, sd, n = cell_mean_sd(cols)
        cells[name] = {"mean": mean, "sd": sd, "n": n}
        del cols
    print(f"cells: {', '.join(cells)}\n", flush=True)

    # Reference |W| statistics over the maskable weights.
    with torch.no_grad():
        all_med = torch.cat([
            get_module(model, n).weight.detach().float().abs().view(-1)[::997].cpu()
            for n in layers]).median().item()

    masks, stats = {}, []
    for label, kw in ESTIMATORS:
        need = {"single": ["lie_prompt", "truth_prompt"],
                "did_position": ["lie_prompt", "truth_prompt", "lie_answer",
                                 "truth_answer"],
                "did_behavior": ["lie_false", "lie_true", "truth_false",
                                 "truth_true"]}[kw["spec"]]
        if any(c not in cells for c in need):
            continue
        colstat = contrast(cells, ratio=args.ratio, **kw)
        pos_frac = (sum((colstat[l] > 0).sum().item() for l in layers)
                    / sum(colstat[l].numel() for l in layers))
        m = mask_from_columns(model, layers, colstat, args.rho, device)
        masks[label] = m

        with torch.no_grad():
            sel, attn_sel, tot_sel = [], 0, 0
            for n in layers:
                W = get_module(model, n).weight.detach().float().abs()
                s = W[m[n].to(W.device)]
                sel.append(s[::max(1, s.numel() // 20000)].cpu())
                k = int(m[n].sum().item())
                tot_sel += k
                if "self_attn" in n:
                    attn_sel += k
            med_sel = torch.cat(sel).median().item()
        stats.append({"label": label, "pos_frac": pos_frac,
                      "med_ratio": med_sel / all_med,
                      "attn_share": attn_sel / max(1, tot_sel)})
        print(f"  {label:<16} built", flush=True)

    ref = masks.get("single")
    print(f"\nmedian |W| over all maskable weights: {all_med:.5f}\n")
    print(f"{'estimator':<16}{'cols with stat>0':>18}{'median |W| vs all':>20}"
          f"{'attn share':>12}{'Jaccard vs single':>19}")
    print("-" * 85)
    for s in stats:
        j = ""
        if ref is not None:
            inter = union = 0
            for n in layers:
                a, b = masks[s["label"]][n], ref[n]
                inter += int((a & b).sum().item())
                union += int((a | b).sum().item())
            j = f"{inter/max(1,union):.3f}"
        print(f"{s['label']:<16}{s['pos_frac']:>17.1%}{s['med_ratio']:>20.2f}"
              f"{s['attn_share']:>12.1%}{j:>19}")


if __name__ == "__main__":
    main()
