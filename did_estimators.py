#!/usr/bin/env python3
"""
Difference-in-differences importance estimators over a factorial design.

What the repo calls DID is a single difference. The score

    S(w) = mean_p |W_ij| * ||x_j^(p)||

is computed under a lie instruction and under a truth instruction, and the two
are subtracted. Both arms are matched on instruction presence, so the subtraction
removes the "any instruction" confound, but it is one difference between two
conditions, not a difference of differences.

The obvious repair does nothing. Adding a no-instruction baseline and taking
(S_lie - S_neutral) - (S_truth - S_neutral) returns S_lie - S_truth exactly,
because both arms subtract the same term. A second difference only adds
information when the baseline differs between arms. This module provides two
that do.

  did_position   within each arm, the shift from reading the instruction to
                 producing the answer:
                     [S_lie(answer)  - S_lie(prompt)]
                   - [S_truth(answer) - S_truth(prompt)]
                 The prompt-position terms carry whatever the model does while
                 reading the words "lie" and "truthfully", which the current
                 estimator cannot separate from deceptive processing. This is
                 the confound that surfaced as the activation-steering baseline
                 answering the literal token "LIE".

  did_behavior   within each arm, the shift from telling the truth to stating a
                 falsehood, on the model's own generations:
                     [S_lie(false)   - S_lie(true)]
                   - [S_truth(false) - S_truth(true)]
                 The truth arm's term is "weights active when the model is
                 simply wrong". Subtracting it separates deception from error,
                 which is the confound the whole pipeline runs on, since its
                 outcome measure is a wrong answer either way.

## Why column norms are enough

The Wanda score is separable: S_ij^(p) = |W_ij| * n_j^(p), where n_j^(p) is the
L2 norm of input activations into column j over the scored positions of prompt p.
The weight magnitude does not depend on the prompt. So every prompt-level
statistic, and every contrast between cells of the design, is a function of the
column norms alone, and the per-weight score is that column statistic times
|W_ij|. Storing an (n_items x in_features) matrix per layer costs megabytes,
against ~5 GB for one full score tensor, and it keeps per-prompt resolution that
the summed estimator throws away.

## Why the t-statistic is not the answer, but standardization is

Ranking weights by a t-statistic looks like the natural fix for an estimator that
sums over prompts and is therefore dominated by the longest one. It cannot work
as stated: since S_ij^(p) = |W_ij| n_j^(p), the |W_ij| cancels from numerator and
denominator, so the per-weight t equals the per-column t and the ranking selects
whole input columns with no weight-level resolution at all. Wanda's per-weight
selectivity comes entirely from |W|.

The usable form keeps the magnitude and standardizes only the activation side:

    score_ij = |W_ij| * (mean_lie(n_j) - mean_truth(n_j)) / sd_pooled(n_j)

a per-column effect size times the weight magnitude. It answers the same
complaint (one long prompt can no longer dominate) without discarding |W|, and
it is scale-free, so it is also a candidate for retiring the hand-tuned `ratio`
knob, whose job is to set a selectivity threshold the raw difference cannot set
for itself.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

EPS = 1e-8


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------

def target_layers(model, skip=("lm_head", "embed_tokens")) -> List[str]:
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name == s or name.endswith("." + s) for s in skip):
            continue
        names.append(name)
    return names


def get_module(model, name: str) -> nn.Linear:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


@torch.no_grad()
def collect_column_norms(model, tok, items, layer_names, device,
                         position="prompt", batch_size=8, max_length=256,
                         desc="") -> Dict[str, torch.Tensor]:
    """Per-item L2 norm of the input activations into each linear layer column.

    items: list of dicts with "prompt" and, when position != "prompt", "answer".
    position: "prompt" scores the prompt tokens only, "answer" the continuation
              only, "all" both. Padding is never scored.

    Returns {layer_name: tensor (n_items, in_features)} on CPU.
    """
    n = len(items)
    store = {name: torch.zeros(n, get_module(model, name).in_features,
                               dtype=torch.float32)
             for name in layer_names}
    state = {"mask": None, "slice": None}

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach().float()                 # (B, T, in)
            m = state["mask"].unsqueeze(-1)             # (B, T, 1)
            sq = (x * x * m).sum(dim=1)                 # (B, in)
            store[name][state["slice"]] = sq.sqrt().cpu()
        return hook

    handles = [get_module(model, n_).register_forward_hook(make_hook(n_))
               for n_ in layer_names]
    # Generation leaves the tokenizer left-padded, which would shift every
    # sequence by its pad count and make the prompt/answer split wrong.
    prev_side, tok.padding_side = tok.padding_side, "right"
    try:
        for s in range(0, n, batch_size):
            batch = items[s:s + batch_size]
            texts, starts = [], []
            for it in batch:
                p_ids = tok(it["prompt"], add_special_tokens=False)["input_ids"]
                starts.append(len(p_ids[:max_length]))
                if position == "prompt" or not it.get("answer"):
                    texts.append(it["prompt"])
                else:
                    texts.append(it["prompt"] + "\n" + it["answer"])
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_length, add_special_tokens=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            attn = enc["attention_mask"].float()
            T = attn.shape[1]
            pos = torch.arange(T, device=device).unsqueeze(0)
            start = torch.tensor(starts, device=device).unsqueeze(1)
            if position == "prompt":
                sel = (pos < start).float()
            elif position == "answer":
                sel = (pos >= start).float()
            else:
                sel = torch.ones_like(attn)
            state["mask"] = sel * attn
            # An item whose answer was truncated away would contribute a zero
            # column; keep it out of the statistics rather than scoring a zero.
            state["slice"] = slice(s, s + len(batch))
            model(**enc)
            del enc
    finally:
        for h in handles:
            h.remove()
        tok.padding_side = prev_side
    if desc:
        print(f"  [{desc}] column norms for {n} items, "
              f"{len(layer_names)} layers", flush=True)
    return store


# ---------------------------------------------------------------------------
# Cell statistics and contrasts
# ---------------------------------------------------------------------------

def cell_mean_sd(cols: Dict[str, torch.Tensor]):
    """-> ({layer: mean over items}, {layer: sd over items}, n_items)."""
    mean, sd, n = {}, {}, 0
    for name, t in cols.items():
        keep = t.abs().sum(dim=1) > 0          # drop empty / truncated items
        v = t[keep] if keep.any() else t
        n = int(keep.sum().item()) if keep.any() else t.shape[0]
        mean[name] = v.mean(dim=0)
        sd[name] = v.std(dim=0, unbiased=True) if v.shape[0] > 1 \
            else torch.zeros_like(mean[name])
    return mean, sd, n


def contrast(cells: Dict[str, dict], spec: str, ratio: float = 0.37,
             standardize: bool = False) -> Dict[str, torch.Tensor]:
    """Per-layer column statistic for one estimator.

    cells maps a cell name to {"mean": {...}, "sd": {...}, "n": int}.
    spec is one of: single, did_position, did_behavior.
    """
    def m(cell, layer):
        return cells[cell]["mean"][layer]

    def pooled_sd(cell_list, layer):
        var = sum(cells[c]["sd"][layer] ** 2 for c in cell_list) / len(cell_list)
        return var.sqrt() + EPS

    out = {}
    layers = list(next(iter(cells.values()))["mean"].keys())
    v = ratio / (1.0 + ratio) if ratio > 0 else 0.0

    for layer in layers:
        if spec == "single":
            # The repo's estimator: a ratio-weighted single difference.
            stat = v * m("lie_prompt", layer) - (1 - v) * m("truth_prompt", layer)
            norm_cells = ["lie_prompt", "truth_prompt"]
        elif spec == "did_position":
            stat = ((m("lie_answer", layer) - m("lie_prompt", layer))
                    - (m("truth_answer", layer) - m("truth_prompt", layer)))
            norm_cells = ["lie_answer", "lie_prompt", "truth_answer", "truth_prompt"]
        elif spec == "did_behavior":
            stat = ((m("lie_false", layer) - m("lie_true", layer))
                    - (m("truth_false", layer) - m("truth_true", layer)))
            norm_cells = ["lie_false", "lie_true", "truth_false", "truth_true"]
        else:
            raise ValueError(f"unknown spec {spec!r}")

        if standardize:
            # Effect size on the activation side only; |W| is reattached in
            # mask_from_columns, so weight-level resolution is preserved.
            stat = stat / pooled_sd(norm_cells, layer)
        out[layer] = stat
    return out


# ---------------------------------------------------------------------------
# Mask construction
# ---------------------------------------------------------------------------

@torch.no_grad()
def mask_from_columns(model, layer_names, colstat, rho, device,
                      reverse=False) -> Dict[str, torch.Tensor]:
    """Top-rho per layer of |W_ij| * colstat_j, the separable Wanda score."""
    masks = {}
    for name in layer_names:
        W = get_module(model, name).weight
        c = colstat[name].to(device, torch.float32)
        score = W.detach().float().abs() * c.unsqueeze(0)
        if reverse:
            score = -score
        flat = score.view(-1)
        k = max(1, int(rho * flat.numel()))
        idx = flat.topk(k).indices
        m = torch.zeros(flat.numel(), dtype=torch.bool, device=device)
        m[idx] = True
        masks[name] = m.view(W.shape).cpu()
        del score, flat, idx, m
    return masks
