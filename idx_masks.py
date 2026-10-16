#!/usr/bin/env python3
"""
Weight masks carried as int32 index sets instead of boolean tensors.

A boolean mask over a model's maskable weights costs one byte per weight, so on
Llama-3.1-8B each mask is ~6.5 GB of host RAM. Scripts that compare many mask
families hold several at once: the stratified second-difference run builds nine
(four contrasts x two ranking modes, plus the magnitude-matched control) and was
OOM-killed at 64 GB before it produced a single steered row.

At rho = 0.04 only 4% of the entries are selected, so the same selection stored as
per-layer int32 indices costs ~1 GB. Rollback gets cheaper too: instead of copying
whole layers, only the selected entries are saved and written back, which is ~0.5 GB
in the model's own dtype rather than a full duplicate of every touched matrix.

The round trip is exact. apply_idx followed by restore_idx returns bitwise-identical
weights, verified on Qwen2.5-0.5B by smoke_idx.py.

Every function takes and returns {layer_name: int32 tensor of flat indices}.
"""

from __future__ import annotations

import torch

from did_estimators import get_module


@torch.no_grad()
def idx_topk(model, layers, colstat, rho, device, wpct=None):
    """Per layer, the indices of the top-rho weights by the ranking rule.

    wpct=None ranks |W_ij| * colstat_j, the separable Wanda score, which is the
    method's own rule. Otherwise it ranks colstat_j alone among the weights below
    the wpct-th percentile of |W| within that layer, which holds magnitude roughly
    fixed so that the contrast rather than the magnitude does the choosing.
    """
    out = {}
    for name in layers:
        W = get_module(model, name).weight
        n = W.numel()
        k = max(1, int(rho * n))
        Wa = W.detach().float().abs().to(device)
        col = colstat[name].to(device)
        if wpct is None:
            s = (Wa * col.unsqueeze(0)).view(-1)
            idx = s.topk(k).indices
            del s
        else:
            flat = Wa.view(-1)
            g = torch.Generator(device=device).manual_seed(12345)
            sample = flat[torch.randperm(n, generator=g, device=device)[:min(n, 200_000)]]
            thresh = torch.quantile(sample, wpct / 100.0)
            eligible = (flat <= thresh).nonzero(as_tuple=True)[0]
            if eligible.numel() <= k:        # stratum smaller than the budget
                idx = eligible
            else:
                s_el = col.unsqueeze(0).expand_as(Wa).reshape(-1)[eligible]
                idx = eligible[s_el.topk(k).indices]
                del s_el
            del flat, eligible
        out[name] = idx.to(torch.int32).cpu()
        del Wa, col, idx
    return out


@torch.no_grad()
def idx_magnitude_matched(model, ref, device, seed=0):
    """Random index set reproducing ref's per-layer |W| decile histogram.

    Decile matching is not exact matching: in practice this lands at roughly 2.3x
    the real mask's median |W|, so it is a strong null rather than a perfect one.
    """
    out = {}
    for li, (name, idx) in enumerate(ref.items()):
        W = get_module(model, name).weight.detach().float().abs().view(-1).to(device)
        n = W.numel()
        g = torch.Generator(device=device).manual_seed(seed * 100003 + li)
        sample = W[torch.randperm(n, generator=g, device=device)[:min(n, 100_000)]]
        qs = torch.quantile(sample, torch.linspace(0, 1, 11, device=device))
        qs[0], qs[-1] = -float("inf"), float("inf")
        bins = torch.bucketize(W, qs[1:-1].contiguous())
        want = torch.bincount(bins[idx.to(device).long()], minlength=10)
        picks = []
        for b in range(10):
            pool = (bins == b).nonzero(as_tuple=True)[0]
            take = int(min(int(want[b].item()), pool.numel()))
            if take:
                sel = torch.randperm(pool.numel(), generator=g, device=device)[:take]
                picks.append(pool[sel])
        out[name] = (torch.cat(picks) if picks
                     else torch.empty(0, dtype=torch.long, device=device)
                     ).to(torch.int32).cpu()
        del W, bins
    return out


@torch.no_grad()
def idx_uniform_random(model, layers, rho, device, seed=0):
    """Uniform random index set per layer, same budget. Not a magnitude control."""
    out = {}
    for li, name in enumerate(layers):
        n = get_module(model, name).weight.numel()
        k = max(1, int(rho * n))
        g = torch.Generator(device=device).manual_seed(seed * 100003 + li)
        out[name] = torch.randperm(n, generator=g,
                                   device=device)[:k].to(torch.int32).cpu()
    return out


@torch.no_grad()
def apply_idx(model, idxs, alpha, saved):
    """W[i] *= (1 + alpha) on the selected entries; stash the old values in saved."""
    for name, idx in idxs.items():
        flat = get_module(model, name).weight.data.view(-1)
        i = idx.to(flat.device).long()
        saved[name] = flat[i].detach().to("cpu", copy=True)
        flat[i] = flat[i] * (1.0 + alpha)
        del i


@torch.no_grad()
def restore_idx(model, idxs, saved):
    """Undo apply_idx exactly. Clears saved."""
    for name, vals in saved.items():
        flat = get_module(model, name).weight.data.view(-1)
        i = idxs[name].to(flat.device).long()
        flat[i] = vals.to(flat.device)
        del i
    saved.clear()


@torch.no_grad()
def idx_diag(model, idxs, ref=None):
    """(median |W| of the selection / median |W| overall, Jaccard against ref).

    Both selections have the same per-layer budget, so a Jaccard of J means they
    share 2J/(1+J) of their weights.
    """
    sel, allw = [], []
    inter = union = 0
    for name, idx in idxs.items():
        W = get_module(model, name).weight.detach().float().abs().view(-1)
        i = idx.to(W.device).long()
        s = W[i]
        if s.numel():
            sel.append(s[:: max(1, s.numel() // 20000)].cpu())
        allw.append(W[::997].cpu())
        if ref is not None:
            r = ref[name].to(W.device)
            hit = int(torch.isin(idx.to(W.device), r).sum().item())
            inter += hit
            union += idx.numel() + r.numel() - hit
        del W, i, s
    rel = (torch.cat(sel).median().item() / torch.cat(allw).median().item()
           if sel else float("nan"))
    return rel, (inter / max(1, union) if ref is not None else float("nan"))
