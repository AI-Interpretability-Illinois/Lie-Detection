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

# Vision-language checkpoints (Gemma-3 4b/12b, for instance) carry a vision tower
# and a projector alongside the language model. Steering those is meaningless for
# a text task and they must not enter the mask.
NON_TEXT = ("vision_tower", "multi_modal_projector", "vision_model",
            "visual", "image_encoder")


def decoder_layers(model):
    """The decoder block list, wherever this architecture keeps it.

    Text-only models expose model.model.layers; Gemma-3's conditional-generation
    wrapper puts the text stack under .language_model, and the exact nesting has
    moved between transformers versions, so the candidates are tried in order.
    """
    for path in ("model.layers", "language_model.model.layers",
                 "language_model.layers", "model.language_model.layers",
                 "transformer.h", "model.decoder.layers"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList) and len(obj):
            return obj
    raise AttributeError(
        "could not locate the decoder layer list; checked model.layers, "
        "language_model[.model].layers, model.language_model.layers, "
        "transformer.h, model.decoder.layers")


def target_layers(model, skip=("lm_head", "embed_tokens")) -> List[str]:
    names = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if any(name == s or name.endswith("." + s) for s in skip):
            continue
        if any(part in name for part in NON_TEXT):
            continue
        names.append(name)
    return names


def checkpoint_dtype(model_id, cache_dir, default=torch.float16):
    """The dtype the checkpoint was released in.

    This matters more than it looks: Gemma is trained and published in bfloat16
    and overflows in float16. Loaded as fp16, gemma-3-4b-it emits nothing but pad
    tokens, because the forward pass has already gone non-finite. Reading the
    dtype off the config avoids guessing per family.
    """
    from transformers import AutoConfig
    try:
        cfg = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    except Exception:
        return default
    for obj in (getattr(cfg, "text_config", None), cfg):
        d = getattr(obj, "torch_dtype", None) or getattr(obj, "dtype", None)
        if isinstance(d, torch.dtype):
            return d
        if isinstance(d, str) and hasattr(torch, d):
            return getattr(torch, d)
    return default


def _cap_gpu_memory(device):
    """Honour LIE_MAX_GPU_GB: cap this process's GPU memory on a shared node.

    Past the cap this process raises an out-of-memory error itself, instead of
    growing into memory another user's training needs. Unset means no cap.
    """
    import os
    cap = os.environ.get("LIE_MAX_GPU_GB")
    if not cap or not torch.cuda.is_available():
        return
    dev = torch.device(device)
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(idx).total_memory / 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) / total), idx)
    print(f"  [load_lm] GPU memory capped at {float(cap):.1f} GB of {total:.1f}", flush=True)


def load_lm(model_id, cache_dir, device, dtype=None):
    """Load a causal LM, falling back to the image-text wrapper.

    Gemma-3 4b and 12b ship as Gemma3ForConditionalGeneration, which
    AutoModelForCausalLM refuses; their text stack is reached through
    decoder_layers(). The 1b checkpoint is a plain Gemma3ForCausalLM.
    dtype=None takes the checkpoint's own dtype.
    """
    _cap_gpu_memory(device)
    from transformers import AutoModelForCausalLM
    if dtype is None:
        dtype = checkpoint_dtype(model_id, cache_dir)
        print(f"  [load_lm] dtype={dtype}", flush=True)
    import os as _os
    if _os.environ.get("LIE_DEVICE_MAP") == "auto":
        return _multi_gpu(model_id, cache_dir, dtype)
    try:
        m = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, cache_dir=cache_dir)
    except (ValueError, KeyError, OSError) as first:
        try:
            from transformers import AutoModelForImageTextToText as Alt
        except ImportError as exc:                     # very old transformers
            raise first from exc
        m = Alt.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir)
        print(f"  [load_lm] {type(m).__name__} via AutoModelForImageTextToText",
              flush=True)
    return m.to(device).eval()


def _multi_gpu(model_id, cache_dir, dtype):
    """LIE_DEVICE_MAP=auto: shard a model that does not fit one card over the visible GPUs (OLMo-2-32B on two
    A6000s). Inputs still go to cuda:0, where the embedding lands; hooks, masks and the belief probes work per
    module device, and .to(device) is skipped."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, cache_dir=cache_dir, device_map="auto")
    print(f"  [load_lm] device_map=auto over {torch.cuda.device_count()} GPUs: "
          f"{sorted(set(str(d) for d in m.hf_device_map.values()))}", flush=True)
    return m.eval()


def get_module(model, name: str) -> nn.Linear:
    mod = model
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


@torch.no_grad()
def collect_column_norms(model, tok, items, layer_names, device,
                         position="prompt", batch_size=8, max_length=256,
                         desc="", rms=False) -> Dict[str, torch.Tensor]:
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
            if rms:                                     # per-token RMS: answers of different
                sq = sq / m.sum(dim=1).clamp(min=1)     # lengths score on the same scale
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
                    # "sep": "" when the prompt already ends with a chat template's
                    # assistant header, so the answer follows it directly
                    texts.append(it["prompt"] + it.get("sep", "\n") + it["answer"])
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
