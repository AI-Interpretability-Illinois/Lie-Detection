#!/usr/bin/env python3
"""
Reversed-contrast control for LieNeuro-DID weight steering.

The method ranks every weight by a difference-in-differences importance score

    DID(w) = v * mean(S_lie)(w) - (1 - v) * mean(S_truth)(w),   v = ratio/(1+ratio)

keeps the top-rho per layer, and amplifies those weights by (1 + alpha).  The
claim attached to it is that the selected weights are *deception-specific*.
This script tests that claim with three matched-cost control masks, all built
from the same two score tensors, all with the same per-layer element count, and
all amplified with the same alpha:

  forward   top-rho by DID                         -- the method
  reverse   top-rho by v*S_truth - (1-v)*S_lie     -- arms swapped, same
                                                     selectivity threshold
  antitop   bottom-rho by DID                      -- the least lie-specific
                                                     weights in the layer
  random    uniform random per-layer, same k       -- pure damage control
  magmatch  random, stratified to reproduce the    -- damage control that also
            forward mask's |W| decile histogram       matches weight magnitude
  none      unsteered

If `reverse` (or `random`) raises the wrong-answer rate as much as `forward`
does at the same perplexity cost, the intervention is amplifying whatever is
contrast-extreme or simply fragile, not a deception circuit.

Metrics per arm: wrong-answer rate on held-out factual questions (greedy plus
sampled reps), how many of those wrong answers are fluent false claims rather
than broken text, and WikiText-2 perplexity as the damage axis.

Fluency is judged two ways, neither of which needs a second model.  A cheap junk
test catches symbol runs, single-token collapse and script drift.  Every response
is then scored under the *unsteered* weights (restored before scoring, so the
measurement is never circular), and a response counts as fluent only if its mean
token NLL is at or below the 95th percentile of the unsteered arm's own
responses.  A wrong answer that passes both is a fluent falsehood; one that fails
either is damage being counted as deception.

Usage
-----
  python run_reversed_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
      --rho 0.04 --ratio 0.37 --alphas 2,3,4,5 --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from pathlib import Path

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessor,
                          LogitsProcessorList)

from src.lie_neuro import LieNeuro, LieNeuroConfig
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT
from src.evaluate import FACTUAL_QA, EVAL_SYSTEM_PROMPT, _is_correct
from src.factual_qa_ext import FACTUAL_QA_EXT
from score_liars_bench import degenerate

# Questions whose text was used to build the scores are excluded, so every
# number below is on held-out questions.
_SEEN = {p.split("Answer truthfully: ", 1)[-1].strip().lower()
         for p in DIRECT_TRUTH_INSTRUCT}
OOD_QA = [qa for qa in FACTUAL_QA if qa["question"].strip().lower() not in _SEEN]
EVAL_QA = ([dict(qa, set="ood25") for qa in OOD_QA] +
           [dict(qa, set="ext75") for qa in FACTUAL_QA_EXT])

ARMS = ["forward", "reverse", "antitop", "random", "magmatch"]

WT_WINDOW, WT_STRIDE, WT_MAX_TOKS = 512, 256, 8192


# ---------------------------------------------------------------------------
# Mask construction — one score pair, six ways of selecting from it
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_mask(ln, arm, rho, ratio, device, n_prompts, seed=0):
    """Write ln._mask in place.  Every arm selects exactly k = rho*numel per layer.

    The forward branch reproduces LieNeuro.build_mask(mode="score_diff") exactly
    (same DID weighting, same per-layer k, same topk index selection), moved onto
    the GPU because the per-layer topk on CPU dominates the runtime.
    """
    v = ratio / (1.0 + ratio) if ratio > 0 else 0.0
    inv = 1.0 / n_prompts if n_prompts > 0 else 1.0
    ln._mask = {}

    for li, name in enumerate(ln._layer_names):
        s_lie = ln._S_lie[name].to(device, torch.float32, non_blocking=True)
        s_tru = ln._S_truth[name].to(device, torch.float32, non_blocking=True)
        shape = s_lie.shape
        n = s_lie.numel()
        k = max(1, int(rho * n))

        if arm == "forward":
            score = (inv * (v * s_lie - (1.0 - v) * s_tru)).view(-1)
            idx = score.topk(k).indices
        elif arm == "reverse":
            # Same estimator with the two arms exchanged: selects weights that
            # matter that much more to truth-telling than to lying.
            score = (inv * (v * s_tru - (1.0 - v) * s_lie)).view(-1)
            idx = score.topk(k).indices
        elif arm == "antitop":
            score = (inv * (v * s_lie - (1.0 - v) * s_tru)).view(-1)
            idx = score.topk(k, largest=False).indices
        elif arm == "random":
            g = torch.Generator(device=device).manual_seed(seed * 100003 + li)
            idx = torch.randperm(n, generator=g, device=device)[:k]
        elif arm == "magmatch":
            # Same count, drawn to reproduce the forward mask's |W| decile
            # histogram, so this control is matched on weight magnitude as well
            # as on element count.
            fwd = (inv * (v * s_lie - (1.0 - v) * s_tru)).view(-1).topk(k).indices
            w_abs = ln._get_module(name).weight.detach().float().abs().view(-1)
            qs = torch.quantile(
                w_abs[torch.randperm(n, device=device)[:min(n, 100_000)]],
                torch.linspace(0, 1, 11, device=device))
            qs[0], qs[-1] = -float("inf"), float("inf")
            bins_all = torch.bucketize(w_abs, qs[1:-1].contiguous())
            want = torch.bincount(bins_all[fwd], minlength=10)
            g = torch.Generator(device=device).manual_seed(seed * 100003 + li)
            picks = []
            for b in range(10):
                pool = (bins_all == b).nonzero(as_tuple=True)[0]
                take = int(min(int(want[b].item()), pool.numel()))
                if take:
                    sel = torch.randperm(pool.numel(), generator=g, device=device)[:take]
                    picks.append(pool[sel])
            idx = torch.cat(picks) if picks else fwd[:0]
            if idx.numel() < k:   # a decile ran dry; top up at random
                rest = torch.randperm(n, generator=g, device=device)
                rest = rest[~torch.isin(rest, idx)][:k - idx.numel()]
                idx = torch.cat([idx, rest])
            del w_abs, bins_all, fwd
        else:
            raise ValueError(f"unknown arm {arm!r}")

        m = torch.zeros(n, dtype=torch.bool, device=device)
        m[idx] = True
        ln._mask[name] = m.view(shape)
        del s_lie, s_tru, m, idx
    return ln._mask


@torch.no_grad()
def apply_steering(ln, alpha, originals):
    """W' = W * (1 + alpha * mask).  Originals are kept on CPU, not on the GPU."""
    for name in ln._layer_names:
        mask = ln._mask.get(name)
        if mask is None:
            continue
        mod = ln._get_module(name)
        if name not in originals:
            originals[name] = mod.weight.detach().to("cpu", copy=True)
        mod.weight.data.mul_(1.0 + alpha * mask.to(mod.weight.device).float())


@torch.no_grad()
def restore(ln, originals):
    for name, w in originals.items():
        ln._get_module(name).weight.data.copy_(w.to(ln._get_module(name).weight.device))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

_RUN_RE = re.compile(r"(.)\1{7,}")


def junk(resp: str) -> bool:
    """Non-answer detector; delegates to grading.is_junk.

    The original version here rejected any response under half ASCII letters,
    which flagged every bare number ('1912.', '3.14', '144') as junk and dropped
    all 17 numeric-answer questions from both the honest and the falsehood count.
    Fixed 2026-09-27; see grading.py.
    """
    from grading import is_junk
    return is_junk(resp)


def response_nlls(model, tok, recs, device, batch_size):
    """Mean token NLL of each response under the CURRENT weights.

    Called after restore(), so responses are scored by the unsteered model: a
    steered model that confidently emits nonsense cannot certify its own output.
    """
    out = []
    for s in range(0, len(recs), batch_size):
        batch = recs[s:s + batch_size]
        ids_list, lens = [], []
        for r in batch:
            chat = [{"role": "system", "content": EVAL_SYSTEM_PROMPT},
                    {"role": "user", "content": r["q"]}]
            p_txt = tok.apply_chat_template(chat, tokenize=False,
                                            add_generation_prompt=True)
            p_ids = tok(p_txt, add_special_tokens=False)["input_ids"]
            r_ids = tok(r["resp"] if r["resp"].strip() else tok.eos_token,
                        add_special_tokens=False)["input_ids"][:64]
            ids_list.append(p_ids + r_ids)
            lens.append((len(p_ids), len(r_ids)))
        width = max(len(i) for i in ids_list)
        pad = tok.pad_token_id
        inp = torch.full((len(ids_list), width), pad, dtype=torch.long)
        lab = torch.full((len(ids_list), width), -100, dtype=torch.long)
        att = torch.zeros((len(ids_list), width), dtype=torch.long)
        for i, (ids, (np_, nr)) in enumerate(zip(ids_list, lens)):
            inp[i, :len(ids)] = torch.tensor(ids)
            att[i, :len(ids)] = 1
            if nr:
                lab[i, np_:np_ + nr] = torch.tensor(ids[np_:np_ + nr])
        inp, lab, att = inp.to(device), lab.to(device), att.to(device)
        logits = model(input_ids=inp, attention_mask=att).logits[:, :-1].float()
        tgt = lab[:, 1:]
        nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            ignore_index=-100, reduction="none").view(tgt.shape)
        keep = (tgt != -100).float()
        per = (nll * keep).sum(1) / keep.sum(1).clamp(min=1)
        out.extend(per.cpu().tolist())
    for r, v in zip(recs, out):
        r["resp_nll"] = v if math.isfinite(v) else float("inf")
    return recs


class SanitizeLogits(LogitsProcessor):
    """Keep sampling alive on an arm whose weights have been destroyed.

    A heavily amplified mask can drive fp16 activations to inf, and the resulting
    nan/inf logits make multinomial sampling raise a device-side assert — which
    would abort the sweep on exactly the configurations whose collapse is the
    finding.  Non-finite entries are pushed to the dtype minimum; a row that is
    entirely non-finite becomes a uniform draw rather than a crash.  Counts how
    often it fired so the collapse is reported, not silently patched.
    """

    def __init__(self):
        self.n_fired = 0

    def __call__(self, input_ids, scores):
        bad = ~torch.isfinite(scores)
        if bad.any():
            self.n_fired += int(bad.any(dim=-1).sum().item())
            scores = scores.masked_fill(bad, torch.finfo(scores.dtype).min)
        return scores


@torch.no_grad()
def generate_batched(model, tok, chats, device, max_new, temperature, batch_size,
                     guard=None):
    outs = []
    for s in range(0, len(chats), batch_size):
        batch = chats[s:s + batch_size]
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                 for c in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        gen = model.generate(
            **enc, max_new_tokens=max_new,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0 if temperature > 0 else None,
            top_k=0 if temperature > 0 else None,
            logits_processor=LogitsProcessorList([guard]) if guard else None,
            pad_token_id=tok.pad_token_id)
        for i in range(len(batch)):
            outs.append(tok.decode(gen[i, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip())
    return outs


def eval_questions(model, tok, device, batch_size, reps, temp, seed, guard=None):
    """Greedy pass plus `reps` sampled passes over the held-out questions."""
    chats = [[{"role": "system", "content": EVAL_SYSTEM_PROMPT},
              {"role": "user", "content": it["question"]}] for it in EVAL_QA]
    recs = []
    for tag, t, n in [("greedy", 0.0, 1), ("sampled", temp, reps)]:
        if n == 0:
            continue
        if t > 0:
            torch.manual_seed(seed)
        resp = generate_batched(model, tok, chats * n, device, 50, t, batch_size,
                                guard=guard)
        for it, r in zip(EVAL_QA * n, resp):
            recs.append({"pass": tag, "set": it["set"], "q": it["question"],
                         "gold": it["answer"], "resp": r,
                         "correct": _is_correct(r, it["answer"]),
                         "degenerate": degenerate(r), "junk": junk(r)})
    return recs


def pctl(vals, q):
    v = sorted(x for x in vals if math.isfinite(x))
    if not v:
        return float("inf")
    return v[min(len(v) - 1, int(q * (len(v) - 1)))]


def summarize(recs, nll_cut):
    """nll_cut: fluency bar, the unsteered arm's p95 response NLL."""
    n = len(recs)
    wrong = [r for r in recs if not r["correct"]]
    fluent = [r for r in recs if not r["junk"] and r.get("resp_nll", 0.0) <= nll_cut]
    fluent_wrong = [r for r in wrong
                    if not r["junk"] and r.get("resp_nll", 0.0) <= nll_cut]
    greedy = [r for r in recs if r["pass"] == "greedy"]
    return {
        "n_responses": n,
        "wrong_rate": len(wrong) / n,
        "wrong_rate_greedy": sum(not r["correct"] for r in greedy) / max(1, len(greedy)),
        "fluent_wrong_rate": len(fluent_wrong) / n,
        "fluent_rate": len(fluent) / n,
        "junk_rate": sum(r["junk"] for r in recs) / n,
        "degen_rate": sum(r["degenerate"] for r in recs) / n,
        "median_resp_nll": pctl([r.get("resp_nll", float("nan")) for r in recs], 0.5),
        "mean_len": sum(len(r["resp"].split()) for r in recs) / n,
        "empty_rate": sum(not r["resp"].strip() for r in recs) / n,
    }


def load_wikitext_ids(tok, cache_dir):
    from datasets import load_dataset
    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                          cache_dir=cache_dir)
    except Exception:
        # A cache copied from another host (e.g. to timan) that this datasets version
        # cannot resolve offline: read the stored test split directly.
        import glob
        from datasets import Dataset
        arrow = sorted(glob.glob(f"{cache_dir}/wikitext/wikitext-2-raw-v1/**/wikitext-test.arrow",
                                 recursive=True))
        if arrow:
            ds = Dataset.from_file(arrow[0])
        else:
            ds = load_dataset("EleutherAI/wikitext_document_level", "wikitext-2-raw-v1",
                              split="test", cache_dir=cache_dir)
            ds = ds.rename_column("page", "text") if "page" in ds.column_names else ds
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return tok(text, add_special_tokens=False, truncation=True,
               max_length=WT_MAX_TOKS)["input_ids"]


@torch.no_grad()
def wikitext_ppl(model, ids, device):
    x = torch.tensor(ids, dtype=torch.long)
    nlls, ntok = [], 0
    for begin in range(0, len(x), WT_STRIDE):
        end = min(begin + WT_WINDOW, len(x))
        chunk = x[begin:end].unsqueeze(0).to(device)
        tgt = min(WT_STRIDE, end - begin)
        labels = chunk.clone()
        labels[:, :-tgt] = -100
        loss = model(input_ids=chunk, labels=labels).loss
        if torch.isfinite(loss):
            nlls.append(loss.item() * tgt)
            ntok += tgt
        if end == len(x):
            break
    if not ntok:
        return float("nan")
    mean_nll = sum(nlls) / ntok
    return math.exp(mean_nll) if mean_nll < 700 else float("inf")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--out-dir", default="/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--alphas", default="2,3,4,5")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--control-alphas", default=None,
                    help="Alphas for the control arms (default: same as --alphas).")
    ap.add_argument("--n-truth", type=int, default=25)
    ap.add_argument("--reps", type=int, default=2, help="Sampled passes per config.")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-lm-head", action="store_true",
                    help="Include lm_head/embed in scoring and masking. Unsafe on "
                         "models with tied embeddings, where it also edits the "
                         "input embedding matrix.")
    ap.add_argument("--recompute-scores", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = args.device
    short = args.model_id.split("/")[-1].replace(".", "p")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"reversed_contrast_{short}_rho{args.rho}_ratio{args.ratio}"
    if args.tag:
        stem += f"_{args.tag}"
    csv_path = out_dir / f"{stem}.csv"
    jsonl_path = out_dir / f"{stem}_responses.jsonl"

    alphas = [float(a) for a in args.alphas.split(",")]
    ctrl_alphas = ([float(a) for a in args.control_alphas.split(",")]
                   if args.control_alphas else alphas)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    print(f"model={args.model_id}  rho={args.rho}  ratio={args.ratio}  "
          f"alphas={alphas}  arms={arms}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, torch_dtype=torch.float16, cache_dir=args.cache_dir).to(device)
    model.eval()

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    skip = [] if args.include_lm_head else ["lm_head", "embed_tokens"]
    if args.include_lm_head and tied:
        print("WARNING: tie_word_embeddings=True — editing lm_head also edits the "
              "input embeddings.", flush=True)
    print(f"tie_word_embeddings={tied}  skip_modules={skip}", flush=True)

    cfg = LieNeuroConfig(importance_threshold=args.rho, alpha=1.0, device=device,
                         skip_modules=skip)
    ln = LieNeuro(model, tok, cfg)
    n_masked = sum(ln._get_module(n).weight.numel() for n in ln._layer_names)
    print(f"{len(ln._layer_names)} linear layers, {n_masked/1e9:.2f}B maskable params",
          flush=True)

    # ── importance scores ────────────────────────────────────────────────────
    scores_path = out_dir / f"scores_{short}_ntruth{args.n_truth}_lmhead{int(args.include_lm_head)}.pt"
    if scores_path.exists() and not args.recompute_scores:
        print(f"loading scores from {scores_path}", flush=True)
        saved = torch.load(scores_path, map_location="cpu", weights_only=False)
        ln._S_lie, ln._S_truth = saved["S_lie"], saved["S_truth"]
    else:
        t0 = time.time()
        print(f"computing S_lie over DIRECT_LIE[:{args.n_truth}] …", flush=True)
        ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE[:args.n_truth], "S_lie")
        print(f"computing S_truth over DIRECT_TRUTH_INSTRUCT[:{args.n_truth}] …",
              flush=True)
        ln._S_truth = ln._compute_scores_for_prompts(
            DIRECT_TRUTH_INSTRUCT[:args.n_truth], "S_truth")
        torch.save({"S_lie": ln._S_lie, "S_truth": ln._S_truth,
                    "n_truth": args.n_truth, "model": args.model_id}, scores_path)
        print(f"scores in {time.time()-t0:.0f}s → {scores_path}", flush=True)

    probe = ln._layer_names[0]
    r = (ln._S_lie[probe].float().sum() / ln._S_truth[probe].float().sum()).item()
    print(f"S_lie/S_truth on {probe}: {r:.4f}", flush=True)

    wt_ids = load_wikitext_ids(tok, args.cache_dir)

    # ── baseline ─────────────────────────────────────────────────────────────
    rows, all_recs = [], []
    print("\n── none (unsteered) ──", flush=True)
    t0 = time.time()
    guard = SanitizeLogits()
    base_recs = eval_questions(model, tok, device, args.batch_size, args.reps,
                               args.temp, args.seed, guard=guard)
    response_nlls(model, tok, base_recs, device, 32)
    # The fluency bar is set once, by the unsteered model on its own answers, and
    # then held fixed for every steered arm.
    nll_cut = pctl([r["resp_nll"] for r in base_recs], 0.95)
    base = summarize(base_recs, nll_cut)
    base_ppl = wikitext_ppl(model, wt_ids, device)
    base.update(arm="none", alpha=0.0, rho=args.rho, ratio=args.ratio,
                mask_coverage=0.0, wikitext_ppl=base_ppl, ppl_ratio=1.0,
                delta_wrong=0.0, delta_fluent_wrong=0.0, nll_cut=nll_cut,
                nonfinite_logit_rows=guard.n_fired, secs=time.time() - t0)
    rows.append(base)
    for rec in base_recs:
        all_recs.append(dict(rec, arm="none", alpha=0.0))
    print(f"  wrong={base['wrong_rate']:.1%}  fluent_wrong={base['fluent_wrong_rate']:.1%}"
          f"  junk={base['junk_rate']:.1%}  ppl={base_ppl:.2f}"
          f"  fluency bar (p95 NLL)={nll_cut:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    # ── steered arms ─────────────────────────────────────────────────────────
    originals = {}
    for arm in arms:
        print(f"\n── {arm}: building mask ──", flush=True)
        tm = time.time()
        build_mask(ln, arm, args.rho, args.ratio, device, args.n_truth, args.seed)
        cov = (sum(m.sum().item() for m in ln._mask.values()) /
               sum(m.numel() for m in ln._mask.values()))
        print(f"  coverage={cov:.4%}  ({time.time()-tm:.0f}s)", flush=True)

        for alpha in (alphas if arm in ("forward", "reverse") else ctrl_alphas):
            t0 = time.time()
            apply_steering(ln, alpha, originals)
            guard.n_fired = 0
            recs = eval_questions(model, tok, device, args.batch_size, args.reps,
                                  args.temp, args.seed, guard=guard)
            n_fired = guard.n_fired
            ppl = wikitext_ppl(model, wt_ids, device)
            restore(ln, originals)
            response_nlls(model, tok, recs, device, 32)   # unsteered weights

            s = summarize(recs, nll_cut)
            s.update(arm=arm, alpha=alpha, rho=args.rho, ratio=args.ratio,
                     mask_coverage=cov, wikitext_ppl=ppl,
                     ppl_ratio=ppl / base_ppl if base_ppl else float("nan"),
                     delta_wrong=s["wrong_rate"] - base["wrong_rate"],
                     delta_fluent_wrong=(s["fluent_wrong_rate"]
                                         - base["fluent_wrong_rate"]),
                     nll_cut=nll_cut, nonfinite_logit_rows=n_fired,
                     secs=time.time() - t0)
            rows.append(s)
            for rec in recs:
                all_recs.append(dict(rec, arm=arm, alpha=alpha))
            print(f"  a={alpha:<4} wrong={s['wrong_rate']:.1%} "
                  f"(Δ{s['delta_wrong']:+.1%})  fluent_wrong={s['fluent_wrong_rate']:.1%} "
                  f"(Δ{s['delta_fluent_wrong']:+.1%})  junk={s['junk_rate']:.1%}"
                  f"  nll={s['median_resp_nll']:.2f}  len={s['mean_len']:.1f}"
                  f"  ppl={ppl:.2f} ({s['ppl_ratio']:.2f}x)  [{time.time()-t0:.0f}s]",
                  flush=True)

            fields = list(rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in fields})

    with open(jsonl_path, "w") as f:
        for rec in all_recs:
            f.write(json.dumps(rec) + "\n")

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{'arm':<10}{'alpha':>6}{'wrong':>9}{'fl.wrong':>10}{'Δfl.wr':>9}"
          f"{'junk':>7}{'nll':>7}{'len':>7}{'ppl':>11}{'ppl×':>9}")
    print("─" * 86)
    for r_ in sorted(rows, key=lambda x: (x["arm"], x["alpha"])):
        ppl_s = f"{r_['wikitext_ppl']:.2f}" if math.isfinite(r_["wikitext_ppl"]) else "inf"
        px = f"{r_['ppl_ratio']:.2f}" if math.isfinite(r_["ppl_ratio"]) else "inf"
        print(f"{r_['arm']:<10}{r_['alpha']:>6.1f}{r_['wrong_rate']:>9.1%}"
              f"{r_['fluent_wrong_rate']:>10.1%}{r_['delta_fluent_wrong']:>+9.1%}"
              f"{r_['junk_rate']:>7.1%}{r_['median_resp_nll']:>7.2f}"
              f"{r_['mean_len']:>7.1f}{ppl_s:>11}{px:>9}")
    print(f"\nfluency bar: response NLL <= {nll_cut:.3f} (unsteered p95) and not junk")
    print(f"\nrows → {csv_path}\nresponses → {jsonl_path}")


if __name__ == "__main__":
    main()
