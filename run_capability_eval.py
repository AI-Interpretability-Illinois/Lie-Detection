#!/usr/bin/env python3
"""
General-capability evaluation for baseline vs LieNeuro-DID-steered models.

Two measures, both independent of any deception framing:

  1. wikitext  — token-level perplexity on the WikiText-103 raw test split,
                 sliding window (2048 ctx / 1024 stride), so every token is
                 predicted with at least 1024 tokens of context.
  2. mmlu      — 5-shot MMLU accuracy over all 57 subjects, scored by comparing
                 the logits of " A"/" B"/" C"/" D" at the single answer position.
                 One forward pass per question, no generation, so a model that
                 has lost the ability to stop still gets a fair score.

Both are run on identical weights to the LIARS' BENCH arms (same rho / alpha /
ratio, lm_head included), so the numbers are directly comparable to that report.

Usage
-----
  CUDA_VISIBLE_DEVICES=0 python run_capability_eval.py --model llama --baseline --device cuda:0
  CUDA_VISIBLE_DEVICES=1 python run_capability_eval.py --model llama --steered  --device cuda:0
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_liars_bench_eval import MODEL_CONFIGS, build_steered_model, CACHE_DIR

OUT_DIR    = Path("outputs/liars_bench")
MAX_LENGTH = 2048
STRIDE     = 1024
N_SHOT     = 5
MMLU_MAX_LEN = 3072   # 5-shot prompts; truncation drops exemplars, not the question


# ── wikitext perplexity ──────────────────────────────────────────────────────
@torch.no_grad()
def wikitext_ppl(model, tok, device):
    ds   = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                        split="test", cache_dir=CACHE_DIR)
    text = "\n\n".join(ds["text"])
    ids  = tok(text, return_tensors="pt").input_ids
    n    = ids.size(1)

    per_tok = np.full(n, np.nan, dtype=np.float32)   # NLL of each scored token
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, n, STRIDE):
        end     = min(begin + MAX_LENGTH, n)
        trg_len = end - prev_end          # tokens newly predicted in this window
        chunk   = ids[:, begin:end].to(device)
        target  = chunk.clone()
        target[:, :-trg_len] = -100       # context tokens are not scored

        out = model(chunk, labels=target)
        # HF shifts by one, so the count of non-ignored label positions is
        # trg_len everywhere except the first window (where it is trg_len - 1).
        n_ctx    = chunk.size(1)
        valid    = (n_ctx - 1) - max(0, (n_ctx - trg_len) - 1)
        nll_sum += out.loss.float().item() * valid

        # same quantity, unreduced, so the excess can be attributed per token
        logits = out.logits[:, :-1].float()
        labels = target[:, 1:]
        tok_nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="none")
        keep = labels.reshape(-1) != -100
        # window position j of the shifted labels is absolute index begin+1+j
        pos = (torch.arange(labels.size(1), device=device) + begin + 1)[keep.view(-1)]
        per_tok[pos.cpu().numpy()] = tok_nll[keep].cpu().numpy()
        n_tokens += valid

        prev_end = end
        if end == n:
            break

    return ({"perplexity": math.exp(nll_sum / n_tokens),
             "nll_per_token": nll_sum / n_tokens,
             "n_tokens": n_tokens},
            {"nll": per_tok, "ids": ids[0].numpy().astype(np.int32)})


# ── MMLU ─────────────────────────────────────────────────────────────────────
LETTERS = ["A", "B", "C", "D"]


def fmt_q(row, with_answer):
    s = row["question"].strip() + "\n"
    for L, c in zip(LETTERS, row["choices"]):
        s += f"{L}. {c}\n"
    s += "Answer:"
    if with_answer:
        s += f" {LETTERS[row['answer']]}\n\n"
    return s


def build_prompts(test, dev):
    """Standard 5-shot MMLU prompts, exemplars drawn from the same subject."""
    shots = {}
    for row in dev:
        shots.setdefault(row["subject"], []).append(row)

    prompts = []
    for row in test:
        subj   = row["subject"]
        header = ("The following are multiple choice questions (with answers) "
                  f"about {subj.replace('_', ' ')}.\n\n")
        body   = "".join(fmt_q(s, True) for s in shots.get(subj, [])[:N_SHOT])
        prompts.append(header + body + fmt_q(row, False))
    return prompts


def letter_ids(tok):
    """Token id for each answer letter as it appears after 'Answer:'."""
    ids = []
    for L in LETTERS:
        enc = tok.encode(f" {L}", add_special_tokens=False)
        ids.append(enc[-1])
    assert len(set(ids)) == 4, f"answer letters are not distinct tokens: {ids}"
    return ids


@torch.no_grad()
def mmlu_acc(model, tok, device, batch_size=8):
    test = load_dataset("cais/mmlu", "all", split="test", cache_dir=CACHE_DIR)
    dev  = load_dataset("cais/mmlu", "all", split="dev",  cache_dir=CACHE_DIR)

    prompts = build_prompts(test, dev)
    gold    = list(test["answer"])
    subjects = list(test["subject"])
    cand    = torch.tensor(letter_ids(tok), device=device)
    lp      = [None] * len(prompts)   # per-item log-softmax over the 4 letters

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    order  = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    preds  = [None] * len(prompts)

    for s in range(0, len(order), batch_size):
        idx   = order[s:s + batch_size]
        batch = [prompts[i] for i in idx]
        enc   = tok(batch, return_tensors="pt", padding=True, truncation=True,
                    max_length=MMLU_MAX_LEN, add_special_tokens=True).to(device)
        logits = model(**enc).logits            # right-padded → last real token
        last   = enc["attention_mask"].sum(1) - 1
        picked = logits[torch.arange(len(idx), device=device), last]
        four   = picked[:, cand].float().log_softmax(-1)
        choice = four.argmax(-1).tolist()
        for j, (i, c) in enumerate(zip(idx, choice)):
            preds[i] = c
            lp[i]    = four[j].cpu().numpy()
        if (s // batch_size) % 100 == 0:
            print(f"  mmlu {s}/{len(order)}", flush=True)

    correct = [int(p == g) for p, g in zip(preds, gold)]
    per_subj = {}
    for subj, c in zip(subjects, correct):
        a = per_subj.setdefault(subj, [0, 0])
        a[0] += c
        a[1] += 1

    return ({"accuracy": sum(correct) / len(correct),
             "n": len(correct),
             "per_subject": {k: v[0] / v[1] for k, v in sorted(per_subj.items())}},
            {"pred": np.array(preds, dtype=np.int8),
             "gold": np.array(gold, dtype=np.int8),
             "subject": np.array(subjects),
             "logprobs": np.stack(lp).astype(np.float32)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--baseline", action="store_true")
    grp.add_argument("--steered",  action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ppl-only", action="store_true",
                    help="Run only the perplexity pass (saves per-token NLL).")
    ap.add_argument("--rho",   type=float, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--ratio", type=float, default=None)
    ap.add_argument("--arm-tag", default=None)
    ap.add_argument("--mmlu-only", action="store_true",
                    help="Skip the perplexity pass and reuse the value already on disk.")
    args = ap.parse_args()

    c = MODEL_CONFIGS[args.model]
    for k in ("rho", "alpha", "ratio"):
        if getattr(args, k) is not None:
            c[k] = getattr(args, k)        # build_steered_model reads the same dict
    arm = args.arm_tag or ("steered" if args.steered else "baseline")
    tag = f"{c['model_short']}_{arm}"

    tok = AutoTokenizer.from_pretrained(c["model_id"], cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side    = "right"  # mmlu reads the last real token explicitly
    tok.truncation_side = "left"   # an over-long prompt loses exemplars, not "Answer:"

    print(f"[{tag}] loading model …", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        c["model_id"], torch_dtype=torch.float16,
        device_map=args.device, cache_dir=CACHE_DIR)
    model.eval()

    if args.steered:
        build_steered_model(model, tok, args.model, args.device)

    prev = OUT_DIR / f"capability_{tag}.json"
    if args.mmlu_only and prev.exists():
        ppl = json.loads(prev.read_text())["wikitext"]
        print(f"[{tag}] reusing ppl={ppl['perplexity']:.3f}", flush=True)
    else:
        t0 = time.time()
        print(f"[{tag}] wikitext-103 perplexity …", flush=True)
        ppl, ppl_items = wikitext_ppl(model, tok, args.device)
        np.savez_compressed(OUT_DIR / f"wikitext_tokens_{tag}.npz", **ppl_items)
        print(f"[{tag}] ppl={ppl['perplexity']:.3f} over {ppl['n_tokens']} tokens "
              f"({time.time() - t0:.0f}s)", flush=True)

    if args.ppl_only:
        prev_j = json.loads(prev.read_text()) if prev.exists() else {}
        out = {"model": c["model_short"], "arm": arm, "rho": c["rho"],
               "alpha": c["alpha"], "ratio": c["ratio"], "wikitext": ppl,
               "mmlu": prev_j.get("mmlu", {})}
        prev.write_text(json.dumps(out, indent=2))
        print(f"[{tag}] ppl-only done → {prev}", flush=True)
        return

    t1 = time.time()
    print(f"[{tag}] MMLU 5-shot …", flush=True)
    mm, per_item = mmlu_acc(model, tok, args.device, batch_size=args.batch_size)
    np.savez_compressed(OUT_DIR / f"mmlu_items_{tag}.npz", **per_item)
    print(f"[{tag}] mmlu={mm['accuracy']:.4f} n={mm['n']} "
          f"({time.time() - t1:.0f}s)", flush=True)

    out = {"model": c["model_short"], "arm": arm,
           "rho": c["rho"], "alpha": c["alpha"], "ratio": c["ratio"],
           "wikitext": ppl, "mmlu": mm}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"capability_{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"[{tag}] → {path}", flush=True)


if __name__ == "__main__":
    main()
