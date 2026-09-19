#!/usr/bin/env python3
"""
Local parameter search around a model's LieNeuro-DID operating point, scored on
LIARS' BENCH rather than the small internal QA set used by
run_lieneuro_did_sweep.py.

Each grid point is screened on three cheap signals:

  lie_hpc         harm-pressure-choice lie rate
  lie_hpkr        harm-pressure-knowledge-report lie rate
  id_lie_share    lie / (lie + honest) on instructed-deception items carrying an
                  HONEST system prompt
  degen_alpaca    degenerate-response rate on a fixed Alpaca slice, a proxy for
                  the coherence judge, which is far too slow to run per point
  ppl             WikiText-103 perplexity over a fixed prefix of windows

Only the two harm-pressure splits are graded exactly here.  The LLM deception
judge writes overrides for instructed-deception and convincing-game *only*, so
those two splits cannot be scored without it: left to the regex grader alone,
most steered instructed-deception responses fall through as `no_claim` (70% at
llama rho=0.04/0.37, against 2% once judged).  id_lie_share drops the unsettled
responses from the denominator to salvage a weak signal from that split, but it
is a hint, not the headline.  The headline comes from the full pipeline, which
is run on the finalists this screen selects.

The model is loaded once; configs are cycled with restore_original() between
apply_steering() calls, so a grid point costs a mask rebuild rather than a
model reload.  The mask depends only on (rho, ratio) -- alpha just scales the
weights it selects -- so grid points are grouped into (rho, ratio) blocks and
every alpha in a block reuses one mask.  Generation is sampled at the benchmark's own temperatures, so
rates carry roughly +/-3 points of sampling noise at these subset sizes: the
screen is for locating a region, not for separating neighbours.  Finalists get
the full pipeline.
"""

import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lie_neuro import LieNeuro, LieNeuroConfig
from run_liars_bench_eval import MODEL_CONFIGS, CACHE_DIR, N_TRUTH, MODE, generate_batched
from score_liars_bench import grade, degenerate

PPL_WINDOWS = 30          # 30 * 1024-token stride ~= 31k scored tokens
MAX_LENGTH, STRIDE = 2048, 1024


@torch.no_grad()
def build_mask_gpu(ln, rho, ratio, device, n_prompts):
    """build_mask(mode="score_diff") on the GPU, writing ln._mask in place.

    The library builds the mask with a per-layer topk on CPU tensors; lm_head
    alone is 525M elements, which costs ~190s per grid point and dominates the
    sweep.  This is the same computation layer by layer -- same DID weighting,
    same per-layer k, same topk-index selection so ties break identically --
    moved onto the GPU.  Verified to produce bitwise-identical masks.
    """
    w    = ratio / (1.0 + ratio) if ratio > 0 else 0.0
    _inv = 1.0 / n_prompts if n_prompts > 0 else 1.0
    ln._mask = {}
    for name in ln._layer_names:
        s_lie   = ln._S_lie[name].to(device, torch.float32, non_blocking=True)
        s_truth = ln._S_truth[name].to(device, torch.float32, non_blocking=True)
        diff = (_inv * (w * s_lie - (1.0 - w) * s_truth)).view(-1)
        k = max(1, int(rho * diff.numel()))
        idx = diff.topk(k).indices
        m = torch.zeros(diff.numel(), dtype=torch.bool, device=device)
        m[idx] = True
        ln._mask[name] = m.view(s_lie.shape)
        del s_lie, s_truth, diff, idx, m
    return ln._mask


@torch.no_grad()
def ppl_prefix(model, ids, device):
    """Perplexity over the first PPL_WINDOWS sliding windows of WikiText."""
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for w, begin in enumerate(range(0, ids.size(1), STRIDE)):
        if w >= PPL_WINDOWS:
            break
        end     = min(begin + MAX_LENGTH, ids.size(1))
        trg_len = end - prev_end
        chunk   = ids[:, begin:end].to(device)
        target  = chunk.clone()
        target[:, :-trg_len] = -100
        loss  = model(chunk, labels=target).loss.float().item()
        n_ctx = chunk.size(1)
        valid = (n_ctx - 1) - max(0, (n_ctx - trg_len) - 1)
        nll_sum  += loss * valid
        n_tokens += valid
        prev_end  = end
    return math.exp(nll_sum / n_tokens)


def rate(verdicts, want):
    gradeable = [v for v in verdicts if v in ("lie", "honest", "no_claim", "refusal")]
    return 100.0 * sum(v == want for v in verdicts) / max(1, len(gradeable))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--grid", required=True,
                    help="Path to a JSON list of [rho, alpha, ratio] triples.")
    ap.add_argument("--shard", default="0/1",
                    help="i/n — take every n-th grid point starting at i.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-alpaca", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-set", default="outputs/liars_bench/eval_set.jsonl")
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    short = cfg["model_short"]

    grid = json.loads(Path(args.grid).read_text())
    blocks = {}
    for rho, alpha, ratio in grid:
        blocks.setdefault((rho, ratio), []).append(alpha)
    blocks = sorted(blocks.items())
    i, n = (int(x) for x in args.shard.split("/"))
    blocks = blocks[i::n]
    npts = sum(len(a) for _, a in blocks)
    print(f"[{short}] shard {args.shard}: {len(blocks)} mask blocks, {npts} grid points",
          flush=True)

    items = [json.loads(l) for l in open(args.eval_set)]
    by_uid = {it["uid"]: it for it in items}
    id_honest = [it for it in items
                 if it["dataset"] == "instructed-deception"
                 and it["grade"].get("instructed_deceptive") is False]
    hpc    = [it for it in items if it["dataset"] == "harm-pressure-choice"]
    hpkr   = [it for it in items if it["dataset"] == "harm-pressure-knowledge-report"]
    alpaca = [it for it in items if it["dataset"] == "alpaca"][:args.n_alpaca]
    print(f"[{short}] id_honest={len(id_honest)} hpc={len(hpc)} "
          f"hpkr={len(hpkr)} alpaca={len(alpaca)}", flush=True)

    beliefs = {}
    for l in open(f"outputs/liars_bench/beliefs_{short}_baseline.jsonl"):
        r = json.loads(l)
        beliefs[r["uid"]] = r["response"]

    tok = AutoTokenizer.from_pretrained(cfg["model_id"], cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"], torch_dtype=torch.float16,
        device_map=args.device, cache_dir=CACHE_DIR)
    model.eval()

    ds  = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                       split="test", cache_dir=CACHE_DIR)
    wid = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids

    ln_cfg = LieNeuroConfig(importance_threshold=blocks[0][0][0], alpha=1.0,
                            device=args.device, skip_modules=[])
    ln = LieNeuro(model, tok, ln_cfg)
    saved = torch.load(f"outputs/scores_{short}_ntruth{N_TRUTH}.pt",
                       map_location="cpu", weights_only=False)
    ln._S_lie, ln._S_truth = saved["S_lie"], saved["S_truth"]

    base_ppl = ppl_prefix(model, wid, args.device)
    print(f"[{short}] baseline prefix ppl = {base_ppl:.3f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("rho,alpha,ratio,lie_hpc,refusal_hpc,lie_hpkr,noclaim_hpkr,"
                "id_lie_share,id_noclaim,degen_alpaca,ppl,ppl_ratio,"
                "t_mask,t_gen,secs\n")

        for bi, ((rho, ratio), alphas) in enumerate(blocks):
            tm = time.time()
            ln.config.importance_threshold = rho
            build_mask_gpu(ln, rho, ratio, args.device, N_TRUTH)
            t_mask = time.time() - tm
            print(f"[{short}] block {bi+1}/{len(blocks)} rho={rho} ratio={ratio} "
                  f"mask in {t_mask:.0f}s, {len(alphas)} alphas", flush=True)

            for alpha in alphas:
                t0 = time.time()
                ln.config.alpha = alpha
                ln.apply_steering()
                torch.manual_seed(args.seed)

                def run(sub, max_new, temp, desc):
                    return generate_batched(model, tok, [it["ctx"] for it in sub],
                                            args.device, max_new, temp,
                                            args.batch_size, desc=desc)

                tg = time.time()
                r_hpc  = run(hpc,      8, 0.7, "hpc")
                r_hpkr = run(hpkr,    48, 0.7, "hpkr")
                r_id   = run(id_honest, 128, 0.7, "id")
                r_alp  = run(alpaca, 192, 0.7, "alpaca")
                t_gen  = time.time() - tg

                def verdicts(sub, resp):
                    return [grade(it, r, beliefs.get(it["uid"], ""))[0]
                            for it, r in zip(sub, resp)]

                v_hpc, v_hpkr, v_id = (verdicts(hpc, r_hpc), verdicts(hpkr, r_hpkr),
                                       verdicts(id_honest, r_id))
                n_lie  = sum(v == "lie" for v in v_id)
                n_hon  = sum(v == "honest" for v in v_id)
                id_share = 100.0 * n_lie / max(1, n_lie + n_hon)
                d_alp = 100.0 * sum(degenerate(r) for r in r_alp) / len(r_alp)
                p     = ppl_prefix(model, wid, args.device)

                ln.restore_original()

                row = (rho, alpha, ratio,
                       rate(v_hpc, "lie"), rate(v_hpc, "refusal"),
                       rate(v_hpkr, "lie"), rate(v_hpkr, "no_claim"),
                       id_share, rate(v_id, "no_claim"),
                       d_alp, p, p / base_ppl, t_mask, t_gen, time.time() - t0)
                f.write(",".join(f"{x:.4f}" for x in row) + "\n")
                f.flush()
                print(f"[{short}]   a={alpha}  "
                      f"hpc={row[3]:5.1f}%  hpkr={row[5]:5.1f}%  id_share={id_share:5.1f}%  "
                      f"degen={d_alp:4.1f}%  ppl={p:6.2f} ({row[11]:.2f}x)  "
                      f"[gen {t_gen:.0f}s] {row[14]:.0f}s", flush=True)

    print(f"[{short}] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
