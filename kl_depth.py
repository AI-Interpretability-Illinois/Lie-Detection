#!/usr/bin/env python3
"""
Is the edit shallow? After Qi et al. (ICLR 2025), who show alignment changes mostly the first
response tokens: for each mask, KL(P_edited || P_unedited) of the next-token distribution at each
of the first K positions of the UNEDITED model's own response, teacher-forced, validation half:
  lb_deception   LIARS' BENCH instructed-deception items (the unedited model's lies)
  lb_control     honesty-instructed items
  multiturn      after a planted lie in an earlier turn (extra_multiturn.jsonl)
Masks: the current hand-written mask at the model's chosen family and strength, and the
response-token designs of mask_designs.py. Writes <out>/kl_depth_<model>.json.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as Fn
from transformers import AutoTokenizer

from did_estimators import load_lm, target_layers
from mask_designs import design_mask_spec, design_norms, handwritten_mask
from run_real_did import apply_mask, restore

O = Path("/work/hdd/bfqt/xiaocong/lie_detection_outputs")
half = lambda uid: "val" if int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--cache-dir", default="/work/hdd/bfqt/xiaocong/hf_cache")
    ap.add_argument("--lb-responses", required=True, help="file with the unedited model's LB responses (arm unsteered)")
    ap.add_argument("--family", required=True)
    ap.add_argument("--alpha", type=float, default=-1.0)
    ap.add_argument("--designs", default="resp:0.04:a0.005,recov:0.04:a0.005")
    ap.add_argument("--design-n", type=int, default=200)
    ap.add_argument("--n-per-split", type=int, default=100)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0, device = time.time(), "cuda"
    short = args.model_id.split("/")[-1].replace(".", "p")
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_lm(args.model_id, args.cache_dir, device)
    layers = target_layers(model)

    lb = {json.loads(l)["uid"]: json.loads(l) for l in open(O / "liars_bench/eval_set.jsonl")}
    mt = {json.loads(l)["uid"]: json.loads(l) for l in open(O / "extra_data/extra_multiturn.jsonl")}
    def resp(path, arm="unsteered"):
        return {r["uid"]: r["resp"] for r in map(json.loads, open(path)) if r["arm"] == arm}
    lbr = resp(args.lb_responses)
    mtr = resp(O / f"liars_bench_honesty_{short}_tmulti_responses.jsonl")
    pick = lambda d, ok: [u for u in sorted(d) if half(u) == "val" and ok(u)][:args.n_per_split]
    splits = {
        "lb_deception": [(lb[u]["ctx"], lbr[u]) for u in pick(lbr, lambda u: u in lb and lb[u]["dataset"] == "instructed-deception" and lb[u]["grade"].get("instructed_deceptive") is True)],
        "lb_control": [(lb[u]["ctx"], lbr[u]) for u in pick(lbr, lambda u: u in lb and lb[u]["dataset"] == "instructed-deception" and lb[u]["grade"].get("instructed_deceptive") is False)],
        "multiturn": [(mt[u]["ctx"], mtr[u]) for u in pick(mtr, lambda u: u in mt)],
    }
    seqs = {}
    for k, pairs in splits.items():
        seqs[k] = []
        for ctx, r in pairs:
            p = tok.apply_chat_template(ctx, tokenize=False, add_generation_prompt=True)
            pi = tok(p, add_special_tokens=False)["input_ids"]
            ri = tok(r, add_special_tokens=False)["input_ids"][:args.K]
            if ri:
                seqs[k].append((pi + ri, len(pi), len(ri)))
        print(f"  {k}: {len(seqs[k])} sequences", flush=True)

    @torch.no_grad()
    def logprobs(seq):
        ids, L, n = seq
        out = model(torch.tensor([ids], device=device)).logits[0, L - 1:L - 1 + n].float()
        return Fn.log_softmax(out, -1)

    base = {k: [logprobs(s).half().cpu() for s in v] for k, v in seqs.items()}
    print(f"  base pass done ({time.time()-t0:.0f}s)", flush=True)
    masks = {f"current:{args.family}": (handwritten_mask(model, tok, layers, device, args.family), args.alpha)}
    dn = {}
    for spec in [x for x in args.designs.split(",") if x]:
        d, rho, r = spec.split(":")
        if d not in dn:
            dn[d] = design_norms(model, tok, layers, device, d, args.design_n)
        masks[f"{d} rho={rho} r={r}"] = (design_mask_spec(model, layers, dn[d], float(rho), r, device), -1.0)
    res = {}
    for name, (m, a) in masks.items():
        orig = {}
        apply_mask(model, m, a, orig)
        res[name] = {}
        for k, v in seqs.items():
            tot, cnt = [0.0] * args.K, [0] * args.K
            for s, b in zip(v, base[k]):
                e = logprobs(s)
                kl = (e.exp() * (e - b.to(device).float())).sum(-1)      # KL(edit || base) per position
                for i, x in enumerate(kl.tolist()):
                    tot[i] += x; cnt[i] += 1
            res[name][k] = [round(t / c, 4) if c else None for t, c in zip(tot, cnt)]
        restore(model, orig)
        print(f"  {name}: " + "  ".join(f"{k} pos1-3 {sum(x for x in res[name][k][:3] if x)/3:.3f} "
                                        f"pos4-24 {sum(x for x in res[name][k][3:] if x)/max(1,len([x for x in res[name][k][3:] if x])):.3f}"
                                        for k in res[name]), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / f"kl_depth_{short}.json").write_text(json.dumps(
        {"model": args.model_id, "family": args.family, "alpha": args.alpha, "K": args.K, "kl": res,
         "n": {k: len(v) for k, v in seqs.items()}}, indent=1))
    print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
