#!/usr/bin/env python3
"""
Mask anatomy for the format-transfer study (J2): where the calibrated masks of the seven build formats
put their deleted weights, how much they overlap, and whether overlap predicts transfer.

For each build format F (the 200 pairs fmt_transfer.py uses): channel norms on lie / truth prompts,
r calibrated to a positive-score share s (as maskc), then per linear layer
  positive set  P_F = the weights of channels with c_j > 0 (they all fall inside the top-rho mask), and
  mask          M_F = the top rho of |W_ij| c_j (what maskc deletes; mostly near-zero padding).
Outputs: pairwise weight-level Jaccard of P and of M across formats; the share of P by decoder layer and
by module type; CAA cosines (from fmt_transfer's cache) for comparison. Forward passes only.
Writes <out>/anatomy_<model>.json.
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from did_estimators import get_module, load_lm, target_layers
import fmt_transfer as FT
from mask_designs import calibrate_ratio


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--out", default="/work/hdd/biro/xiaocong/lie_out/fmt_runs")
    ap.add_argument("--shares", default="0.0025,0.005")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--n-build", type=int, default=200)
    ap.add_argument("--score-batch", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    args.draw_seed = 0
    short = Path(args.model_id).name.replace(".", "p")
    out = Path(args.out)
    dev = torch.device(args.device)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_lm(args.model_id, args.cache_dir, dev).to(dev).eval()
    layers = target_layers(model)
    F = FT.FORMATS
    npath = out / f"anatomy_norms_{short}.pt"
    if npath.exists():
        norms = torch.load(npath)
    else:
        norms = {f: FT.norms_for(model, tok, layers, dev, f, args) for f in F}
        torch.save(norms, npath)
    res = {"model": short, "layers": len(layers), "shares": {}}
    for share in map(float, args.shares.split(",")):
        r = {f: calibrate_ratio(model, layers, {"F": norms[f]["lie"], "T": norms[f]["instruct"]}, share) for f in F}
        v = {f: r[f] / (1 + r[f]) for f in F}
        inter_p = defaultdict(float); inter_m = defaultdict(float)
        size_p = defaultdict(float); size_m = defaultdict(float)
        by_layer = {f: defaultdict(float) for f in F}
        by_mod = {f: defaultdict(float) for f in F}
        for name in layers:
            W = get_module(model, name).weight.detach().float().abs()
            n = W.numel(); k = max(1, int(args.rho * n))
            rows = W.shape[0]
            m = re.search(r"layers\.(\d+)\.", name)
            li = int(m.group(1)) if m else -1
            mod = name.split(".")[-1]
            pos, msk = {}, {}
            for f in F:
                c = (v[f] * norms[f]["lie"][name] - (1 - v[f]) * norms[f]["instruct"][name]).to(dev).float()
                pos[f] = c > 0                                    # channels; weights = channels x rows
                s = (W * c.unsqueeze(0)).view(-1)
                mm = torch.zeros(n, dtype=torch.bool, device=dev)
                mm[s.topk(k).indices] = True
                msk[f] = mm
                np_ = float(pos[f].sum()) * rows
                size_p[f] += np_; size_m[f] += k
                by_layer[f][li] += np_; by_mod[f][mod] += np_
            for i, a in enumerate(F):
                for b in F[i + 1:]:
                    inter_p[(a, b)] += float((pos[a] & pos[b]).sum()) * rows
                    inter_m[(a, b)] += float((msk[a] & msk[b]).sum())
            del W, msk
        jac = lambda inter, size, a, b: inter[(a, b)] / max(1.0, size[a] + size[b] - inter[(a, b)])
        tot = sum(get_module(model, nm).weight.numel() for nm in layers)
        res["shares"][str(share)] = dict(
            r={f: round(r[f], 4) for f in F},
            positive_share={f: size_p[f] / tot for f in F},
            jaccard_positive={f"{a}~{b}": round(jac(inter_p, size_p, a, b), 4) for i, a in enumerate(F) for b in F[i + 1:]},
            jaccard_mask={f"{a}~{b}": round(jac(inter_m, size_m, a, b), 4) for i, a in enumerate(F) for b in F[i + 1:]},
            positive_by_layer={f: {str(li): x / max(1.0, size_p[f]) for li, x in sorted(by_layer[f].items())} for f in F},
            positive_by_module={f: {md: x / max(1.0, size_p[f]) for md, x in sorted(by_mod[f].items())} for f in F})
        jp = res["shares"][str(share)]["jaccard_positive"]
        print(f"share {share}: positive-set Jaccard {min(jp.values()):.3f}-{max(jp.values()):.3f}; "
              f"r {min(r.values()):.3f}-{max(r.values()):.3f}", flush=True)
    cpath = next(iter(sorted(out.glob(f"caa_{short}_L*_n{args.n_build}.pt"))), None)
    if cpath:
        res["caa_cos"] = torch.load(cpath)["cos"]
    (out / f"anatomy_{short}.json").write_text(json.dumps(res, indent=1))
    print(f"-> {out / f'anatomy_{short}.json'}")


if __name__ == "__main__":
    main()
