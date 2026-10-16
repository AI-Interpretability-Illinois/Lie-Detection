#!/usr/bin/env python3
"""
Mask anatomy (extras plan items 2 and 4), per model, from the same 25-prompt sets as the main table (no generation):

  concentration   how much of the positive contrast mass sits in few channels: share of sum(c_j, c_j>0) held by the
                  top 1% / 4% of channels, and the share of channels with c_j > 0;
  outliers        "massive-activation" channels (mean truthful-prompt norm above the layer's 99th percentile):
                  their share of the positive-score weights and of the whole 4% mask, against the 1% base rate;
  normalised      a scale-free contrast, c'_j = n^F_j / n^T_j - 1/r, which ranks channels by the ratio instead of
                  the difference: Jaccard of its positive set with the difference contrast's, and the outlier share
                  under it. (The behavioural test of this variant is run_liars_bench_honesty.py --contrast ratio.)
  cn_vs_ours      the CreativityNeuro rule (top-rho by |W| n^F on pooled-token norms, minus top-rho by |W| n^T) against
                  our mask at the same rho: Jaccard of the two weight sets; the share of CN's weights whose contrast
                  score is positive under our rule (the rest of our 4% is near-zero padding); the split of each mask
                  between attention and MLP matrices and the Jaccard within each; the depth profile (share of each
                  mask per layer index); threshold sensitivity: Jaccard of each rule's 4% mask with its own 1/2/8%
                  masks and Jaccard between the rules at those budgets.
Writes <out>/qwen_anatomy_<model>.json and prints one line per family.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from did_estimators import cell_mean_sd, collect_column_norms, get_module, load_lm, target_layers
from src.prompts import (DIRECT_FICTION_FALSE, DIRECT_HONEST_ERROR, DIRECT_LIE, DIRECT_TRUTH_INSTRUCT,
                         TF_ERROR, TF_FICTION, TF_LIE, TF_TRUTH)

FAMS = {"ours": (["lie"], ["instruct"]), "pooled": (["lie", "fiction", "error"], ["instruct"]),
        "fiction": (["fiction"], ["instruct"]), "fictf": (["tf_fiction"], ["tf_truth"]),
        "pooltf": (["tf_lie", "tf_fiction", "tf_error"], ["tf_truth"]),
        "poolmix": (["lie", "fiction", "error", "tf_lie", "tf_fiction", "tf_error"], ["instruct", "tf_truth"])}
SETS = {"lie": DIRECT_LIE, "instruct": DIRECT_TRUTH_INSTRUCT, "fiction": DIRECT_FICTION_FALSE, "error": DIRECT_HONEST_ERROR,
        "tf_lie": TF_LIE, "tf_fiction": TF_FICTION, "tf_error": TF_ERROR, "tf_truth": TF_TRUTH}
RHOS = (0.01, 0.02, 0.08)       # threshold-sensitivity budgets besides the 4% of the paper


def topk_set(score, k):
    m = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    m[score.view(-1).topk(k).indices] = True
    return m


def module_kind(name):
    return "attn" if re.search(r"attn|attention", name) else ("mlp" if re.search(r"mlp|feed_forward|ffn", name) else "other")


@torch.no_grad()
def matrix_stats(W, c, cr, nFr, nTr, nT_mean, rho, ratio):
    """W: |weights| (rows x cols, float, on device); c / cr: difference / ratio contrast per column (mean norms);
    nFr / nTr: pooled-token (rms) treatment / control norms per column; nT_mean: mean truthful norms (outlier test)."""
    dev = W.device; rows, cols = W.shape; n = W.numel(); k = max(1, int(rho * n))
    c, cr, nFr, nTr, nT_mean = (t.to(dev).float() for t in (c, cr, nFr, nTr, nT_mean))
    s_o = (W * c.unsqueeze(0))
    M_o = topk_set(s_o, k)
    M_cn = topk_set(W * nFr.unsqueeze(0), k) & ~topk_set(W * nTr.unsqueeze(0), k)
    pos = (c > 0); posr = (cr > 0)
    pos_w = pos.unsqueeze(0).expand(rows, cols).reshape(-1)
    thr = torch.quantile(nT_mean, 0.99); outl = nT_mean > thr
    outl_w = outl.unsqueeze(0).expand(rows, cols).reshape(-1)
    cp = c.clamp(min=0); srt = cp.sort(descending=True).values
    st = dict(n=n, rows=rows, k=k,
              pos_ch=int(pos.sum()), n_ch=c.numel(), tot_pos=float(cp.sum()),
              top1=float(srt[:max(1, c.numel() // 100)].sum()), top4=float(srt[:max(1, c.numel() * 4 // 100)].sum()),
              pos_w=int(pos.sum()) * rows, out_pos_w=int((pos & outl).sum()) * rows, out_rows=int(outl.sum()) * rows,
              posr_w=int(posr.sum()) * rows, out_posr_w=int((posr & outl).sum()) * rows,
              jac_i=int((pos & posr).sum()) * rows, jac_u=int((pos | posr).sum()) * rows,
              out_mask_w=int((M_o & outl_w).sum()),
              cn_w=int(M_cn.sum()), cn_and_o=int((M_cn & M_o).sum()), cn_or_o=int((M_cn | M_o).sum()),
              cn_pos=int((M_cn & pos_w).sum()), out_cn_w=int((M_cn & outl_w).sum()))
    for r in RHOS:
        kk = max(1, int(r * n))
        M_o2 = topk_set(s_o, kk)
        M_cn2 = topk_set(W * nFr.unsqueeze(0), kk) & ~topk_set(W * nTr.unsqueeze(0), kk)
        st[f"o4_o{r}_i"] = int((M_o & M_o2).sum()); st[f"o4_o{r}_u"] = int((M_o | M_o2).sum())
        st[f"cn4_cn{r}_i"] = int((M_cn & M_cn2).sum()); st[f"cn4_cn{r}_u"] = int((M_cn | M_cn2).sum())
        st[f"o{r}_cn{r}_i"] = int((M_o2 & M_cn2).sum()); st[f"o{r}_cn{r}_u"] = int((M_o2 | M_cn2).sum())
        st[f"cn{r}_w"] = int(M_cn2.sum())
        del M_o2, M_cn2
    del s_o, M_o, M_cn, pos_w, outl_w
    return st


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    ap.add_argument("--out", default="/work/nvme/biro/xiaocong/lie_sweep/anatomy")
    ap.add_argument("--rho", type=float, default=0.04)
    ap.add_argument("--ratio", type=float, default=0.37)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    short = Path(args.model_id).name.replace(".", "p")
    dev = torch.device(args.device)
    tok = __import__("transformers").AutoTokenizer.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_lm(args.model_id, args.cache_dir, dev)
    layers = target_layers(model)
    norms, rms = {}, {}
    for k, prompts in SETS.items():
        cols = collect_column_norms(model, tok, [{"prompt": p} for p in prompts[:args.n]], layers, dev, position="prompt", batch_size=2)
        norms[k], _, _ = cell_mean_sd(cols)
        rms[k] = {ly: (t * t).mean(dim=0).sqrt() for ly, t in cols.items()}      # pooled-token norm, as Wanda / CreativityNeuro
        del cols
    v = args.ratio / (1 + args.ratio)
    depth = lambda ly: int(re.search(r"layers\.(\d+)\.", ly).group(1)) if re.search(r"layers\.(\d+)\.", ly) else -1
    n_layers = max(depth(ly) for ly in layers) + 1
    res = {"model": short, "rho": args.rho, "ratio": args.ratio, "rhos": list(RHOS), "n_layers": n_layers, "families": {}}
    for fam, (tk, ck) in FAMS.items():
        avg = lambda src, keys, ly: sum(src[k][ly] for k in keys) / len(keys)
        tot = {}; kinds = {"attn": {}, "mlp": {}, "other": {}}; prof_o = [0] * n_layers; prof_cn = [0] * n_layers
        for ly in layers:
            nF, nT = avg(norms, tk, ly), avg(norms, ck, ly)
            c = v * nF - (1 - v) * nT
            cr = nF / nT.clamp(min=1e-8) - 1.0 / args.ratio
            W = get_module(model, ly).weight.detach().float().abs().to(dev)
            st = matrix_stats(W, c, cr, avg(rms, tk, ly), avg(rms, ck, ly), nT, args.rho, args.ratio)
            del W
            for key, val in st.items():
                tot[key] = tot.get(key, 0) + val
                kd = kinds[module_kind(ly)]; kd[key] = kd.get(key, 0) + val
            if depth(ly) >= 0:
                prof_o[depth(ly)] += st["k"]; prof_cn[depth(ly)] += st["cn_w"]
        T = tot
        r = dict(positive_channel_share=T["pos_ch"] / T["n_ch"], positive_mass_top1pct=T["top1"] / max(T["tot_pos"], 1e-9),
                 positive_mass_top4pct=T["top4"] / max(T["tot_pos"], 1e-9),
                 positive_weight_share=T["pos_w"] / T["n"], outlier_base_rate=T["out_rows"] / T["n"],
                 outlier_share_of_positive=T["out_pos_w"] / max(T["pos_w"], 1), outlier_share_of_mask=T["out_mask_w"] / max(T["k"], 1),
                 ratio_positive_weight_share=T["posr_w"] / T["n"], ratio_outlier_share_of_positive=T["out_posr_w"] / max(T["posr_w"], 1),
                 jaccard_diff_vs_ratio_positive=T["jac_i"] / max(T["jac_u"], 1),
                 # CN vs ours
                 cn_weight_share=T["cn_w"] / T["n"], ours_weight_share=T["k"] / T["n"],
                 jaccard_cn_vs_ours=T["cn_and_o"] / max(T["cn_or_o"], 1),
                 cn_in_ours_share=T["cn_and_o"] / max(T["cn_w"], 1),                 # share of CN's weights that our 4% mask also holds
                 cn_positive_share=T["cn_pos"] / max(T["cn_w"], 1),                   # ... whose contrast score is positive under our rule
                 positive_in_cn_share=T["cn_pos"] / max(T["pos_w"], 1),               # share of our positive-score weights that CN holds
                 outlier_share_of_cn=T["out_cn_w"] / max(T["cn_w"], 1),
                 module_split={kind: dict(ours_share=d.get("k", 0) / max(T["k"], 1), cn_share=d.get("cn_w", 0) / max(T["cn_w"], 1),
                                          weights_share=d.get("n", 0) / T["n"],
                                          jaccard=d.get("cn_and_o", 0) / max(d.get("cn_or_o", 0), 1),
                                          cn_positive_share=d.get("cn_pos", 0) / max(d.get("cn_w", 0), 1))
                               for kind, d in kinds.items() if d},
                 depth_profile_ours=[x / max(T["k"], 1) for x in prof_o], depth_profile_cn=[x / max(T["cn_w"], 1) for x in prof_cn],
                 threshold={str(rr): dict(ours4_vs_ours=T[f"o4_o{rr}_i"] / max(T[f"o4_o{rr}_u"], 1),
                                          cn4_vs_cn=T[f"cn4_cn{rr}_i"] / max(T[f"cn4_cn{rr}_u"], 1),
                                          ours_vs_cn=T[f"o{rr}_cn{rr}_i"] / max(T[f"o{rr}_cn{rr}_u"], 1),
                                          cn_weight_share=T[f"cn{rr}_w"] / T["n"]) for rr in RHOS})
        res["families"][fam] = r
        print(f"{short} {fam:8s} pos-ch {100*r['positive_channel_share']:.2f}%  mass top1% {r['positive_mass_top1pct']:.2f} top4% {r['positive_mass_top4pct']:.2f}  "
              f"outlier: base {100*r['outlier_base_rate']:.1f}% pos {100*r['outlier_share_of_positive']:.1f}% mask {100*r['outlier_share_of_mask']:.1f}% cn {100*r['outlier_share_of_cn']:.1f}%  "
              f"ratio: outlier {100*r['ratio_outlier_share_of_positive']:.1f}% J {r['jaccard_diff_vs_ratio_positive']:.2f}  "
              f"cn: {100*r['cn_weight_share']:.2f}% of weights, J(cn,ours) {r['jaccard_cn_vs_ours']:.3f}, cn in ours {100*r['cn_in_ours_share']:.1f}%, "
              f"cn positive {100*r['cn_positive_share']:.1f}%, attn share ours/cn {100*r['module_split'].get('attn', {}).get('ours_share', 0):.0f}/"
              f"{100*r['module_split'].get('attn', {}).get('cn_share', 0):.0f}%", flush=True)
    (out / f"qwen_anatomy_{short}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
