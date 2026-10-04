#!/usr/bin/env python3
"""
Why does the edit do less on Qwen? (extras plan item 4, after the judge audit). Three per-model diagnostics of the
contrast that selects the mask, computed from the same 25-prompt sets as the main table (no generation):

  concentration   how much of the positive contrast mass sits in few channels: share of sum(c_j, c_j>0) held by the
                  top 1% / 4% of channels, and the share of channels with c_j > 0;
  outliers        "massive-activation" channels (mean truthful-prompt norm above the layer's 99th percentile):
                  their share of the positive-score weights and of the whole 4% mask, against the 1% base rate;
  normalised      a scale-free contrast, c'_j = n^F_j / n^T_j - 1/r, which ranks channels by the ratio instead of
                  the difference: Jaccard of its positive set with the difference contrast's, and the outlier share
                  under it. (The behavioural test of this variant is run_liars_bench_honesty.py --contrast ratio.)
Also the headroom on LIARS' BENCH (unedited honest rate and honesty-instructed control) read from the judged file.
Writes <out>/qwen_anatomy_<model>.json and prints one line per family.
"""
import argparse
import json
import os
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
    norms = {}
    for k, prompts in SETS.items():
        cols = collect_column_norms(model, tok, [{"prompt": p} for p in prompts[:args.n]], layers, dev, position="prompt", batch_size=2)
        norms[k], _, _ = cell_mean_sd(cols); del cols
    v = args.ratio / (1 + args.ratio)
    res = {"model": short, "families": {}}
    for fam, (tk, ck) in FAMS.items():
        nF = {ly: sum(norms[k][ly] for k in tk) / len(tk) for ly in layers}
        nT = {ly: sum(norms[k][ly] for k in ck) / len(ck) for ly in layers}
        pos_mass = tot_pos = 0.0; n_ch = n_pos = 0
        top1 = top4 = 0.0
        all_c = []
        W_out = 0; pos_w = 0; mask_w = 0; out_pos_w = 0; out_mask_w = 0; out_rows = 0
        jac_i = jac_u = 0.0; out_pos_ratio = 0.0; pos_ratio_w = 0.0
        for ly in layers:
            c = v * nF[ly] - (1 - v) * nT[ly]
            cr = nF[ly] / nT[ly].clamp(min=1e-8) - 1.0 / args.ratio
            W = get_module(model, ly).weight.detach().float().abs().to(dev)
            rows = W.shape[0]; n = W.numel(); k = max(1, int(args.rho * n))
            # concentration
            cp = c.clamp(min=0); srt = cp.sort(descending=True).values
            tot_pos += float(cp.sum()); n_ch += c.numel(); n_pos += int((c > 0).sum())
            top1 += float(srt[:max(1, c.numel() // 100)].sum()); top4 += float(srt[:max(1, c.numel() * 4 // 100)].sum())
            # outlier channels: truthful-prompt norm above the layer's 99th percentile
            thr = torch.quantile(nT[ly].float(), 0.99)
            outl = nT[ly] > thr
            pos = c > 0; posr = cr > 0
            pos_w += int(pos.sum()) * rows; out_pos_w += int((pos & outl).sum()) * rows
            pos_ratio_w += int(posr.sum()) * rows; out_pos_ratio += int((posr & outl).sum()) * rows
            jac_i += int((pos & posr).sum()) * rows; jac_u += int((pos | posr).sum()) * rows
            s = (W * c.to(dev).unsqueeze(0)).view(-1)
            idx = s.topk(k).indices
            cols_of = idx % W.shape[1]
            mask_w += k; out_mask_w += int(outl.to(dev)[cols_of].sum()); out_rows += int(outl.sum()) * rows; W_out += n
            del W, s, idx
        res["families"][fam] = dict(
            positive_channel_share=n_pos / n_ch, positive_mass_top1pct=top1 / max(tot_pos, 1e-9), positive_mass_top4pct=top4 / max(tot_pos, 1e-9),
            positive_weight_share=pos_w / W_out, outlier_base_rate=out_rows / W_out,
            outlier_share_of_positive=out_pos_w / max(pos_w, 1), outlier_share_of_mask=out_mask_w / max(mask_w, 1),
            ratio_positive_weight_share=pos_ratio_w / W_out, ratio_outlier_share_of_positive=out_pos_ratio / max(pos_ratio_w, 1),
            jaccard_diff_vs_ratio_positive=jac_i / max(jac_u, 1))
        r = res["families"][fam]
        print(f"{short} {fam:8s} pos-ch {100*r['positive_channel_share']:.2f}%  mass top1% {r['positive_mass_top1pct']:.2f} top4% {r['positive_mass_top4pct']:.2f}  "
              f"outlier: base {100*r['outlier_base_rate']:.1f}% pos {100*r['outlier_share_of_positive']:.1f}% mask {100*r['outlier_share_of_mask']:.1f}%  "
              f"ratio-contrast: outlier {100*r['ratio_outlier_share_of_positive']:.1f}% jaccard {r['jaccard_diff_vs_ratio_positive']:.2f}", flush=True)
    (out / f"qwen_anatomy_{short}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
