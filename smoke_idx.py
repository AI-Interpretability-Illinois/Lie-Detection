"""Round-trip check for the index-set mask machinery in run_instructed_falsehood."""
import torch
from transformers import AutoTokenizer
from did_estimators import (cell_mean_sd, collect_column_norms, get_module,
                            load_lm, target_layers)
from src.prompts import DIRECT_LIE, DIRECT_TRUTH_INSTRUCT
from run_instructed_falsehood import (apply_idx, idx_diag, idx_from_colstat,
                                      idx_magnitude_matched, restore_idx)

MID = "Qwen/Qwen2.5-0.5B-Instruct"
CD = "/work/hdd/bfqt/xiaocong/hf_cache"
dev = "cuda:0"
tok = AutoTokenizer.from_pretrained(MID, cache_dir=CD)
tok.pad_token = tok.pad_token or tok.eos_token
m = load_lm(MID, CD, dev)
L = target_layers(m)
print(f"{len(L)} maskable matrices")

def norms(ps):
    c = collect_column_norms(m, tok, [{"prompt": p} for p in ps[:8]], L, dev,
                             position="prompt", batch_size=4)
    mean, _, _ = cell_mean_sd(c)
    return mean

v = 0.37 / 1.37
nl, nt = norms(DIRECT_LIE), norms(DIRECT_TRUTH_INSTRUCT)
cs = {k: v * nl[k] - (1 - v) * nt[k] for k in L}
rho = 0.04
idxs = idx_from_colstat(m, L, cs, rho, dev)
mm = idx_magnitude_matched(m, idxs, dev, 0)

k_want = sum(max(1, int(rho * get_module(m, n).weight.numel())) for n in L)
print(f"selected {sum(i.numel() for i in idxs.values())} / want {k_want}")
print(f"magmatch {sum(i.numel() for i in mm.values())} indices")
print("dtype", next(iter(idxs.values())).dtype,
      "unique per layer:",
      all(i.unique().numel() == i.numel() for i in list(idxs.values())[:5]))
rel, _ = idx_diag(m, idxs)
rel_mm, jac = idx_diag(m, mm, ref=idxs)
print(f"median|W| rel: did {rel:.4f}x  magmatch {rel_mm:.4f}x  Jaccard {jac:.4f}")

ref = {n: get_module(m, n).weight.detach().clone() for n in L[:6]}
saved = {}
apply_idx(m, idxs, -1.0, saved)
nz = sum(int((get_module(m, n).weight.detach() != ref[n]).sum()) for n in L[:6])
exp = sum(idxs[n].numel() for n in L[:6])
zeroed = sum(int((get_module(m, n).weight.detach().view(-1)[idxs[n].to(dev).long()]
                  == 0).sum()) for n in L[:6])
print(f"alpha=-1: changed {nz} of expected <= {exp}; zeroed {zeroed}/{exp}")
restore_idx(m, idxs, saved)
bad = sum(int((get_module(m, n).weight.detach() != ref[n]).sum()) for n in L[:6])
print(f"after restore, differing entries = {bad}  -> {'EXACT' if bad == 0 else 'MISMATCH'}")
