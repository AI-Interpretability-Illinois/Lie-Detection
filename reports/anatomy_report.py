#!/usr/bin/env python3
"""
Why does the edit do less on Qwen? (extras plan item 4) Reads the qwen_anatomy.py JSONs (ANAT_DIR) and the main table and
writes reports/anatomy_report.md: per model, for the family of the mask arm chosen on LIARS' BENCH,

  * the headroom (unedited honest rate and the honest-prompt control on the test half) and the mask's lead;
  * contrast concentration: share of channels with a positive contrast, share of the positive mass in the top 1% / 4% of
    channels, share of the 4% weight budget that carries a positive score (the rest is near-zero padding);
  * outlier channels (mean truthful-prompt norm above the layer's 99th percentile): their share of the positive-score
    weights and of the whole mask, against the 1% base rate; the same share under the scale-free ratio contrast, and the
    Jaccard between the difference and ratio contrasts' positive sets;
  * where the mask sits: attention share against attention's share of all weights, and the depth profile summarised as the
    share in the first / middle / last third.

The behavioural counterpart (masks built with --contrast ratio on Qwen-7B and Llama) is read from main_table_ratio.json
when present and printed as a second table against main_table_final.json.
"""
import glob
import json
import os
from pathlib import Path

R = Path("/u/xiaocong/lie_detection/reports")
ANAT = Path(os.environ.get("ANAT_DIR", "/work/nvme/biro/xiaocong/lie_sweep/anatomy"))
SHORT = {"Llama-3.1-8B": "Meta-Llama-3p1-8B-Instruct", "Phi-4-mini": "Phi-4-mini-instruct", "Falcon3-3B": "Falcon3-3B-Instruct",
         "OLMo-2-1B": "OLMo-2-0425-1B-Instruct", "Qwen2.5-3B": "Qwen2p5-3B-Instruct", "Qwen2.5-7B": "Qwen2p5-7B-Instruct",
         "Qwen2.5-1.5B": "Qwen2p5-1p5B-Instruct"}
ORDER = ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-1.5B", "Qwen2.5-3B", "Qwen2.5-7B"]


def thirds(prof):
    n = len(prof); a = sum(prof[: n // 3]); b = sum(prof[n // 3: 2 * n // 3]); c = sum(prof[2 * n // 3:])
    return f"{100 * a:.0f} / {100 * b:.0f} / {100 * c:.0f}"


def main():
    tab = {r["model"]: r for r in json.load(open(R / "main_table_final.json")) if not r.get("pending")}
    files = {Path(p).stem.replace("qwen_anatomy_", ""): json.load(open(p)) for p in glob.glob(str(ANAT / "qwen_anatomy_*.json"))}
    out = ["# Mask anatomy: why the edit does less on the Qwen models", "",
           "Family = the mask arm chosen on LIARS' BENCH validation. Headroom = unedited honest rate -> honest-prompt control, test half. "
           "Positive = channels/weights with a positive contrast score (the rest of the 4% budget is near-zero padding). "
           "Outlier channels = truthful-prompt norm above the layer's 99th percentile (base rate 1% of weights).", "",
           "| model | family | unedited -> honest prompt | lead | positive channels | positive mass in top 1% / 4% of channels | positive weights (% of all) | outlier share: positive / mask | ratio contrast: outlier share, J(diff, ratio) | attention share of mask (of all weights) | depth thirds (first / mid / last) |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in ORDER:
        r = tab.get(name); a = files.get(SHORT[name])
        if not r:
            continue
        c = r["lb"]["cells"]; L = r["lb"]["lead"]
        arm = c.get("mask", {}).get("arm") or ""
        fam = next((f for f in ("poolmix", "pooltf", "fictf", "pooled", "fiction") if f in arm), "pooled")
        head = f"{c['unsteered']['v']:.1f} -> {c['prompt']['v']:.1f}" if c.get("prompt", {}).get("v") is not None else f"{c['unsteered']['v']:.1f} -> –"
        lead = f"{L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}]" if L else "–"
        if not a:
            out.append(f"| {name} | {fam} | {head} | {lead} | pending | | | | | | |"); continue
        d = a["families"][fam]; ms = d["module_split"].get("attn", {})
        out.append(f"| {name} | {fam} | {head} | {lead} | {100 * d['positive_channel_share']:.2f}% | {d['positive_mass_top1pct']:.2f} / {d['positive_mass_top4pct']:.2f} | "
                   f"{100 * d['positive_weight_share']:.2f}% | {100 * d['outlier_share_of_positive']:.0f}% / {100 * d['outlier_share_of_mask']:.1f}% | "
                   f"{100 * d['ratio_outlier_share_of_positive']:.0f}%, {d['jaccard_diff_vs_ratio_positive']:.2f} | {100 * ms.get('ours_share', 0):.0f}% ({100 * ms.get('weights_share', 0):.0f}%) | "
                   f"{thirds(d['depth_profile_ours'])} |")
    rp = R / "main_table_ratio.json"
    if rp.exists():
        rat = {r["model"]: r for r in json.load(open(rp)) if not r.get("pending")}
        if rat:
            out += ["", "## Ratio contrast (n_F / n_T - 1/r) in place of the difference contrast, same protocol", "",
                    "| model | bench | mask (difference) | mask (ratio) | lead (difference) | lead (ratio) |", "|---|---|---|---|---|---|"]
            for name, rr in rat.items():
                for b in ("lb", "ail"):
                    f0 = tab[name][b]; f1 = rr[b]
                    m0 = f0["cells"].get("mask", {}).get("v"); m1 = f1["cells"].get("mask", {}).get("v")
                    L0 = f0["lead"]; L1 = f1["lead"]
                    fmt = lambda L: f"{L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}] vs {L['vs']}" if L else "–"
                    out.append(f"| {name} | {b} | {m0 if m0 is not None else '–'} | {m1 if m1 is not None else '–'} | {fmt(L0)} | {fmt(L1)} |")
    (R / "anatomy_report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
