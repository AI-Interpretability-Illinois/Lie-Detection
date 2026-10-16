#!/usr/bin/env python3
"""
CreativityNeuro baseline report (extras plan item 2). Reads reports/main_table_cn.json (build_final_table.py --cn) and,
when present, the anatomy JSONs (qwen_anatomy.py, /work/nvme/biro/xiaocong/lie_sweep/anatomy) and writes
reports/cn_report.md:

  1. per model and benchmark, the test-half honest rate of the four cells of the 2x2 (selection rule x activation
     statistic): ours = contrast score on per-prompt-mean norms, rs = contrast score on pooled-token (rms) norms,
     sd = set difference on mean norms, cn = set difference on rms norms (CreativityNeuro's rule), each at the strength
     chosen on validation, with the paired lead of ours over cn and its bootstrap interval;
  2. the 2x2 averaged over models, with the two marginal effects (rule, statistic);
  3. mask anatomy: how much of CN's weight set our mask holds, the share of CN's weights with a positive contrast
     score, the attention/MLP split of both masks, and the threshold sensitivity.
"""
import glob
import json
from pathlib import Path

R = Path("/u/xiaocong/lie_detection/reports")
ANAT = Path("/work/nvme/biro/xiaocong/lie_sweep/anatomy")
SHORT = {"Llama-3.1-8B": "Meta-Llama-3p1-8B-Instruct", "Phi-4-mini": "Phi-4-mini-instruct", "Falcon3-3B": "Falcon3-3B-Instruct",
         "OLMo-2-1B": "OLMo-2-0425-1B-Instruct", "Qwen2.5-3B": "Qwen2p5-3B-Instruct", "Qwen2.5-7B": "Qwen2p5-7B-Instruct",
         "Qwen2.5-1.5B": "Qwen2p5-1p5B-Instruct"}
CELLS = [("mask", "ours (score, mean)"), ("rs", "score, rms"), ("sd", "set diff, mean"), ("cn", "CN (set diff, rms)")]


def cell(c, f):
    x = c.get(f)
    if x is None:
        return "pending"
    if x["v"] is None:
        return f"– ({', '.join(x['fails'])})"
    return f"{x['v']:.1f}"


def main():
    tab = [r for r in json.load(open(R / "main_table_cn.json")) if not r.get("pending")]
    out = ["# CreativityNeuro-style set-difference baseline against the contrast mask", "",
           "Test-half honest rate (%), arm chosen on the validation half within each family; '–' = every strength failed a "
           "usability gate (perplexity / control / polarity). Lead = paired difference ours − CN on common test items, 95% bootstrap interval.", ""]
    for bench, name in (("lb", "LIARS' BENCH"), ("ail", "AI Liar")):
        out += [f"## {name}", "", "| model | unedited | " + " | ".join(l for _, l in CELLS) + " | ours − CN |", "|---|---|" + "---|" * len(CELLS) + "---|"]
        for r in tab:
            c = r[bench]["cells"]; L = r[bench].get("lead_cn")
            lead = f"{L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}]" if L else "–"
            out.append(f"| {r['model']} | {cell(c, 'unsteered')} | " + " | ".join(cell(c, f) for f, _ in CELLS) + f" | {lead} |")
        # 2x2 over the models where all four cells are usable
        full = [r for r in tab if all(r[bench]["cells"].get(f, {}).get("v") is not None for f, _ in CELLS)]
        if full:
            m = {f: sum(r[bench]["cells"][f]["v"] for r in full) / len(full) for f, _ in CELLS}
            out += ["", f"2×2 over the {len(full)} model(s) with all four cells usable (mean honest rate):", "",
                    "| | mean norms | rms norms |", "|---|---|---|",
                    f"| contrast score | {m['mask']:.1f} | {m['rs']:.1f} |", f"| set difference | {m['sd']:.1f} | {m['cn']:.1f} |", "",
                    f"rule effect (score − set difference, averaged over statistics): {((m['mask'] + m['rs']) - (m['sd'] + m['cn'])) / 2:+.1f} points; "
                    f"statistic effect (mean − rms, averaged over rules): {((m['mask'] + m['sd']) - (m['rs'] + m['cn'])) / 2:+.1f} points."]
        out.append("")
    files = {Path(p).stem.replace("qwen_anatomy_", ""): json.load(open(p)) for p in glob.glob(str(ANAT / "qwen_anatomy_*.json"))}
    if files:
        out += ["## Mask anatomy (qwen_anatomy.py), family of the mask arm chosen on LIARS' BENCH", "",
                "| model | family | CN weights (% of matrix) | ours ∩ CN / CN | CN weights with positive contrast score | attention share ours / CN | J(ours, CN) at ρ = 1 / 2 / 4 / 8 % | ours ρ=4 vs ρ=1 / 2 / 8 |",
                "|---|---|---|---|---|---|---|---|"]
        for r in tab:
            a = files.get(SHORT.get(r["model"], ""))
            if not a:
                continue
            arm = r["lb"]["cells"].get("mask", {}).get("arm") or ""
            fam = next((f for f in ("poolmix", "pooltf", "fictf", "pooled", "fiction") if f in arm), "pooled")
            d = a["families"][fam]; th = d["threshold"]; ms = d["module_split"].get("attn", {})
            out.append(f"| {r['model']} | {fam} | {100 * d['cn_weight_share']:.2f} | {100 * d['cn_in_ours_share']:.0f}% | {100 * d['cn_positive_share']:.0f}% | "
                       f"{100 * ms.get('ours_share', 0):.0f} / {100 * ms.get('cn_share', 0):.0f}% | "
                       f"{th['0.01']['ours_vs_cn']:.2f} / {th['0.02']['ours_vs_cn']:.2f} / {d['jaccard_cn_vs_ours']:.2f} / {th['0.08']['ours_vs_cn']:.2f} | "
                       f"{th['0.01']['ours4_vs_ours']:.2f} / {th['0.02']['ours4_vs_ours']:.2f} / {th['0.08']['ours4_vs_ours']:.2f} |")
        out.append("")
    (R / "cn_report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
