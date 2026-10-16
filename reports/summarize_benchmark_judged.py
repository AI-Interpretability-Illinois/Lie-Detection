#!/usr/bin/env python3
"""
Cross-model table for LIARS' BENCH deception-instructed items, hybrid-judged.

For each model: the honest rate of the best usable arm in each family, where the
families are the weight masks built from falsehood instructions (fiction, pooled),
the published lie-built mask, activation steering (best of the swept layers and
strengths), the honest system prompt, the magnitude-matched random mask, and the
unsteered model. Usable means perplexity under 1.15x and an honest rate on the
honesty-instructed control half within 5 points of unsteered, since an arm that
breaks the model's ordinary honesty is not a result.
"""

import csv
import glob
import os

OUT = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
JUDGE = "Qwen2p5-7B-Instruct_v2"


def family(arm):
    if arm.startswith("weight:fiction") or arm.startswith("weight:pooled"):
        return "falsehood mask"
    if arm.startswith("weight:ours"):
        return "lie mask (published)"
    if arm.startswith("actsteer"):
        return "activation steering"
    if arm.startswith("prompt:honest"):
        return "honest prompt"
    if arm.startswith("weight:magmatch"):
        return "random mask control"
    if arm == "unsteered":
        return "unsteered"
    return None


FAMS = ["falsehood mask", "lie mask (published)", "activation steering",
        "honest prompt", "random mask control", "unsteered"]

print(f"{'model':<28}" + "".join(f"{f:>22}" for f in FAMS) + f"{'lead':>7}")
for p in sorted(glob.glob(f"{OUT}/liars_bench_honesty_*_idfix2_judged_{JUDGE}.csv")):
    model = os.path.basename(p).replace("liars_bench_honesty_", "").split("_idfix2")[0]
    judged = [r for r in csv.DictReader(open(p)) if r["grader"] == "hybrid"]
    ppl = {r["arm"]: float(r["ppl_ratio"])
           for r in csv.DictReader(open(f"{OUT}/liars_bench_honesty_{model}_idfix2.csv"))}
    base = next(r for r in judged if r["arm"] == "unsteered")
    ctrl0 = float(base["idt_honest_rate"])
    best, loose = {}, {}
    for r in judged:
        f = family(r["arm"])
        if f is None:
            continue
        h = float(r["id_honest_rate"])
        if ppl.get(r["arm"], 9) < 1.15:          # perplexity bar only
            if f not in loose or h > loose[f][0]:
                loose[f] = (h, r["arm"])
        ok = ppl.get(r["arm"], 9) < 1.15 and float(r["idt_honest_rate"]) >= ctrl0 - 0.05
        if not ok:
            continue
        if f not in best or h > best[f][0]:
            best[f] = (h, r["arm"])
    cells = []
    for f in FAMS:
        cells.append(f"{best[f][0]*100:>21.1f}%" if f in best else f"{'none usable':>22}")
    others = [best[f][0] for f in FAMS[1:] if f in best]
    lead = (best["falsehood mask"][0] - max(others)) * 100 if "falsehood mask" in best else float("nan")
    lo = [loose[f][0] for f in FAMS[1:] if f in loose]
    lead_loose = ((loose["falsehood mask"][0] - max(lo)) * 100
                  if "falsehood mask" in loose else float("nan"))
    print(f"{model:<28}" + "".join(cells) + f"{lead:>+7.1f}"
          f"   (perplexity bar only: {lead_loose:+.1f})")
print("\nlead = best falsehood mask minus the best of every other family, usable arms only.\n"
      "The bracketed lead drops the control-half rule and keeps only the perplexity bar.")
