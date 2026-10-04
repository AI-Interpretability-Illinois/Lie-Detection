#!/usr/bin/env python3
"""
Model-size study (extras plan, Oct 4): does the mask's gain change with model size within a family?

Input: the main-table JSONs (reports/main_table_final.json for the seven original models, plus
reports/main_table_<tag>.json files produced by build_final_table.py for the new sizes) and a size table below.
For each family (Qwen2.5-Instruct, OLMo-2-Instruct, Dolly-v2) and benchmark, it regresses
  gain = honest(mask) - honest(best other)      and      gain0 = honest(mask) - honest(unedited)
on log10(parameters), with a paired bootstrap over test items for each point (from the per-item files) and a
case-resampling bootstrap over models for the slope, and prints slope with 95% CI. With three to five points
per family the slope CIs are wide; the script reports them rather than a verdict.

Analysis fixed before the new runs (2026-10-04): slope on log10 params, per family, per benchmark; the
pre-registered question is whether the 95% interval of the slope excludes zero.
"""
import json
import math
import random
import sys
from pathlib import Path

PARAMS = {"Qwen2.5-0.5B": 0.49e9, "Qwen2.5-1.5B": 1.54e9, "Qwen2.5-3B": 3.09e9, "Qwen2.5-7B": 7.6e9, "Qwen2.5-14B": 14.7e9,
          "OLMo-2-1B": 1.5e9, "OLMo-2-7B": 7.3e9, "OLMo-2-13B": 13.7e9, "OLMo-2-32B": 32e9,
          "Falcon3-1B": 1.67e9, "Falcon3-3B": 3.23e9, "Falcon3-7B": 7.46e9, "Falcon3-10B": 10.3e9,
          "Dolly-v2-3B": 2.8e9, "Dolly-v2-7B": 6.9e9, "Dolly-v2-12B": 12e9}   # Dolly-v2 left the Hub (Oct 2026); kept for the record
FAMILY = {k: k.split("-")[0] if not k.startswith("OLMo") else "OLMo-2" for k in PARAMS}
FAMILY.update({k: "Dolly-v2" for k in PARAMS if k.startswith("Dolly")})


def slope(xs, ys):
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else float("nan")


def main():
    rows = {}
    for p in [Path("/u/xiaocong/lie_detection/reports/main_table_final.json")] + [Path(a) for a in sys.argv[1:]]:
        if p.exists():
            for r in json.load(open(p)):
                if not r.get("pending") and r["model"] in PARAMS:
                    rows[r["model"]] = r
    for fam in ("Qwen2.5", "OLMo-2", "Falcon3", "Dolly-v2"):
        ms = [m for m in rows if FAMILY[m] == fam]
        if len(ms) < 2:
            print(f"{fam}: {len(ms)} model(s), nothing to fit"); continue
        for bench in ("lb", "ail"):
            pts = []
            for m in ms:
                c = rows[m][bench]["cells"]; L = rows[m][bench]["lead"]
                if c.get("mask", {}).get("v") is None or c.get("unsteered", {}).get("v") is None:
                    continue
                pts.append((math.log10(PARAMS[m]), (L["diff"] if L else float("nan")), c["mask"]["v"] - c["unsteered"]["v"], m,
                            (L["lo"], L["hi"]) if L else None))
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            for label, idx in (("gain vs best other", 1), ("gain vs unedited", 2)):
                ys = [p[idx] for p in pts]
                if any(y != y for y in ys):
                    continue
                b = slope(xs, ys)
                rnd = random.Random(0); boots = []
                for _ in range(2000):
                    samp = [rnd.choice(pts) for _ in pts]
                    if len({s[3] for s in samp}) < 2:
                        continue
                    boots.append(slope([s[0] for s in samp], [s[idx] for s in samp]))
                boots.sort()
                lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots)) - 1]
                print(f"{fam:9s} {bench:3s} {label:19s} n={len(pts)}  slope per decade of params {b:+.1f} points  "
                      f"[{lo:+.1f}, {hi:+.1f}]   points: " + ", ".join(f"{p[3]} {p[idx]:+.1f}" for p in pts))


if __name__ == "__main__":
    main()
