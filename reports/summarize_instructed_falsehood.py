#!/usr/bin/env python3
"""Tables for the instructed-falsehood report, straight from the run CSVs."""
import csv, glob, os, sys
from collections import defaultdict

OUT = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
ARMS = ["lie", "fiction", "error", "pooled", "para", "magmatch"]
AGG = {"heldout3": "lie-flavoured framings (comparable with earlier reports)",
       "falsehood2": "the two new falsehood framings",
       "heldout_all": "all five non-direct framings"}

rows = defaultdict(list)
for p in sorted(glob.glob(f"{OUT}/instructed_falsehood_*.csv")):
    model = os.path.basename(p).split("_")[2]
    for r in csv.DictReader(open(p)):
        rows[model].append(r)

if not rows:
    sys.exit("no instructed_falsehood_*.csv yet")

for agg, desc in AGG.items():
    print(f"\n### {agg} — {desc}")
    print(f"{'model':<26}{'knob':>6}" + "".join(f"{a:>10}" for a in ARMS)
          + f"{'unsteered':>11}")
    for model in sorted(rows):
        base = next((float(r[agg]) for r in rows[model] if r["arm"] == "none"), None)
        for knob in sorted({float(r["knob"]) for r in rows[model] if r["arm"] != "none"}):
            cells = []
            for a in ARMS:
                hit = [r for r in rows[model]
                       if r["arm"] == a and abs(float(r["knob"]) - knob) < 1e-9]
                cells.append(f"{float(hit[0][agg])*100:>9.1f}" if hit else f"{'-':>9}")
            print(f"{model:<26}{knob:>6.2f}" + "".join(c + " " for c in cells)
                  + (f"{base*100:>10.1f}" if base is not None else f"{'-':>10}"))

print("\n### damage at the strongest knob (-1.0 where present)")
print(f"{'model':<26}{'arm':<10}{'ppl':>8}{'neutral':>9}{'med|W|':>9}{'Jac vs lie':>12}")
for model in sorted(rows):
    ks = {float(r["knob"]) for r in rows[model] if r["arm"] != "none"}
    if not ks:
        continue
    k = min(ks)
    for a in ARMS:
        hit = [r for r in rows[model]
               if r["arm"] == a and abs(float(r["knob"]) - k) < 1e-9]
        if not hit:
            continue
        r = hit[0]
        j = r.get("jaccard_vs_lie", "")
        j = f"{float(j):.3f}" if j not in ("", "nan") else "1.000"
        print(f"{model:<26}{a:<10}{float(r['ppl_ratio']):>8.2f}"
              f"{float(r['neutral_acc'])*100:>9.1f}{float(r['median_W_rel']):>9.4f}{j:>12}")
