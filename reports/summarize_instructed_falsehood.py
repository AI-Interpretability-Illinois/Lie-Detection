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
    print("  ! = neutral accuracy more than 5 points below unsteered; not a result")
    for model in sorted(rows):
        base = next((float(r[agg]) for r in rows[model] if r["arm"] == "none"), None)
        nbase = next((float(r["neutral_acc"]) for r in rows[model]
                      if r["arm"] == "none"), None)
        for knob in sorted({float(r["knob"]) for r in rows[model] if r["arm"] != "none"}):
            cells = []
            for a in ARMS:
                hit = [r for r in rows[model]
                       if r["arm"] == a and abs(float(r["knob"]) - knob) < 1e-9]
                if not hit:
                    cells.append(f"{'-':>9}")
                    continue
                # A falsehood rate is only interpretable on a model that can still
                # answer. Perplexity misses this: gemma-3-4b at -1.0 sits at 1.12x
                # perplexity with neutral accuracy down 51 points.
                broke = (nbase is not None
                         and float(hit[0]["neutral_acc"]) < nbase - 0.05)
                cells.append(f"{float(hit[0][agg])*100:>8.1f}" + ("!" if broke else " "))
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


def best_usable():
    """Per model and arm, the lowest falsehood rate among USABLE knobs.

    Usable means perplexity under 1.15x and neutral accuracy within 5 points of
    unsteered. Without the second gate a broken model reads as a perfect result:
    gemma-3-4b's fiction arm at -1.0 is 1.12x perplexity with 44.5% neutral.
    """
    print(f"\n### best usable configuration per arm "
          f"(rate over all five non-direct framings, and the knob)")
    print(f"{'model':<26}{'unsteered':>10}{'lie arm':>10}{'best falsehood':>22}"
          f"{'magmatch':>10}{'gain':>7}")
    wins = 0
    for model in sorted(rows):
        rs = rows[model]
        base = next((r for r in rs if r["arm"] == "none"), None)
        if base is None:
            continue
        nb = float(base["neutral_acc"])
        got = {}
        for a in ARMS:
            ok = [r for r in rs if r["arm"] == a
                  and float(r["ppl_ratio"]) < 1.15
                  and float(r["neutral_acc"]) >= nb - 0.05]
            got[a] = (min(float(r["heldout_all"]) for r in ok) if ok else None)
        fb = [(a, got[a]) for a in ("fiction", "error", "pooled") if got[a] is not None]
        b = min(fb, key=lambda x: x[1]) if fb else None
        gain = (got["lie"] - b[1]) * 100 if (b and got["lie"] is not None) else float("nan")
        if gain > 0:
            wins += 1
        print(f"{model:<26}{float(base['heldout_all'])*100:>10.1f}"
              f"{(got['lie'] or float('nan'))*100:>10.1f}"
              f"{(f'{b[1]*100:.1f} ({b[0]})' if b else '-'):>22}"
              f"{(got['magmatch'] or float('nan'))*100:>10.1f}{gain:>7.1f}")
    print(f"\nfalsehood arm beats the lie arm on {wins}/{len(rows)} models")


best_usable()
