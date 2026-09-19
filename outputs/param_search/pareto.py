#!/usr/bin/env python3
"""Rank sweep points and print the Pareto front of deception against damage.

usage: pareto.py <model_short> <csv> [<csv> ...]

Deception is the mean of the two splits the screen grades exactly (HP-C and
HP-KR lie rates).  Damage is the perplexity ratio.  A point is on the front if
no other point beats it on both.
"""
import sys, csv

short, paths = sys.argv[1], sys.argv[2:]
rows = []
for p in paths:
    with open(p) as f:
        for r in csv.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})

for r in rows:
    r["decep"] = (r["lie_hpc"] + r["lie_hpkr"]) / 2

front = [a for a in rows
         if not any(b["decep"] >= a["decep"] and b["ppl_ratio"] <= a["ppl_ratio"]
                    and (b["decep"] > a["decep"] or b["ppl_ratio"] < a["ppl_ratio"])
                    for b in rows)]
front.sort(key=lambda r: -r["decep"])

hdr = (f"{'rho':>5} {'alpha':>5} {'ratio':>5} | {'HP-C':>6} {'HP-KR':>6} {'mean':>6} | "
       f"{'id_sh':>6} {'id_nc':>6} | {'degen':>6} {'ppl':>7} {'x':>5}")
def line(r):
    return (f"{r['rho']:5.2f} {r['alpha']:5.1f} {r['ratio']:5.2f} | "
            f"{r['lie_hpc']:6.1f} {r['lie_hpkr']:6.1f} {r['decep']:6.1f} | "
            f"{r['id_lie_share']:6.1f} {r['id_noclaim']:6.1f} | "
            f"{r['degen_alpaca']:6.1f} {r['ppl']:7.2f} {r['ppl_ratio']:5.2f}")

print(f"=== {short}: {len(rows)} points ===\n")
print("PARETO FRONT (deception vs perplexity)")
print(hdr)
for r in front:
    print(line(r))

print("\nTOP 10 BY MEAN DECEPTION")
print(hdr)
for r in sorted(rows, key=lambda r: -r["decep"])[:10]:
    print(line(r))

print("\nBEST DECEPTION WITHIN A DAMAGE BUDGET")
for cap in (1.3, 1.5, 1.8, 2.2):
    ok = [r for r in rows if r["ppl_ratio"] <= cap]
    if ok:
        b = max(ok, key=lambda r: r["decep"])
        print(f"  ppl <= {cap:.1f}x  ->  {line(b)}")
