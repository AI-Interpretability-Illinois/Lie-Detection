#!/usr/bin/env python3
"""Two-proportion z-tests between two arms, parsed straight out of report.txt.

usage: compare_arms.py <arm_a> <arm_b> [report.txt]
Prints, for every section of the report, arm_a vs arm_b on each verdict column.
"""
import re, sys, math

arm_a, arm_b = sys.argv[1], sys.argv[2]
path = sys.argv[3] if len(sys.argv) > 3 else "outputs/liars_bench/report.txt"

sections, cur, cols = {}, None, None
for line in open(path):
    if line.startswith("### "):
        cur = line[4:].strip()
        sections[cur] = {}
        continue
    if cur is None:
        continue
    if line.lstrip().startswith("model / arm"):
        cols = line.split()[4:]        # drop "model / arm" (3 tokens) and "n"
        sections[cur]["_cols"] = cols
        continue
    m = re.match(r"^(\S+) / (\S+)\s+(\S+)\s+(.*)$", line.rstrip())
    if m and "_cols" in sections[cur]:
        model, arm, rest = m.group(1), m.group(2), m.group(4).split()
        if cur.startswith("coherence rate"):          # different shape, parsed below
            continue
        n = int(m.group(3))
        sections[cur][f"{model}_{arm}"] = (n, [float(x.rstrip('%')) for x in rest])

# coherence table has a different shape; parse it separately
coh = {}
cur = None
for line in open(path):
    if line.startswith("### "):
        cur = line[4:].strip()
    if cur and cur.startswith("coherence rate"):
        m = re.match(r"^(\S+) / (\S+)\s+(\S+)\s+(\d+)\s+([\d.]+)%", line.rstrip())
        if m:
            coh[(f"{m.group(1)}_{m.group(2)}", m.group(3))] = (int(m.group(4)), float(m.group(5)))

def z(p1, n1, p2, n2):
    p1, p2 = p1 / 100, p2 / 100
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se > 0 else float("nan")

def flag(zz):
    return "***" if abs(zz) > 2.58 else ("*" if abs(zz) > 1.96 else "")

print(f"{arm_a}  vs  {arm_b}\n")
for name, tbl in sections.items():
    if name.startswith("coherence rate") or arm_a not in tbl or arm_b not in tbl:
        continue
    na, va = tbl[arm_a]
    nb, vb = tbl[arm_b]
    print(f"### {name}   (n {na} vs {nb})")
    for c, a, b in zip(tbl["_cols"], va, vb):
        if a == 0 and b == 0:
            continue
        zz = z(a, na, b, nb)
        print(f"  {c:>10}  {a:6.1f}%  {b:6.1f}%   d={a-b:+6.1f}  z={zz:+6.2f} {flag(zz)}")
    print()

print("### coherence")
for ds in ("alpaca", "convincing-game", "instructed-deception"):
    if (arm_a, ds) in coh and (arm_b, ds) in coh:
        (na, a), (nb, b) = coh[(arm_a, ds)], coh[(arm_b, ds)]
        zz = z(a, na, b, nb)
        print(f"  {ds:>22}  {a:6.1f}%  {b:6.1f}%   d={a-b:+6.1f}  z={zz:+6.2f} {flag(zz)}")
