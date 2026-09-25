#!/usr/bin/env python3
"""
Analysis for run_reversed_contrast.py.

The raw wrong-rate that run_reversed_contrast.py prints is not the quantity to
compare arms on, for two reasons this script corrects:

  * Baseline errors.  The unsteered model already misses ~8% of the questions.
    Only items it answered correctly can be flipped, so rates are computed over
    that subset.

  * Substring-match artifacts.  `_is_correct` is a substring test, so a verbose
    but correct answer ("299,792 kilometers per second" for gold "300000",
    "Amperage" for "ampere") scores as wrong.  Steering makes the model verbose,
    which inflates its wrong-rate without it lying once.  Flips are therefore
    split by response length: a flip of six words or fewer that passes the junk
    test is a clean falsehood ("Canada" for the largest country), and longer
    flips are reported separately as unreliable.

Arms are then compared at matched WikiText perplexity rather than matched alpha,
because alpha buys damage at very different rates for different masks and the
whole question is whether the lie-specific mask beats a control at equal damage.

Usage
-----
  python analyze_reversed_contrast.py \
      /work/hdd/bfqt/xiaocong/lie_detection_outputs 'reversed_contrast_Qwen2p5-1p5B*'
"""

import csv
import glob
import json
import random
import sys
from pathlib import Path

TERSE_WORDS = 6
BANDS = {"~1.03x": (1.00, 1.06), "~1.14x": (1.07, 1.20), "~1.34x": (1.28, 1.42),
         "~1.79x": (1.70, 1.92), "~3.0x": (2.5, 3.6)}


def tag_of(path):
    for t in ("seed1", "hialpha"):
        if t in path:
            return t
    return "base"


def boot_ci(k, n, B=2000, seed=0):
    if n == 0:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    xs = [1] * k + [0] * (n - k)
    draws = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(B))
    return 100 * draws[int(0.025 * B)], 100 * draws[int(0.975 * B)]


def main():
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1
                   else "/work/hdd/bfqt/xiaocong/lie_detection_outputs")
    pattern = sys.argv[2] if len(sys.argv) > 2 else "reversed_contrast_*"

    recs, ppl = [], {}
    for f in sorted(glob.glob(str(out_dir / (pattern + "_responses.jsonl")))):
        for line in open(f):
            r = json.loads(line)
            r["tag"] = tag_of(f)
            recs.append(r)
    for f in sorted(glob.glob(str(out_dir / (pattern + ".csv")))):
        if f.endswith("_responses.jsonl"):
            continue
        for r in csv.DictReader(open(f)):
            ppl[(tag_of(f), r["arm"], float(r["alpha"]))] = float(r["ppl_ratio"])
    if not recs:
        sys.exit(f"no responses matched {pattern} under {out_dir}")

    # Baseline answers come from the primary run; every rate is over the items
    # it got right.
    base = {r["q"]: r for r in recs
            if r["arm"] == "none" and r["pass"] == "greedy" and r["tag"] == "base"}

    groups = {}
    for r in recs:
        if r["pass"] != "greedy" or r["arm"] == "none":
            continue
        groups.setdefault((r["tag"], r["arm"], r["alpha"]), []).append(r)

    print(f"{'arm':<9} {'seed':<5}{'alpha':>6} | {'ppl x':>9} | "
          f"{'clean flips (95% CI)':<24} | {'verbose':>8}{'junk':>7}")
    print("-" * 78)
    table = []
    for key in sorted(groups, key=lambda k: (k[1], k[0], k[2])):
        rs = groups[key]
        ok = [r for r in rs if base[r["q"]]["correct"]]
        flips = [r for r in ok if not r["correct"]]
        clean = [r for r in flips
                 if len(r["resp"].split()) <= TERSE_WORDS and not r["junk"]]
        verbose = [r for r in flips if len(r["resp"].split()) > TERSE_WORDS]
        lo, hi = boot_ci(len(clean), len(ok))
        rate = 100 * len(clean) / max(1, len(ok))
        p = ppl.get(key, float("nan"))
        seed = "1" if key[0] == "seed1" else "0"
        table.append((key[1], seed, key[2], p, rate, lo, hi))
        pstr = f"{p:9.2f}" if p == p and p < 1e4 else f"{p:9.1e}"
        print(f"{key[1]:<9} {seed:<5}{key[2]:>6.1f} | {pstr} | "
              f"{rate:>5.1f}%  [{lo:4.1f}, {hi:4.1f}]        | "
              f"{100*len(verbose)/max(1,len(ok)):>7.1f}%"
              f"{100*sum(r['junk'] for r in rs)/len(rs):>6.1f}%")

    print("\nMatched-damage comparison — clean flips at similar WikiText ppl ratio")
    print("(the lie-specific mask has to beat these controls to be lie-specific)")
    for band, (lo_, hi_) in BANDS.items():
        sel = [t for t in table if lo_ <= t[3] <= hi_]
        if sel:
            cells = "   ".join(f"{t[0][:4]}(a{t[2]:g},s{t[1]}) {t[4]:.1f}%"
                               for t in sorted(sel, key=lambda t: -t[4]))
            print(f"  {band:<8}{cells}")


if __name__ == "__main__":
    main()
