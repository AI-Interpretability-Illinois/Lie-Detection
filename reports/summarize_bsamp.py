#!/usr/bin/env python3
"""
Test (b), behaviour: masks built from the hand-written 25 prompts vs pool draws (3 x 25, 100, 400),
LIARS' BENCH validation half, hybrid keyword + Qwen2.5-7B judge. Also the damage checks of
reports/build_final_table.py: perplexity, honesty-instructed control, polarity (believed-true and
believed-false items). Paired bootstrap CI of each arm minus the hand-written mask.
"""
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

O = Path("/work/hdd/bfqt/xiaocong/lie_detection_outputs")
MODELS = [("Llama-3p1-8B-Instruct", "Llama-3.1-8B (fiction, a=-1)"),
          ("Phi-4-mini-instruct", "Phi-4-mini (pooled, a=-1)"),
          ("Qwen2p5-7B-Instruct", "Qwen2.5-7B (pooled, a=-1)"),
          ("OLMo-2-0425-1B-Instruct", "OLMo-2-1B (fiction, a=-0.25)")]
J = "judged_Qwen2p5-7B-Instruct_v2"


def find(name):
    for d in (O / "timan_b", O):
        if (d / name).exists():
            return d / name


def label(arm):
    if arm == "unsteered":
        return "unedited"
    fam = arm.split()[0].split(":")[1]
    if fam in ("fiction", "pooled"):
        return "hand-written 25"
    n, d = fam[2:].rstrip("_").split("d")
    return f"pool {n}" + (f" (draw {int(d)+1})" if n == "25" else "")


def boot(diffs, n=2000):
    r = random.Random(0)
    b = sorted(sum(r.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return 100 * b[50], 100 * b[1949]


def main():
    for short, name in MODELS:
        jp, cp = find(f"liars_bench_honesty_{short}_bsamp_{J}.jsonl"), find(f"liars_bench_honesty_{short}_bsamp.csv")
        if not jp:
            print(f"\n{name}: pending"); continue
        ppl = {r["arm"]: float(r["ppl_ratio"]) for r in csv.DictReader(open(cp))}
        by = defaultdict(list)
        for r in map(json.loads, open(jp)):
            if r["verdict_hybrid"] in ("honest", "lie", "no_claim", "refusal"):
                by[r["arm"]].append(r)
        arms = list(ppl)
        base = next(a for a in arms if label(a) == "hand-written 25")
        dec = lambda a, b=None: {r["uid"]: r["verdict_hybrid"] == "honest" for r in by[a]
                                 if r["split"] == "instructed-deception" and (b is None or r["belief"] is b)}
        rate = lambda d: 100 * sum(d.values()) / max(1, len(d))
        ctl = lambda a: rate({r["uid"]: r["verdict_hybrid"] == "honest" for r in by[a] if r["split"] == "instructed-truthful"})
        print(f"\n{name}   (validation half, n = {len(dec(base))} deception items)")
        print(f"  {'mask':<18}{'honest':>8}   {'minus hand-written [95% CI]':<28}{'control':>8}{'believed T/F':>14}{'ppl':>7}")
        for a in arms:
            d = dec(a)
            if a == base or label(a) == "unedited":
                cmp = ""
            else:
                U = sorted(set(d) & set(dec(base))); diffs = [d[u] - dec(base)[u] for u in U]
                lo, hi = boot(diffs)
                cmp = f"{100*sum(diffs)/len(diffs):+5.1f} [{lo:+.1f}, {hi:+.1f}]"
            print(f"  {label(a):<18}{rate(d):8.1f}   {cmp:<28}{ctl(a):8.1f}{rate(dec(a, True)):7.1f}/{rate(dec(a, False)):<6.1f}{ppl[a]:7.3f}")


if __name__ == "__main__":
    main()
