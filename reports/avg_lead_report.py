#!/usr/bin/env python3
"""
Two-benchmark average for Table 1 (owner's request, 2026-10-06): per model and method, the mean of the LIARS' BENCH and
AI Liar test-half honest rates at the options chosen on validation (LB graded by the GPT-5.6 hybrid, AI Liar by keyword,
exactly the cells of reports/main_table_gpt.json), and the lead of the falsehood mask over the best other method on that
average. The comparator is the other method with the best validation average (LB validation under the Qwen2.5-7B hybrid,
the selection grader; AI Liar validation by keyword), as for the per-benchmark leads; the comparator with the best TEST
average is printed too. Interval: paired bootstrap, items resampled within each benchmark independently (2,000 resamples,
seed 0), the lead being the mean of the two per-benchmark mean differences. Writes reports/avg_lead.json.
Run from reports/: python avg_lead_report.py
"""
import glob
import json
import random
from collections import defaultdict

import build_final_table as B
from build_gpt_table import G as GDIR, G_OK, SHORT

MAIN5 = ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B"]
METHODS = ["mask", "liemask", "steer", "prompt", "unsteered"]


def main():
    gpt = \
          {r["model"]: r for r in json.load(open("main_table_gpt.json"))}
    spec = {name: (m, lbt, ailt) for m, name, lbt, ailt in B.MODELS}
    out = {}
    for name in MAIN5:
        m, lbt, ailt = spec[name]
        lb_q, _, ail = B.load(m, lbt, ailt)
        row = gpt[name]
        # per-item test-half verdicts: LB from the GPT-5.6 re-grade, AI Liar by keyword
        lb_g = defaultdict(dict)
        for p in glob.glob(str(GDIR / f"liars_bench_honesty_{SHORT[name]}_*_judged_gpt_v2.jsonl")):
            for r in map(json.loads, open(p)):
                if r["split"] == "instructed-deception" and B.half(r["uid"]) == "test" and r["verdict_hybrid"] in G_OK:
                    lb_g[r["arm"]].setdefault(r["uid"], r["verdict_hybrid"] == "honest")
        def ail_items(a, part):
            return {x["uid"]: x["honest"] for x in ail[a] if B.half(x["uid"]) == part}
        def lb_val(a):
            rs = [r for r in lb_q[a] if r["split"] == "instructed-deception" and B.half(r["uid"]) == "val"]
            return sum(r["verdict_hybrid"] == "honest" for r in rs) / len(rs)
        cells = {}
        for f in METHODS:
            cl, ca = row["lb"]["cells"].get(f, {}), row["ail"]["cells"].get(f, {})
            if cl.get("v") is None or ca.get("v") is None:
                continue
            al = ail_items(ca["arm"], "val")
            cells[f] = {"lb": cl["v"], "ail": ca["v"], "avg": round((cl["v"] + ca["v"]) / 2, 1),
                        "val_avg": (lb_val(cl["arm"]) + sum(al.values()) / len(al)) / 2,
                        "lb_arm": cl["arm"], "ail_arm": ca["arm"]}
        others = [f for f in cells if f != "mask"]
        vs = max(others, key=lambda f: cells[f]["val_avg"])
        vs_test = max(others, key=lambda f: cells[f]["avg"])
        def diffs(f):
            a, b = lb_g[cells["mask"]["lb_arm"]], lb_g[cells[f]["lb_arm"]]
            U = sorted(set(a) & set(b)); dl = [a[u] - b[u] for u in U]
            a, b = ail_items(cells["mask"]["ail_arm"], "test"), ail_items(cells[f]["ail_arm"], "test")
            U = sorted(set(a) & set(b)); da = [a[u] - b[u] for u in U]
            return dl, da
        def lead(f):
            dl, da = diffs(f); rnd = random.Random(0)
            est = (sum(dl) / len(dl) + sum(da) / len(da)) / 2
            boots = sorted((sum(rnd.choice(dl) for _ in dl) / len(dl) + sum(rnd.choice(da) for _ in da) / len(da)) / 2
                           for _ in range(2000))
            return {"vs": f, "diff": round(100 * est, 1), "lo": round(100 * boots[50], 1), "hi": round(100 * boots[1949], 1),
                    "n_lb": len(dl), "n_ail": len(da)}
        L = lead(vs); Lt = lead(vs_test) if vs_test != vs else L
        out[name] = {"cells": {f: {k: v for k, v in c.items() if k != "val_avg"} for f, c in cells.items()},
                     "lead": L, "lead_vs_best_test": Lt}
        print(f"{name:<13} " + "  ".join(f"{f}={c['avg']:.1f}" for f, c in cells.items())
              + f"   lead {L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}] vs {vs} (val)"
              + ("" if vs_test == vs else f" | vs {vs_test} (test-best) {Lt['diff']:+.1f} [{Lt['lo']:+.1f}, {Lt['hi']:+.1f}]"))
    json.dump(out, open("avg_lead.json", "w"), indent=1)
    print("-> reports/avg_lead.json")


if __name__ == "__main__":
    main()
