#!/usr/bin/env python3
"""
Main table under the GPT-5.6 judge (owner's decision, 2026-10-04): the arms stay as chosen on the validation half with the
Qwen2.5-7B hybrid grader (reports/main_table_final.json), and every LIARS' BENCH test-half honest rate, lead and
interval is recomputed from the GPT-5.6 re-grade of those arms (judge_gpt.py, GPT_DIR/*_judged_gpt_v2.jsonl: keyword
parser where it settles, GPT-5.6 otherwise, beliefs shared). AI Liar is graded by keyword only and is copied unchanged.
Writes reports/main_table_gpt.json with the same structure as main_table_final.json (plus "grader" fields) and prints
both graders side by side.
"""
import glob
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

R = Path("/u/xiaocong/lie_detection/reports")
G = Path(os.environ.get("GPT_DIR", "/work/nvme/biro/xiaocong/lie_out_gpt"))
SHORT = {"Llama-3.1-8B": "Meta-Llama-3p1-8B-Instruct", "Phi-4-mini": "Phi-4-mini-instruct", "Falcon3-3B": "Falcon3-3B-Instruct",
         "OLMo-2-1B": "OLMo-2-0425-1B-Instruct", "Qwen2.5-3B": "Qwen2p5-3B-Instruct", "Qwen2.5-7B": "Qwen2p5-7B-Instruct",
         "Qwen2.5-1.5B": "Qwen2p5-1p5B-Instruct"}
G_OK = ("honest", "lie", "no_claim", "refusal")
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"


def lead(pm, pb):
    U = sorted(set(pm) & set(pb))
    if not U:
        return None
    diffs = [pm[u] - pb[u] for u in U]; rnd = random.Random(0)
    boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
    return {"diff": round(100 * sum(diffs) / len(diffs), 1), "lo": round(100 * boots[50], 1), "hi": round(100 * boots[1949], 1), "n": len(U)}


def main():
    # --sweep: the same recomputation for the size-study models (judge_gpt.py --sweep), so a size series sits on
    # Table 1's grader. Input reports/main_table_sweep.json, output reports/main_table_sweep_gpt.json.
    sweep = "--sweep" in sys.argv
    if sweep:
        SHORT.update(dict((b, a) for a, b in [("Qwen2p5-0p5B-Instruct", "Qwen2.5-0.5B"), ("Qwen2p5-14B-Instruct", "Qwen2.5-14B"),
                                              ("OLMo-2-1124-7B-Instruct", "OLMo-2-7B"), ("OLMo-2-1124-13B-Instruct", "OLMo-2-13B"),
                                              ("OLMo-2-0325-32B-Instruct", "OLMo-2-32B"), ("Falcon3-1B-Instruct", "Falcon3-1B"),
                                              ("Falcon3-7B-Instruct", "Falcon3-7B"), ("Falcon3-10B-Instruct", "Falcon3-10B")]))
    tab = json.load(open(R / ("main_table_sweep.json" if sweep else "main_table_final.json")))
    out = []
    for row in tab:
        if row.get("pending"):
            out.append(row); continue
        model = SHORT[row["model"]]
        per = defaultdict(dict)          # (arm) -> uid -> honest (GPT hybrid, test half, deception split)
        for p in glob.glob(str(G / f"liars_bench_honesty_{model}_*_judged_gpt_v2.jsonl")):
            for r in map(json.loads, open(p)):
                if r["split"] == "instructed-deception" and half(r["uid"]) == "test" and r["verdict_hybrid"] in G_OK:
                    per[r["arm"]].setdefault(r["uid"], r["verdict_hybrid"] == "honest")
        new = {"model": row["model"], "grader": {"lb": "gpt-5.6 hybrid (test half; arms chosen on validation with the Qwen2.5-7B hybrid)", "ail": "keyword"}}
        lb = row["lb"]; cells = {}
        for f, c in lb["cells"].items():
            if c.get("v") is None or c["arm"] not in per:
                cells[f] = dict(c, v7b=c.get("v")); continue
            d = per[c["arm"]]
            cells[f] = dict(c, v=round(100 * sum(d.values()) / len(d), 1), v7b=c["v"], n=len(d))
        L = lb.get("lead"); L2 = None
        if L and "mask" in cells and L["vs"] in cells:
            L2 = lead(per[cells["mask"]["arm"]], per[cells[L["vs"]]["arm"]])
            if L2:
                L2["vs"] = L["vs"]; L2["lead7b"] = {k: L[k] for k in ("diff", "lo", "hi")}
        new["lb"] = {"cells": cells, "lead": L2 or L}
        new["ail"] = row["ail"]
        out.append(new)
        fmt = lambda c: f"{c['v']:5.1f}" if c.get("v") is not None else "   - "
        print(f"{row['model']:<13} lb GPT  " + "  ".join(fmt(cells.get(f, {})) for f in ("mask", "liemask", "steer", "prompt", "unsteered"))
              + (f"   lead {L2['diff']:+.1f} [{L2['lo']:+.1f}, {L2['hi']:+.1f}] vs {L2['vs']}" if L2 else ""))
        print(f"{'':<13} lb 7B   " + "  ".join((f"{c['v7b']:5.1f}" if c.get('v7b') is not None else "   - ") for c in (cells.get(f, {}) for f in ("mask", "liemask", "steer", "prompt", "unsteered")))
              + (f"   lead {L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}] vs {L['vs']}" if L else ""))
    name = "main_table_sweep_gpt.json" if sweep else "main_table_gpt.json"
    json.dump(out, open(R / name, "w"), indent=1)
    print("-> reports/" + name)


if __name__ == "__main__":
    main()
