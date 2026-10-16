#!/usr/bin/env python3
"""
Post-hoc comparison for the Discussion (added 2026-10-06, after the main table was fixed): when the attacker writes the
prompt, the honest-prompt baseline is not available to the defender, so the edit's relevant comparators are the methods
that live outside the prompt. For every main-table model this prints, on the LIARS' BENCH test half under the GPT-5.6
hybrid grade (the same files and arms as reports/build_gpt_table.py), the paired lead of the mask over activation steering
and over the lie-instruction mask, each with a paired bootstrap 95% interval (2,000 resamples, seed 0).
Writes reports/steer_lead.json.
"""
import json
from pathlib import Path

from build_gpt_table import G, G_OK, R, SHORT, half, lead
import glob
from collections import defaultdict


def main():
    tab = json.load(open(R / "main_table_gpt.json"))
    out = {}
    for row in tab:
        model = SHORT[row["model"]]
        per = defaultdict(dict)
        for p in glob.glob(str(G / f"liars_bench_honesty_{model}_*_judged_gpt_v2.jsonl")):
            for r in map(json.loads, open(p)):
                if r["split"] == "instructed-deception" and half(r["uid"]) == "test" and r["verdict_hybrid"] in G_OK:
                    per[r["arm"]].setdefault(r["uid"], r["verdict_hybrid"] == "honest")
        cells = row["lb"]["cells"]; res = {}
        for f in ("steer", "liemask", "unsteered"):
            c = cells.get(f, {})
            if c.get("v") is None or c.get("arm") not in per:
                res[f] = None; continue
            res[f] = lead(per[cells["mask"]["arm"]], per[c["arm"]])
        out[row["model"]] = res
        fmt = lambda L: "   --" if not L else f"{L['diff']:+5.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}]"
        print(f"{row['model']:<13} vs steer {fmt(res['steer']):<22} vs lie-mask {fmt(res['liemask']):<22} vs unedited {fmt(res['unsteered'])}")
    json.dump(out, open(R / "steer_lead.json", "w"), indent=1)
    print("-> reports/steer_lead.json")


if __name__ == "__main__":
    main()
