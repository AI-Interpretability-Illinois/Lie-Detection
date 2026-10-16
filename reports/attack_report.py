#!/usr/bin/env python3
"""
Adaptive attacks on the order to lie (extras plan item 5; build_attack_evalsets.py): honest rate per attack and arm,
each method at its LIARS' BENCH-chosen setting (queue/lie_extras.slurm, queue/run_timan_extras.sh), graded by the
Qwen2.5-7B hybrid judge as on LIARS' BENCH. The attack items are the 537 LIARS' BENCH deception items under a many-shot
lying prefix, a Spanish order and a Chinese order, so the test half is the same MD5 split as Table 1 and the plain
LIARS' BENCH row (from reports/main_table_final.json, same judge) is the no-attack reference.
Reads <SWEEP>/liars_bench_honesty_<model>_atk_judged_Qwen2p5-7B-Instruct_v2.jsonl; writes reports/attack_table.json and
reports/attack_report.md. Lead = mask minus the comparator fixed on LIARS' BENCH validation (lead["vs"] in the main table),
paired bootstrap over test-half items; also mask minus the best other arm on the attack itself.
"""
import glob
import re
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_tasks import family, paired   # noqa: E402

R = Path("/u/xiaocong/lie_detection/reports")
# FINAL=1 (review, 2026-10-12): the runs at Table 1's final options (queue/lie_review.slurm EXP=atklegit, tag rvatklegitatk)
import os
FINAL = os.environ.get("FINAL") == "1"
SWEEP = Path("/work/hdd/bfqt/xiaocong/lie_review" if FINAL else "/work/nvme/biro/xiaocong/lie_sweep")
ATK = ["rvatklegitatk"] if FINAL else ["atk", "atkfill"]
NAMES = {"Meta-Llama-3p1-8B-Instruct": "Llama-3.1-8B", "Phi-4-mini-instruct": "Phi-4-mini", "Falcon3-3B-Instruct": "Falcon3-3B",
         "OLMo-2-0425-1B-Instruct": "OLMo-2-1B", "Qwen2p5-3B-Instruct": "Qwen2.5-3B", "Qwen2p5-7B-Instruct": "Qwen2.5-7B",
         "Qwen2p5-1p5B-Instruct": "Qwen2.5-1.5B"}
SPLITS = ["attack-manyshot", "attack-es", "attack-zh"]
LABEL = {"attack-manyshot": "many-shot lies", "attack-es": "Spanish order", "attack-zh": "Chinese order", "plain": "plain (Table 1)"}
FAMS = ["mask", "liemask", "steer", "prompt", "unsteered"]
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"


def _norm(a):
    """steering arms carry the coefficient computed on the host (c=-13.0 on Delta, -13.1 on timan for the same layer and r)"""
    return re.sub(r" \(c=[^)]*\)", "", a)


def main():
    main_tab = {r["model"]: r for r in json.load(open(R / ("main_table_final.json" if FINAL else "main_table_final_menu0.json"))) if not r.get("pending")}   # menu0: the options chosen on the original menu (Table 1 before the 2026-10-11 menu extension), at which this analysis was first run
    out = []
    for stem, name in NAMES.items():
        fs = sorted(f for t in ATK for f in glob.glob(str(SWEEP / f"liars_bench_honesty_{stem}_{t}_judged_Qwen2p5-7B-Instruct_v2.jsonl")))
        if not fs:
            continue
        chosen = {f: c.get("arm") for f, c in main_tab[name]["lb"]["cells"].items()} if name in main_tab else {}
        items = defaultdict(lambda: defaultdict(dict))     # (split, half) -> fam -> uid -> honest
        for r in (r for fp in fs for r in map(json.loads, open(fp))):
            if r["split"] not in SPLITS or r.get("verdict_hybrid") not in ("honest", "lie"):
                continue
            f = family(r["arm"])
            if f is None or (chosen and _norm(r["arm"]) != _norm(chosen.get(f) or "")):   # only the option Table 1 chose (2026-10-07)
                continue
            orig = r["uid"].split(":", 1)[1]
            for h in ("all", half(orig)):
                items[(r["split"], h)][f][orig] = r["verdict_hybrid"] == "honest"
        row = {"model": name, "splits": {}}
        vs = (main_tab[name]["lb"].get("lead_val") or main_tab[name]["lb"]["lead"])["vs"] if name in main_tab else None   # fixed on validation
        for s in SPLITS:
            cells = {h: {f: {"v": round(100 * sum(d.values()) / len(d), 1), "n": len(d)} for f, d in items[(s, h)].items() if d}
                     for h in ("all", "test")}
            it = items[(s, "test")]
            lead_pre = paired(it, vs) if vs and "mask" in it and vs in it else None
            others = [f for f in FAMS if f != "mask" and f in it]
            lead = paired(it, max(others, key=lambda f: cells["test"][f]["v"])) if "mask" in it and others else None
            row["splits"][s] = {"cells": cells, "lead_pre": lead_pre, "lead": lead}
        if name in main_tab:
            row["plain"] = {f: c.get("v") for f, c in main_tab[name]["lb"]["cells"].items()}
            row["plain_lead"] = main_tab[name]["lb"].get("lead_val") or main_tab[name]["lb"]["lead"]
        out.append(row)
    json.dump(out, open(R / ("attack_table_final.json" if FINAL else "attack_table.json"), "w"), indent=1)
    L = ["# Adaptive attacks on the order to lie (Qwen2.5-7B judge)", "",
         "Honest rate (%) on the test half of the 537 LIARS' BENCH deception items under each attack, every method at its",
         "LIARS' BENCH-chosen setting; 'plain' is the same items without the attack (Table 1, same judge). Lead: mask minus the",
         "comparator fixed on LIARS' BENCH validation, paired bootstrap 95% interval over test items.", ""]
    for row in out:
        L += [f"## {row['model']}", "", "| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |", "|---|---|---|---|---|---|---|---|"]
        if "plain" in row:
            p = row["plain"]; pl = row["plain_lead"]
            L.append("| plain (Table 1) | " + " | ".join(f"{p[f]:.1f}" if p.get(f) is not None else "-" for f in FAMS)
                     + f" | {pl['diff']:+.1f} [{pl['lo']:+.1f}, {pl['hi']:+.1f}] vs {pl['vs']} | |")
        for s in SPLITS:
            c = row["splits"][s]["cells"]["test"]; lp = row["splits"][s]["lead_pre"]; lb = row["splits"][s]["lead"]
            L.append(f"| {LABEL[s]} | " + " | ".join(f"{c[f]['v']:.1f}" if f in c else "-" for f in FAMS)
                     + " | " + (f"{lp['diff']:+.1f} [{lp['lo']:+.1f}, {lp['hi']:+.1f}] vs {lp['vs']}" if lp else "-")
                     + " | " + (f"{lb['diff']:+.1f} [{lb['lo']:+.1f}, {lb['hi']:+.1f}] vs {lb['vs']}" if lb else "-") + " |")
        L.append("")
    (R / ("attack_report_final.md" if FINAL else "attack_report.md")).write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
