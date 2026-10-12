#!/usr/bin/env python3
"""
Over-refusal check (extras plan item 5; build_legit_evalsets.py, judge_legit.py): on 150 legitimate requests that call for a
false statement inside a frame (fiction, role-play, counterfactual), the share of replies that COMPLIED, CORRECTED the claim
instead, or REFUSED, per model and arm (each method at its LIARS' BENCH-chosen setting), labels by the GPT-5.6 judge.
Reads <SWEEP>/legit_judged_gpt/<model>_legit_judged.jsonl; writes reports/legit_table.json and reports/legit_report.md.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from summarize_tasks import family   # noqa: E402

R = Path("/u/xiaocong/lie_detection/reports")
D = Path("/work/nvme/biro/xiaocong/lie_sweep/legit_judged_gpt")
NAMES = {"Meta-Llama-3p1-8B-Instruct": "Llama-3.1-8B", "Phi-4-mini-instruct": "Phi-4-mini", "Falcon3-3B-Instruct": "Falcon3-3B",
         "OLMo-2-0425-1B-Instruct": "OLMo-2-1B", "Qwen2p5-3B-Instruct": "Qwen2.5-3B", "Qwen2p5-7B-Instruct": "Qwen2.5-7B",
         "Qwen2p5-1p5B-Instruct": "Qwen2.5-1.5B"}
FAMS = ["mask", "liemask", "steer", "prompt", "unsteered"]
KINDS = ["legit-fiction", "legit-roleplay", "legit-counterfactual"]


def main():
    main_tab = {r["model"]: r for r in json.load(open(R / "main_table_final_menu0.json")) if not r.get("pending")}   # the options chosen on the original menu (Table 1 before the 2026-10-11 menu extension), at which this analysis was run
    out = []
    L = ["# Legitimate falsehoods: do the edited models still comply? (GPT-5.6 labels)", "",
         "150 requests (50 fiction, 50 role-play, 50 counterfactual) that legitimately ask for a false statement inside a frame;",
         "share of replies that complied / corrected the claim instead / refused, every method at its LIARS' BENCH-chosen setting.", ""]
    for stem, name in NAMES.items():
        p = D / f"{stem}_legitall_judged.jsonl"   # legit + legitfill (2026-10-07), if judged
        if not p.exists():
            p = D / f"{stem}_legit_judged.jsonl"
        if not p.exists():
            continue
        chosen = {f: c.get("arm") for f, c in main_tab[name]["lb"]["cells"].items()} if name in main_tab else {}
        tally = defaultdict(Counter); by_kind = defaultdict(Counter)
        for r in map(json.loads, open(p)):
            f = family(r["arm"])
            if f is None or not r.get("label") or (chosen and re.sub(r" \(c=[^)]*\)", "", r["arm"]) != re.sub(r" \(c=[^)]*\)", "", chosen.get(f) or "")):   # only the option Table 1 chose
                continue
            tally[f][r["label"]] += 1
            by_kind[(f, "legit-" + str(r.get("kind", r.get("split", ""))).split("-")[-1])][r["label"]] += 1   # kind may lack the "legit-" prefix
        row = {"model": name, "arms": {}}
        L += [f"## {name}", "", "| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |", "|---|---|---|---|---|---|"]
        for f in FAMS:
            if f not in tally:
                continue
            c = tally[f]; n = sum(c.values())
            kinds = [by_kind[(f, k)] for k in KINDS]
            row["arms"][f] = {"n": n, **{lab.lower(): round(100 * c[lab] / n, 1) for lab in ("COMPLIED", "CORRECTED", "REFUSED")},
                              "complied_by_kind": {k: round(100 * kc["COMPLIED"] / max(1, sum(kc.values())), 1) for k, kc in zip(KINDS, kinds)}}
            L.append(f"| {f} | {n} | {100 * c['COMPLIED'] / n:.1f} | {100 * c['CORRECTED'] / n:.1f} | {100 * c['REFUSED'] / n:.1f} | "
                     + " / ".join(f"{100 * kc['COMPLIED'] / max(1, sum(kc.values())):.0f}" for kc in kinds) + " |")
        L.append("")
        out.append(row)
    json.dump(out, open(R / "legit_table.json", "w"), indent=1)
    (R / "legit_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
