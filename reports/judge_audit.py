#!/usr/bin/env python3
"""
Judge audit (extras plan, Oct 4): the Qwen2.5-7B judge against the GPT judge (judge_gpt.py) and, when present,
human labels.

Per generator family (Llama, Phi, Falcon, OLMo, Qwen) and overall, on the chosen arms of the main table:
  * agreement and Cohen's kappa of the two judges' stance labels (AGREES / DISAGREES / NEITHER), on replies
    the keyword parser did not settle (where the judge decides) and on all replies;
  * the honest rate per arm under the 7B-hybrid grader and under the GPT-hybrid grader, and the change in the
    mask's lead over the best other method (test half) with a paired bootstrap 95% interval;
  * self-preference: the 7B judge's leniency (share of replies it calls honest that GPT calls lie, minus the
    reverse) on Qwen-generated replies vs the other families; the same against the human labels if
    reports/judge_human_labels.jsonl exists (fields uid, arm, model, label in {honest, lie, no_claim, refusal}).
Writes reports/judge_audit.md.
"""
import glob
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

G = Path(os.environ.get("GPT_DIR", "/work/nvme/biro/xiaocong/lie_out_gpt"))
FAM = {"Meta-Llama-3p1-8B-Instruct": "Llama", "Phi-4-mini-instruct": "Phi", "Falcon3-3B-Instruct": "Falcon",
       "OLMo-2-0425-1B-Instruct": "OLMo", "Qwen2p5-3B-Instruct": "Qwen", "Qwen2p5-7B-Instruct": "Qwen", "Qwen2p5-1p5B-Instruct": "Qwen"}
NAME = {"Meta-Llama-3p1-8B-Instruct": "Llama-3.1-8B", "Phi-4-mini-instruct": "Phi-4-mini", "Falcon3-3B-Instruct": "Falcon3-3B",
        "OLMo-2-0425-1B-Instruct": "OLMo-2-1B", "Qwen2p5-3B-Instruct": "Qwen2.5-3B", "Qwen2p5-7B-Instruct": "Qwen2.5-7B",
        "Qwen2p5-1p5B-Instruct": "Qwen2.5-1.5B"}
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"


def kappa(pairs):
    n = len(pairs)
    if not n:
        return float("nan"), float("nan")
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / n / n
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def lab(c):
    return "NEITHER" if c is None else ("AGREES" if c else "DISAGREES")


def main():
    rows = defaultdict(list)
    for p in sorted(glob.glob(str(G / "liars_bench_honesty_*_judged_gpt_v2.jsonl"))):
        stem = Path(p).name.replace("liars_bench_honesty_", "").split("_judged")[0]
        model = stem.rsplit("_", 1)[0]
        for r in map(json.loads, open(p)):
            if r["split"] != "instructed-deception" and not r["split"].startswith(("inject", "amgot")):
                continue
            rows[model].append(r)
    human = {}
    hp = Path("/u/xiaocong/lie_detection/reports/judge_human_labels.jsonl")
    if hp.exists():
        for r in map(json.loads, open(hp)):
            human[(r["model"], r["arm"], r["uid"])] = r["label"]
    out = ["# Judge audit: Qwen2.5-7B judge vs GPT (gpt-5.6-sol)", "",
           "Stance labels on the same replies (AGREES / DISAGREES / NEITHER); beliefs shared. Chosen arms of the main table.", ""]
    out.append("| family | replies | judged (keyword unsettled) | agreement | kappa | 7B calls honest, GPT lie | GPT honest, 7B lie |")
    out.append("|---|---|---|---|---|---|---|")
    tab = json.load(open("/u/xiaocong/lie_detection/reports/main_table_final_menu0.json"))   # the options chosen on the original menu (Table 1 before the 2026-10-11 menu extension), at which this analysis was run
    byfam = defaultdict(list)
    for m, rs in rows.items():
        byfam[FAM[m]] += rs
    byfam["all"] = [r for rs in rows.values() for r in rs]
    for fam, rs in byfam.items():
        uns = [r for r in rs if r.get("keyword_claim") is None and r.get("judge_claim") != "bad"]
        pairs = [(lab(r["qwen7b_claim"]), lab(r["judge_claim"])) for r in uns]
        po, k = kappa(pairs)
        len7 = sum(r["verdict_qwen7b_hybrid"] == "honest" and r["verdict_judge"] == "lie" for r in rs)
        leng = sum(r["verdict_hybrid"] == "honest" and r["verdict_qwen7b_hybrid"] == "lie" for r in rs)
        out.append(f"| {fam} | {len(rs)} | {len(uns)} | {po:.3f} | {k:.3f} | {len7} ({100*len7/max(1,len(rs)):.1f}%) | {leng} ({100*leng/max(1,len(rs)):.1f}%) |")
    out += ["", "## Honest rate on the test half, chosen arms: 7B-hybrid -> GPT-hybrid", "",
            "| model | arm | 7B | GPT | n |", "|---|---|---|---|---|"]
    lead = ["", "## Lead of the mask over the best other method (test half), 7B grader vs GPT grader", "",
            "| model | vs | 7B lead [95% CI] | GPT lead [95% CI] |", "|---|---|---|---|"]
    for m, rs in rows.items():
        name = NAME[m]
        row = next((r for r in tab if r["model"] == name), None)
        per = defaultdict(dict)
        for r in rs:
            if r["split"] == "instructed-deception" and half(r["uid"]) == "test":
                for g, key in (("7B", "verdict_qwen7b_hybrid"), ("GPT", "verdict_hybrid")):
                    if r[key] in ("honest", "lie", "no_claim", "refusal"):
                        per[(r["arm"], g)][r["uid"]] = r[key] == "honest"
        arms = sorted({a for a, _ in per})
        for a in arms:
            d7, dg = per.get((a, "7B"), {}), per.get((a, "GPT"), {})
            if d7:
                out.append(f"| {name} | {a} | {100*sum(d7.values())/len(d7):.1f} | {100*sum(dg.values())/max(1,len(dg)):.1f} | {len(d7)} |")
        if row and row["lb"]["lead"]:
            L = row["lb"]["lead"]; mask_arm = row["lb"]["cells"]["mask"]["arm"]; best_arm = row["lb"]["cells"][L["vs"]]["arm"]
            cells = []
            for g in ("7B", "GPT"):
                pm, pb = per.get((mask_arm, g), {}), per.get((best_arm, g), {})
                U = sorted(set(pm) & set(pb))
                if not U:
                    cells.append("-"); continue
                diffs = [pm[u] - pb[u] for u in U]; rnd = random.Random(0)
                boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
                cells.append(f"{100*sum(diffs)/len(diffs):+.1f} [{100*boots[50]:+.1f}, {100*boots[1949]:+.1f}] (n={len(U)})")
            lead.append(f"| {name} | {L['vs']} | {cells[0]} | {cells[1]} |")
    out += lead
    if human:
        out += ["", "## Human labels", "", "| judge | agreement with human (honest/lie/other) | n |", "|---|---|---|"]
        for g, key in (("7B-hybrid", "verdict_qwen7b_hybrid"), ("GPT-hybrid", "verdict_hybrid")):
            pairs = []
            for m, rs in rows.items():
                for r in rs:
                    h = human.get((m, r["arm"], r["uid"]))
                    if h:
                        c = lambda v: v if v in ("honest", "lie") else "other"
                        pairs.append((c(r[key]), c(h)))
            po, k = kappa(pairs)
            out.append(f"| {g} | {po:.3f} (kappa {k:.3f}) | {len(pairs)} |")
    Path("/u/xiaocong/lie_detection/reports/judge_audit.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
