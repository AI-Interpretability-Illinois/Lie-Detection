#!/usr/bin/env python3
"""
Five further tasks, each method at the one setting per model chosen on the LIARS' BENCH
validation half (queue/make_task_jobs.py); nothing re-tuned per task, all items used.

Metric: honest rate among gradeable items (items where the unedited model's belief is
known). Lead = our mask minus the best other method on THIS task (liemask, steering,
honest prompt, unedited), chosen by point estimate, which favours the baselines; and
lead_pre = our mask minus the comparator fixed in advance (the best other method on the
LIARS' BENCH validation half). Paired bootstrap 95% intervals over items gradeable under both arms.
Writes reports/tasks_table.json and prints markdown tables.
"""
import json
import os
import random
from collections import defaultdict

OUT = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
# GRADER=gpt: the injected-document and AM/GoT columns from the GPT-5.6 re-grade (judge_gpt.py --tasks; files
# <GPT_DIR>/liars_bench_honesty_<model>_t{inject,amgot}_judged_gpt_v2.jsonl, verdict_hybrid = keyword parser where it
# settles, GPT-5.6 otherwise), as Table 1 since v4.3; the other tasks are keyword- or MASK-judge-graded and unchanged.
# Output then goes to reports/tasks_table_gpt.json.
GPT = os.environ.get("GRADER", "") == "gpt"
GPT_DIR = os.environ.get("GPT_DIR", "/work/nvme/biro/xiaocong/lie_out_gpt")
MODELS = [("Meta-Llama-3p1-8B-Instruct", "Llama-3.1-8B"), ("Phi-4-mini-instruct", "Phi-4-mini"),
          ("Falcon3-3B-Instruct", "Falcon3-3B"), ("OLMo-2-0425-1B-Instruct", "OLMo-2-1B"),
          ("Qwen2p5-3B-Instruct", "Qwen2.5-3B"), ("Qwen2p5-7B-Instruct", "Qwen2.5-7B"),
          ("Qwen2p5-1p5B-Instruct", "Qwen2.5-1.5B")]
FAMS = ["mask", "liemask", "steer", "prompt", "unsteered", "random"]
OTHERS = ["liemask", "steer", "prompt", "unsteered"]
MASK_GROUP = {"mask-continuations": "instructed", "mask-disinformation": "instructed",
              "mask-statistics": "instructed", "mask-known-facts": "incentive",
              "mask-doubling-down-known-facts": "incentive", "mask-provided-facts": "incentive"}
# task key -> (file suffix, verdict field, split -> bucket)
TASKS = {
    "sandbag": ("tsandbag_responses", "verdict",
                {"sandbag-mmlu": "sandbag", "sandbag-wmdp": "sandbag", "mmlu-plain": "mmlu_plain"}),
    "inject": ("tinject_judged_Qwen2p5-7B-Instruct_v2", "verdict_hybrid",
               {"inject-deception": "inject", "inject-control": "inject_control"}),
    "mask": ("tmask_maskjudged_Qwen2p5-7B-Instruct_v1", "verdict_mask",
             {s: "mask_" + g for s, g in MASK_GROUP.items()}),
    "amgot": ("tamgot_judged_Qwen2p5-7B-Instruct_v2", "verdict_hybrid", {"amgot-deception": "amgot"}),
    "multi": ("tmulti_responses", "verdict", {"multiturn": "multiturn"}),
}
BUCKETS = ["sandbag", "inject", "mask_instructed", "amgot", "multiturn",
           "mask_incentive", "inject_control", "mmlu_plain"]
LABEL = {"sandbag": "Sandbagging (MMLU+WMDP)", "inject": "Injected instruction",
         "mask_instructed": "MASK, instructed", "amgot": "Azaria-Mitchell / GoT",
         "multiturn": "Multi-turn", "mask_incentive": "MASK, incentive (no instruction)",
         "inject_control": "Injection control (no injection)", "mmlu_plain": "Plain MMLU (capability)"}


def family(a):
    if a.startswith(("weight:fiction", "weight:pooled", "weight:fictf", "weight:pooltf", "weight:poolmix")):
        return "mask"
    for p, f in (("weight:ours", "liemask"), ("weight:magmatch", "random"),
                 ("actsteer", "steer"), ("prompt:honest", "prompt"), ("unsteered", "unsteered")):
        if a.startswith(p):
            return f


def boot(diffs, n=2000):
    rnd = random.Random(0)
    b = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return b[int(0.025 * n)], b[int(0.975 * n) - 1]


def paired(it, other):
    pm, pb = it["mask"], it[other]
    U = sorted(set(pm) & set(pb)); diffs = [pm[u] - pb[u] for u in U]
    lo, hi = boot(diffs)
    return {"vs": other, "diff": round(100 * sum(diffs) / len(diffs), 1),
            "lo": round(100 * lo, 1), "hi": round(100 * hi, 1), "n": len(U)}


def main():
    # pre-registered comparator: the best other method on the LIARS' BENCH validation half
    pre = {r["model"]: r["lb"]["lead"]["vs"]
           for r in json.load(open("/u/xiaocong/lie_detection/reports/main_table_final.json"))}
    table = []
    for short, name in MODELS:
        items = defaultdict(lambda: defaultdict(dict))     # bucket -> fam -> uid -> honest
        arms = {}
        for task, (suf, key, smap) in TASKS.items():
            path = f"{OUT}/liars_bench_honesty_{short}_{suf}.jsonl"
            if GPT and task in ("inject", "amgot"):
                g = f"{GPT_DIR}/liars_bench_honesty_{short}_t{task}_judged_gpt_v2.jsonl"
                if os.path.exists(g):
                    path = g
                else:
                    print(f"no GPT re-grade for {short} {task}: Qwen2.5-7B grades used")
            for l in open(path):
                r = json.loads(l)
                f, b, v = family(r["arm"]), smap.get(r["split"]), r.get(key)
                if f is None or b is None or v in (None, "ungradeable", "deferred"):
                    continue
                items[b][f][r["uid"]] = v == "honest"
                arms[f] = r["arm"]
        row = {"model": name, "arms": arms, "tasks": {}}
        for b in BUCKETS:
            cell = {f: {"v": round(100 * sum(d.values()) / len(d), 1), "n": len(d)}
                    for f, d in items[b].items() if d}
            lead = None
            others = [f for f in OTHERS if f in cell]
            lead_pre = None
            if "mask" in cell and others and b not in ("inject_control", "mmlu_plain"):
                lead = paired(items[b], max(others, key=lambda f: cell[f]["v"]))
                if pre[name] in cell:
                    lead_pre = paired(items[b], pre[name])
            row["tasks"][b] = {"cells": cell, "lead": lead, "lead_pre": lead_pre}
        table.append(row)
    json.dump(table, open(f"/u/xiaocong/lie_detection/reports/tasks_table{'_gpt' if GPT else ''}.json", "w"), indent=1)

    fmt = lambda c, f: f"{c[f]['v']:.1f}" if f in c else "-"
    print("## Lead of our mask over the best other method (95% CI)\n")
    print("| model | " + " | ".join(LABEL[b] for b in BUCKETS[:6]) + " |")
    print("|---" * 7 + "|")
    for row in table:
        cells = []
        for b in BUCKETS[:6]:
            L = row["tasks"][b]["lead"]
            cells.append(f"{L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}] vs {L['vs']}" if L else "-")
        print(f"| {row['model']} | " + " | ".join(cells) + " |")
    print("\n## Lead over the comparator fixed on LIARS' BENCH validation (95% CI)\n")
    print("| model | " + " | ".join(LABEL[b] for b in BUCKETS[:6]) + " |")
    print("|---" * 7 + "|")
    for row in table:
        cells = []
        for b in BUCKETS[:6]:
            L = row["tasks"][b]["lead_pre"]
            cells.append(f"{L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}] vs {L['vs']}" if L else "-")
        print(f"| {row['model']} | " + " | ".join(cells) + " |")
    for b in BUCKETS:
        print(f"\n## {LABEL[b]}: honest rate % (n)\n")
        print("| model | our mask | lie mask | steering | honest prompt | unedited | random mask |")
        print("|---|---|---|---|---|---|---|")
        for row in table:
            c = row["tasks"][b]["cells"]
            print(f"| {row['model']} | " + " | ".join(fmt(c, f) for f in FAMS)
                  + f" |  n={c.get('unsteered', {}).get('n', '-')}")


if __name__ == "__main__":
    main()
