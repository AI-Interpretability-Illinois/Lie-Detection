#!/usr/bin/env python3
"""The task averages of model_avg_report.table2 for the runs at Table 1's final options (review, 2026-10-12): mean over the
five models of the per-model lead over the comparator fixed in advance (tasks_table_final.json lead_pre), items resampled
within each model; plain-MMLU change of the mask and the random mask. Injected / AM-GoT graded by the Qwen2.5-7B hybrid.
Writes reports/model_avg_tasks_final.json ({"table2": ...}, the shape make_tables.tasks_table reads)."""
import json
import os
from collections import defaultdict

os.environ["FINAL"] = "1"
import summarize_tasks as T                        # noqa: E402
from model_avg_report import MAIN5, boot_mean_over_models   # noqa: E402

pre = {r["model"]: r["tasks"]["sandbag"]["lead_pre"]["vs"] for r in json.load(open("tasks_table_final.json"))}
items = {}
for short, name in T.MODELS[:5]:
    it = defaultdict(lambda: defaultdict(dict))
    for task, (suf, key, smap) in T.TASKS.items():
        path = f"{T.OUT if task == 'mask' else T.RVD}/liars_bench_honesty_{short}_{suf}.jsonl"
        for l in open(path):
            r = json.loads(l)
            f, b, v = T.family(r["arm"]), smap.get(r["split"]), r.get(key)
            if f is None or b is None or v in (None, "ungradeable", "deferred"):
                continue
            it[b][f][r["uid"]] = v == "honest"
    items[name] = it
out = {}
for b in ("sandbag", "inject", "mask_instructed", "amgot", "multiturn", "mask_incentive"):
    dl = []
    for n in MAIN5:
        a, c = items[n][b]["mask"], items[n][b][pre[n]]
        U = sorted(set(a) & set(c)); dl.append([a[u] - c[u] for u in U])
    out[b] = boot_mean_over_models(dl)
    print(f"{b:16s} mean lead over models {out[b]['diff']:+.1f} [{out[b]['lo']:+.1f}, {out[b]['hi']:+.1f}]")
acc = lambda n, f: 100 * sum(items[n]["mmlu_plain"][f].values()) / len(items[n]["mmlu_plain"][f])
for f in ("mask", "random", "steer", "prompt", "liemask"):
    d = sum(acc(n, f) - acc(n, "unsteered") for n in MAIN5) / len(MAIN5)
    out[f"mmlu_{f}"] = round(d, 1); print(f"mmlu delta {f}: {d:+.2f}")
json.dump({"table2": out}, open("model_avg_tasks_final.json", "w"), indent=1)
