#!/usr/bin/env python3
"""
Averages over the five main-table models (owner's request, 2026-10-06: "average on models for each task").

Table 1 (LIARS' BENCH, GPT-5.6 hybrid; AI Liar, keyword; test halves, the arms of reports/main_table_gpt.json):
  each method's honest rate averaged over the five models. A method with no usable option on a model ("--" in Table 1)
  counts as the unedited model there, since it would leave the model unedited. The lead of the mask is taken over the
  other method with the best five-model average on the TEST halves, i.e. the best baseline in the average row (owner,
  2026-10-11); "lead_val" keeps the earlier comparator, the best five-model average on the validation halves (LIARS'
  BENCH validation under the Qwen2.5-7B hybrid, the selection grader; AI Liar by keyword). Interval: items resampled
  within each model (paired across methods), the lead being the mean over models of the per-model mean differences;
  2,000 resamples, seed 0.
Table 2 (further tasks): for each task, the mean over the five models of the lead over the comparator fixed in advance
  (as in Table 2), same bootstrap; and the mean change in plain MMLU accuracy under the mask and the random mask.
Writes reports/model_avg.json. Run from reports/.
"""
import glob
import json
import os
import random
from collections import defaultdict

import build_final_table as B
from build_gpt_table import G as GDIR, G_OK, SHORT

MAIN5 = ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B"]
METHODS = ["mask", "liemask", "steer", "prompt", "unsteered"]


def boot_mean_over_models(diff_lists, n=2000):
    rnd = random.Random(0)
    est = sum(sum(d) / len(d) for d in diff_lists) / len(diff_lists)
    bs = sorted(sum(sum(rnd.choice(d) for _ in d) / len(d) for d in diff_lists) / len(diff_lists) for _ in range(n))
    return {"diff": round(100 * est, 1), "lo": round(100 * bs[50], 1), "hi": round(100 * bs[1949], 1)}


def table1():
    gpt = {r["model"]: r for r in json.load(open("main_table_gpt.json"))}
    spec = {name: (m, lbt, ailt) for m, name, lbt, ailt in B.MODELS}
    test = {"lb": {}, "ail": {}}; val = {"lb": {}, "ail": {}}; rate = {"lb": {}, "ail": {}}
    for name in MAIN5:
        m, lbt, ailt = spec[name]
        lb_q, _, ail = B.load(m, lbt, ailt)
        lb_g = defaultdict(dict)
        for p in glob.glob(str(GDIR / f"liars_bench_honesty_{SHORT[name]}_*_judged_gpt_v2.jsonl")):
            for r in map(json.loads, open(p)):
                if r["split"] == "instructed-deception" and B.half(r["uid"]) == "test" and r["verdict_hybrid"] in G_OK:
                    lb_g[r["arm"]].setdefault(r["uid"], r["verdict_hybrid"] == "honest")
        for bench in ("lb", "ail"):
            cells = gpt[name][bench]["cells"]
            for f in METHODS:
                c = cells.get(f, {})
                arm = c["arm"] if c.get("v") is not None else cells["unsteered"]["arm"]   # no usable option -> unedited
                if bench == "lb":
                    t = lb_g[arm]
                    vs = [r for r in lb_q[arm] if r["split"] == "instructed-deception" and B.half(r["uid"]) == "val"]
                    v = sum(r["verdict_hybrid"] == "honest" for r in vs) / len(vs)
                else:
                    t = {x["uid"]: x["honest"] for x in ail[arm] if B.half(x["uid"]) == "test"}
                    vv = [x["honest"] for x in ail[arm] if B.half(x["uid"]) == "val"]; v = sum(vv) / len(vv)
                test[bench][(name, f)] = t; val[bench][(name, f)] = v
                rate[bench][(name, f)] = c["v"] if c.get("v") is not None else cells["unsteered"]["v"]
    out = {}
    for bench in ("lb", "ail"):
        mean = {f: sum(rate[bench][(n, f)] for n in MAIN5) / len(MAIN5) for f in METHODS}
        vmean = {f: sum(val[bench][(n, f)] for n in MAIN5) / len(MAIN5) for f in METHODS}
        others = [f for f in METHODS if f != "mask"]
        vs = max(others, key=vmean.get); vs_t = max(others, key=mean.get)
        def lead(f):
            dl = []
            for n in MAIN5:
                a, b = test[bench][(n, "mask")], test[bench][(n, f)]
                U = sorted(set(a) & set(b)); dl.append([a[u] - b[u] for u in U])
            return dict(boot_mean_over_models(dl), vs=f)
        Lt = lead(vs_t); L = Lt if vs == vs_t else lead(vs)
        out[bench] = {"mean": {f: round(v, 2) for f, v in mean.items()}, "lead": Lt, "lead_val": L}
        print(f"{bench:4s} " + "  ".join(f"{f}={mean[f]:.2f}" for f in METHODS)
              + f"   lead {Lt['diff']:+.1f} [{Lt['lo']:+.1f}, {Lt['hi']:+.1f}] vs {vs_t} (test-best)"
              + ("" if vs_t == vs else f" | vs {vs} (val-best) {L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}]"))
    return out


def table2():
    import summarize_tasks as T
    # Table 2's comparators were fixed in advance (the best other method on LIARS' BENCH validation at the time of the
    # task runs); the 2026-10-06 fill of empty Table 1 cells made the lie mask Falcon3-3B's best other method, but it was
    # never run on these tasks, so the comparators stay as recorded in tasks_table_gpt.json.
    pre = {r["model"]: r["tasks"]["sandbag"]["lead_pre"]["vs"] for r in json.load(open("tasks_table_gpt.json"))}
    items = {}
    for short, name in T.MODELS:
        if name not in MAIN5:
            continue
        it = defaultdict(lambda: defaultdict(dict))
        for task, (suf, key, smap) in T.TASKS.items():
            path = f"{T.OUT}/liars_bench_honesty_{short}_{suf}.jsonl"
            if task in ("inject", "amgot"):
                g = f"{T.GPT_DIR}/liars_bench_honesty_{short}_t{task}_judged_gpt_v2.jsonl"
                assert os.path.exists(g), g
                path = g
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
    for f in ("mask", "random"):
        d = sum(acc(n, f) - acc(n, "unsteered") for n in MAIN5) / len(MAIN5)
        out[f"mmlu_{f}"] = round(d, 1); print(f"mmlu delta {f}: {d:+.2f}")
    return out


if __name__ == "__main__":
    res = {"table1": table1(), "table2": table2()}
    json.dump(res, open("model_avg.json", "w"), indent=1)
    print("-> reports/model_avg.json")
