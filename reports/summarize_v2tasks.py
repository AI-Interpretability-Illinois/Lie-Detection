#!/usr/bin/env python3
"""J4: the calibrated masks (queue/lie_v2tasks.slurm, tags v2sandbag / v2multi / v2inject) on the tasks that
exposed the 25-prompt mask's costs, next to that mask and the unedited model from the earlier task runs
(reports/tasks_tables.md; tags tsandbag / tmulti / tinject in /work/hdd/bfqt/xiaocong/lie_detection_outputs).
Prints, per model and arm: plain-MMLU accuracy (capability), sandbagging honest %, multi-turn honest %, and
injected-instruction honest % (hybrid judge)."""
import json
import os
from collections import defaultdict

B = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
O = "/work/hdd/biro/xiaocong/lie_out"
J = "judged_Qwen2p5-7B-Instruct_v2"
MODELS = ["Meta-Llama-3p1-8B-Instruct", "Phi-4-mini-instruct", "Qwen2p5-3B-Instruct", "Qwen2p5-7B-Instruct"]


def rows(path, key="verdict"):
    out = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    if not os.path.exists(path):
        return out
    for r in map(json.loads, open(path)):
        v = r.get(key)
        if v in (None, "ungradeable", "deferred"):
            continue
        ds = r.get("dataset") or r.get("split")
        c = out[r["arm"]][ds]
        c[0] += v == "honest"; c[1] += 1
    return out


def pct(c):
    return f"{100 * c[0] / c[1]:5.1f}" if c[1] else "    -"


for m in MODELS:
    print(f"\n#### {m}")
    tabs = {}
    for name, old, new, key in (("sandbag", "tsandbag", "v2sandbag", "verdict"),
                                ("multi", "tmulti", "v2multi", "verdict"),
                                ("inject", "tinject", "v2inject", "verdict_hybrid")):
        suffix = f"_{J}.jsonl" if key == "verdict_hybrid" else "_responses.jsonl"
        a = rows(f"{B}/liars_bench_honesty_{m}_{old}{suffix}", key)
        b = rows(f"{O}/liars_bench_honesty_{m}_{new}{suffix}", key)
        tabs[name] = {**a, **b}
    arms = sorted({a for t in tabs.values() for a in t}, key=lambda a: (not a.startswith("weight:fm"), a))
    print(f"  {'arm':34s} {'MMLU-plain':>10} {'sandbag':>8} {'multi':>7} {'inject':>7} {'inj-ctl':>8}")
    for a in arms:
        s = tabs["sandbag"].get(a, {}); mt = tabs["multi"].get(a, {}); ij = tabs["inject"].get(a, {})
        plain = s.get("mmlu-plain", [0, 0])
        sb = [sum(s.get(k, [0, 0])[0] for k in s if k.startswith("sandbag")), sum(s.get(k, [0, 0])[1] for k in s if k.startswith("sandbag"))]
        mu = mt.get("multiturn", [0, 0])
        inj = ij.get("inject-deception", [0, 0]); ctl = ij.get("inject-control", [0, 0])
        print(f"  {a:34s} {pct(plain):>10} {pct(sb):>8} {pct(mu):>7} {pct(inj):>7} {pct(ctl):>8}")
