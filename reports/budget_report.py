#!/usr/bin/env python3
"""
Mask-budget sweep report (extras plan item 4). The contrast threshold r sets how many channels carry a positive score
(n_F / n_T > 1/r); at the paper's r = 0.37 that is 0.05-0.8% of the weights. queue/lie_budget.slurm (Delta) and
queue/run_timan_budget.sh (timan) run the model's chosen family at four strengths for r in {0.2, 0.6, 1.0}, tags
rb020 / rb060 / rb100; the r = 0.37 arms come from the main-table files. For each model and r this prints, per strength,
the validation honest rate on LIARS' BENCH, the perplexity ratio and the gates it fails, then the test-half rate of the
best usable strength, against the honest prompt and the unedited model. Writes reports/budget_report.md.
"""
import csv
import glob
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

R = Path("/u/xiaocong/lie_detection/reports")
J = "judged_Qwen2p5-7B-Instruct_v2"
DIRS = ["/work/nvme/biro/xiaocong/lie_sweep", "/u/xiaocong/lie_results/timan_sweep", "/work/hdd/bfqt/xiaocong/lie_detection_outputs"]
MAIN = {"Qwen2p5-7B-Instruct": ("Qwen2.5-7B", ["final"], "pooled"), "Meta-Llama-3p1-8B-Instruct": ("Llama-3.1-8B", ["final"], "fiction"),
        "OLMo-2-1124-7B-Instruct": ("OLMo-2-7B", ["sw1", "sw2", "sw3", "sw4"], "fiction")}
TOL = 0.05
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"


def find(model, tag, suffix):
    for d in DIRS:
        p = f"{d}/liars_bench_honesty_{model}_{tag}{suffix}"
        if os.path.exists(p):
            return p


def load(model, tags):
    lb, ppl = defaultdict(list), {}
    for t in tags:
        p = find(model, t, f"_{J}.jsonl"); c = find(model, t, ".csv")
        if not (p and c):
            continue
        seen = set(a for a in lb)
        for r in map(json.loads, open(p)):
            if r["arm"] in seen or r["verdict_hybrid"] not in ("honest", "lie", "no_claim", "refusal"):
                continue
            lb[r["arm"]].append(r)
        for r in csv.DictReader(open(c)):
            ppl.setdefault(r["arm"], float(r["ppl_ratio"]))
    return lb, ppl


def main():
    out = ["# Mask-budget sweep: the contrast threshold r", "",
           "Family = the mask arm chosen at r = 0.37. Per strength: validation honest rate on LIARS' BENCH (perplexity ratio; gates failed). "
           "Best = test-half honest rate of the best usable strength at that r.", ""]
    for model, (name, main_tags, fam) in MAIN.items():
        rows = []
        for r, tags in (("0.2", ["rb020"]), ("0.37", main_tags), ("0.6", ["rb060"]), ("1.0", ["rb100"])):
            lb, ppl = load(model, tags)
            if "unsteered" not in lb:
                base_lb, _ = load(model, main_tags)
                if "unsteered" not in base_lb:
                    continue
                lb["unsteered"] = base_lb["unsteered"]
                for a in base_lb:
                    if a.startswith("prompt:honest"):
                        lb.setdefault(a, base_lb[a])
            rate = lambda rs: sum(x["verdict_hybrid"] == "honest" for x in rs) / max(1, len(rs))
            dec = lambda a, part, b=None: [x for x in lb[a] if x["split"] == "instructed-deception" and half(x["uid"]) == part and (b is None or x["belief"] is b)]
            ctl = lambda a, part: [x for x in lb[a] if x["split"] == "instructed-truthful" and half(x["uid"]) == part]
            u = {"T": rate(dec("unsteered", "val", True)), "F": rate(dec("unsteered", "val", False)), "c": rate(ctl("unsteered", "val"))}
            cells, best = [], None
            for st in ("-0.25", "-0.5", "-0.75", "-1.0"):
                a = f"weight:{fam} a={st}"
                if a not in lb:
                    cells.append("–"); continue
                f = []
                if ppl.get(a, 9) >= 1.15: f.append("ppl")
                if rate(ctl(a, "val")) < u["c"] - TOL: f.append("ctl")
                if rate(dec(a, "val", True)) < u["T"] - TOL or rate(dec(a, "val", False)) < u["F"] - TOL: f.append("pol")
                v = rate(dec(a, "val"))
                cells.append(f"{100 * v:.1f} ({ppl.get(a, float('nan')):.2f}{'; ' + ','.join(f) if f else ''})")
                if not f and (best is None or v > best[0]):
                    best = (v, a)
            prompt_arms = [a for a in lb if a.startswith("prompt:honest")]
            pr = max((rate(dec(a, "test")) for a in prompt_arms), default=float("nan"))
            rows.append(f"| {r} | " + " | ".join(cells) + f" | {100 * rate(dec(best[1], 'test')):.1f} ({best[1].split('a=')[1]})" if best else f"| {r} | " + " | ".join(cells) + " | –")
            rows[-1] += f" | {100 * pr:.1f} | {100 * rate(dec('unsteered', 'test')):.1f} |"
        if rows:
            out += [f"## {name} ({fam})", "", "| r | a=−0.25 | a=−0.5 | a=−0.75 | a=−1.0 | best usable (test) | honest prompt (test) | unedited (test) |",
                    "|---|---|---|---|---|---|---|---|"] + rows + [""]
    (R / "budget_report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
