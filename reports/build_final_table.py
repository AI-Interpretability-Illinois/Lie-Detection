#!/usr/bin/env python3
"""
Final results table: seven models, same options for every method, settings chosen
on a validation half and reported on a test half.

Protocol (fixed before the final runs finished):
  * Options per model: six mask variants at four strengths, three steering layers
    at two strengths, three honest system-prompt wordings, the unedited model.
  * Items of each benchmark are split in half by an MD5 hash of the item id.
  * An arm is usable when, on the LIARS' BENCH validation half: perplexity < 1.15x;
    honesty-instructed control within 5 points of unsteered; honest rate on
    believed-true and on believed-false deception items each within 5 points of
    unsteered. Every arm is the same edit on both benchmarks, so these checks
    gate its use on AI Liar too (AI Liar has no control half of its own).
  * For each benchmark and method, the usable arm with the best validation honest
    rate is chosen; its test-half honest rate is reported.
  * The mask's lead over the best other method (chosen the same way) gets a paired
    bootstrap 95% interval over test items.
Gemma-3-4B is excluded (see reports/polarity_fix_report.md).
Writes reports/main_table_final.json.
"""
import csv
import hashlib
import json
import os
import random
from collections import defaultdict

OUT = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
J = "judged_Qwen2p5-7B-Instruct_v2"
MODELS = [  # short name, label, LB tags, AI Liar tags (merged in order)
    ("Meta-Llama-3p1-8B-Instruct", "Llama-3.1-8B", ["final"], ["finalail"]),
    ("Phi-4-mini-instruct", "Phi-4-mini", ["fixlb", "finaladd"], ["fixail", "finailadd"]),
    ("Falcon3-3B-Instruct", "Falcon3-3B", ["fixlb", "finaladd"], ["fixail", "finailadd"]),
    ("OLMo-2-0425-1B-Instruct", "OLMo-2-1B", ["final"], ["finalail"]),
    ("Qwen2p5-3B-Instruct", "Qwen2.5-3B", ["final"], ["finalail"]),
    ("Qwen2p5-7B-Instruct", "Qwen2.5-7B", ["final"], ["finalail"]),
    ("Qwen2p5-1p5B-Instruct", "Qwen2.5-1.5B", ["fixlb", "finaladd"], ["fixail", "finailadd"]),
]
G = ("honest", "lie", "no_claim", "refusal")
TOL = 0.05
FAMS = ["mask", "liemask", "steer", "prompt", "unsteered"]


def family(a):
    if a.startswith(("weight:fiction", "weight:pooled", "weight:fictf", "weight:pooltf", "weight:poolmix")):
        return "mask"
    if a.startswith("weight:ours"):
        return "liemask"
    if a.startswith("actsteer"):
        return "steer"
    if a.startswith("prompt:honest"):
        return "prompt"
    if a == "unsteered":
        return "unsteered"


def half(uid):
    return "val" if int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2 == 0 else "test"


items_ail = {json.loads(l)["uid"]: json.loads(l) for l in open(f"{OUT}/extra_data/extra_ailiar.jsonl")}


def load(model, lb_tags, ail_tags):
    lb, ppl, ail, seen = defaultdict(list), {}, defaultdict(list), set()
    for t in lb_tags:
        p = f"{OUT}/liars_bench_honesty_{model}_{t}_{J}.jsonl"
        c = f"{OUT}/liars_bench_honesty_{model}_{t}.csv"
        if not (os.path.exists(p) and os.path.exists(c)):
            return None
        new = set()
        for r in map(json.loads, open(p)):
            if r["arm"] in seen or r["verdict_hybrid"] not in G:
                continue
            lb[r["arm"]].append(r); new.add(r["arm"])
        for r in csv.DictReader(open(c)):
            ppl.setdefault(r["arm"], float(r["ppl_ratio"]))
        seen |= new
    seen = set()
    for t in ail_tags:
        p = f"{OUT}/liars_bench_honesty_{model}_{t}_responses.jsonl"
        if not os.path.exists(p):
            return None
        new = set()
        for r in map(json.loads, open(p)):
            v = r.get("verdict")
            if r["arm"] in seen or v in (None, "ungradeable") or items_ail[r["uid"]]["grade"]["near_train"]:
                continue
            ail[r["arm"]].append({"uid": r["uid"], "honest": v == "honest"}); new.add(r["arm"])
        seen |= new
    return lb, ppl, ail


def main():
    table = []
    for model, name, lbt, ailt in MODELS:
        d = load(model, lbt, ailt)
        if d is None:
            print(f"{name:<14} pending"); table.append({"model": name, "pending": True}); continue
        lb, ppl, ail = d
        rate = lambda rs: sum(r["verdict_hybrid"] == "honest" for r in rs) / max(1, len(rs))
        def dec(a, part, b=None):
            return [r for r in lb[a] if r["split"] == "instructed-deception" and half(r["uid"]) == part
                    and (b is None or r["belief"] is b)]
        def ctl(a, part):
            return [r for r in lb[a] if r["split"] == "instructed-truthful" and half(r["uid"]) == part]
        u = {"all": rate(dec("unsteered", "val")), "T": rate(dec("unsteered", "val", True)),
             "F": rate(dec("unsteered", "val", False)), "c": rate(ctl("unsteered", "val"))}
        def fails(a):
            f = []
            if ppl.get(a, 9) >= 1.15: f.append("perplexity")
            if rate(ctl(a, "val")) < u["c"] - TOL: f.append("control")
            if rate(dec(a, "val", True)) < u["T"] - TOL or rate(dec(a, "val", False)) < u["F"] - TOL:
                f.append("polarity")
            return f
        score = {
            "lb": {part: (lambda a, part=part: rate(dec(a, part))) for part in ("val", "test")},
            "ail": {part: (lambda a, part=part: (sum(x["honest"] for x in ail[a] if half(x["uid"]) == part)
                                                 / max(1, sum(half(x["uid"]) == part for x in ail[a]))))
                    for part in ("val", "test")},
        }
        per_item = {
            "lb": lambda a: {r["uid"]: r["verdict_hybrid"] == "honest" for r in dec(a, "test")},
            "ail": lambda a: {x["uid"]: x["honest"] for x in ail[a] if half(x["uid"]) == "test"},
        }
        row = {"model": name}
        for bench, arms in (("lb", lb), ("ail", ail)):
            chosen, why = {}, {}
            for a in arms:
                f = family(a)
                if not f: continue
                if fails(a):
                    if f not in why or score[bench]["val"](a) > score[bench]["val"](why[f][0]):
                        why[f] = (a, fails(a))
                    continue
                if f not in chosen or score[bench]["val"](a) > score[bench]["val"](chosen[f]):
                    chosen[f] = a
            cells = {}
            for f in FAMS:
                if f in chosen:
                    a = chosen[f]
                    cells[f] = {"v": round(score[bench]["test"](a) * 100, 1), "arm": a}
                elif f in why:
                    cells[f] = {"v": None, "arm": why[f][0], "fails": why[f][1]}
            lead = None
            others = [f for f in FAMS[1:] if f in chosen]
            if "mask" in chosen and others:
                best = max(others, key=lambda f: score[bench]["val"](chosen[f]))
                pm, pb = per_item[bench](chosen["mask"]), per_item[bench](chosen[best])
                U = sorted(set(pm) & set(pb)); diffs = [pm[x] - pb[x] for x in U]
                rnd = random.Random(0)
                boots = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
                lead = {"vs": best, "diff": round(sum(diffs) / len(diffs) * 100, 1),
                        "lo": round(boots[50] * 100, 1), "hi": round(boots[1949] * 100, 1), "n": len(U)}
            row[bench] = {"cells": cells, "lead": lead}
        table.append(row)
        for bench in ("lb", "ail"):
            c = row[bench]["cells"]; L = row[bench]["lead"]
            fmt = lambda f: (f"{c[f]['v']:5.1f}" if f in c and c[f]["v"] is not None else "   - ")
            print(f"{name:<14}{bench:>4}  " + "  ".join(fmt(f) for f in FAMS)
                  + (f"   lead {L['diff']:+.1f} [{L['lo']:+.1f}, {L['hi']:+.1f}] vs {L['vs']}" if L else ""))
    json.dump(table, open("/u/xiaocong/lie_detection/reports/main_table_final.json", "w"), indent=1)
    print("columns: our mask, lie mask, steering, honest prompt, unsteered (test-half honest rate, %)")


if __name__ == "__main__":
    main()
