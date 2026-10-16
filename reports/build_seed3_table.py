#!/usr/bin/env python3
"""
Table 1 over three sampling seeds (owner, 2026-10-12: report Table 1 as the mean over seeds 0-2, which the review asked
for). Options: those of reports/main_table_final.json (chosen on the validation half with seed 0, unchanged). Test-half
honest rates: LIARS' BENCH graded by the GPT-5.6 hybrid (seed 0: the Table 1 grades in GPT_DIR; seeds 1-2: judge_gpt.py
--tags rvseeds1 / rvseeds2 in GPT_DIR/review_seeds), AI Liar by keyword (seed 0: the main runs; seeds 1-2: the review runs
in RV). A method's rate is the mean over the three seeds. The lead of ours is over the best other method in the row on
this mean (the rule of Table 1); its interval is a paired bootstrap over test items, each item carrying its seed-averaged
difference (2,000 resamples). The average row: each method's mean over the five models; lead over the best other average,
items resampled within each model. Also counts, per benchmark and baseline, the models on which ours is ahead.
Writes reports/main_table_gpt_seed3.json and reports/model_avg_seed3.json (the formats make_tables.main_table reads).
"""
import glob
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import build_final_table as B   # noqa: E402

R = Path("/u/xiaocong/lie_detection/reports")
GDIR = Path(os.environ.get("GPT_DIR", "/work/nvme/biro/xiaocong/lie_out_gpt"))
RV = "/work/hdd/bfqt/xiaocong/lie_review"
G_OK = ("honest", "lie", "no_claim", "refusal")
F = ["mask", "liemask", "steer", "prompt", "unsteered"]
MAIN5 = [m for m in B.MODELS if m[1] in ("Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B")]
norm = lambda a: re.sub(r" \(c=[^)]*\)", "", a)


def lb_gpt(files):
    out = defaultdict(dict)
    for p in files:
        for r in map(json.loads, open(p)):
            if r["split"] == "instructed-deception" and B.half(r["uid"]) == "test" and r["verdict_hybrid"] in G_OK:
                out[norm(r["arm"])].setdefault(r["uid"], r["verdict_hybrid"] == "honest")
    return out


def ail_kw(path):
    out = defaultdict(dict)
    for r in map(json.loads, open(path)):
        v = r.get("verdict")
        if v in (None, "ungradeable") or B.items_ail[r["uid"]]["grade"]["near_train"] or B.half(r["uid"]) != "test":
            continue
        out[norm(r["arm"])][r["uid"]] = v == "honest"
    return out


def boot(item_diffs, n=2000):
    rnd = random.Random(0); d = list(item_diffs)
    bs = sorted(sum(rnd.choice(d) for _ in d) / len(d) for _ in range(n))
    return round(100 * sum(d) / len(d), 1), round(100 * bs[50], 1), round(100 * bs[1949], 1)


def main():
    t1 = {r["model"]: r for r in json.load(open(R / "main_table_final.json"))}
    rows, per = [], {}
    for S, name, lbt, ailt in MAIN5:
        seeds = {"lb": [], "ail": []}
        main_files = [p for p in glob.glob(str(GDIR / f"liars_bench_honesty_{S}_*_judged_gpt_v2.jsonl"))]
        seeds["lb"].append(lb_gpt(main_files))
        for sd in (1, 2):
            seeds["lb"].append(lb_gpt(glob.glob(str(GDIR / "review_seeds" / f"liars_bench_honesty_{S}_rvseeds{sd}_judged_gpt_v2.jsonl"))))
        _, _, ail0 = B.load(S, lbt, ailt)
        seeds["ail"].append({norm(a): {x["uid"]: x["honest"] for x in xs if B.half(x["uid"]) == "test"} for a, xs in ail0.items()})
        for sd in (1, 2):
            seeds["ail"].append(ail_kw(f"{RV}/liars_bench_honesty_{S}_rvseeds{sd}ail_responses.jsonl"))
        row = {"model": name}
        per[name] = {}
        for b in ("lb", "ail"):
            cells = t1[name][b]["cells"]
            arms = {f: norm(c["arm"]) for f, c in cells.items() if c.get("v") is not None}
            item = {}          # f -> uid -> mean over seeds
            for f, a in arms.items():
                acc = defaultdict(list)
                for sd in range(3):
                    for u, h in seeds[b][sd].get(a, {}).items():
                        acc[u].append(h)
                missing = [sd for sd in range(3) if not seeds[b][sd].get(a)]
                if missing:
                    raise SystemExit(f"{name} {b} {f} ({a}): no replies for seeds {missing}")
                item[f] = {u: sum(v) / len(v) for u, v in acc.items()}
            rate = {f: 100 * sum(d.values()) / len(d) for f, d in item.items()}
            others = [f for f in item if f != "mask"]
            vs = max(others, key=rate.get)
            U = sorted(set(item["mask"]) & set(item[vs]))
            diff, lo, hi = boot([item["mask"][u] - item[vs][u] for u in U])
            row[b] = {"cells": {f: {"v": round(rate[f], 1), "arm": cells[f]["arm"]} for f in item},
                      "lead": {"vs": vs, "diff": diff, "lo": lo, "hi": hi, "n": len(U)}, "seeds": 3}
            per[name][b] = item
            print(f"{name:<13}{b:>4} " + "  ".join(f"{f}={rate[f]:.1f}" for f in F if f in rate) + f"  lead {diff:+.1f} [{lo:+.1f},{hi:+.1f}] vs {vs}")
        rows.append(row)
    json.dump(rows, open(R / "main_table_gpt_seed3.json", "w"), indent=1)
    out = {"table1": {}, "wins": {}}
    for b in ("lb", "ail"):
        names = [r["model"] for r in rows]
        mean = {f: sum((next(r for r in rows if r["model"] == n)[b]["cells"].get(f) or
                        next(r for r in rows if r["model"] == n)[b]["cells"]["unsteered"])["v"] for n in names) / len(names) for f in F}
        vs = max([f for f in F if f != "mask"], key=mean.get)
        rnd = random.Random(0)
        dls = []
        for n in names:
            it = per[n][b]; o = it["mask"]; c = it.get(vs) or it["unsteered"]
            dls.append([o[u] - c[u] for u in sorted(set(o) & set(c))])
        est = sum(sum(d) / len(d) for d in dls) / len(dls)
        bs = sorted(sum(sum(rnd.choice(d) for _ in d) / len(d) for d in dls) / len(dls) for _ in range(2000))
        out["table1"][b] = {"mean": {f: round(v, 2) for f, v in mean.items()},
                            "lead": {"vs": vs, "diff": round(100 * est, 1), "lo": round(100 * bs[50], 1), "hi": round(100 * bs[1949], 1)}}
        out["wins"][b] = {f: sum(1 for r in rows if r[b]["cells"]["mask"]["v"] > (r[b]["cells"].get(f) or r[b]["cells"]["unsteered"])["v"])
                          for f in F if f != "mask"}
        L = out["table1"][b]["lead"]
        print(f"AVERAGE {b:>4} " + "  ".join(f"{f}={mean[f]:.2f}" for f in F) + f"  lead {L['diff']:+.1f} [{L['lo']:+.1f},{L['hi']:+.1f}] vs {vs}"
              + f"  wins {out['wins'][b]}")
    json.dump(out, open(R / "model_avg_seed3.json", "w"), indent=1)


if __name__ == "__main__":
    main()
