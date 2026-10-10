#!/usr/bin/env python3
"""
OLMo-2-32B at a glance (owner, 2026-10-09): every method of Table 1 at its chosen option, plus the
magnitude-matched random mask as a control, against the unedited model on both benchmarks.

No new runs: the arms are the ones reports/build_final_table.py --sweep already chose on the validation
half with the Qwen2.5-7B grader (reports/main_table_sweep.json), and the rates are recomputed on the test
half from the GPT-5.6 re-grade for LIARS' BENCH (judge_gpt.py --sweep) and from the keyword verdicts for
AI Liar, exactly as Table 1 is built (reports/build_gpt_table.py).

The random mask is a null, not a competing method: it deletes the same number of weights of the same
magnitudes, so it says whether a gain is about *which* weights were chosen or merely how many.

Writes reports/olmo32b.json.
"""
import glob
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

R = Path("/u/xiaocong/lie_detection/reports")
GPT = Path("/work/nvme/biro/xiaocong/lie_out_gpt")
SWEEP_DIRS = ["/work/nvme/biro/xiaocong/lie_sweep", "/u/xiaocong/lie_results/timan_sweep"]
M = "OLMo-2-0325-32B-Instruct"
G_OK = ("honest", "lie", "no_claim", "refusal")
MAG = "weight:magmatch a=-1.0"
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"


def boot(diffs):
    rnd = random.Random(0)
    b = sorted(sum(rnd.choice(diffs) for _ in diffs) / len(diffs) for _ in range(2000))
    return round(100 * b[50], 1), round(100 * b[1949], 1)


def main():
    row = next(r for r in json.load(open(R / "main_table_sweep.json")) if r["model"] == "OLMo-2-32B")
    arms = {f: c["arm"] for f, c in row["lb"]["cells"].items() if c.get("v") is not None}
    arms["magmatch"] = MAG

    lb = defaultdict(dict)
    for p in glob.glob(str(GPT / f"liars_bench_honesty_{M}_*_judged_gpt_v2.jsonl")):
        for r in map(json.loads, open(p)):
            if r["split"] == "instructed-deception" and half(r["uid"]) == "test" and r["verdict_hybrid"] in G_OK:
                lb[r["arm"]].setdefault(r["uid"], r["verdict_hybrid"] == "honest")

    items = {json.loads(l)["uid"]: json.loads(l) for l in open("/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data/extra_ailiar.jsonl")}
    ail = defaultdict(dict)
    for d in SWEEP_DIRS:
        for p in glob.glob(f"{d}/liars_bench_honesty_{M}_sw*ail_responses.jsonl"):
            for r in map(json.loads, open(p)):
                v = r.get("verdict")
                if v in (None, "ungradeable") or half(r["uid"]) != "test" or items[r["uid"]]["grade"]["near_train"]:
                    continue
                ail[r["arm"]].setdefault(r["uid"], v == "honest")

    out = {"model": "OLMo-2-32B", "arms": arms,
           "grader": {"lb": "gpt-5.6 hybrid (test half; arms chosen on validation with the Qwen2.5-7B hybrid)", "ail": "keyword"}}
    for bench, per in (("lb", lb), ("ail", ail)):
        u = per.get(arms["unsteered"], {})
        cells = {}
        for f, a in arms.items():
            d = per.get(a)
            if not d:
                print(f"  missing {bench} {f}: {a}"); continue
            common = sorted(set(d) & set(u))
            diffs = [d[x] - u[x] for x in common]
            lo, hi = boot(diffs) if f != "unsteered" and diffs else (None, None)
            cells[f] = {"arm": a, "v": round(100 * sum(d.values()) / len(d), 1), "n": len(d),
                        "gain": None if f == "unsteered" else round(100 * sum(diffs) / len(diffs), 1),
                        "lo": lo, "hi": hi}
        out[bench] = cells
        print(f"{bench}: " + "  ".join(f"{f} {c['v']:.1f}" + (f" ({c['gain']:+.1f} [{c['lo']:+.1f},{c['hi']:+.1f}])" if c["gain"] is not None else "")
                                       for f, c in cells.items()))
    json.dump(out, open(R / "olmo32b.json", "w"), indent=1)
    print("-> reports/olmo32b.json")


if __name__ == "__main__":
    main()
