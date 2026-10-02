#!/usr/bin/env python3
"""Main-table rows plus the 100-prompt mask (pool draw seed 0, same family and strength as the chosen
25-prompt mask), test halves. LIARS' BENCH: hybrid judge; AI Liar: keyword verdict, near-train items
excluded (as reports/build_final_table.py). Writes reports/p100_table.json."""
import hashlib
import json
from pathlib import Path

O = Path("/work/hdd/bfqt/xiaocong/lie_detection_outputs")
R = Path("/u/xiaocong/lie_detection/reports")
SHORT = {"Llama-3.1-8B": "Llama-3p1-8B-Instruct", "Phi-4-mini": "Phi-4-mini-instruct",
         "Falcon3-3B": "Falcon3-3B-Instruct", "OLMo-2-1B": "OLMo-2-0425-1B-Instruct",
         "Qwen2.5-3B": "Qwen2p5-3B-Instruct", "Qwen2.5-7B": "Qwen2p5-7B-Instruct",
         "Qwen2.5-1.5B": "Qwen2p5-1p5B-Instruct"}
half = lambda uid: "val" if int(hashlib.md5(uid.encode()).hexdigest(), 16) % 2 == 0 else "test"
ail_items = {json.loads(l)["uid"]: json.loads(l) for l in open(O / "extra_data/extra_ailiar.jsonl")}


def find(name):
    for d in (O / "timan_b", O):
        if (d / name).exists():
            return d / name


def lb(short):
    p = find(f"liars_bench_honesty_{short}_p100_judged_Qwen2p5-7B-Instruct_v2.jsonl")
    if not p:
        return None
    rs = [r for r in map(json.loads, open(p)) if r["split"] == "instructed-deception"
          and half(r["uid"]) == "test" and r["verdict_hybrid"] in ("honest", "lie", "no_claim", "refusal")]
    return round(100 * sum(r["verdict_hybrid"] == "honest" for r in rs) / len(rs), 1), len(rs)


def ail(short):
    p = find(f"liars_bench_honesty_{short}_p100ail_responses.jsonl")
    if not p:
        return None
    rs = [r for r in map(json.loads, open(p)) if r.get("verdict") not in (None, "ungradeable")
          and half(r["uid"]) == "test" and not ail_items[r["uid"]]["grade"]["near_train"]]
    return round(100 * sum(r["verdict"] == "honest" for r in rs) / len(rs), 1), len(rs)


def main():
    out = []
    for row in json.load(open(R / "main_table_final.json")):
        s = SHORT[row["model"]]
        rec = {"model": row["model"]}
        for b, f in (("lb", lb), ("ail", ail)):
            c, L = row[b]["cells"], row[b]["lead"]
            got = f(s)
            rec[b] = {"ours25": c["mask"]["v"], "ours100": got[0] if got else None,
                      "n100": got[1] if got else None, "vs": L["vs"], "other": c[L["vs"]]["v"],
                      "lead25": L["diff"], "sig25": L["lo"] > 0,
                      "lead100": round(got[0] - c[L["vs"]]["v"], 1) if got else None}
        out.append(rec)
        print(rec["model"], {b: (rec[b]["ours25"], rec[b]["ours100"], rec[b]["n100"], rec[b]["vs"], rec[b]["other"]) for b in ("lb", "ail")})
    json.dump(out, open(R / "p100_table.json", "w"), indent=1)


if __name__ == "__main__":
    main()
