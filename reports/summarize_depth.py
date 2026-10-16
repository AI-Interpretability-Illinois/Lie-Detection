#!/usr/bin/env python3
"""Response-token (resp) and recovery (recov) masks vs the current mask, validation halves:
LIARS' BENCH (hybrid judge), AI Liar (keyword, near-train excluded), multi-turn (inline grader).
Paired bootstrap 95% CI of each arm minus unedited."""
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

O = Path("/work/hdd/biro/xiaocong/lie_out")
X = Path("/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data")
MODELS = ["Meta-Llama-3p1-8B-Instruct", "Phi-4-mini-instruct", "Falcon3-3B-Instruct"]
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"
near = {json.loads(l)["uid"]: json.loads(l)["grade"]["near_train"] for l in open(X / "extra_ailiar.jsonl")}


def boot(d, n=2000):
    r = random.Random(0)
    b = sorted(sum(r.choice(d) for _ in d) / len(d) for _ in range(n))
    return 100 * b[50], 100 * b[1949]


def per_item(path, key, keep):
    out = defaultdict(dict)
    for r in map(json.loads, open(path)):
        v = r.get(key)
        if v in (None, "ungradeable", "deferred") or not keep(r):
            continue
        out[r["arm"]][r["uid"]] = v == "honest"
    return out


def label(a):
    if a == "unsteered":
        return "unedited"
    f = a.split()[0].split(":")[1]
    return f if not f.startswith(("resp", "recov")) else \
        f"{f[:-8] if f.startswith('recov') else f[:-8]} {int(f[-4:-1]) / 100:.2f}%".replace("040a", " ").strip()


def main():
    for m in MODELS:
        files = {"LIARS' BENCH": (O / f"liars_bench_honesty_{m}_depth_judged_Qwen2p5-7B-Instruct_v2.jsonl", "verdict_hybrid",
                                  lambda r: r["split"] == "instructed-deception"),
                 "AI Liar": (O / f"liars_bench_honesty_{m}_depthail_responses.jsonl", "verdict",
                             lambda r: not near.get(r["uid"], False)),
                 "multi-turn": (O / f"liars_bench_honesty_{m}_depthmt_responses.jsonl", "verdict", lambda r: True)}
        if not files["LIARS' BENCH"][0].exists():
            print(f"\n{m}: pending"); continue
        ppl = {r["arm"]: float(r["ppl_ratio"]) for r in csv.DictReader(open(O / f"liars_bench_honesty_{m}_depth.csv"))}
        ctl = per_item(files["LIARS' BENCH"][0], "verdict_hybrid", lambda r: r["split"] == "instructed-truthful")
        data = {b: per_item(p, k, f) for b, (p, k, f) in files.items() if p.exists()}
        print(f"\n{m}   honest %, validation half; [95% CI of arm - unedited]")
        print(f"  {'mask':<18}" + "".join(f"{b:>30}" for b in data) + f"{'control':>9}{'ppl':>7}")
        for a in ppl:
            cells = []
            for b, d in data.items():
                x = d[a]; rate = 100 * sum(x.values()) / len(x)
                if a == "unsteered":
                    cells.append(f"{rate:>12.1f}{'':>18}")
                else:
                    U = sorted(set(x) & set(d["unsteered"])); df = [x[u] - d["unsteered"][u] for u in U]
                    lo, hi = boot(df)
                    cells.append(f"{rate:>12.1f} [{lo:+5.1f},{hi:+5.1f}]".ljust(30))
            c = ctl[a]
            print(f"  {label(a):<18}" + "".join(cells) + f"{100*sum(c.values())/len(c):9.1f}{ppl[a]:7.3f}")


if __name__ == "__main__":
    main()
