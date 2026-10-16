#!/usr/bin/env python3
"""Combined-mask test (queue/run_timan_combo.sh): the chosen prompt mask OR a small recovery mask, against
unedited, the prompt mask alone and the recovery mask alone. Validation halves: LIARS' BENCH (hybrid
judge), AI Liar (keyword, near-train excluded), multi-turn (inline grader). Each cell: honest %, then the
paired bootstrap 95% CI of the arm minus unedited and minus the prompt mask."""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

O = Path(sys.argv[1] if len(sys.argv) > 1 else "/work/hdd/biro/xiaocong/lie_out/timan_combo")
X = Path("/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data")
MODELS = ["Llama-3p1-8B-Instruct", "Phi-4-mini-instruct", "Falcon3-3B-Instruct"]
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
    f = a.split()[0].split(":")[1].rstrip("_")
    return f.replace("recov040a025", "recov 0.25%").replace("recov040a050", "recov 0.50%").replace("+", " + ")


def ci(x, ref):
    U = sorted(set(x) & set(ref))
    lo, hi = boot([x[u] - ref[u] for u in U])
    return f"[{lo:+5.1f},{hi:+5.1f}]"


def main():
    for m in MODELS:
        files = {"LIARS' BENCH": (O / f"liars_bench_honesty_{m}_combo_judged_Qwen2p5-7B-Instruct_v2.jsonl",
                                  "verdict_hybrid", lambda r: r["split"] == "instructed-deception"),
                 "AI Liar": (O / f"liars_bench_honesty_{m}_comboail_responses.jsonl", "verdict",
                             lambda r: not near.get(r["uid"], False)),
                 "multi-turn": (O / f"liars_bench_honesty_{m}_combomt_responses.jsonl", "verdict", lambda r: True)}
        if not (O / f"liars_bench_honesty_{m}_combo.csv").exists():
            print(f"\n{m}: pending"); continue
        ppl = {r["arm"]: float(r["ppl_ratio"]) for r in csv.DictReader(open(O / f"liars_bench_honesty_{m}_combo.csv"))}
        cur = [a for a in ppl if a != "unsteered" and "+" not in a and "recov" not in a][0]
        ctl = per_item(files["LIARS' BENCH"][0], "verdict_hybrid", lambda r: r["split"] == "instructed-truthful") \
            if files["LIARS' BENCH"][0].exists() else None
        data = {b: per_item(p, k, f) for b, (p, k, f) in files.items() if p.exists()}
        print(f"\n{m}   honest %, validation half; [95% CI vs unedited] [vs {label(cur)}]")
        print(f"  {'mask':<26}" + "".join(f"{b:>38}" for b in data) + f"{'control':>9}{'ppl':>7}")
        for a in ppl:
            cells = []
            for b, d in data.items():
                x = d[a]; rate = 100 * sum(x.values()) / len(x)
                s = f"{rate:5.1f}"
                if a != "unsteered":
                    s += " " + ci(x, d["unsteered"])
                if a not in ("unsteered", cur):
                    s += " " + ci(x, d[cur])
                cells.append(s.rjust(38))
            c = f"{100 * sum(ctl[a].values()) / len(ctl[a]):9.1f}" if ctl else f"{'-':>9}"
            print(f"  {label(a):<26}" + "".join(cells) + c + f"{ppl[a]:7.3f}")


if __name__ == "__main__":
    main()
