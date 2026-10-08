#!/usr/bin/env python3
"""
Sampling-seed replication of Table 1 (queue/lie_seed.slurm): the LIARS' BENCH-chosen arms of every main-table model
regenerated with torch seeds 1 and 2 (replies are sampled at temperature 0.7) and judged by the Qwen2.5-7B hybrid, against
the paper's run (seed 0 = reports/main_table_final.json, same judge). Reports, per model and arm family, the test-half honest
rate under each seed, the spread (max - min) and the standard deviation across seeds, and the lead of the falsehood mask over
the comparator fixed on validation under each seed. Writes reports/seed_table.json and reports/seed_report.md.
"""
import glob
import re
import hashlib
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from summarize_tasks import family, paired   # noqa: E402

R = Path("/u/xiaocong/lie_detection/reports")
SWEEP = Path("/work/nvme/biro/xiaocong/lie_sweep")
NAMES = {"Meta-Llama-3p1-8B-Instruct": "Llama-3.1-8B", "Phi-4-mini-instruct": "Phi-4-mini", "Falcon3-3B-Instruct": "Falcon3-3B",
         "OLMo-2-0425-1B-Instruct": "OLMo-2-1B", "Qwen2p5-3B-Instruct": "Qwen2.5-3B", "Qwen2p5-7B-Instruct": "Qwen2.5-7B",
         "Qwen2p5-1p5B-Instruct": "Qwen2.5-1.5B"}
FAMS = ["mask", "liemask", "steer", "prompt", "unsteered"]
SEEDS = [1, 2]
G_OK = ("honest", "lie", "no_claim", "refusal")   # gradeable verdicts, as the main table (no claim / refusal count as not honest)
TAGS0 = {"Meta-Llama-3p1-8B-Instruct": ["final"], "Phi-4-mini-instruct": ["fixlb", "finaladd"], "Falcon3-3B-Instruct": ["fixlb", "finaladd", "swfill"],
         "OLMo-2-0425-1B-Instruct": ["final", "swfill"], "Qwen2p5-3B-Instruct": ["final"], "Qwen2p5-7B-Instruct": ["final"],
         "Qwen2p5-1p5B-Instruct": ["fixlb", "finaladd"]}   # the main table's LB tags per model (reports/build_final_table.py MODELS)
half = lambda u: "val" if int(hashlib.md5(u.encode()).hexdigest(), 16) % 2 == 0 else "test"


def main():
    main_tab = {r["model"]: r for r in json.load(open(R / "main_table_final.json")) if not r.get("pending")}
    out = []
    L = ["# Sampling-seed replication of Table 1 (Qwen2.5-7B judge, LIARS' BENCH test half)", "",
         "Seed 0 is the paper's run; seeds 1 and 2 regenerate the same arms (temperature 0.7) with a different torch seed.",
         "Spread = max - min over the seeds available; SD = sample standard deviation. Lead: falsehood mask minus the comparator",
         "fixed on LIARS' BENCH validation, paired bootstrap 95% interval over test items, under each seed.", ""]
    for stem, name in NAMES.items():
        if name not in main_tab:
            continue
        rates = {0: {f: c.get("v") for f, c in main_tab[name]["lb"]["cells"].items()}}
        leads = {0: main_tab[name]["lb"]["lead"]}
        vs = leads[0]["vs"] if leads[0] else None
        for s in SEEDS:
            fs = sorted(glob.glob(str(SWEEP / f"liars_bench_honesty_{stem}_seed{s}_judged_Qwen2p5-7B-Instruct_v2.jsonl"))
                        + glob.glob(str(SWEEP / f"liars_bench_honesty_{stem}_seed{s}fill_judged_Qwen2p5-7B-Instruct_v2.jsonl")))
            if not fs:
                continue
            per = defaultdict(dict)
            for fp in fs:
                for r in map(json.loads, open(fp)):
                    if r["split"] != "instructed-deception" or half(r["uid"]) != "test" or r.get("verdict_hybrid") not in G_OK:
                        continue
                    f = family(r["arm"])
                    if f and re.sub(r" \(c=[^)]*\)", "", r["arm"]) == re.sub(r" \(c=[^)]*\)", "", main_tab[name]["lb"]["cells"].get(f, {}).get("arm") or ""):   # only the option Table 1 chose
                        per[f][r["uid"]] = r["verdict_hybrid"] == "honest"
            rates[s] = {f: round(100 * sum(d.values()) / len(d), 1) for f, d in per.items() if d}
            leads[s] = paired(per, vs) if vs and "mask" in per and vs in per else None
        # pooled over seeds: each test item's honesty averaged over the seeds that have it (seed 0 from the main-table files),
        # then the same paired bootstrap over items; a per-item mean over three samples halves the sampling noise
        pooled = None
        p0 = {}
        # seed 0 = the main table's own files: the model's LB tags in order, first file holding an arm wins (as build_final_table.py)
        seen0 = set()
        for t in TAGS0[stem]:
            p = f"/work/hdd/bfqt/xiaocong/lie_detection_outputs/liars_bench_honesty_{stem}_{t}_judged_Qwen2p5-7B-Instruct_v2.jsonl"
            if not Path(p).exists():   # sweep-dir tags (swfill: the 2026-10-06 milder options)
                p = str(SWEEP / Path(p).name)
            if not Path(p).exists():
                continue
            new = set()
            for r in map(json.loads, open(p)):
                if r["arm"] in seen0:
                    continue
                new.add(r["arm"])
                if r["split"] == "instructed-deception" and half(r["uid"]) == "test" and r.get("verdict_hybrid") in G_OK:
                    f = family(r["arm"])
                    if f and re.sub(r" \(c=[^)]*\)", "", r["arm"]) == re.sub(r" \(c=[^)]*\)", "", main_tab[name]["lb"]["cells"].get(f, {}).get("arm") or ""):
                        p0.setdefault(f, {})[r["uid"]] = r["verdict_hybrid"] == "honest"
            seen0 |= new
        per_seed = {0: p0}
        for s in SEEDS:
            fs = sorted(glob.glob(str(SWEEP / f"liars_bench_honesty_{stem}_seed{s}_judged_Qwen2p5-7B-Instruct_v2.jsonl"))
                        + glob.glob(str(SWEEP / f"liars_bench_honesty_{stem}_seed{s}fill_judged_Qwen2p5-7B-Instruct_v2.jsonl")))
            if fs:
                d = defaultdict(dict)
                for fp in fs:
                    for r in map(json.loads, open(fp)):
                        if r["split"] == "instructed-deception" and half(r["uid"]) == "test" and r.get("verdict_hybrid") in G_OK:
                            f = family(r["arm"])
                            if f and re.sub(r" \(c=[^)]*\)", "", r["arm"]) == re.sub(r" \(c=[^)]*\)", "", main_tab[name]["lb"]["cells"].get(f, {}).get("arm") or ""):
                                d[f][r["uid"]] = r["verdict_hybrid"] == "honest"
                per_seed[s] = d
        if vs and len(per_seed) > 1:
            agg = {}
            for f in ("mask", vs):
                acc = defaultdict(list)
                for s, d in per_seed.items():
                    for u, v in d.get(f, {}).items():
                        acc[u].append(float(v))
                agg[f] = {u: sum(x) / len(x) for u, x in acc.items()}
            if agg.get("mask") and agg.get(vs):
                pooled = paired(agg, vs); pooled["seeds"] = len(per_seed)
        row = {"model": name, "rates": rates, "leads": leads, "vs": vs, "pooled_lead": pooled,
               "seed0_files_rate": {f: round(100 * sum(d.values()) / len(d), 1) for f, d in p0.items() if d}}   # should match rates[0]
        out.append(row)
        seeds = sorted(rates)
        L += [f"## {name}", "", "| arm | " + " | ".join(f"seed {s}" for s in seeds) + " | spread | SD |", "|---" * (len(seeds) + 3) + "|"]
        for f in FAMS:
            vals = [rates[s].get(f) for s in seeds]
            have = [v for v in vals if v is not None]
            spread = f"{max(have) - min(have):.1f}" if len(have) > 1 else "-"
            sd = f"{statistics.stdev(have):.1f}" if len(have) > 1 else "-"
            L.append(f"| {f} | " + " | ".join(f"{v:.1f}" if v is not None else "-" for v in vals) + f" | {spread} | {sd} |")
        L.append("| lead vs " + str(vs) + " | " + " | ".join(f"{leads[s]['diff']:+.1f} [{leads[s]['lo']:+.1f}, {leads[s]['hi']:+.1f}]" if leads.get(s) else "-" for s in seeds) + " | | |")
        if pooled:
            L.append(f"| lead, pooled over {pooled['seeds']} seeds | {pooled['diff']:+.1f} [{pooled['lo']:+.1f}, {pooled['hi']:+.1f}] (n={pooled['n']}) |" + " |" * (len(seeds) + 1))
        L.append("")
    json.dump(out, open(R / "seed_table.json", "w"), indent=1)
    (R / "seed_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
