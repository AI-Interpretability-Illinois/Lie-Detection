#!/usr/bin/env python3
"""
Sampling-seed replication of Table 1 (queue/lie_seed.slurm): the LIARS' BENCH-chosen arms of every main-table model
regenerated with torch seeds 1 and 2 (replies are sampled at temperature 0.7) and judged by the Qwen2.5-7B hybrid, against
the paper's run (seed 0 = reports/main_table_final.json, same judge). Reports, per model and arm family, the test-half honest
rate under each seed, the spread (max - min) and the standard deviation across seeds, and the lead of the falsehood mask over
the comparator fixed on validation under each seed. Writes reports/seed_table.json and reports/seed_report.md.
"""
import glob
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
            fs = glob.glob(str(SWEEP / f"liars_bench_honesty_{stem}_seed{s}_judged_Qwen2p5-7B-Instruct_v2.jsonl"))
            if not fs:
                continue
            per = defaultdict(dict)
            for r in map(json.loads, open(fs[0])):
                if r["split"] != "instructed-deception" or half(r["uid"]) != "test" or r.get("verdict_hybrid") not in ("honest", "lie"):
                    continue
                f = family(r["arm"])
                if f:
                    per[f][r["uid"]] = r["verdict_hybrid"] == "honest"
            rates[s] = {f: round(100 * sum(d.values()) / len(d), 1) for f, d in per.items() if d}
            leads[s] = paired(per, vs) if vs and "mask" in per and vs in per else None
        row = {"model": name, "rates": rates, "leads": leads, "vs": vs}
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
        L.append("")
    json.dump(out, open(R / "seed_table.json", "w"), indent=1)
    (R / "seed_report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
