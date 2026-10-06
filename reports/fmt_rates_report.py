#!/usr/bin/env python3
"""
Format study in honest rates (owner's request, 2026-10-06: the 'share of the gap closed' metric was hard to read).
From the fmt_summarize.py summaries (each build format at the strength chosen on validation), per model, method and half:
  single-turn: mean honest rate (%) under the order to lie over the off-diagonal single-turn cells (build i != eval j,
               i over all seven build formats, j over the six single-turn formats), ALL formats (no 10-point-gap filter);
  multi-turn:  mean over the seven build formats of the honest rate on the multi-turn format;
  uned. / ctrl.: the unedited model's honest rate under the order to lie / under the honesty-instructed control, averaged
               over the same cells;
  worst:       the lowest honest rate over those off-diagonal single-turn cells (36 per model and method);
  spread:      for the build-format paragraph, each build row's mean honest rate over the six single-turn eval formats
               (diagonal included), max - min over the seven rows.
Writes reports/fmt_rates.json. (Adapted from a scratch script of the dagformer session, which averaged only over formats
with a gap of at least 10 points.)
"""
import json
from pathlib import Path

FORMATS = ["F1_sys", "F2_user", "F3_inject", "F4_mc", "F5_tf", "F6_multi", "F7_long"]
ST = [f for f in FORMATS if f != "F6_multi"]
A = Path("/work/hdd/biro/xiaocong/lie_out/fmt_runs"); B = Path("/work/hdd/biro/xiaocong/lie_out/fmt_runs_timan")
FILES = {"Phi-4-mini": {"val": A / "summary_Phi-4-mini-instruct_val.json", "test": A / "summary_Phi-4-mini-instruct_test.json"},
         "Llama-3.1-8B": {"val": A / "summary_Meta-Llama-3p1-8B-Instruct_val.json", "test": B / "summary_Llama-3p1-8B-Instruct_test.json"},
         "Qwen2.5-7B": {"val": A / "summary_Qwen2p5-7B-Instruct_val.json", "test": B / "summary_Qwen2p5-7B-Instruct_test.json"},
         "Falcon3-3B": {"val": A / "summary_Falcon3-3B-Instruct_val.json", "test": B / "summary_Falcon3-3B-Instruct_test.json"}}
mean = lambda xs: sum(xs) / len(xs)


def main():
    out = {}
    for m, hv in FILES.items():
        for half, p in hv.items():
            S = json.load(open(p)); U, C = S["unedited"]["U"], S["unedited"]["C"]
            off = [(i, j) for i in FORMATS for j in ST if i != j]
            res = {"uned": {"st": mean([U[j] for _, j in off]), "multi": U["F6_multi"]},
                   "ctrl": {"st": mean([C[j] for _, j in off]), "multi": C["F6_multi"]}}
            for meth in ("maskc", "caa"):
                pick = S[meth]["row_pick"]
                rate = lambda i, j: S["arms"][f"{meth}:{i}:{pick[i]}"]["dec"][j]
                rows = {i.split("_")[1]: mean([rate(i, j) for j in ST]) for i in FORMATS}
                res[meth] = {"st": mean([rate(i, j) for i, j in off]), "multi": mean([rate(i, "F6_multi") for i in FORMATS]),
                             "worst": min(rate(i, j) for i, j in off),
                             "rows": rows, "spread": max(rows.values()) - min(rows.values()),
                             "multi_rows": {i.split("_")[1]: rate(i, "F6_multi") for i in FORMATS}}
            res["U_per_format"] = {j.split("_")[1]: U[j] for j in FORMATS}
            res["C_per_format"] = {j.split("_")[1]: C[j] for j in FORMATS}
            out.setdefault(m, {})[half] = res
            r = res
            print(f"{m:13s} {half:4s} single-turn uned {r['uned']['st']:5.1f} mask {r['maskc']['st']:5.1f} caa {r['caa']['st']:5.1f} "
                  f"ctrl {r['ctrl']['st']:5.1f} | multi uned {r['uned']['multi']:5.1f} mask {r['maskc']['multi']:5.1f} "
                  f"caa {r['caa']['multi']:5.1f} ctrl {r['ctrl']['multi']:5.1f} | spread mask {r['maskc']['spread']:4.1f} "
                  f"caa {r['caa']['spread']:4.1f}")
            for meth in ("maskc", "caa"):
                rows = r[meth]["rows"]; lo = min(rows, key=rows.get); hi = max(rows, key=rows.get)
                print(f"      {meth:5s} rows {lo} {rows[lo]:.1f} .. {hi} {rows[hi]:.1f}; multi by build: "
                      + " ".join(f"{k}:{v:.0f}" for k, v in r[meth]["multi_rows"].items()))
            print("      U " + " ".join(f"{k}:{v:.0f}" for k, v in r["U_per_format"].items()) + " | C "
                  + " ".join(f"{k}:{v:.0f}" for k, v in r["C_per_format"].items()))
    json.dump(out, open(Path(__file__).parent / "fmt_rates.json", "w"), indent=1)


if __name__ == "__main__":
    main()
