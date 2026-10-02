#!/usr/bin/env python3
"""Tables for test (b) from outputs/controls/mask_sample_size_<model>.json (mask_sample_size.py)."""
import json
from pathlib import Path

D = Path("/u/xiaocong/lie_detection/outputs/controls")
ORDER = ["Llama-3p1-8B-Instruct", "Phi-4-mini-instruct", "Falcon3-3B-Instruct",
         "OLMo-2-0425-1B-Instruct", "Qwen2p5-3B-Instruct", "Qwen2p5-7B-Instruct", "Qwen2p5-1p5B-Instruct"]
NAME = dict(zip(ORDER, ["Llama-3.1-8B", "Phi-4-mini", "Falcon3-3B", "OLMo-2-1B", "Qwen2.5-3B",
                        "Qwen2.5-7B", "Qwen2.5-1.5B"]))


def main():
    res = {m: json.loads((D / f"mask_sample_size_{m}.json").read_text())
           for m in ORDER if (D / f"mask_sample_size_{m}.json").exists()}
    for fam in ("fiction", "pooled"):
        print(f"\n### {fam}: agreement of two masks from disjoint prompt draws (weight-weighted Jaccard)\n")
        ns = next(iter(res.values()))["ns"]
        print("| model | " + " | ".join(f"n={n}" for n in ns) + " | positive-score weights only: "
              + " | ".join(f"n={n}" for n in ns) + " |")
        print("|---" * (2 * len(ns) + 1) + "|")
        for m, r in res.items():
            J = r["jaccard"]
            print(f"| {NAME[m]} | " + " | ".join(f"{J[f'{fam} | split | {n}']:.2f}" for n in ns) + " | "
                  + " | ".join(f"{J[f'{fam} | split_signal | {n}']:.2f}" for n in ns) + " |")
        print(f"\n### {fam}: the hand-written 25 prompts, and what the 4% budget is made of\n")
        print("| model | hand-written 25 vs 400-prompt mask (positive-score) | random 25 vs 400 (positive-score) "
              "| padding share r=0.37 | positive-score weights, % of matrix | padding share at r = 1 / 0.667 / 0.5 / 0.25 "
              "| overlap with r=0.37 mask at r = 1 / 0.667 / 0.5 / 0.25 |")
        print("|---|---|---|---|---|---|---|")
        for m, r in res.items():
            J, P = r["jaccard"], r["padding_share"]
            pad = P[f"{fam} | {r['ratio']}"]
            xs = (1.0, 0.667, 0.5, 0.25)
            pads = " / ".join("%.2f" % P["%s | %s" % (fam, x)] for x in xs)
            laps = " / ".join("%.2f" % J["%s | lambda | %s" % (fam, x)] for x in xs)
            print(f"| {NAME[m]} | {J[f'{fam} | orig_vs_ref']:.2f} ({J[f'{fam} | orig_vs_ref_signal']:.2f}) "
                  f"| {J[f'{fam} | vs_ref | 25']:.2f} ({J[f'{fam} | vs_ref_signal | 25']:.2f}) "
                  f"| {pad:.2f} | {100 * r['rho'] * (1 - pad):.2f} "
                  f"| {pads} | {laps} |")


if __name__ == "__main__":
    main()
