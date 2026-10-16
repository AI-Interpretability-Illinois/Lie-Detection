#!/usr/bin/env python3
"""
Queue jobs for the polarity fix on the three models where no mask passed the
damage checks (Qwen2.5-1.5B, Gemma-3-4B, Falcon3-3B), plus Phi-4-mini as a check
that the fix does not hurt a model where the method already works.

Per model:
  fixlb    LIARS' BENCH (deception-instructed + honesty-instructed control), every
           mask family at -0.5 / -0.75 / -1.0 including the polarity-balanced
           families, plus the same baselines as the main table
  judge    hybrid keyword + LLM-judge grading of fixlb (Qwen2.5-7B judge)
  fixail   How to Catch an AI Liar with the same arms

Every job carries its own GPU memory cap (LIE_MAX_GPU_GB), for running beside the
DAGFormer training on gpua047: 12 GB for generation, 17 GB for the 7B judge.
"""

from pathlib import Path

REPO = Path("/u/xiaocong/lie_detection")
Q = REPO / "queue"
X = "/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data"
MODELS = [("Qwen/Qwen2.5-1.5B-Instruct", "7,14,21"),
          ("google/gemma-3-4b-it", "8,17,25"),
          ("tiiuae/Falcon3-3B-Instruct", "5,11,16"),
          ("microsoft/Phi-4-mini-instruct", "8,16,24")]      # check: already works
FAMS = ["ours", "pooled", "fiction", "fictf", "pooltf", "poolmix"]


def short(m):
    return m.split("/")[-1].replace(".", "p")


def marker(n):
    return f"{Q}/markers/{n}.ok"


def arms(layers):
    a = ["unsteered", "prompt", "magmatch100"]
    a += [f"{f}{s}" for f in FAMS for s in ("050", "075", "100")]
    a += [f"asrL{L}_{r}" for L in layers.split(",") for r in ("-0.18", "-0.5")]
    return ",".join(a)


def write(name, mem, cap, body, needs=()):
    lines = ["#!/bin/bash", f"# MEM_GB: {mem}", f"# DONE_IF: {marker(name)}"]
    lines += [f"# NEEDS: {n}" for n in needs]
    lines += ["set -eo pipefail", f"cd {REPO}",
              "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
              f"export LIE_MAX_GPU_GB={cap}", body.strip(), f"touch {marker(name)}", ""]
    (Q / "pending" / f"{name}.sh").write_text("\n".join(lines))


def main():
    n = 0
    for i, (m, layers) in enumerate(MODELS):
        s = short(m)
        common = (f"python run_liars_bench_honesty.py --model-id {m} --rho 0.04 --ratio 0.37 "
                  f"--falsehood-arms --polarity-arms --arms '{arms(layers)}' "
                  f"--as-layers {layers} --as-rel-coeffs=-0.18,-0.5 "
                  f"--batch-size 16 --score-batch 2")
        g = f"0{i}a_fixlb_{s}"
        write(g, 12, 12, common + " --splits instructed-deception,instructed-truthful,alpaca "
                                  "--n-id 537 --tag fixlb")
        write(f"0{i}b_fixjudge_{s}", 17, 17,
              f"python judge_claims.py --responses liars_bench_honesty_{s}_fixlb_responses.jsonl "
              f"--judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2",
              needs=[marker(g)])
        write(f"0{i}c_fixail_{s}", 12, 12,
              common + f" --eval-set {X}/extra_ailiar.jsonl --belief-tag ailiar --tag fixail")
        n += 3
    print(f"wrote {n} jobs")


if __name__ == "__main__":
    main()
