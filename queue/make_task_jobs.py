#!/usr/bin/env python3
"""
Jobs for five further tasks, each method at the ONE setting per model chosen on the
LIARS' BENCH validation half (reports/main_table_final.json). Nothing is re-tuned per
task, so these results measure whether those choices carry over.

Order (file names sort into run order):
  2s  instructed sandbagging (MMLU + WMDP) and plain-MMLU accuracy   graded inline
  3i  injected deception instructions (+ no-injection control)       + judge
  4m  MASK, instructed and incentive groups                          + MASK judge
  5a  Azaria-Mitchell / Geometry of Truth statements                 + judge
  6t  keeping a lie going over a second turn                         graded inline
Arms per model: unsteered, magnitude-matched random mask, and the chosen setting of
our mask, the published lie mask, steering and the honest prompt, where one passed
the damage checks.
"""
import json
import re
from pathlib import Path

REPO = Path("/u/xiaocong/lie_detection")
Q = REPO / "queue"
OUT = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
X = f"{OUT}/extra_data"
B = f"{OUT}/liars_bench"
QJ, LJ = "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODELS = {  # label -> model id, steering layers, gen memory, batch
    "Llama-3.1-8B": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "8,14,21", 18, 8),
    "Phi-4-mini": ("microsoft/Phi-4-mini-instruct", "8,16,24", 12, 16),
    "Falcon3-3B": ("tiiuae/Falcon3-3B-Instruct", "5,11,16", 12, 16),
    "OLMo-2-1B": ("allenai/OLMo-2-0425-1B-Instruct", "4,8,12", 12, 16),
    "Qwen2.5-3B": ("Qwen/Qwen2.5-3B-Instruct", "9,18,27", 12, 16),
    "Qwen2.5-7B": ("Qwen/Qwen2.5-7B-Instruct", "7,14,21", 18, 8),
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", "7,14,21", 12, 16),
}


def arm_key(label):
    if label == "unsteered":
        return "unsteered"
    m = re.match(r"weight:(\w+) a=(-?[\d.]+)", label)
    if m:
        return f"{m.group(1)}{int(round(abs(float(m.group(2))) * 100)):03d}"
    m = re.match(r"actsteer:L(\d+) r=(-?[\d.]+)", label)
    if m:
        return f"asrL{m.group(1)}_{m.group(2)}"
    m = re.match(r"prompt:honest(\d?)", label)
    if m:
        return "prompt" + m.group(1)
    raise ValueError(label)


def short(m):
    return m.split("/")[-1].replace(".", "p")


def marker(n):
    return f"{Q}/markers/{n}.ok"


def write(name, mem, body, needs=()):
    lines = ["#!/bin/bash", f"# MEM_GB: {mem}", f"# DONE_IF: {marker(name)}"]
    lines += [f"# NEEDS: {n}" for n in needs]
    lines += ["set -eo pipefail", f"cd {REPO}",
              "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
              f"export LIE_MAX_GPU_GB={mem}", body.strip(), f"touch {marker(name)}", ""]
    (Q / "pending" / f"{name}.sh").write_text("\n".join(lines))


def main():
    table = {r["model"]: r for r in json.load(open(REPO / "reports/main_table_final.json"))}
    n = 0
    for i, (label, (mid, layers, mem, bs)) in enumerate(MODELS.items()):
        cells = table[label]["lb"]["cells"]
        arms = ["unsteered", "magmatch100"] + [arm_key(c["arm"]) for f, c in cells.items()
                                               if f != "unsteered" and c["v"] is not None]
        s = short(mid)
        base = (f"python run_liars_bench_honesty.py --model-id {mid} --rho 0.04 --ratio 0.37 "
                f"--falsehood-arms --polarity-arms --arms '{','.join(arms)}' "
                f"--as-layers {layers} --as-rel-coeffs=-0.18,-0.5 --score-batch 2")
        mask_bs = max(4, bs // 2)
        # 2s sandbagging
        write(f"2s{i}_sandbag_{s}", mem,
              base + f" --batch-size {bs} --eval-set {X}/extra_sandbag.jsonl --belief-tag sandbag --tag tsandbag")
        # 3i injected instructions + judge
        g = f"3i{i}a_inject_{s}"
        write(g, mem, base + f" --batch-size {bs} --eval-set {X}/extra_inject.jsonl --belief-tag inject --tag tinject")
        write(f"3i{i}b_injectjudge_{s}", 17,
              f"python judge_claims.py --responses liars_bench_honesty_{s}_tinject_responses.jsonl "
              f"--splits inject-deception,inject-control --beliefs {B}/beliefs_{s}_inject.jsonl "
              f"--eval-set {X}/extra_inject.jsonl --judge {QJ} --batch-size 8 --tag v2", needs=[marker(g)])
        # 4m MASK + judge (a Llama cross-judge on Qwen2.5-7B for inter-judge agreement)
        g = f"4m{i}a_mask_{s}"
        write(g, mem, base + f" --batch-size {mask_bs} --eval-set {X}/extra_mask.jsonl "
                             f"--belief-tag tmask --n-per-split 80 --tag tmask")
        jm = (f"python judge_mask.py --responses liars_bench_honesty_{s}_tmask_responses.jsonl "
              f"--beliefs {B}/beliefs_{s}_tmask.jsonl")
        write(f"4m{i}b_maskjudge_{s}", 17, jm + f" --judge {QJ} --batch-size 8 --tag v1", needs=[marker(g)])
        if label == "Qwen2.5-7B":
            write(f"4m{i}c_maskjudge_llama_{s}", 18, jm + f" --judge {LJ} --batch-size 4 --tag v1",
                  needs=[marker(g)])
            n += 1
        # 5a Azaria-Mitchell / GoT + judge
        g = f"5a{i}a_amgot_{s}"
        write(g, mem, base + f" --batch-size {bs} --eval-set {X}/extra_amgot.jsonl --belief-tag tamgot --tag tamgot")
        write(f"5a{i}b_amgotjudge_{s}", 17,
              f"python judge_claims.py --responses liars_bench_honesty_{s}_tamgot_responses.jsonl "
              f"--splits amgot-deception --beliefs {B}/beliefs_{s}_tamgot.jsonl "
              f"--eval-set {X}/extra_amgot.jsonl --judge {QJ} --batch-size 8 --tag v2", needs=[marker(g)])
        # 6t multi-turn
        write(f"6t{i}_multiturn_{s}", mem,
              base + f" --batch-size {bs} --eval-set {X}/extra_multiturn.jsonl --belief-tag multiturn --tag tmulti")
        n += 9
        print(f"{label:<14} arms: {','.join(arms)}")
    print(f"wrote {n} jobs")


if __name__ == "__main__":
    main()
