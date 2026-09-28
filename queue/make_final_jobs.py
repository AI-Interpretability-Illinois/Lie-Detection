#!/usr/bin/env python3
"""
Jobs for the final table: every model gets the same options for every method.

  masks     six families (ours, pooled, fiction, fictf, pooltf, poolmix) at
            -0.25 / -0.5 / -0.75 / -1.0, plus the magnitude-matched control
  steering  three layers x two strengths per model, as before
  prompting three honest system-prompt wordings (prompt, prompt2, prompt3)

Full runs ("final" tags) for the four models that only had strength -1.0 and one
prompt wording. Add-on runs ("finaladd") for the three models from the polarity-fix
rerun, which already have three strengths: they need the -0.25 strength and the two
new prompt wordings only. The table script merges tags.

GPU caps for sharing gpua047 with the DAGFormer training: 18 GB for 7-8B generation,
17 GB for the 7B judge, 12 GB for small-model generation.
"""
from pathlib import Path

REPO = Path("/u/xiaocong/lie_detection")
Q = REPO / "queue"
X = "/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data"
FAMS = ["ours", "pooled", "fiction", "fictf", "pooltf", "poolmix"]
FULL = [("meta-llama/Meta-Llama-3.1-8B-Instruct", "8,14,21", 18, 8),
        ("Qwen/Qwen2.5-7B-Instruct", "7,14,21", 18, 8),
        ("Qwen/Qwen2.5-3B-Instruct", "9,18,27", 12, 16),
        ("allenai/OLMo-2-0425-1B-Instruct", "4,8,12", 12, 16)]
ADD = [("microsoft/Phi-4-mini-instruct", "8,16,24", 12, 16),
       ("tiiuae/Falcon3-3B-Instruct", "5,11,16", 12, 16),
       ("Qwen/Qwen2.5-1.5B-Instruct", "7,14,21", 12, 16)]


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
    n = 0
    for group, models, tag_lb, tag_ail in (("f", FULL, "final", "finalail"),
                                          ("g", ADD, "finaladd", "finailadd")):
        for i, (m, layers, mem, bs) in enumerate(models):
            s = short(m)
            if group == "f":
                arms = (["unsteered", "prompt", "prompt2", "prompt3", "magmatch100"]
                        + [f"{f}{k}" for f in FAMS for k in ("025", "050", "075", "100")]
                        + [f"asrL{L}_{r}" for L in layers.split(",") for r in ("-0.18", "-0.5")])
            else:
                arms = ["unsteered", "prompt2", "prompt3"] + [f"{f}025" for f in FAMS]
            common = (f"python run_liars_bench_honesty.py --model-id {m} --rho 0.04 --ratio 0.37 "
                      f"--falsehood-arms --polarity-arms --arms '{','.join(arms)}' "
                      f"--as-layers {layers} --as-rel-coeffs=-0.18,-0.5 "
                      f"--batch-size {bs} --score-batch 2")
            a = f"1{group}{i}a_{tag_lb}_{s}"
            write(a, mem, common + f" --splits instructed-deception,instructed-truthful "
                                   f"--n-id 537 --tag {tag_lb}")
            write(f"1{group}{i}b_judge_{tag_lb}_{s}", 17,
                  f"python judge_claims.py --responses liars_bench_honesty_{s}_{tag_lb}_responses.jsonl "
                  f"--judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2", needs=[marker(a)])
            write(f"1{group}{i}c_{tag_ail}_{s}", mem,
                  common + f" --eval-set {X}/extra_ailiar.jsonl --belief-tag ailiar --tag {tag_ail}")
            n += 3
    print(f"wrote {n} jobs")


if __name__ == "__main__":
    main()
