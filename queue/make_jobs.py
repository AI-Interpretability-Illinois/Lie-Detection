#!/usr/bin/env python3
"""
Write queue job files for the extra-dataset evaluation plan.

Priority order (file names sort into run order):
  0x  resume the work killed at 02:13 when the allocation ended: judges not yet
      run, OLMo's interrupted benchmark run, and Qwen2.5-1.5B's
  1x  How to Catch an AI Liar, eight models (graded inline, no judge)
  2x  MASK, eight models, each followed by the MASK judge; a second judge from a
      different family on four models so inter-judge agreement can be reported
  3x  Azaria-Mitchell / Geometry of Truth, eight models, each followed by the
      hybrid true/false judge
Within a dataset the three main-table models go first.
"""

from pathlib import Path

REPO = Path("/u/xiaocong/lie_detection")
Q = REPO / "queue"
OUT = "/work/hdd/bfqt/xiaocong/lie_detection_outputs"
X = f"{OUT}/extra_data"
B = f"{OUT}/liars_bench"
QJ, LJ = "Qwen/Qwen2.5-7B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct"

# model id, activation layers, memory for generation, generation batch, MASK batch
MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B-Instruct", "8,14,21", 20, 16, 8),
    ("microsoft/Phi-4-mini-instruct",         "8,16,24", 12, 16, 12),
    ("Qwen/Qwen2.5-7B-Instruct",              "7,14,21", 20, 16, 8),
    ("google/gemma-3-4b-it",                  "8,17,25", 14, 16, 12),
    ("Qwen/Qwen2.5-3B-Instruct",              "9,18,27", 10, 24, 12),
    ("tiiuae/Falcon3-3B-Instruct",            "5,11,16", 10, 24, 12),
    ("allenai/OLMo-2-0425-1B-Instruct",       "4,8,12",   8, 24, 16),
    ("Qwen/Qwen2.5-1.5B-Instruct",            "7,14,21",  8, 24, 16),
]
MASK_ARMS = "unsteered,ours100,pooled100,fiction100,magmatch100,prompt"


def short(m):
    return m.split("/")[-1].replace(".", "p")


def marker(name):
    return f"{Q}/markers/{name}.ok"


def arms(layers):
    a = MASK_ARMS
    for L in layers.split(","):
        a += f",asrL{L}_-0.18,asrL{L}_-0.5"
    return a


def write(name, mem, body, needs=()):
    lines = ["#!/bin/bash", f"# MEM_GB: {mem}", f"# DONE_IF: {marker(name)}"]
    lines += [f"# NEEDS: {n}" for n in needs]
    lines += ["set -eo pipefail", f"cd {REPO}", body.strip(), f"touch {marker(name)}", ""]
    (Q / "pending" / f"{name}.sh").write_text("\n".join(lines))


def gen_cmd(m, layers, bs, tag, evalset=None, splits=None, extra=""):
    c = (f"python run_liars_bench_honesty.py --model-id {m} --rho 0.04 --ratio 0.37 "
         f"--falsehood-arms --arms '{arms(layers)}' --as-layers {layers} "
         f"--as-rel-coeffs=-0.18,-0.5 --batch-size {bs} --score-batch 2 --tag {tag}")
    if evalset:
        c += f" --eval-set {X}/extra_{evalset}.jsonl --belief-tag {evalset}"
    if splits:
        c += f" --splits {splits}"
    return c + (" " + extra if extra else "")


def main():
    n = 0
    # ── 0x: resume ───────────────────────────────────────────────────────────
    def judge_lb(m, j):
        return (f"python judge_claims.py --responses liars_bench_honesty_{short(m)}_idfix2_responses.jsonl "
                f"--judge {j} --batch-size {16 if j == LJ else 32} --tag v2")
    write("00_judge_qwenjudge_gemma-3-4b-it", 17, judge_lb("google/gemma-3-4b-it", QJ))
    write("01_judge_qwenjudge_Qwen2p5-3B", 17, judge_lb("Qwen/Qwen2.5-3B-Instruct", QJ))
    write("02_judge_qwenjudge_Falcon3-3B", 17, judge_lb("tiiuae/Falcon3-3B-Instruct", QJ))
    write("03_judge_llamajudge_Qwen2p5-7B", 20, judge_lb("Qwen/Qwen2.5-7B-Instruct", LJ))
    write("04_judge_llamajudge_Qwen2p5-3B", 20, judge_lb("Qwen/Qwen2.5-3B-Instruct", LJ))
    for i, (m, layers) in enumerate([("allenai/OLMo-2-0425-1B-Instruct", "4,8,12"),
                                     ("Qwen/Qwen2.5-1.5B-Instruct", "7,14,21")]):
        g = f"05_lb_idfix2_{short(m)}"
        write(g, 8, gen_cmd(m, layers, 24, "idfix2",
                            splits="instructed-deception,instructed-truthful,alpaca",
                            extra="--n-id 537"))
        write(f"06_judge_qwenjudge_{short(m)}", 17, judge_lb(m, QJ), needs=[marker(g)])
        n += 2
    n += 5

    # ── 1x: AI Liar ──────────────────────────────────────────────────────────
    for i, (m, layers, mem, bs, _) in enumerate(MODELS):
        write(f"1{i}_ailiar_{short(m)}", mem, gen_cmd(m, layers, bs, "ailiar", "ailiar"))
        n += 1

    # ── 2x: MASK ─────────────────────────────────────────────────────────────
    for i, (m, layers, mem, _, mbs) in enumerate(MODELS):
        g = f"2{i}a_mask_{short(m)}"
        write(g, mem + 2, gen_cmd(m, layers, mbs, "mask", "mask", extra="--n-per-split 80"))
        resp = f"liars_bench_honesty_{short(m)}_mask_responses.jsonl"
        bel = f"{B}/beliefs_{short(m)}_mask.jsonl"
        write(f"2{i}b_maskjudge_qwen_{short(m)}", 18,
              f"python judge_mask.py --responses {resp} --beliefs {bel} --judge {QJ} "
              f"--batch-size 12 --tag v1",
              needs=[marker(g)])
        n += 2
        # a second judge from another family: on the Qwen models (so no model is
        # graded only by its own family) and on two others for agreement
        if short(m) in ("Qwen2p5-7B-Instruct", "Qwen2p5-3B-Instruct",
                        "Phi-4-mini-instruct", "Qwen2p5-1p5B-Instruct"):
            write(f"2{i}c_maskjudge_llama_{short(m)}", 21,
                  f"python judge_mask.py --responses {resp} --beliefs {bel} --judge {LJ} "
                  f"--batch-size 8 --tag v1",
                  needs=[marker(g)])
            n += 1

    # ── 3x: Azaria-Mitchell / Geometry of Truth ──────────────────────────────
    for i, (m, layers, mem, bs, _) in enumerate(MODELS):
        g = f"3{i}a_amgot_{short(m)}"
        write(g, mem, gen_cmd(m, layers, bs, "amgot", "amgot"))
        j = QJ if not short(m).startswith("Qwen") else LJ
        write(f"3{i}b_amgotjudge_{short(m)}", 20 if j == LJ else 17,
              f"python judge_claims.py --responses liars_bench_honesty_{short(m)}_amgot_responses.jsonl "
              f"--splits amgot-deception --beliefs {B}/beliefs_{short(m)}_amgot.jsonl "
              f"--eval-set {X}/extra_amgot.jsonl "
              f"--judge {j} --batch-size {16 if j == LJ else 32} --tag v2",
              needs=[marker(g)])
        n += 2
    print(f"wrote {n} jobs to {Q/'pending'}")


if __name__ == "__main__":
    main()
