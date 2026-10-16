#!/bin/bash
# LLM-judge re-grading of the corrected LIARS' BENCH runs, one GPU, in sequence.
# Qwen2.5-7B judges every model; Llama-3.1-8B also judges the Qwen runs, so no
# model is only ever graded by its own family.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-3}
run() {
  echo "=== $(date '+%H:%M') judge=$2 on $1"
  nice -n 10 python judge_claims.py --responses "$1" --judge "$2" --batch-size 32 --tag v2 2>&1 \
    | grep -vE "Warning|warnings.warn|Loading checkpoint|torch_dtype"
}
for m in Qwen2p5-7B-Instruct Phi-4-mini-instruct; do
  run liars_bench_honesty_${m}_idfix2_responses.jsonl Qwen/Qwen2.5-7B-Instruct
done
run liars_bench_honesty_Qwen2p5-7B-Instruct_idfix2_responses.jsonl meta-llama/Meta-Llama-3.1-8B-Instruct
# wait for the runs still going, then judge them too
for m in gemma-3-4b-it Qwen2p5-3B-Instruct; do
  until grep -qE "rows →|Traceback" slurm_logs/lb_idfix2_$( [ $m = gemma-3-4b-it ] && echo gemma4b || echo qwen3b ).log 2>/dev/null; do sleep 60; done
  run liars_bench_honesty_${m}_idfix2_responses.jsonl Qwen/Qwen2.5-7B-Instruct
done
run liars_bench_honesty_Qwen2p5-3B-Instruct_idfix2_responses.jsonl meta-llama/Meta-Llama-3.1-8B-Instruct
echo "=== all judged"
