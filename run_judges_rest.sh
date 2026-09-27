#!/bin/bash
# Judge the last three corrected-benchmark runs once they exist, after the
# first judge chain has released this GPU.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-3}
# (no longer waits for the first judge chain: runs on its own GPU)
for m in Falcon3-3B-Instruct OLMo-2-0425-1B-Instruct Qwen2p5-1p5B-Instruct; do
  until grep -qE "rows →|Traceback" slurm_logs/lb_idfix2_$m.log 2>/dev/null; do sleep 60; done
  echo "=== $(date '+%H:%M') judge on $m"
  nice -n 10 python judge_claims.py --responses liars_bench_honesty_${m}_idfix2_responses.jsonl \
    --judge Qwen/Qwen2.5-7B-Instruct --batch-size 32 --tag v2 2>&1 \
    | grep -vE "Warning|warnings.warn|Loading checkpoint|torch_dtype"
done
echo "=== all judged"
