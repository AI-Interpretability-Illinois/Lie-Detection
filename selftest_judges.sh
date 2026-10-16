#!/bin/bash
source ~/.bashrc >/dev/null 2>&1; conda activate modularity 2>/dev/null
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${GPU:-2}
B=/work/hdd/bfqt/xiaocong/lie_detection_outputs/liars_bench
for J in meta-llama/Meta-Llama-3.1-8B-Instruct; do
  echo "##### $J"
  python judge_mask.py --responses liars_bench_honesty_Qwen2p5-0p5B-Instruct_smoke_mask_responses.jsonl \
    --beliefs $B/beliefs_Qwen2p5-0p5B-Instruct_smoke_mask.jsonl --judge $J --selftest-n 120 \
    --selftest-only --min-selftest 0 --batch-size 8 2>&1 | grep -E "self-test|Traceback|Error"
done
echo "##### SELFTEST DONE"
