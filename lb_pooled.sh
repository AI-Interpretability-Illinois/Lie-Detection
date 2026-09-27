#!/bin/bash
# LIARS' BENCH with the pooled-falsehood treatment arm, on one GPU.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-1}
nice -n 10 python run_liars_bench_honesty.py \
  --model-id "${MID:-meta-llama/Meta-Llama-3.1-8B-Instruct}" \
  --rho 0.04 --ratio 0.37 --falsehood-arms \
  --arms unsteered,ours100,pooled100,fiction100,magmatch100,prompt \
  --batch-size "${BS:-8}" --score-batch 2 --tag pooled
