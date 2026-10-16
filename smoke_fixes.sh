#!/bin/bash
# Smoke test for the 2026-09-27 evaluation fixes, on Qwen2.5-0.5B.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-2}
echo "##### A: benchmark runner, both instructed halves"
python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --splits instructed-deception,instructed-truthful --n-id 24 --n-idt 16 \
  --arms unsteered,ours100 --batch-size 16 --score-batch 4 --tag smokefix 2>&1 \
  | grep -vE "Warning|warnings.warn|Loading checkpoint"
echo "##### B: pressure metric end to end"
python run_instructed_falsehood.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
  --arms lie,magmatch --knobs=-1.0 --batch-size 32 --score-batch 4 --tag smokefix 2>&1 \
  | grep -vE "Warning|warnings.warn|Loading checkpoint"
echo "##### done"
