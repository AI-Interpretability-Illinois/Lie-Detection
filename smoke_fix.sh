#!/bin/bash
source ~/.bashrc >/dev/null 2>&1; conda activate modularity 2>/dev/null
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-1} LIE_MAX_GPU_GB=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-0.5B-Instruct --rho 0.04 --ratio 0.37 \
  --falsehood-arms --polarity-arms --arms unsteered,prompt2,prompt3,fictf025,ours025 \
  --splits instructed-deception,instructed-truthful --n-id 16 --n-idt 8 \
  --batch-size 16 --score-batch 4 --tag smokefix3 2>&1 | grep -vE "Warning|warnings.warn|Loading checkpoint|torch_dtype"
echo "##### SMOKE DONE"
