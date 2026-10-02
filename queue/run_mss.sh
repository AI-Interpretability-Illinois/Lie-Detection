#!/bin/bash
# usage: GPU=1 run_mss.sh model_id [model_id ...]  -- mask sample-size test, one model after another
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_MAX_GPU_GB=${LIE_MAX_GPU_GB:-10}
cd /u/xiaocong/lie_detection
for m in "$@"; do
  s=$(basename "$m")
  echo "$(date '+%F %T') start $m on gpu$GPU"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python mask_sample_size.py --model-id "$m" --batch 8 > queue/logs/mss_$s.log 2>&1
  echo "$(date '+%F %T') end $m rc=$?"
done
