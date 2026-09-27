#!/bin/bash
# Usage: GPU=2 BS=12 SB=2 ./if_runner.sh <model-id> [<model-id> ...]
# Runs the instructed-falsehood benchmark on each model in turn on one GPU.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-0}
mkdir -p slurm_logs
for MID in "$@"; do
  SHORT=$(basename "$MID" | tr '.' 'p')
  LOG=slurm_logs/instrfalse_${SHORT}.log
  echo "=== $(date '+%F %H:%M') start $MID on GPU ${GPU:-0} -> $LOG"
  nice -n 10 python run_instructed_falsehood.py \
      --model-id "$MID" --rho 0.04 --ratio 0.37 \
      --knobs="-0.5,-0.75,-1.0" \
      --batch-size "${BS:-24}" --score-batch "${SB:-4}" \
      --tag resw > "$LOG" 2>&1
  echo "=== $(date '+%F %H:%M') exit $? for $MID"
done
