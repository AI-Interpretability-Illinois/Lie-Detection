#!/bin/bash
# Qwen2.5-3B-Instruct: the pressure-metric suite on one GPU, in sequence.
#   1. instructed-falsehood benchmark (six arms, damping to full ablation)
#   2. head-to-head against activation steering swept over depth
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-1}
MID=Qwen/Qwen2.5-3B-Instruct
echo "=== $(date '+%F %H:%M') instructed falsehood"
nice -n 10 python run_instructed_falsehood.py --model-id $MID --rho 0.04 --ratio 0.37 \
  --knobs="-0.5,-0.75,-1.0" --batch-size 32 --score-batch 4 --tag resw \
  > slurm_logs/instrfalse_Qwen2p5-3B-Instruct.log 2>&1
echo "=== $(date '+%F %H:%M') exit $? ; head-to-head"
nice -n 10 python run_prior_work_baselines.py --model-id $MID --rho 0.04 --ratio 0.37 \
  --variants score_diff_instruct --damp=-1.0 --amp=3.0 \
  --as-layers 6,12,18,24,30 --as-rel-coeffs=-0.06,-0.18,-0.5 \
  --batch-size 32 --score-batch 4 --tag q3b_h2h \
  > slurm_logs/h2h_qwen3b.log 2>&1
echo "=== $(date '+%F %H:%M') exit $?"
