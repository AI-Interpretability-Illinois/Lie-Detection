#!/bin/bash
#SBATCH --job-name=lie_did
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/lie_did_%j.out
#SBATCH --error=slurm_logs/lie_did_%j.out
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
set -u   # after bashrc: /etc/bashrc trips -u on BASHRCSOURCED
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
cd /u/xiaocong/lie_detection
python run_real_did.py --model-id "${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}" \
    --rho "${RHO:-0.04}" --ratio "${RATIO:-0.37}" --n-truth 25 \
    --k-samples "${KS:-10}" --reps 1 --score-batch 8
