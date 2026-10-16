#!/bin/bash
#SBATCH --job-name=lie_dose
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/lie_dose_%j.out
#SBATCH --error=slurm_logs/lie_dose_%j.out
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
set -u   # after bashrc: /etc/bashrc trips -u on BASHRCSOURCED
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_CACHE_DIR=/work/hdd/bfqt/xiaocong/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
cd /u/xiaocong/lie_detection
python run_dose_response.py --model-id "${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}" \
    --rho 0.04 --ratio 0.37 --knobs="${KNOBS}" --tag "${TAG}" \
    --batch-size 48 --score-batch 8
