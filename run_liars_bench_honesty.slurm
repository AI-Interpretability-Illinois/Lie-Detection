#!/bin/bash
#SBATCH --job-name=lie_lb
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/lie_lb_%j.out
#SBATCH --error=slurm_logs/lie_lb_%j.out
set -u
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_CACHE_DIR=/work/hdd/bfqt/xiaocong/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
cd /u/xiaocong/lie_detection
python run_liars_bench_honesty.py --model-id "${MODEL_ID:-Qwen/Qwen2.5-1.5B-Instruct}" \
    --rho "${RHO:-0.04}" --ratio "${RATIO:-0.37}" --n-hp 800 --n-id 400 --n-alpaca 150
