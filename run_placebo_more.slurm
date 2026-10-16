#!/bin/bash
#SBATCH --job-name=lie_plac2
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/lie_plac2_%j.out
#SBATCH --error=slurm_logs/lie_plac2_%j.out
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
set -u
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_CACHE_DIR=/work/hdd/bfqt/xiaocong/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
cd /u/xiaocong/lie_detection
echo "########## placebo: Qwen2.5-7B-Instruct (repo's Qwen operating point)"
python run_placebo_contrast.py --model-id Qwen/Qwen2.5-7B-Instruct \
    --rho 0.06 --ratio 0.35 --damp -0.75 --amp 3.0 --batch-size 32 --score-batch 4 --tag q7b
echo "########## placebo: gemma-3-4b-it"
python run_placebo_contrast.py --model-id google/gemma-3-4b-it \
    --rho 0.04 --ratio 0.37 --damp -0.75 --amp 3.0 --batch-size 24 --score-batch 4 --tag g4b
echo "########## placebo: Falcon3-3B-Instruct (mask harmed this model)"
python run_placebo_contrast.py --model-id tiiuae/Falcon3-3B-Instruct \
    --rho 0.04 --ratio 0.37 --damp -0.75 --amp 3.0 --batch-size 24 --score-batch 4 --tag falcon3b
