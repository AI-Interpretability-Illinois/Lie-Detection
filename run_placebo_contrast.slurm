#!/bin/bash
#SBATCH --job-name=lie_placebo
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --output=slurm_logs/lie_placebo_%j.out
#SBATCH --error=slurm_logs/lie_placebo_%j.out
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
set -u   # after bashrc: /etc/bashrc trips -u on BASHRCSOURCED
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_CACHE_DIR=/work/hdd/bfqt/xiaocong/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
cd /u/xiaocong/lie_detection

# Cheap smoke first: this runner has never been executed on a GPU, and a failure
# after the 8B model loads costs far more than three minutes here.
echo "########## SMOKE: Qwen2.5-0.5B-Instruct"
python run_placebo_contrast.py --model-id Qwen/Qwen2.5-0.5B-Instruct \
    --rho 0.04 --ratio 0.37 --damp -0.75 --amp 3.0 --reps 0 \
    --batch-size 48 --score-batch 8 --tag smoke || { echo "SMOKE FAILED, stopping"; exit 1; }

echo "########## placebo contrast: Qwen2.5-1.5B-Instruct"
python run_placebo_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --rho 0.04 --ratio 0.37 --damp -0.75 --amp 3.0 \
    --batch-size 48 --score-batch 8 --tag q15

echo "########## placebo contrast: Llama-3.1-8B-Instruct"
python run_placebo_contrast.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --rho 0.04 --ratio 0.37 --damp -0.75 --amp 3.0 \
    --batch-size 32 --score-batch 4 --tag llama8b
