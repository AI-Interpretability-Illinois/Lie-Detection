#!/bin/bash
#SBATCH --job-name=lie_strat
#SBATCH --account=bfqt-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=05:00:00
#SBATCH --output=slurm_logs/lie_strat_%j.out
#SBATCH --error=slurm_logs/lie_strat_%j.out
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
set -u
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_CACHE_DIR=/work/hdd/bfqt/xiaocong/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
cd /u/xiaocong/lie_detection
for W in ${WPCTS}; do
  echo "########## stratified DID2: ${MODEL_ID} wpct=$W"
  python run_stratified_did2.py --model-id "${MODEL_ID}" --rho "${RHO}" --ratio "${RATIO}" \
      --wpct "$W" --batch-size "${BS}" --score-batch "${SB}"
done
