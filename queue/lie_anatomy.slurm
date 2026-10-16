#!/bin/bash
#SBATCH --job-name=lie_anat2
#SBATCH --time=02:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%j.out
# Qwen diagnosis (extras plan item 4): qwen_anatomy.py on the seven main-table models.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /u/xiaocong/lie_detection
# MODELS (env, space-separated) overrides the default seven; the five small ones ran on timan108 GPU2 on Oct 4
for m in ${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct Qwen/Qwen2.5-7B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct microsoft/Phi-4-mini-instruct tiiuae/Falcon3-3B-Instruct allenai/OLMo-2-0425-1B-Instruct}; do
  echo "$(date '+%F %T') start $m"; python qwen_anatomy.py --model-id $m --cache-dir $HF_HOME 2>&1 | grep -vE "Warning|warn"; echo "$(date '+%F %T') end $m rc=$?"
done
