#!/bin/bash
#SBATCH --job-name=lie_anat
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# J2: mask anatomy and overlap (fmt_anatomy.py) for the four format-study models, one after another.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FMT_OUT=/work/hdd/biro/xiaocong/lie_out/fmt_suite
cd /u/xiaocong/lie_detection
for m in microsoft/Phi-4-mini-instruct tiiuae/Falcon3-3B-Instruct Qwen/Qwen2.5-7B-Instruct meta-llama/Meta-Llama-3.1-8B-Instruct; do
  echo "$(date '+%F %T') start $m"
  python fmt_anatomy.py --model-id $m --cache-dir $HF_HOME >> /work/hdd/biro/xiaocong/lie_out/logs/anat_$(basename $m).log 2>&1
  echo "$(date '+%F %T') end $m rc=$?"
done
