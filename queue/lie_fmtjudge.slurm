#!/bin/bash
#SBATCH --job-name=lie_fmtjudge
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# One long-form judge pass over every fmt_*.jsonl in the runs dir (cached replies are skipped), after a batch.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FMT_OUT=/work/hdd/biro/xiaocong/lie_out/fmt_suite
cd /u/xiaocong/lie_detection
echo "$(date '+%F %T') $SLURM_JOB_ID judge on $(hostname -s)"
python fmt_judge.py --runs ${RUNS:-/work/hdd/biro/xiaocong/lie_out/fmt_runs} --cache-dir $HF_HOME
echo "$(date '+%F %T') end judge rc=$?"
