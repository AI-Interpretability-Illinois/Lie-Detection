#!/bin/bash
#SBATCH --job-name=lie_fmt
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# Format-transfer study (fmt_transfer.py) on more models: one shard (SHARD=i/n) of every model in MODELS,
# validation halves, then the long-form judge once every shard has finished (fmt_judge.py).
# Env: SHARD (e.g. 0/2); MODELS (optional) = ";"-separated "<model_id> <caa layer> <caa rel strengths>";
# METHODS, FMT_ARGS (extra fmt_transfer.py options); BATCH (marker name, default delta); NOJUDGE=1 skips
# the judge (one judge job runs after the batch instead).
# Restart-safe: finished arms are skipped.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FMT_OUT=/work/hdd/biro/xiaocong/lie_out/fmt_suite
O=/work/hdd/biro/xiaocong/lie_out/fmt_runs; mkdir -p $O
cd /u/xiaocong/lie_detection
S=${SHARD:-0/1}; n=${S#*/}
MODELS=${MODELS:-"meta-llama/Meta-Llama-3.1-8B-Instruct 8 -0.25,-0.5,-0.75,-1.0;Qwen/Qwen2.5-7B-Instruct 14 -0.09,-0.18,-0.27,-0.36;tiiuae/Falcon3-3B-Instruct 5 -0.09,-0.18,-0.27,-0.36"}
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): shard $S"
IFS=';' read -ra SPECS <<< "$MODELS"
for spec in "${SPECS[@]}"; do
  set -- $spec
  s=$(basename $1 | tr . p)
  echo "$(date '+%F %T') start $s (layer $2, caa $3)"
  # maskc (calibrated r), not the fixed-r mask: built from 200 pairs that one is nearly inert (pilot)
  python fmt_transfer.py --model-id $1 --cache-dir $HF_HOME --out $O --shard $S --layer $2 --caa-rel=$3 \
    --methods ${METHODS:-unsteered,maskc,caa} $FMT_ARGS \
    >> /work/hdd/biro/xiaocong/lie_out/logs/fmt_${s}_${S/\//of}.log 2>&1
  echo "$(date '+%F %T') end $s rc=$?"
done
B=${BATCH:-delta}
touch $O/.done_${B}_${S/\//of}
if [ -z "$NOJUDGE" ] && [ "$(ls $O/.done_${B}_* 2>/dev/null | wc -l)" -ge "$n" ]; then
  echo "$(date '+%F %T') start judge"
  python fmt_judge.py --runs $O --cache-dir $HF_HOME >> /work/hdd/biro/xiaocong/lie_out/logs/fmt_judge_delta.log 2>&1
  echo "$(date '+%F %T') end judge rc=$?"
fi
echo "$(date '+%F %T') all done"
