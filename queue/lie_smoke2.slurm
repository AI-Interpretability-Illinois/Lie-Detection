#!/bin/bash
#SBATCH --job-name=lie_smoke2
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# Smoke for the analysis batch (Phi-4-mini, tiny sizes): POOL / LOO sources, directional ablation, mask
# anatomy, and run_liars_bench_honesty.py --fmt-masks. Exits non-zero on the first failure, so dependent
# jobs (afterok) never start.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -e -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/hdd/biro/xiaocong/lie_out; export LIE_OUT=$O FMT_OUT=$O/fmt_suite
SM=$O/fmt_smoke2; rm -rf $SM; mkdir -p $SM
cd /u/xiaocong/lie_detection
M=microsoft/Phi-4-mini-instruct
echo "$(date '+%F %T') 1 fmt_transfer: POOL, LOO, dirab"
python fmt_transfer.py --model-id $M --cache-dir $HF_HOME --out $SM --layer 16 --n-build 16 --n-eval 6 --batch 16 \
  --arms unsteered,maskc:POOL:0.005,caa:LOO-F6_multi:-0.27,dirab:F2_user:1.0,dirab:POOL:2.0 --suffix _x
echo "$(date '+%F %T') 2 fmt_transfer: draw seed"
python fmt_transfer.py --model-id $M --cache-dir $HF_HOME --out $SM --layer 16 --n-build 16 --n-eval 6 --batch 16 \
  --arms maskc:F2_user:0.005,caa:F2_user:-0.27 --draw-seed 1
echo "$(date '+%F %T') 3 fmt_judge"
python fmt_judge.py --runs $SM --cache-dir $HF_HOME
python reports/fmt_summarize.py $SM --model Phi-4-mini-instruct --tags ,_x | head -5
echo "$(date '+%F %T') 4 fmt_anatomy"
python fmt_anatomy.py --model-id $M --cache-dir $HF_HOME --out $SM --n-build 16
echo "$(date '+%F %T') 5 runner --fmt-masks"
python run_liars_bench_honesty.py --model-id $M --out-dir $O --rho 0.04 --ratio 0.37 --arms fmuser050_100,fmpool025_100 \
  --fmt-masks F2_user:0.005,POOL:0.0025 --fmt-n-build 16 --batch-size 8 --score-batch 2 \
  --splits instructed-deception,instructed-truthful --n-id 12 --tag v2smoke
echo "$(date '+%F %T') smoke ok"
