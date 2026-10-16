#!/bin/bash
# usage: GPU=1 run_p100.sh <model_id> <lb_family> <lb_strength> <ail_family> <ail_strength> [batch] [model_id ...]
# 100-prompt masks (pool draw seed 0) at each model's chosen LIARS' BENCH and AI Liar setting, test halves.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_MAX_GPU_GB=${LIE_MAX_GPU_GB:-10}
cd /u/xiaocong/lie_detection
X=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
declare -A P=([fiction]=pf [pooled]=pp [fictf]=pft [pooltf]=ppt [poolmix]=ppm)
while [ $# -ge 5 ]; do
  m=$1; lf=$2; ls=$3; af=$4; as=$5; bs=${6:-8}; shift 6
  s=$(basename $m | tr . p)
  echo "$(date '+%F %T') start $m LB ${P[$lf]}100d0_$ls on gpu$GPU"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python run_liars_bench_honesty.py --model-id $m --rho 0.04 --ratio 0.37 \
    --arms ${P[$lf]}100d0_$ls --pool-draws $lf:100:0 --half test \
    --splits instructed-deception,instructed-truthful --n-id 537 --batch-size $bs --score-batch 2 --tag p100 \
    > queue/logs/p100_$s.log 2>&1
  echo "$(date '+%F %T') end LB rc=$?; start AIL ${P[$af]}100d0_$as"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python run_liars_bench_honesty.py --model-id $m --rho 0.04 --ratio 0.37 \
    --arms ${P[$af]}100d0_$as --pool-draws $af:100:0 --half test \
    --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --batch-size $bs --score-batch 2 --tag p100ail \
    > queue/logs/p100ail_$s.log 2>&1
  echo "$(date '+%F %T') end AIL rc=$?"
done
