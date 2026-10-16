#!/bin/bash
# usage: GPU=1 run_bsamp.sh <model_id> <family pooled|fiction> <strength 100|025|...> [batch]
# Sample-size test (b), behaviour: masks from the hand-written 25 prompts vs pool draws, LIARS' BENCH validation half.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_MAX_GPU_GB=${LIE_MAX_GPU_GB:-10}
cd /u/xiaocong/lie_detection
m=$1; fam=$2; st=$3; bs=${4:-8}
p=$([ "$fam" = pooled ] && echo pp || echo pf)
draws="$fam:25:0,$fam:25:1,$fam:25:2,$fam:100:0,$fam:400:0"
arms="unsteered,$fam$st,${p}25d0_$st,${p}25d1_$st,${p}25d2_$st,${p}100d0_$st,${p}400d0_$st"
echo "$(date '+%F %T') start $m $arms on gpu$GPU"
CUDA_VISIBLE_DEVICES=$GPU nice -n 10 python run_liars_bench_honesty.py --model-id "$m" --rho 0.04 --ratio 0.37 \
  --falsehood-arms --arms "$arms" --pool-draws "$draws" --half val \
  --splits instructed-deception,instructed-truthful --n-id 537 --batch-size $bs --score-batch 2 --tag bsamp
echo "$(date '+%F %T') end $m rc=$?"
