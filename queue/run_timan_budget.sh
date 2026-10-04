#!/bin/bash
# Mask-budget sweep over the contrast threshold r on a timan host (same as queue/lie_budget.slurm on Delta): the model's
# family at four strengths for R in {0.2, 0.6, 1.0} (0.37 is the main table), LIARS' BENCH + the Qwen2.5-7B judge, tags
# rb020 / rb060 / rb100 in $T/out. usage: GPU=1 CAP=18 PY=<python> bash queue/run_timan_budget.sh <model_id> <family> [bs]
T=/srv/local/xy51/lie; PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin/python}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
[ "${CAP:-0}" != 0 ] && export LIE_MAX_GPU_GB=$CAP
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=/srv/local/xy51/hf_models/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
M=$1; FAM=$2; BS=${3:-8}; S=$(basename $M | tr . p)
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; nice -n 10 "$@" > $T/logs/${n}_${S}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
arms="${FAM}025,${FAM}050,${FAM}075,${FAM}100"
for r in 0.2 0.6 1.0; do
  tag=rb$(printf "%03d" $($PY -c "print(round($r*100))"))
  step ${tag}_lb $T/out/liars_bench_honesty_${S}_${tag}.csv $PY run_liars_bench_honesty.py --model-id $M --cache-dir $HF_HUB_CACHE --out-dir $T/out --rho 0.04 --ratio $r \
    --falsehood-arms --polarity-arms --arms $arms --batch-size $BS --score-batch 2 \
    --eval-set $T/out/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $tag
  step ${tag}_judge $T/out/liars_bench_honesty_${S}_${tag}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses liars_bench_honesty_${S}_${tag}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $T/out/liars_bench/eval_set.jsonl
done
echo "$(date '+%F %T') budget sweep done $M"
