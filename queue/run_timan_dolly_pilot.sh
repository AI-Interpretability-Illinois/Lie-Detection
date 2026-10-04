#!/bin/bash
# Dolly-v2 pilot (extras plan): does unedited dolly-v2-3b lie on >= 30% of LIARS' BENCH validation items? Runs the
# unedited arm on the validation half and the Qwen2.5-7B judge; co-tenant on timan1 GPU2 beside the CN run.
T=/srv/local/xy51/lie; PY=/home/xy51/anaconda3/envs/modularity/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-2} LIE_MAX_GPU_GB=${CAP:-12}
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
M=${M:-databricks/dolly-v2-3b}; S=$(basename $M | tr . p)
echo "$(date '+%F %T') pilot $M"
nice -n 10 $PY run_liars_bench_honesty.py --model-id $M --cache-dir $HF_HOME --out-dir $T/out --rho 0.04 --ratio 0.37 --arms unsteered \
  --eval-set $T/out/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --half val --batch-size 8 --score-batch 2 --tag pilot > $T/logs/pilot_$S.log 2>&1
echo "$(date '+%F %T') generation rc=$?"
LIE_MAX_GPU_GB=20 nice -n 10 $PY judge_claims.py --responses liars_bench_honesty_${S}_pilot_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $T/out/liars_bench/eval_set.jsonl > $T/logs/pilot_judge_$S.log 2>&1
echo "$(date '+%F %T') judge rc=$?"; grep -A3 "HYBRID" $T/logs/pilot_judge_$S.log | head -5
