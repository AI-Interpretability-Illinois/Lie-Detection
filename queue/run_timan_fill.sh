#!/bin/bash
# timan1 version of queue/lie_fill.slurm (2026-10-06, coordinator xiaocong-76: share timan1 GPU2 with dagformer's 75M run,
# <=30 GB GPU memory, hard stop before 03:00 Oct 7). Milder options for the empty Table 1 cells: OLMo-2-1B steering
# r=-0.05/-0.1 at layers 4/8/12; Falcon3-3B lie mask alpha=-0.10/-0.15 and honest prompts 4/5. LIARS' BENCH + AI Liar +
# the Qwen2.5-7B judge, tag swfill(+ail). Code: $T/fill_repo (a copy of the Delta repo); outputs: $T/fill_out, rsynced
# afterwards to Delta's /work/nvme/biro/xiaocong/lie_sweep. Every step runs under `timeout` up to DEADLINE.
T=/srv/local/xy51/lie; F=$T/fill_out; PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin/python}
DEADLINE=${DEADLINE:-"2026-10-07 02:30"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 LIE_MAX_GPU_GB=30
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=/srv/local/xy51/hf_models/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$F LIE_CONTROLS=$T/controls
cd $T/fill_repo
echo "$(date '+%F %T') fill on $(hostname -s) GPU $CUDA_VISIBLE_DEVICES pid $$, deadline $DEADLINE"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  local left=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) )); [ $left -le 60 ] && { echo "$(date '+%F %T') deadline: skip $n"; return; }
  echo "$(date '+%F %T') start $n (${left}s left)"; timeout $left nice -n 10 "$@" > $T/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
SPECS="allenai/OLMo-2-0425-1B-Instruct|asrL4_-0.05,asrL4_-0.1,asrL8_-0.05,asrL8_-0.1,asrL12_-0.05,asrL12_-0.1|4,8,12
tiiuae/Falcon3-3B-Instruct|ours010,ours015,prompt4,prompt5|5"
TG=swfill
while IFS='|' read -r MODEL ARMS LAYERS; do
  S=$(basename $MODEL | tr . p)
  R="$PY run_liars_bench_honesty.py --model-id $MODEL --cache-dir $HF_HUB_CACHE --out-dir $F --rho 0.04 --ratio 0.37 --arms $ARMS --as-layers $LAYERS --as-rel-coeffs=-0.05,-0.1 --batch-size 8 --score-batch 2"
  step ${TG}_lb_${S} $F/liars_bench_honesty_${S}_${TG}.csv $R --eval-set $F/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $TG
  step ${TG}_ail_${S} $F/liars_bench_honesty_${S}_${TG}ail.csv $R --eval-set $F/extra_data/extra_ailiar.jsonl --belief-tag ailiar --tag ${TG}ail
  step ${TG}_judge_${S} $F/liars_bench_honesty_${S}_${TG}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses liars_bench_honesty_${S}_${TG}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $F/liars_bench/eval_set.jsonl
done <<< "$SPECS"
echo "$(date '+%F %T') all done"
