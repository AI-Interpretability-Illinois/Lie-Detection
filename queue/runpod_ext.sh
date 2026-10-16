#!/bin/bash
# runpod1 copy of Delta's queue/lie_steerx.slurm (T=swsteer: steering r = -0.3/-0.75/-1 at each model's three layers) and
# queue/lie_alphax.slurm (T=swalpha: mask alpha = -1.125/-1.25 for every mask family and the lie-instruction mask), owner
# 2026-10-11. Same code, eval sets, beliefs and steering vectors as Delta (md5-checked); models at Delta's cached revisions.
# LB (deception + truthful control) + AI Liar + the Qwen2.5-7B judge. Usage: run_ext.sh <model key> <swsteer|swalpha> <gpu>
set -o pipefail
source /workspace/env.sh >/dev/null 2>&1
L=/workspace/lie; O=$L/out
export HF_HOME=$L/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$O CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$3
KEY=$1; T=$2
case $KEY in
  llama)  MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct; LAYERS=8,14,21 ;;
  phi)    MODEL=microsoft/Phi-4-mini-instruct; LAYERS=8,16,24 ;;
  falcon) MODEL=tiiuae/Falcon3-3B-Instruct; LAYERS=5,11,16 ;;
  olmo)   MODEL=allenai/OLMo-2-0425-1B-Instruct; LAYERS=4,8,12 ;;
  qwen3b) MODEL=Qwen/Qwen2.5-3B-Instruct; LAYERS=9,18,27 ;;
  *) echo "unknown model $KEY"; exit 1 ;;
esac
S=$(basename $MODEL | tr . p)
if [ "$T" = swsteer ]; then
  RS="-0.3,-0.75,-1"
  ARMS=$(for l in ${LAYERS//,/ }; do for r in ${RS//,/ }; do printf "asrL%s_%s," $l $r; done; done)
  EXTRA="--as-layers $LAYERS --as-rel-coeffs=$RS"
elif [ "$T" = swalpha ]; then
  ARMS=$(for f in ours fiction pooled fictf pooltf poolmix; do printf "%s112,%s125," $f $f; done)
  EXTRA="--falsehood-arms --polarity-arms"
else echo "unknown extension $T"; exit 1; fi
ARMS=${ARMS%,}
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $L/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
cd $L/repo
echo "$(date '+%F %T') $S $T on GPU $3: $ARMS"
R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir $HF_HOME --out-dir $O --rho 0.04 --ratio 0.37 --arms $ARMS $EXTRA --batch-size 16 --score-batch 2"
step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R --eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $T
step ${T}_ail_${S} $O/liars_bench_honesty_${S}_${T}ail.csv $R --eval-set $O/extra_data/extra_ailiar.jsonl --belief-tag ailiar --tag ${T}ail
step ${T}_judge_${S} $O/liars_bench_honesty_${S}_${T}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_${T}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME --eval-set $O/liars_bench/eval_set.jsonl
echo "$(date '+%F %T') $S $T all done"
