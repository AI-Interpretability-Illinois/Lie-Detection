#!/bin/bash
# timan1 lane for the review's MMLU-per-option pass (queue/lie_review.slurm EXP=mmlu), grant #69: GPU1 and GPU2 until
# 20:30 CDT 2026-10-12. Usage: run_timan_review_mmlu.sh <gpu> <model key>... Same model revisions as Delta (checked),
# Delta's steering vectors. Outputs: /srv/local/xy51/lie/review_out, rsynced to Delta's /work/hdd/bfqt/xiaocong/lie_review.
T=/srv/local/xy51/lie; O=$T/review_out; PY=/home/xy51/anaconda3/envs/modularity/bin/python
DEADLINE=${DEADLINE:-"2026-10-12 20:25"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1; shift
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache HF_HOME=/srv/local/xy51/hf_models
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$O
cd $T/review_repo
for KEY in "$@"; do
  case $KEY in
    phi)    MODEL=microsoft/Phi-4-mini-instruct; L=8,16,24; C=$T/hf ;;
    falcon) MODEL=tiiuae/Falcon3-3B-Instruct; L=5,11,16; C=/srv/local/xy51/hf_models/hub ;;
    olmo)   MODEL=allenai/OLMo-2-0425-1B-Instruct; L=4,8,12; C=/srv/local/xy51/hf_models/hub ;;
    qwen3b) MODEL=Qwen/Qwen2.5-3B-Instruct; L=9,18,27; C=/srv/local/xy51/hf_models/hub ;;
  esac
  S=$(basename $MODEL | tr . p); TG=rvmmlu
  A="unsteered"
  for f in ours fiction pooled fictf pooltf poolmix; do for a in 025 050 075 100 112 125; do A="$A,$f$a"; done; done
  for l in ${L//,/ }; do for r in -0.18 -0.3 -0.5 -0.75 -1; do A="$A,asrL${l}_$r"; done; done
  A="$A,prompt,prompt2,prompt3"
  [ $KEY = falcon ] && A="$A,ours010,ours015,prompt4,prompt5"
  [ $KEY = olmo ] && for l in ${L//,/ }; do A="$A,asrL${l}_-0.05,asrL${l}_-0.1"; done
  [ -e $O/liars_bench_honesty_${S}_${TG}.csv ] && { echo "skip $S"; continue; }
  left=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) )); [ $left -le 60 ] && { echo "deadline: skip $S"; continue; }
  echo "$(date '+%F %T') start $S on GPU $CUDA_VISIBLE_DEVICES (${left}s left)"
  timeout $left nice -n 10 $PY run_liars_bench_honesty.py --model-id $MODEL --cache-dir $C --out-dir $O --rho 0.04 --ratio 0.37 \
    --falsehood-arms --polarity-arms --as-layers $L --as-rel-coeffs=-0.05,-0.1,-0.18,-0.3,-0.5,-0.75,-1 --batch-size 16 --score-batch 2 \
    --eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 \
    --no-gen --mmlu 2000 --arms $A --tag $TG > $O/logs/${TG}_${S}.log 2>&1
  echo "$(date '+%F %T') end $S rc=$?"
done
