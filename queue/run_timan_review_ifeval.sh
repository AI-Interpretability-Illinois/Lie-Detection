#!/bin/bash
# timan1 lane for the review's IFEval check (queue/lie_review.slurm EXP=ifeval) at Table 1's LIARS' BENCH-chosen options,
# grant #69 (GPU1/GPU2 until 20:30 CDT 2026-10-12). Waits for the lane's seed runs (pid WAITPID). Prompts from
# review_out/ifeval_541.jsonl (exported from Delta's HF cache); langdetect/immutabledict from /srv/local/xy51/pylib.
T=/srv/local/xy51/lie; O=$T/review_out; PY=/home/xy51/anaconda3/envs/modularity/bin/python
DEADLINE=${DEADLINE:-"2026-10-12 20:28"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1; shift
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache HF_HOME=/srv/local/xy51/hf_models PYTHONPATH=/srv/local/xy51/pylib
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$O IFEVAL_JSONL=$O/ifeval_541.jsonl
cd $T/review_repo
[ -n "$WAITPID" ] && while kill -0 $WAITPID 2>/dev/null; do sleep 20; done
for KEY in "$@"; do
  case $KEY in
    phi)    MODEL=microsoft/Phi-4-mini-instruct; L=8,16,24; C=$T/hf; CLB=pooled112,ours050,asrL16_-0.18,prompt2 ;;
    falcon) MODEL=tiiuae/Falcon3-3B-Instruct; L=5,11,16; C=/srv/local/xy51/hf_models/hub; CLB=poolmix125,ours015,asrL5_-0.3,prompt5 ;;
    olmo)   MODEL=allenai/OLMo-2-0425-1B-Instruct; L=4,8,12; C=/srv/local/xy51/hf_models/hub; CLB=pooled125,ours125,asrL12_-0.1,prompt ;;
    qwen3b) MODEL=Qwen/Qwen2.5-3B-Instruct; L=9,18,27; C=/srv/local/xy51/hf_models/hub; CLB=pooltf112,ours100,asrL27_-0.18,prompt ;;
  esac
  S=$(basename $MODEL | tr . p); TG=rvifeval
  [ -e $O/liars_bench_honesty_${S}_${TG}.csv ] && grep -q prompt $O/liars_bench_honesty_${S}_${TG}.csv && { echo "skip $S"; continue; }
  left=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) )); [ $left -le 60 ] && { echo "deadline: skip $S"; continue; }
  echo "$(date '+%F %T') start $S on GPU $CUDA_VISIBLE_DEVICES (${left}s left)"
  timeout $left nice -n 10 $PY run_liars_bench_honesty.py --model-id $MODEL --cache-dir $C --out-dir $O --rho 0.04 --ratio 0.37 \
    --falsehood-arms --polarity-arms --as-layers $L --as-rel-coeffs=-0.05,-0.1,-0.18,-0.3,-0.5,-0.75,-1 --batch-size 32 --score-batch 2 \
    --eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 \
    --no-gen --ifeval 541 --arms unsteered,$CLB --tag $TG > $O/logs/${TG}_${S}.log 2>&1
  echo "$(date '+%F %T') end $S rc=$?"
done
