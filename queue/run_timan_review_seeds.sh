#!/bin/bash
# timan1 lane for the review's seed replication at Table 1's final options (queue/lie_review.slurm EXP=seeds), grant #69
# (GPU1/GPU2 until 20:30 CDT 2026-10-12). Waits for the lane's MMLU pass (pid WAITPID) first. Usage:
#   WAITPID=<pid> run_timan_review_seeds.sh <gpu> <model key>...
T=/srv/local/xy51/lie; O=$T/review_out; PY=/home/xy51/anaconda3/envs/modularity/bin/python
DEADLINE=${DEADLINE:-"2026-10-12 20:25"}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$1; shift
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache HF_HOME=/srv/local/xy51/hf_models
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$O
cd $T/review_repo
[ -n "$WAITPID" ] && while kill -0 $WAITPID 2>/dev/null; do sleep 30; done
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  local left=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) )); [ $left -le 60 ] && { echo "$(date '+%F %T') deadline: skip $n"; return; }
  echo "$(date '+%F %T') start $n"; timeout $left nice -n 10 "$@" > $O/logs/${n}.log 2>&1; echo "$(date '+%F %T') end $n rc=$?"; }
for KEY in "$@"; do
  case $KEY in
    phi)    MODEL=microsoft/Phi-4-mini-instruct; L=8,16,24; C=$T/hf
            CLB=pooled112,ours050,asrL16_-0.18,prompt2; CAIL=fiction100,ours050,asrL16_-0.18,prompt2 ;;
    falcon) MODEL=tiiuae/Falcon3-3B-Instruct; L=5,11,16; C=/srv/local/xy51/hf_models/hub
            CLB=poolmix125,ours015,asrL5_-0.3,prompt5; CAIL=poolmix112,ours015,asrL5_-0.18,prompt5 ;;
    olmo)   MODEL=allenai/OLMo-2-0425-1B-Instruct; L=4,8,12; C=/srv/local/xy51/hf_models/hub
            CLB=pooled125,ours125,asrL12_-0.1,prompt; CAIL=pooled125,ours125,asrL12_-0.1,prompt ;;
    qwen3b) MODEL=Qwen/Qwen2.5-3B-Instruct; L=9,18,27; C=/srv/local/xy51/hf_models/hub
            CLB=pooltf112,ours100,asrL27_-0.18,prompt; CAIL=poolmix112,ours100,asrL9_-0.5,prompt ;;
  esac
  S=$(basename $MODEL | tr . p)
  R="$PY run_liars_bench_honesty.py --model-id $MODEL --cache-dir $C --out-dir $O --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --as-layers $L --as-rel-coeffs=-0.05,-0.1,-0.18,-0.3,-0.5,-0.75,-1 --batch-size 16 --score-batch 2"
  for sd in 1 2; do T2=rvseeds$sd
    step ${T2}_lb_${S} $O/liars_bench_honesty_${S}_${T2}.csv $R --eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --seed $sd --arms unsteered,$CLB --tag $T2
    step ${T2}_judge_${S} $O/liars_bench_honesty_${S}_${T2}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
      $PY judge_claims.py --responses liars_bench_honesty_${S}_${T2}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $O/liars_bench/eval_set.jsonl
    step ${T2}_ail_${S} $O/liars_bench_honesty_${S}_${T2}ail.csv $R --eval-set $O/extra_data/extra_ailiar.jsonl --belief-tag ailiar --seed $sd --arms unsteered,$CAIL --tag ${T2}ail
  done
done
echo "$(date '+%F %T') seeds lane done"
