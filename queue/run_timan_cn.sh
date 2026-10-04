#!/bin/bash
# CreativityNeuro-style baselines (extras plan, Oct 4) on timan1 GPU2 (grant #37): set-difference masks on
# pooled-token (rms) norms with our six prompt families and four strengths ("cn" arms, the faithful baseline),
# plus the 2x2 decomposition on the model's chosen families: sd (set difference, per-prompt-mean norms) and rs
# (our graded contrast, rms norms). Arms run in groups of 8 so at most 8 boolean masks live in host memory.
# Each group: LIARS' BENCH (all items) + AI Liar + the Qwen2.5-7B judge on LIARS' BENCH; tags cn<group>.
# usage: GPU=2 CAP=24 bash run_timan_cn.sh "<model_id> <fam1> <fam2>" ...   (fam1/fam2: chosen LB / AIL families)
T=/srv/local/xy51/lie
PY=/home/xy51/anaconda3/envs/modularity/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-2} LIE_MAX_GPU_GB=${CAP:-24}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
X=$T/out/extra_data
free_mib() { nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }
step() { local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date '+%F %T') skip $name"; return; fi
  until [ "$(free_mib)" -ge $(( (LIE_MAX_GPU_GB + 3) * 1024 )) ]; do echo "$(date '+%F %T') wait $name: $(free_mib) MiB free"; sleep 120; done
  echo "$(date '+%F %T') start $name ($(free_mib) MiB free)"
  nice -n 10 "$@" > $T/logs/$name.log 2>&1 & local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(free_mib)" -lt 1500 ]; then pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null; echo "$(date '+%F %T') backed off $name"; sleep 60; fi
    sleep 10; done
  wait $pid; echo "$(date '+%F %T') end $name rc=$?"; }
ST="025 050 075 100"
for pass in 1 2; do
for spec in "$@"; do
  set -- $spec; m=$1; f1=$2; f2=$3; s=$(basename $m | tr . p)
  groups=("A:cnours,cnpooled" "B:cnfiction,cnfictf" "C:cnpooltf,cnpoolmix" "D:sd$f1,rs$f1")
  [ "$f2" != "$f1" ] && groups+=("E:sd$f2,rs$f2")
  for g in "${groups[@]}"; do
    tag=cn${g%%:*}; fams=${g#*:}; arms=""
    for fam in ${fams//,/ }; do for st in $ST; do arms="$arms,$fam$st"; done; done
    arms=${arms#,}
    R="$PY run_liars_bench_honesty.py --model-id $m --cache-dir $T/hf --out-dir $T/out --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $arms --batch-size 8 --score-batch 2"
    step ${tag}_lb_$s $T/out/liars_bench_honesty_${s}_${tag}.csv $R --eval-set $T/out/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $tag
    step ${tag}_ail_$s $T/out/liars_bench_honesty_${s}_${tag}ail.csv $R --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --tag ${tag}ail
    step ${tag}_judge_$s $T/out/liars_bench_honesty_${s}_${tag}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
      $PY judge_claims.py --responses liars_bench_honesty_${s}_${tag}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $T/out/liars_bench/eval_set.jsonl
  done
done
done
echo "$(date '+%F %T') all done"
