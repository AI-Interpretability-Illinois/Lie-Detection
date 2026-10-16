#!/bin/bash
# 100-prompt masks on timan1 GPU1 (owner request 2026-09-30): Llama-3.1-8B and Qwen2.5-7B at their chosen
# LIARS' BENCH and AI Liar settings, test halves, then the Qwen2.5-7B judge on every LB p100 file present.
# Restart-safe: a step whose output exists is skipped.
T=/srv/local/xy51/lie
PY=/home/xy51/anaconda3/envs/modularity/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1} LIE_MAX_GPU_GB=${CAP:-20}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
J="--judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $T/out/liars_bench/eval_set.jsonl"
free_mib() { nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }
step() {  # step <name> <output that marks it done> <command...>
  # GPU1 is shared (DAGFormer SFT ~12 GB, another user up to ~27 GB): start only with CAP+3 GB
  # free, and kill our job if the card drops below 1.5 GB free; a later pass retries it.
  local name=$1 done=$2; shift 2
  if [ -e "$done" ]; then echo "$(date '+%F %T') skip $name"; return; fi
  until [ "$(free_mib)" -ge $(( (LIE_MAX_GPU_GB + 3) * 1024 )) ]; do
    echo "$(date '+%F %T') wait $name: $(free_mib) MiB free"; sleep 120; done
  echo "$(date '+%F %T') start $name ($(free_mib) MiB free)"
  nice -n 10 "$@" > $T/logs/$name.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(free_mib)" -lt 1500 ]; then
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null
      echo "$(date '+%F %T') backed off $name: card below 1.5 GB free"; sleep 60
    fi
    sleep 10
  done
  wait $pid; echo "$(date '+%F %T') end $name rc=$?"
}
X=$T/out/extra_data
p100_args() {  # p100_args <model_id> <family> <strength> <lb|ail>
  declare -A P=([fiction]=pf [pooled]=pp [fictf]=pft [pooltf]=ppt [poolmix]=ppm)
  local common="run_liars_bench_honesty.py --model-id $1 --cache-dir $T/hf --out-dir $T/out --rho 0.04 --ratio 0.37 --arms ${P[$2]}100d0_$3 --pool-draws $2:100:0 --half test --batch-size 8 --score-batch 2"
  if [ "$4" = lb ]; then
    echo $common --eval-set $T/out/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag p100
  else
    echo $common --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --tag p100ail
  fi
}
for pass in 1 2 3; do
for spec in "meta-llama/Llama-3.1-8B-Instruct fiction 100 pooled 100" "Qwen/Qwen2.5-7B-Instruct pooled 100 pooled 075"; do
  set -- $spec; m=$1; s=$(basename $m | tr . p)
  step p100_$s $T/out/liars_bench_honesty_${s}_p100.csv $PY $(p100_args $m $2 $3 lb)
  step p100ail_$s $T/out/liars_bench_honesty_${s}_p100ail.csv $PY $(p100_args $m $4 $5 ail)
done
for r in $T/out/liars_bench_honesty_*_p100_responses.jsonl; do
  [ -e "$r" ] || continue
  s=$(basename $r _p100_responses.jsonl); s=${s#liars_bench_honesty_}
  step judge_p100_$s $T/out/liars_bench_honesty_${s}_p100_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses $(basename $r) $J
done
done
echo "$(date '+%F %T') all done"
