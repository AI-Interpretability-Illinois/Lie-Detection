#!/bin/bash
# Test (b) on timan1 GPU1 (owner OK 2026-09-30): judge the small-model runs, then Llama-3.1-8B and
# Qwen2.5-7B stability + behaviour + judge. Restart-safe: a step whose output exists is skipped.
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
bsamp_args() {  # bsamp_args <model_id> <family> <strength>: runner arguments (no spaces inside any)
  local p=$([ "$2" = pooled ] && echo pp || echo pf)
  echo run_liars_bench_honesty.py --model-id $1 --cache-dir $T/hf --out-dir $T/out \
    --eval-set $T/out/liars_bench/eval_set.jsonl --rho 0.04 --ratio 0.37 --falsehood-arms \
    --arms unsteered,$2$3,${p}25d0_$3,${p}25d1_$3,${p}25d2_$3,${p}100d0_$3,${p}400d0_$3 \
    --pool-draws $2:25:0,$2:25:1,$2:25:2,$2:100:0,$2:400:0 --half val \
    --splits instructed-deception,instructed-truthful --n-id 537 --batch-size 8 --score-batch 2 --tag bsamp
}
for pass in 1 2 3; do
for s in Phi-4-mini-instruct OLMo-2-0425-1B-Instruct; do
  step judge_$s $T/out/liars_bench_honesty_${s}_bsamp_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses liars_bench_honesty_${s}_bsamp_responses.jsonl $J
done
for spec in "meta-llama/Llama-3.1-8B-Instruct fiction 100" "Qwen/Qwen2.5-7B-Instruct pooled 100"; do
  set -- $spec; m=$1; s=$(basename $m | tr . p)
  step mss_$s $T/controls/mask_sample_size_$s.json $PY mask_sample_size.py --model-id $m --cache-dir $T/hf --batch 8
  step bsamp_$s $T/out/liars_bench_honesty_${s}_bsamp.csv $PY $(bsamp_args $m $2 $3)
  step judge_$s $T/out/liars_bench_honesty_${s}_bsamp_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses liars_bench_honesty_${s}_bsamp_responses.jsonl $J
done
done
echo "$(date '+%F %T') all done"
