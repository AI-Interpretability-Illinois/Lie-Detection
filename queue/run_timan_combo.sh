#!/bin/bash
# Combined-mask test on timan1 (owner OK 2026-10-01 18:40 CDT): the chosen prompt mask OR a small recovery
# mask (mask_designs.py recov, calibrated r at 0.25% / 0.5% positive share, rho 4%), full deletion,
# against unedited, the prompt mask alone and the recovery mask alone. Validation halves of LIARS' BENCH,
# AI Liar and multi-turn, settings as queue/lie_depth.slurm; then the Qwen2.5-7B judge on LIARS' BENCH.
# usage: GPU=<id> CAP=<GB> bash run_timan_combo.sh "<model_id> <family> <strength>" ["..."]
# Restart-safe: a step whose output exists is skipped.
T=/srv/local/xy51/lie
PY=/home/xy51/anaconda3/envs/modularity/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1} LIE_MAX_GPU_GB=${CAP:-20}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
X=$T/out/extra_data
free_mib() { nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }
step() {  # step <name> <output that marks it done> <command...>
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
  wait $pid; local rc=$?; echo "$(date '+%F %T') end $name rc=$rc"
}
DES=recov:0.04:a0.0025,recov:0.04:a0.005
SPECS=("$@")
for pass in 1 2; do
for spec in "${SPECS[@]}"; do
  set -- $spec; m=$1; fam=$2; st=$3; s=$(basename $m | tr . p)
  ARMS=unsteered,$fam$st,recov040a025_100,$fam+recov040a025_100,$fam+recov040a050_100
  R="$PY run_liars_bench_honesty.py --model-id $m --cache-dir $T/hf --out-dir $T/out --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $ARMS --designs $DES --combine $fam+recov040a025,$fam+recov040a050 --half val --batch-size 8 --score-batch 2"
  step combo_lb_$s $T/out/liars_bench_honesty_${s}_combo.csv $R --eval-set $T/out/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag combo
  step combo_ail_$s $T/out/liars_bench_honesty_${s}_comboail.csv $R --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --tag comboail
  step combo_mt_$s $T/out/liars_bench_honesty_${s}_combomt.csv $R --eval-set $X/extra_multiturn.jsonl --belief-tag multiturn --tag combomt
  step combo_judge_$s $T/out/liars_bench_honesty_${s}_combo_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses liars_bench_honesty_${s}_combo_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $T/out/liars_bench/eval_set.jsonl
done
done
echo "$(date '+%F %T') all done"
