#!/bin/bash
# Corrected LIARS' BENCH (both instructed halves) for the remaining models of the
# instructed-falsehood set, after the gemma run on this GPU finishes.
cd /u/xiaocong/lie_detection
while pgrep -u "$USER" -f "run_liars_bench_honest[y].py --model-id google/gemma-3-4b-it" > /dev/null; do sleep 60; done
L='unsteered,ours100,pooled100,fiction100,magmatch100,prompt'
SP='--splits instructed-deception,instructed-truthful,alpaca --n-id 537'
go() {  # model, three layers
  local M=$1 a=$2 b=$3 c=$4
  echo "=== $(date '+%H:%M') $M"
  GPU=${GPU:-0} BS=24 MID=$M TAGX=idfix2 \
  ARMS="$L,asrL${a}_-0.18,asrL${a}_-0.5,asrL${b}_-0.18,asrL${b}_-0.5,asrL${c}_-0.18,asrL${c}_-0.5" \
  ASWEEP="--as-layers $a,$b,$c --as-rel-coeffs=-0.18,-0.5 $SP" \
  ./lb_pooled.sh > slurm_logs/lb_idfix2_$(basename $M | tr '.' 'p').log 2>&1
  echo "=== exit $?"
}
go tiiuae/Falcon3-3B-Instruct 5 11 16
go allenai/OLMo-2-0425-1B-Instruct 4 8 12
go Qwen/Qwen2.5-1.5B-Instruct 7 14 21
