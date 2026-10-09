#!/bin/bash
# Pythia size ablation on timan108 (owner request 2026-10-08 21:25; docs/pythia_size_ablation.md). One lane:
#   bash run_pythia_lane.sh <cards> <size> [<size> ...]     cards = 0, 1 or 0+1 (two cards, model sharded)
# Per size: sweep_pythia.sh (doc arms + the six steering arms, all LIARS BENCH splits), then the Qwen2.5-7B judge (v2)
# on the same card(s). Coordinator grant #58: timan108 GPU0+GPU1 until 2026-10-09 14:00.
T=/srv/local/xy51/lie; R=$T/pythia_repo; O=$T/pythia_out; PY=/srv/local/xy51/envs/modularity310/bin/python
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PY CACHE=/srv/local/xy51/hf_models/hub OUTDIR=$O PYROOT=/srv/local/xy51/pythia_final LOGDIR=$T/logs_pythia
export STEER_ARMS=1 TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cards=$1; shift; cd $R
for s in "$@"; do
  tag=pythia$(echo $s | tr . p); short=pythia-$(echo $s | tr . p)-deduped-step143000
  if [ -e $O/liars_bench_honesty_${short}_${tag}.csv ]; then echo "$(date "+%F %T") skip run $s"
  else echo "$(date "+%F %T") run $s on $cards"; GPUS=$cards SIZES=$s bash sweep_pythia.sh; echo "$(date "+%F %T") run $s done"; fi
  [ -e $O/liars_bench_honesty_${short}_${tag}_responses.jsonl ] || { echo "$(date "+%F %T") NO RESPONSES for $s"; continue; }
  if ls $O/liars_bench_honesty_${short}_${tag}_judged_Qwen2p5-7B-Instruct_v2.* >/dev/null 2>&1; then echo "$(date "+%F %T") skip judge $s"; continue; fi
  echo "$(date "+%F %T") judge $s"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${cards%%+*} LIE_OUT=$O HF_HOME=/srv/local/xy51/hf_models nice -n 10 \
    $PY judge_claims.py --responses liars_bench_honesty_${short}_${tag}_responses.jsonl --eval-set $O/liars_bench/eval_set.jsonl \
    --beliefs $O/liars_bench/beliefs_${short}_${tag}.jsonl --judge Qwen/Qwen2.5-7B-Instruct --cache-dir $T/hf --batch-size 8 --tag v2 \
    > $LOGDIR/judge_${tag}.log 2>&1
  echo "$(date "+%F %T") judge $s rc=$?"
done
echo "$(date "+%F %T") lane $cards done"
