#!/bin/bash
# Format-transfer pilot on timan1 (owner OK 2026-10-01 20:50): Phi-4-mini, fmt_transfer.py arms
# (unsteered, mask:<F>:<alpha>, caa:<F>:<r>) on the validation halves of the 7 formats, sharded over GPUs,
# then the long-form judge (fmt_judge.py) on the GPU that finishes last.
# usage: GPU=<id> SHARD=<i>/<n> bash run_timan_fmt.sh            (one shard; judge runs if all shards done)
#        GPU=<id> bash run_timan_fmt.sh smoke                     (8 facts per format, 3 arms, judge)
# Extra fmt_transfer.py options go in FMT_ARGS (e.g. FMT_ARGS="--half test --arms ...").
# M / LAYER: model id and CAA layer (default Phi-4-mini, 16). NOJUDGE=1: skip the judge (run it once later).
T=/srv/local/xy51/lie
PY=/home/xy51/anaconda3/envs/modularity/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1} LIE_MAX_GPU_GB=${CAP:-16}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True FMT_OUT=$T/out/fmt_suite
cd $T/lie_detection
M=${M:-microsoft/Phi-4-mini-instruct}; LAYER=${LAYER:-16}; MS=$(basename $M | tr . p)
free_mib() { nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=memory.free --format=csv,noheader,nounits | xargs; }
run() {  # run <name> <command...>: wait for memory, back off if the card fills up
  local name=$1; shift
  until [ "$(free_mib)" -ge $(( (LIE_MAX_GPU_GB + 3) * 1024 )) ]; do
    echo "$(date '+%F %T') wait $name: $(free_mib) MiB free"; sleep 120; done
  echo "$(date '+%F %T') start $name ($(free_mib) MiB free)"
  nice -n 10 "$@" >> $T/logs/$name.log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if [ "$(free_mib)" -lt 1500 ]; then
      pkill -TERM -P $pid 2>/dev/null; kill -TERM $pid 2>/dev/null
      echo "$(date '+%F %T') backed off $name: card below 1.5 GB free"; sleep 60
    fi
    sleep 10
  done
  wait $pid; local rc=$?; echo "$(date '+%F %T') end $name rc=$rc"; return $rc
}
if [ "$1" = smoke ]; then
  rm -rf $T/out/fmt_smoke
  run fmt_smoke $PY fmt_transfer.py --model-id $M --cache-dir $T/hf --out $T/out/fmt_smoke \
      --arms unsteered,mask:F2_user:-1.0,caa:F6_multi:-0.18 --n-build 16 --n-eval 8 --batch 16 &&
  LIE_MAX_GPU_GB=${JCAP:-24} run fmt_smoke_judge $PY fmt_judge.py --runs $T/out/fmt_smoke --cache-dir $T/hf --batch-size 8
  echo "$(date '+%F %T') smoke done"; exit
fi
S=${SHARD:-0/1}; n=${S#*/}; H=val; [[ "$FMT_ARGS" == *"--half test"* ]] && H=test
for pass in 1 2; do
  run fmt_${H}_${MS}_${S/\//of} $PY fmt_transfer.py --model-id $M --cache-dir $T/hf --out $T/out/fmt_runs --shard $S --layer $LAYER $FMT_ARGS && break
done
touch $T/out/fmt_runs/.done_${H}_${MS}_${S/\//of}
if [ -z "$NOJUDGE" ] && [ "$(ls $T/out/fmt_runs/.done_${H}_${MS}_* 2>/dev/null | wc -l)" -ge "$n" ]; then
  # the 7B judge needs more than the generation cap (bf16 weights alone are 15 GB)
  LIE_MAX_GPU_GB=${JCAP:-24} run fmt_judge_$H $PY fmt_judge.py --runs $T/out/fmt_runs --cache-dir $T/hf --batch-size 8
fi
echo "$(date '+%F %T') all done ($S)"
