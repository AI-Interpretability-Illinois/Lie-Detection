#!/bin/bash
# One worker per GPU, pulling from queue/pending in name order.
#
# A job is a bash script with optional header lines:
#   # MEM_GB: <n>        free GPU memory it needs before it may start (default 20)
#   # NEEDS: <path>      a file that must exist first (repeatable), e.g. the
#                        completion marker of the generation job a judge reads
#   # DONE_IF: <path>    skip the job if this exists (its own completion marker)
#
# Restart-safe: on start, jobs this GPU's previous worker left in running/ go back
# to pending/, and jobs whose DONE_IF marker exists are skipped, so after a
# disconnect it is enough to run queue/start_workers.sh again.
# Stop all workers with: touch ~/.stop_lie_queue
GPU=$1
Q=/u/xiaocong/lie_detection/queue
WID="$(hostname -s)-gpu$GPU"
# Optional limits for shared nodes: skip jobs declaring more than MAX_MEM_GB, and
# cap each job's GPU memory at LIE_MAX_GPU_GB (read by did_estimators.load_lm).
MAX_MEM_GB=${MAX_MEM_GB:-999}
export LIE_MAX_GPU_GB=${LIE_MAX_GPU_GB:-}
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd /u/xiaocong/lie_detection
for f in "$Q"/running/*.$WID; do
  [ -e "$f" ] || continue
  b=$(basename "$f" .$WID); mv "$f" "$Q/pending/$b"; echo "$(date '+%F %H:%M') requeued $b"
done
echo $$ > "$Q/worker.$WID.pid"
while [ ! -e ~/.stop_lie_queue ]; do
  job=""
  for f in $(ls "$Q/pending" 2>/dev/null | sort); do
    p="$Q/pending/$f"; [ -f "$p" ] || continue
    d=$(grep -m1 '^# DONE_IF:' "$p" | cut -d: -f2- | xargs)
    if [ -n "$d" ] && [ -e "$d" ]; then mv "$p" "$Q/done/$f" 2>/dev/null && echo "$(date '+%F %H:%M') already done $f"; continue; fi
    ok=1
    while read -r n; do [ -z "$n" ] || [ -e "$n" ] || ok=0; done < <(grep '^# NEEDS:' "$p" | cut -d: -f2- | xargs -n1 2>/dev/null)
    [ $ok = 1 ] || continue
    need=$(grep -m1 '^# MEM_GB:' "$p" | cut -d: -f2 | xargs); need=${need:-20}
    [ "$need" -le "$MAX_MEM_GB" ] || continue
    free=$(nvidia-smi --id=$GPU --query-gpu=memory.free --format=csv,noheader,nounits | xargs)
    [ -n "$free" ] && [ $((free / 1024)) -ge "$need" ] || continue
    mv "$p" "$Q/running/$f.$WID" 2>/dev/null || continue     # another worker won
    job=$f; break
  done
  if [ -z "$job" ]; then sleep 60; continue; fi
  echo "$(date '+%F %H:%M') start $job"
  CUDA_VISIBLE_DEVICES=$GPU nice -n 10 bash "$Q/running/$job.$WID" > "$Q/logs/$job.log" 2>&1
  rc=$?
  if [ $rc = 0 ]; then mv "$Q/running/$job.$WID" "$Q/done/$job"; else mv "$Q/running/$job.$WID" "$Q/failed/$job"; fi
  echo "$(date '+%F %H:%M') end $job rc=$rc"
done
echo "$(date '+%F %H:%M') stop file present, exiting"
