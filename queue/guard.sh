#!/bin/bash
# Guard for running the queue beside someone else's training on the same GPUs.
#
# 1. Settle window: when the set of other GPU processes changes (their job
#    restarting, or handing over to a new run), pause new job starts for SETTLE
#    seconds by setting the queue's stop file. Running jobs continue; idle workers
#    exit. After the window the guard restarts the workers.
# 2. Low-memory kill: if a GPU running one of our jobs drops below LOW_MIB free,
#    kill our job on that GPU so the other run gets the memory, requeue it, and
#    open a settle window.
#
# Our processes are identified by command line (same Unix user as the training).
# Start on the node:  setsid nohup queue/guard.sh >> queue/logs/guard.log 2>&1 &
# It exits when ~/.stop_lie_guard exists.
Q=/u/xiaocong/lie_detection/queue
LOW_MIB=${LOW_MIB:-1500}
SETTLE=${SETTLE:-600}
WORKER_GPUS=${WORKER_GPUS:-"1 2 3"}
WORKER_MAX_MEM=${WORKER_MAX_MEM:-18}
OURS="run_liars_bench_honest[y].py|judge_claim[s].py|judge_mas[k].py"
prev="__init__"
paused_until=0
requeue=""
echo "$(date '+%F %H:%M:%S') guard up: low=${LOW_MIB}MiB settle=${SETTLE}s gpus=[$WORKER_GPUS]"
while [ ! -e ~/.stop_lie_guard ]; do
  now=$(date +%s)
  mine=" $(pgrep -u "$USER" -f "$OURS" | tr '\n' ' ') "
  apps=$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader,nounits 2>/dev/null)
  others=$(echo "$apps" | awk -F', *' 'NF>=2{print $1}' | while read -r p; do
             case "$mine" in *" $p "*) ;; *) echo "$p" ;; esac; done | sort -u | tr '\n' ' ')

  # 1. other processes changed: pause new starts
  if [ "$prev" != "__init__" ] && [ "$others" != "$prev" ]; then
    paused_until=$((now + SETTLE))
    touch ~/.stop_lie_queue
    echo "$(date '+%F %H:%M:%S') other GPU processes changed [$prev] -> [$others]; no new jobs until $(date -d @$paused_until '+%H:%M:%S')"
  fi
  prev="$others"

  # 2. low memory on a GPU where one of our jobs runs: kill ours there
  while IFS=', ' read -r idx uuid free; do
    [ -n "$free" ] || continue
    if [ "$free" -lt "$LOW_MIB" ]; then
      for p in $(echo "$apps" | awk -F', *' -v u="$uuid" '$2==u{print $1}'); do
        case "$mine" in *" $p "*)
          job=$(ls "$Q/running" 2>/dev/null | grep "gpu$idx\$" | head -1)
          kill "$p" 2>/dev/null && echo "$(date '+%F %H:%M:%S') gpu$idx free ${free}MiB: killed our pid $p (${job:-unknown job})"
          [ -n "$job" ] && requeue="$requeue ${job%.*}"
          paused_until=$((now + SETTLE)); touch ~/.stop_lie_queue ;;
        esac
      done
    fi
  done < <(nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader,nounits 2>/dev/null)

  # requeue jobs we killed once the worker has moved them to failed/
  for j in $requeue; do
    if [ -e "$Q/failed/$j" ]; then
      mv "$Q/failed/$j" "$Q/pending/$j" && echo "$(date '+%F %H:%M:%S') requeued $j"
      requeue=$(echo "$requeue" | sed "s/ $j//")
    fi
  done

  # settle window over: resume the workers
  if [ "$paused_until" -gt 0 ] && [ "$now" -ge "$paused_until" ]; then
    paused_until=0
    GPUS="$WORKER_GPUS" MAX_MEM_GB="$WORKER_MAX_MEM" "$Q/start_workers.sh" | sed "s/^/$(date '+%F %H:%M:%S') resume: /"
  fi
  sleep 5
done
echo "$(date '+%F %H:%M:%S') guard stop file present, exiting"
