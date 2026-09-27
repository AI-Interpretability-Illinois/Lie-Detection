#!/bin/bash
# Start a worker on each GPU unless one is already alive. Safe to run repeatedly,
# including after a disconnect: run it on the node, or
#   ssh gpua047 /u/xiaocong/lie_detection/queue/start_workers.sh
Q=/u/xiaocong/lie_detection/queue
rm -f ~/.stop_lie_queue
for g in ${GPUS:-0 1 2 3}; do
  pf="$Q/worker.gpu$g.pid"
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then echo "gpu$g worker alive ($(cat "$pf"))"; continue; fi
  setsid nohup bash "$Q/worker.sh" $g < /dev/null >> "$Q/logs/worker.gpu$g.log" 2>&1 &
  echo "gpu$g worker started"
done
