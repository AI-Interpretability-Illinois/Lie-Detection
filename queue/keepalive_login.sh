#!/bin/bash
# Runs on the LOGIN node. Processes launched on gpua047 by ssh are adopted into
# whichever SLURM job holds the node, and die when that job ends (seen at 02:13
# and 23:04 on 2026-09-27, both training-job handovers). This loop brings the
# queue back after each handover, without touching the training:
#   - only while a resv_train* job is RUNNING on the node,
#   - only when the ssh session sees all 4 GPUs (adopted into the training job,
#     not the 1-GPU keepalive filler, which lives an hour at most),
#   - only after that training job has run SETTLE_MIN minutes.
# Exits when every fix job has its completion marker, or when ~/.stop_lie_keepalive exists.
NODE=${NODE:-gpua047}
Q=/u/xiaocong/lie_detection/queue
SETTLE_MIN=${SETTLE_MIN:-10}
log() { echo "$(date '+%F %H:%M:%S') $*"; }
log "keepalive up for $NODE (settle ${SETTLE_MIN} min)"
while [ ! -e ~/.stop_lie_keepalive ]; do
  if [ "$(ls $Q/markers/0?[abc]_fix*.ok 2>/dev/null | wc -l)" -ge 12 ]; then
    log "all fix jobs done; exiting"; exit 0
  fi
  # a training job running on the node, and for how long (minutes)
  el=$(squeue -u "$USER" -w "$NODE" -h -t RUNNING -o "%j %M" 2>/dev/null | awk '$1 ~ /^resv_train/ {print $2; exit}')
  if [ -n "$el" ]; then
    mins=$(echo "$el" | awk -F'[-:]' '{ if (NF==2) print $1; else if (NF==3) print $1*60+$2; else print 9999 }')
    state=$(timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
      'echo "$(nvidia-smi -L 2>/dev/null | wc -l) $(pgrep -u $USER -f "queue/worke[r].sh" | wc -l) $(pgrep -u $USER -f "queue/guar[d].sh" | wc -l)"' 2>/dev/null)
    read -r gpus workers guard <<< "$state"
    if [ "${gpus:-0}" -eq 4 ] && [ "${mins:-0}" -ge "$SETTLE_MIN" ] && { [ "${workers:-0}" -eq 0 ] || [ "${guard:-0}" -eq 0 ]; }; then
      log "training up ${mins} min, 4 GPUs visible, workers=${workers} guard=${guard}: relaunching"
      timeout 40 ssh -o BatchMode=yes -o ConnectTimeout=10 "$NODE" \
        "cd /u/xiaocong/lie_detection && rm -f ~/.stop_lie_guard && \
         { pgrep -u \$USER -f 'queue/guar[d].sh' >/dev/null || setsid nohup ./queue/guard.sh < /dev/null >> queue/logs/guard.log 2>&1 & } ; \
         GPUS='1 2 3' MAX_MEM_GB=17 ./queue/start_workers.sh" 2>&1 | sed 's/^/    /'
    fi
  fi
  sleep 120
done
log "stop file present; exiting"
