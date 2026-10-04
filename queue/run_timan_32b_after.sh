#!/bin/bash
# OLMo-2-32B size point across timan1 GPU1+GPU2 (grants #37/#38, coordinator OK 08:59 Oct 4), started automatically once
# the 7B (GPU1) and 13B (GPU2) sweeps have ended and both cards have been idle for 3 consecutive minutes.
# usage: nohup setsid bash queue/run_timan_32b_after.sh > /srv/local/xy51/lie/logs/gpu12_32b.log 2>&1 < /dev/null &
T=/srv/local/xy51/lie; cd $T/lie_detection
idle() { local u1 u2; u1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' '); u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 | tr -d ' '); [ "$u1" -lt 1500 ] && [ "$u2" -lt 1500 ]; }
ok=0
while [ $ok -lt 3 ]; do
  if ! pgrep -f "run_timan_swee[p]" >/dev/null && idle; then ok=$((ok+1)); else ok=0; fi
  sleep 60
done
echo "$(date '+%F %T') 32B START (GPU1+GPU2 idle, 7B/13B sweeps ended)"
GPU=1,2 LIE_DEVICE_MAP=auto CAP=0 bash queue/run_timan_sweep.sh allenai/OLMo-2-0325-32B-Instruct 16,32,48 2
echo "$(date '+%F %T') 32B END"
