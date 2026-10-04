#!/bin/bash
# OLMo-2-32B size point across timan1 GPU1+GPU2 (grants #37/#38, coordinator OK 08:59 Oct 4), started automatically once
# no sweep of ours is running and both cards have had no compute process at all for 3 consecutive minutes (a foreign
# 1.4 GB process at 100% util appeared on GPU1 at 16:25; the earlier < 1.5 GB test would have launched beside it).
# usage: nohup setsid bash queue/run_timan_32b_after.sh > /srv/local/xy51/lie/logs/gpu12_32b.log 2>&1 < /dev/null &
T=/srv/local/xy51/lie; cd $T/lie_detection
procs() { nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c -E "$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1)"; }
ok=0; last=""
while [ $ok -lt 3 ]; do
  p1=$(procs 1); p2=$(procs 2); busy=$(pgrep -f "run_timan_swee[p]" | wc -l)
  if [ "$p1" -eq 0 ] && [ "$p2" -eq 0 ] && [ "$busy" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  state="sweeps=$busy gpu1_procs=$p1 gpu2_procs=$p2"; [ "$state" != "$last" ] && echo "$(date '+%F %T') $state" && last=$state
  sleep 60
done
echo "$(date '+%F %T') 32B START (GPU1+GPU2 free of all processes, sweeps ended)"
GPU=1,2 LIE_DEVICE_MAP=auto CAP=0 bash queue/run_timan_sweep_v2.sh allenai/OLMo-2-0325-32B-Instruct 16,32,48 2
echo "$(date '+%F %T') 32B END"
