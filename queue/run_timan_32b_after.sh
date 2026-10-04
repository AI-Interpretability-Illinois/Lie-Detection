#!/bin/bash
# OLMo-2-32B size point across timan1 GPU1+GPU2 (grants #37/#38, coordinator OK 08:59 Oct 4), started automatically once
# no sweep of ours is running and, for 3 consecutive minutes, each card carries at most 3 GB of foreign use (coordinator's
# rule of 16:31: a small co-tenant such as yurenh2's 1.4 GB job is accepted, since the 32B needs ~65 GB across 2 x 48 GB).
# usage: nohup setsid bash queue/run_timan_32b_after.sh > /srv/local/xy51/lie/logs/gpu12_32b.log 2>&1 < /dev/null &
T=/srv/local/xy51/lie; cd $T/lie_detection
used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -d ' '; }
ok=0; last=""
while [ $ok -lt 3 ]; do
  u1=$(used 1); u2=$(used 2); busy=$(pgrep -f "run_timan_swee[p]" | wc -l)
  if [ "$u1" -le 3072 ] && [ "$u2" -le 3072 ] && [ "$busy" -eq 0 ]; then ok=$((ok+1)); else ok=0; fi
  state="sweeps=$busy gpu1=${u1}MiB gpu2=${u2}MiB"; [ "$state" != "$last" ] && echo "$(date '+%F %T') $state" && last=$state
  sleep 60
done
echo "$(date '+%F %T') 32B START (sweeps ended; foreign use gpu1=${u1} gpu2=${u2} MiB)"
GPU=1,2 LIE_DEVICE_MAP=auto CAP=0 bash queue/run_timan_sweep_v2.sh allenai/OLMo-2-0325-32B-Instruct 16,32,48 2
echo "$(date '+%F %T') 32B END"
