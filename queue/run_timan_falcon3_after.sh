#!/bin/bash
# Falcon3-7B (GPU1) and Falcon3-10B (GPU2) size points on timan1, started together once the 32B run has ended AND the
# flag /srv/local/xy51/lie/GRANT_F3 exists (touched after the coordinator confirms the cards stay with lie-steering).
# usage: nohup setsid bash queue/run_timan_falcon3_after.sh > /srv/local/xy51/lie/logs/falcon3_after.log 2>&1 < /dev/null &
T=/srv/local/xy51/lie; cd $T/lie_detection
until grep -q "32B END" $T/logs/gpu12_32b.log 2>/dev/null && [ -e $T/GRANT_F3 ]; do sleep 120; done
echo "$(date '+%F %T') 32B ended and flag present; starting Falcon3-7B on GPU1 and Falcon3-10B on GPU2"
GPU=1 CAP=40 bash queue/run_timan_sweep_v2.sh tiiuae/Falcon3-7B-Instruct 7,14,21 8 > $T/logs/f3_7b.log 2>&1 &
GPU=2 CAP=44 bash queue/run_timan_sweep_v2.sh tiiuae/Falcon3-10B-Instruct 10,20,30 4 > $T/logs/f3_10b.log 2>&1 &
wait
echo "$(date '+%F %T') Falcon3-7B/10B done"
