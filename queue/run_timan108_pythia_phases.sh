#!/bin/bash
# Pythia size ablation on timan108 (grant #58: GPU0+GPU1 until 2026-10-09 14:00), single task (owner 21:51).
# 12b first, sharded over both cards (the largest and least certain run), then one lane per card, balanced by size.
T=/srv/local/xy51/lie; L=$T/logs_pythia
echo "$(date "+%F %T") 12b on 0+1"
bash $T/run_pythia_lane.sh 0+1 12b > $L/lb_lane01.log 2>&1
echo "$(date "+%F %T") 12b lane over -> two lanes"
bash $T/run_pythia_lane.sh 0 6.9b 1b 160m 70m > $L/lb_lane0.log 2>&1 &
bash $T/run_pythia_lane.sh 1 2.8b 1.4b 410m > $L/lb_lane1.log 2>&1 &
wait
echo "$(date "+%F %T") all sizes done"
