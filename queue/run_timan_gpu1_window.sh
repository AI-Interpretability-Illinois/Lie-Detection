#!/bin/bash
# timan1 GPU1 window (grant #38 part 1, earliest 08:30 Oct 4): run the size-sweep points that fit on one 48 GB card
# (OLMo-2-7B, OLMo-2-13B) and, if the model is present, the Dolly pilot. Launch gates, all required (coordinator's
# conditions): the start time, the grant flag file (touched by the lie-steering session when the coordinator grants
# the card), and GPU1 below 1.5 GB with no compute process for 10 consecutive one-minute checks (swe-memory's W1
# has a retry pass after its main pass; a one-off dip must not launch into that gap).
# usage: nohup bash queue/run_timan_gpu1_window.sh > /srv/local/xy51/lie/logs/gpu1_window.log 2>&1 &
T=/srv/local/xy51/lie; cd $T/lie_detection
START=${START:-"2026-10-04 08:30"}; FLAG=${FLAG:-$T/GRANT_GPU1}
until [ $(date +%s) -ge $(date -d "$START" +%s) ] && [ -e "$FLAG" ]; do sleep 60; done
echo "$(date '+%F %T') window open and grant flag present; waiting for GPU1 to be idle 10 minutes"
ok=0
while [ $ok -lt 10 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1 | grep -c .)
  if [ "$used" -lt 1500 ] && [ "$procs" -eq 0 ]; then ok=$((ok+1)); else [ $ok -gt 0 ] && echo "$(date '+%F %T') GPU1 busy again (${used} MiB, $procs procs) after $ok idle minutes"; ok=0; fi
  sleep 60
done
echo "$(date '+%F %T') GPU1 idle for 10 minutes (${used} MiB used), starting"
if [ -d /srv/local/xy51/hf_models/hub/models--databricks--dolly-v2-3b/snapshots ]; then
  GPU=1 CAP=30 bash queue/run_timan_dolly_pilot.sh
fi
GPU=1 CAP=40 bash queue/run_timan_sweep.sh allenai/OLMo-2-1124-7B-Instruct 8,16,24 8
GPU=1 CAP=44 bash queue/run_timan_sweep.sh allenai/OLMo-2-1124-13B-Instruct 10,20,30 4
echo "$(date '+%F %T') GPU1 window script done"
