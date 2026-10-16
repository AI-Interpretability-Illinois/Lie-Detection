#!/bin/bash
# fourth pass on timan1 GPU1 (2026-10-04 22:45): Falcon3-3B generation only (SKIP_JUDGE=1; the Qwen-7B judge step runs later
# on timan108), deadline 23:06 so that the card is clear before the 13B ends and the 32B waiter fires. Waits for the download.
cd /srv/local/xy51/lie
deadline=$(date -d "${DEADLINE:-23:06}" +%s)
m=tiiuae/Falcon3-3B-Instruct
until grep -q "done $m" logs/download_small.log; do grep -q "FAILED $m" logs/download_small.log && { echo "download failed"; exit 1; }; sleep 15; done
left=$(( deadline - $(date +%s) ))
[ $left -le 300 ] && { echo "$(date '+%F %T') skip: ${left}s left"; exit 0; }
echo "$(date '+%F %T') launching $m (${left}s to deadline)"
GPU=1 CAP=40 BS=16 SKIP_JUDGE=1 timeout -k 60 ${left}s bash lie_detection/queue/run_timan_extras.sh $m unsteered,fictf100,asr-0.18,prompt2 5 -0.18
echo "$(date '+%F %T') chain4 end rc=$?"
