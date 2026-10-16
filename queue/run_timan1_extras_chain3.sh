#!/bin/bash
# third pass on timan1 GPU1 (2026-10-04): Qwen2.5-3B extras after chain2, same 22:55 deadline; waits for the download
cd /srv/local/xy51/lie
deadline=$(date -d "${DEADLINE:-22:55}" +%s)
m=Qwen/Qwen2.5-3B-Instruct
while pgrep -f "run_timan1_extras_chain2.sh|run_timan1_extras_chain.sh" > /dev/null; do sleep 20; done
until grep -q "done $m" logs/download_small.log; do grep -q "FAILED $m" logs/download_small.log && { echo "download failed"; exit 1; }; sleep 20; done
left=$(( deadline - $(date +%s) ))
[ $left -le 300 ] && { echo "$(date "+%F %T") skip: ${left}s left"; exit 0; }
echo "$(date "+%F %T") launching $m (${left}s to deadline)"
GPU=1 CAP=40 BS=16 timeout -k 60 ${left}s bash lie_detection/queue/run_timan_extras.sh $m unsteered,pooltf100,asr-0.18,prompt 27 -0.18
echo "$(date "+%F %T") chain3 end rc=$?"
