#!/bin/bash
# Second pass on timan1 GPU1 (2026-10-04): waits until the first chain (run_timan1_extras_chain.sh) is gone, then reruns the
# Qwen2.5-1.5B extras; the atk steps are skipped (done files exist) and only legit_gen runs, with the runner fix that
# exempts freeform splits from the belief gate. Same hard deadline as the first chain.
cd /srv/local/xy51/lie
deadline=$(date -d "${DEADLINE:-22:55}" +%s)
while pgrep -f run_timan1_extras_chain.sh > /dev/null; do sleep 30; done
left=$(( deadline - $(date +%s) ))
[ $left -le 120 ] && { echo "$(date '+%F %T') skip: ${left}s left"; exit 0; }
echo "$(date '+%F %T') launching Qwen2.5-1.5B legit pass (${left}s to deadline)"
GPU=1 CAP=40 BS=16 timeout -k 60 ${left}s bash lie_detection/queue/run_timan_extras.sh Qwen/Qwen2.5-1.5B-Instruct unsteered,fictf100,asr-0.18,prompt3 14 -0.18
echo "$(date '+%F %T') chain2 end rc=$?"
