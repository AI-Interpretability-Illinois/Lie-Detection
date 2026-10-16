#!/bin/bash
# Fill the idle timan1 GPU1 window before the OLMo-2-32B run (coordinator's suggestion, 2026-10-04 21:18): run the
# extras (attack + legit sets) for small main-table models one after another, each under a hard deadline so that the
# 32B start is not delayed. Waits for each model's download (logs/download_small.log, "done <model>" lines).
# usage: DEADLINE="22:55" bash queue/run_timan1_extras_chain.sh      (run from /srv/local/xy51/lie)
cd /srv/local/xy51/lie
DL=logs/download_small.log
deadline=$(date -d "${DEADLINE:-22:55}" +%s)
run() { local m=$1; shift
  until grep -q "done $m" $DL; do grep -q "FAILED $m" $DL && { echo "$(date '+%F %T') download failed $m"; return; }; sleep 20; done
  local left=$(( deadline - $(date +%s) ))
  [ $left -le 300 ] && { echo "$(date '+%F %T') skip $m: ${left}s left"; return; }
  echo "$(date '+%F %T') launching $m (${left}s to deadline)"
  GPU=1 CAP=40 BS=16 timeout -k 60 ${left}s bash lie_detection/queue/run_timan_extras.sh $m "$@"
  echo "$(date '+%F %T') finished $m rc=$?"
}
# steering arm names: with ONE layer in --as-layers the runner names the arm "asr<r>" (asr-0.18), not "asrL<L>_<r>"
run Qwen/Qwen2.5-1.5B-Instruct unsteered,fictf100,asr-0.18,prompt3 14 -0.18
run allenai/OLMo-2-0425-1B-Instruct unsteered,fiction025,asr-0.5,prompt 12 -0.5
echo "$(date '+%F %T') chain end"
