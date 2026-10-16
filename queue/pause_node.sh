#!/bin/bash
Q=/u/xiaocong/lie_detection/queue
# guard first, so it does not restart workers
for p in $(pgrep -u "$USER" -f "queue/guar[d].sh"); do kill "$p" && echo "killed guard $p"; done
for pf in $Q/worker.gpua047-gpu*.pid; do
  w=$(cat "$pf"); kill -0 "$w" 2>/dev/null || continue
  g=$(ps -o pgid= -p "$w" | xargs)
  kill -TERM -- -"$g" && echo "killed worker $w (group $g)"
done
sleep 20
left=$(pgrep -u "$USER" -af "run_liars_bench_honest[y].py|judge_claim[s].py|judge_mas[k].py|queue/worke[r].sh")
[ -n "$left" ] && { echo "still alive, KILL:"; echo "$left"; pkill -KILL -u "$USER" -f "run_liars_bench_honest[y].py|judge_claim[s].py|judge_mas[k].py|queue/worke[r].sh"; sleep 5; }
for f in $Q/running/*.gpua047-gpu*; do [ -e "$f" ] || continue; b=$(basename "$f"); b=${b%.gpua047-gpu*}; mv "$f" "$Q/pending/$b" && echo "requeued $b"; done
for f in $Q/failed/*; do [ -e "$f" ] || continue; echo "in failed: $(basename $f)"; done
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
pgrep -u "$USER" -af "python" | grep -v grep | cut -c1-120
