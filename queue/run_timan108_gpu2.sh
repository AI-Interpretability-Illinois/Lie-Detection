#!/bin/bash
# timan108 GPU2 (A5000 24 GB, grant #41, from 12:03 Oct 4): CreativityNeuro groups for Qwen2.5-3B (Delta 22655131 cancelled
# once this is loading). Qwen2.5-1.5B stayed on Delta: its job 22655130 started on gpua038 at 12:01 before the grant.
# zixuans8 keeps a few GB on this card, so the cap is 18 GB.
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=2 CAP=18 CN_ONLY=1 bash queue/run_timan_sweep.sh Qwen/Qwen2.5-3B-Instruct 9,18,27 8
echo "$(date '+%F %T') timan108 GPU2 work done"
