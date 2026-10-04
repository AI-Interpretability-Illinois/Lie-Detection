#!/bin/bash
# timan108 GPU2 (A5000 24 GB, grant #41, after the anatomy run): CreativityNeuro groups for Falcon3-3B (Delta 22655132
# cancelled once this is loading). Cap 18 GB because zixuans8 keeps up to ~8 GB on this card.
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=2 CAP=18 CN_ONLY=1 bash queue/run_timan_sweep.sh tiiuae/Falcon3-3B-Instruct 5,11,16 8
echo "$(date '+%F %T') timan108 GPU2 Falcon CN done"
