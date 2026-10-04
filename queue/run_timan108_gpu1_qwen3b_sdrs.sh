#!/bin/bash
# timan108 GPU1 (A5000 24 GB, grant #42 reused): the sd/rs cells of the 2x2 for Qwen2.5-3B (pooltf on LB, poolmix on AI Liar).
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=1 CAP=18 SDRS=pooltf,poolmix bash queue/run_timan_sweep_v3.sh Qwen/Qwen2.5-3B-Instruct 9,18,27 8
echo "$(date '+%F %T') timan108 GPU1 Qwen-3B sd/rs done"
