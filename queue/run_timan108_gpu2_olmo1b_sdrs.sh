#!/bin/bash
# timan108 GPU2 (A5000 24 GB): the sd/rs cells of the 2x2 for OLMo-2-1B's chosen families (fiction on LB, pooled on AI Liar).
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=2 CAP=18 SDRS=fiction,pooled bash queue/run_timan_sweep_v3.sh allenai/OLMo-2-0425-1B-Instruct 4,8,12 8
echo "$(date '+%F %T') timan108 GPU2 OLMo-1B sd/rs done"
