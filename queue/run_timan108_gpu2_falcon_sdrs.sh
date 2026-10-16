#!/bin/bash
# timan108 GPU2 (grant #41, if still held): the sd/rs cells of the 2x2 for Falcon3-3B's chosen family (fictf on both benchmarks).
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=2 CAP=18 SDRS=fictf bash queue/run_timan_sweep_v3.sh tiiuae/Falcon3-3B-Instruct 5,11,16 8
echo "$(date '+%F %T') timan108 GPU2 Falcon sd/rs done"
