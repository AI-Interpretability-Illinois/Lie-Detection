#!/bin/bash
# The GPU1 window's work, launched directly once the grant is in hand (the waiter's first pass failed on a cache path).
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=1 CAP=40 bash queue/run_timan_sweep.sh allenai/OLMo-2-1124-7B-Instruct 8,16,24 8
GPU=1 CAP=44 bash queue/run_timan_sweep.sh allenai/OLMo-2-1124-13B-Instruct 10,20,30 4
echo "$(date '+%F %T') GPU1 chain done"
