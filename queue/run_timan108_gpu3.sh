#!/bin/bash
# timan108 GPU3 (A5000 24 GB, grant #40, from 11:38 Oct 4): the Qwen2.5-0.5B size point and the CreativityNeuro groups
# for OLMo-2-1B, moved off the congested gpua038 queue (their Delta copies 22655127 / 22655129 are cancelled once this
# is running). Same layout as timan1 (/srv/local/xy51/lie, /srv/local/xy51/hf_models/hub); the judge model is read from
# the shared home cache through /srv/local/xy51/lie/hf. Interpreter: the host's own env (the anaconda one has no torch).
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; cd $T/lie_detection
GPU=3 CAP=20 bash queue/run_timan_sweep.sh Qwen/Qwen2.5-0.5B-Instruct 6,12,18 8
GPU=3 CAP=20 CN_ONLY=1 bash queue/run_timan_sweep.sh allenai/OLMo-2-0425-1B-Instruct 4,8,12 8
echo "$(date '+%F %T') timan108 GPU3 work done"
