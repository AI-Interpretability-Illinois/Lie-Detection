#!/bin/bash
# J1 (analysis plan 2026-10-02): format-study test halves at the validation-chosen strengths, on timan1.
# GPU1: Llama-3.1-8B; GPU2: Qwen2.5-7B then Falcon3-3B; then one judge pass on GPU2 after both.
# usage: bash run_timan_j1.sh gpu1 | gpu2
T=/srv/local/xy51/lie; cd $T
LL="unsteered,maskc:F1_sys:0.0025,maskc:F2_user:0.0025,maskc:F3_inject:0.0025,maskc:F4_mc:0.0025,maskc:F5_tf:0.0025,maskc:F6_multi:0.0025,maskc:F7_long:0.0025,caa:F1_sys:-0.5,caa:F2_user:-0.5,caa:F3_inject:-0.5,caa:F4_mc:-0.25,caa:F5_tf:-0.25,caa:F6_multi:-0.5,caa:F7_long:-0.5"
QW="unsteered,maskc:F1_sys:0.0025,maskc:F2_user:0.005,maskc:F3_inject:0.0025,maskc:F4_mc:0.0025,maskc:F5_tf:0.005,maskc:F6_multi:0.0025,maskc:F7_long:0.005,caa:F1_sys:-0.27,caa:F2_user:-0.27,caa:F3_inject:-0.18,caa:F4_mc:-0.27,caa:F5_tf:-0.27,caa:F6_multi:-0.18,caa:F7_long:-0.27"
FA="unsteered,maskc:F1_sys:0.005,maskc:F2_user:0.0025,maskc:F3_inject:0.0025,maskc:F4_mc:0.005,maskc:F5_tf:0.005,maskc:F6_multi:0.0025,maskc:F7_long:0.0025,caa:F1_sys:-0.09,caa:F2_user:-0.36,caa:F3_inject:-0.09,caa:F4_mc:-0.36,caa:F5_tf:-0.09,caa:F6_multi:-0.36,caa:F7_long:-0.36"
D=$T/out/fmt_runs
if [ "$1" = gpu1 ]; then
  GPU=1 CAP=24 NOJUDGE=1 M=meta-llama/Llama-3.1-8B-Instruct LAYER=8 FMT_ARGS="--half test --arms $LL" bash run_timan_fmt.sh
  touch $D/.j1_gpu1
else
  GPU=2 CAP=24 NOJUDGE=1 M=Qwen/Qwen2.5-7B-Instruct LAYER=14 FMT_ARGS="--half test --arms $QW" bash run_timan_fmt.sh
  GPU=2 CAP=16 NOJUDGE=1 M=tiiuae/Falcon3-3B-Instruct LAYER=5 FMT_ARGS="--half test --arms $FA" bash run_timan_fmt.sh
  until [ -e $D/.j1_gpu1 ]; do sleep 60; done
  export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 LIE_MAX_GPU_GB=24 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FMT_OUT=$T/out/fmt_suite PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  cd $T/lie_detection && /home/xy51/anaconda3/envs/modularity/bin/python fmt_judge.py --runs $D --cache-dir $T/hf --batch-size 8 >> $T/logs/fmt_judge_j1.log 2>&1
  echo "$(date '+%F %T') j1 judge rc=$?"
fi
echo "$(date '+%F %T') j1 $1 all done"
