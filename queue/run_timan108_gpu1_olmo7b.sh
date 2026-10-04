#!/bin/bash
# timan108 GPU1 (A5000 24 GB): OLMo-2-7B contrast-threshold sweep (family fiction, the arm chosen at r = 0.37) and its mask
# anatomy. Launch only after the coordinator's grant.
export PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; H=/srv/local/xy51/hf_models/hub; cd $T/lie_detection
GPU=1 CAP=18 bash queue/run_timan_budget.sh allenai/OLMo-2-1124-7B-Instruct fiction 8
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LIE_MAX_GPU_GB=18 HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=$H HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
nice -n 10 $PY qwen_anatomy.py --model-id allenai/OLMo-2-1124-7B-Instruct --cache-dir $H --out $T/anatomy > $T/logs/anatomy_OLMo-2-1124-7B-Instruct.log 2>&1; echo "$(date '+%F %T') anatomy rc=$?"
echo "$(date '+%F %T') timan108 GPU1 OLMo-7B work done"
