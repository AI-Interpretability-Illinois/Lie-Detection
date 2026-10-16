#!/bin/bash
# timan108 GPU2 (A5000 24 GB, grant #41): mask anatomy (qwen_anatomy.py) for the five main-table models that fit beside
# zixuans8's ~8 GB; Llama-3.1-8B and Qwen2.5-7B stay in the Delta job (lie_anatomy.slurm, MODELS=...). Results go to
# /srv/local/xy51/lie/anatomy and are rsynced to /work/nvme/biro/xiaocong/lie_sweep/anatomy on Delta.
PY=/srv/local/xy51/envs/modularity310/bin/python
T=/srv/local/xy51/lie; H=/srv/local/xy51/hf_models/hub; cd $T/lie_detection
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 LIE_MAX_GPU_GB=${CAP:-15}
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=$H HF_XET_CACHE=/srv/local/xy51/hf_models/xet TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache
unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in microsoft/Phi-4-mini-instruct tiiuae/Falcon3-3B-Instruct; do
  [ -d $H/models--${m//\//--}/snapshots ] || $PY -c "from huggingface_hub import snapshot_download; snapshot_download('$m', cache_dir='$H')" 2>&1 | tail -1
done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct allenai/OLMo-2-0425-1B-Instruct tiiuae/Falcon3-3B-Instruct microsoft/Phi-4-mini-instruct; do
  s=$(basename $m | tr . p); [ -e $T/anatomy/qwen_anatomy_$s.json ] && { echo "skip $m"; continue; }
  echo "$(date '+%F %T') start $m"; nice -n 10 $PY qwen_anatomy.py --model-id $m --cache-dir $H --out $T/anatomy > $T/logs/anatomy_$s.log 2>&1; echo "$(date '+%F %T') end $m rc=$?"
done
echo "$(date '+%F %T') timan108 GPU2 anatomy done"
