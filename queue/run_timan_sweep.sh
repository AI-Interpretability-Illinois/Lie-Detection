#!/bin/bash
# Size-sweep points on timan1 (extras plan): the main-table protocol on one model, in arm groups (as
# queue/lie_sweep.slurm), models from /srv/local/xy51/hf_models. usage: GPU=1 CAP=30 bash run_timan_sweep.sh <model_id> <layers> [bs]
# For OLMo-2-32B: GPU=1,2 LIE_DEVICE_MAP=auto CAP=0 (no cap; the model is sharded over both cards).
T=/srv/local/xy51/lie; PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin/python}   # timan108: /srv/local/xy51/envs/modularity310/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
[ "${CAP:-0}" != 0 ] && export LIE_MAX_GPU_GB=$CAP
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=/srv/local/xy51/hf_models/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE   # ~/.bashrc points these at a full /shared disk
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
M=$1; LAYERS=$2; BS=${3:-8}; S=$(basename $M | tr . p); X=$T/out/extra_data
ST="025 050 075 100"
groups=("sw1:ours,pooled:magmatch100" "sw2:fiction,fictf:" "sw3:pooltf,poolmix:" "sw4:::steer")
[ -n "$CN" ] && groups+=("sw5:cnours,cnpooled:" "sw6:cnfiction,cnfictf:" "sw7:cnpooltf,cnpoolmix:")
# CN_ONLY=1: only the CreativityNeuro groups (main-table models that already have the rest), as queue/lie_sweep.slurm
[ -n "$CN_ONLY" ] && groups=("sw5:cnours,cnpooled:" "sw6:cnfiction,cnfictf:" "sw7:cnpooltf,cnpoolmix:")
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; nice -n 10 "$@" > $T/logs/${n}_${S}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
for g in "${groups[@]}"; do
  IFS=':' read -r tag fams extra kind <<< "$g"
  arms=""
  if [ "$kind" = steer ]; then
    arms="unsteered,prompt,prompt2,prompt3"; for L in ${LAYERS//,/ }; do for r in -0.18 -0.5; do arms="$arms,asrL${L}_$r"; done; done
  else
    for fam in ${fams//,/ }; do for st in $ST; do arms="$arms,$fam$st"; done; done; arms=${arms#,}; [ -n "$extra" ] && arms="$arms,$extra"
  fi
  R="$PY run_liars_bench_honesty.py --model-id $M --cache-dir $HF_HUB_CACHE --out-dir $T/out --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $arms --as-layers $LAYERS --as-rel-coeffs=-0.18,-0.5 --batch-size $BS --score-batch 2"
  step ${tag}_lb $T/out/liars_bench_honesty_${S}_${tag}.csv $R --eval-set $T/out/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $tag
  step ${tag}_ail $T/out/liars_bench_honesty_${S}_${tag}ail.csv $R --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --tag ${tag}ail
  step ${tag}_judge $T/out/liars_bench_honesty_${S}_${tag}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    $PY judge_claims.py --responses liars_bench_honesty_${S}_${tag}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf --eval-set $T/out/liars_bench/eval_set.jsonl
done
echo "$(date '+%F %T') all done $M"
