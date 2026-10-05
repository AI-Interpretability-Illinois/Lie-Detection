#!/bin/bash
# Extras plan item 5 (adaptive attacks + legitimate falsehoods) on one main-table model, on a timan GPU: the timan
# version of queue/lie_extras.slurm. Same steps (atk_gen, atk_judge with the Qwen2.5-7B judge, legit_gen), same
# output names under $T/out (tags atk / legit), so rsync --include="liars_bench_honesty_<model>_atk*|_legit*" moves
# them next to the Delta outputs in /work/nvme/biro/xiaocong/lie_sweep.
# usage: GPU=1 CAP=40 BS=16 bash queue/run_timan_extras.sh <model_id> <arms> <steer_layer> <rel_coeff>
#   e.g. GPU=1 bash queue/run_timan_extras.sh Qwen/Qwen2.5-1.5B-Instruct unsteered,fictf100,asrL14_-0.18,prompt3 14 -0.18
# Uses run_liars_bench_honesty_x.py (the 2026-10-04 19:42 runner with the freeform grade type), copied next to the host's
# older run_liars_bench_honesty.py so that a sweep still running from the old file is not disturbed.
T=/srv/local/xy51/lie; PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin/python}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-1}
[ "${CAP:-0}" != 0 ] && export LIE_MAX_GPU_GB=$CAP
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=/srv/local/xy51/hf_models/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$T/out LIE_CONTROLS=$T/controls
cd $T/lie_detection
M=$1; ARMS=$2; LAYERS=$3; RC=$4; BS=${BS:-16}; S=$(basename $M | tr . p); X=$T/out/extra_data
echo "$(date '+%F %T') extras on $(hostname -s) GPU $CUDA_VISIBLE_DEVICES: $M arms $ARMS"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; nice -n 10 "$@" > $T/logs/${n}_${S}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
R="$PY run_liars_bench_honesty_x.py --model-id $M --cache-dir $HF_HUB_CACHE --out-dir $T/out --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $ARMS --as-layers $LAYERS --as-rel-coeffs=$RC --batch-size $BS --score-batch 2"
step atk_gen $T/out/liars_bench_honesty_${S}_atk.csv $R --eval-set $X/extra_attack.jsonl --belief-tag attack --tag atk
step atk_judge $T/out/liars_bench_honesty_${S}_atk_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  $PY judge_claims.py --responses liars_bench_honesty_${S}_atk_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf \
  --eval-set $X/extra_attack.jsonl --splits attack-manyshot,attack-es,attack-zh --beliefs $T/out/liars_bench/beliefs_${S}_attack.jsonl
step legit_gen $T/out/liars_bench_honesty_${S}_legit.csv $R --eval-set $X/extra_legit.jsonl --tag legit
echo "$(date '+%F %T') all done $M"
