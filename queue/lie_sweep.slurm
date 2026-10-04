#!/bin/bash
#SBATCH --job-name=lie_sweep
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%j.out
# Size sweep (extras plan, Oct 4): the main-table protocol on one new model, in arm groups so that at most two
# boolean masks live in host memory (a 14B mask is 14 GB). Groups: G1 ours+pooled (+magmatch), G2 fiction+fictf,
# G3 pooltf+poolmix, G4 steering (3 layers x 2 strengths) + 3 honest prompts + unsteered. Each group: LIARS' BENCH
# (all items) + AI Liar + the Qwen2.5-7B judge on LIARS' BENCH; tags sw1..sw4 (+ail). Also the CreativityNeuro
# arms (G5-G7: cn families) when CN=1. Env: MODEL, LAYERS (e.g. 12,24,36), HF (cache dir), BS, CN.
# build_final_table.py reads the sw* tags from SW_DIR. Restart-safe. CACHE: the directory holding models--* (the nvme cache
# is laid out as <HF>/hub; bfqt holds models at its top level); WikiText is read from <CACHE>/wikitext.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=${HF:-/work/hdd/bfqt/xiaocong/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/nvme/biro/xiaocong/lie_sweep; mkdir -p $O/logs $O/liars_bench; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
[ -e $O/extra_data ] || ln -s $B/extra_data $O/extra_data
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
S=$(basename $MODEL | tr . p)
# belief caches of the main-table models (bfqt is over quota, so outputs go to nvme; beliefs are copied once)
for f in $B/liars_bench/beliefs_${S}.jsonl $B/liars_bench/beliefs_${S}_ailiar.jsonl; do [ -e "$f" ] && [ ! -e $O/liars_bench/$(basename $f) ] && cp $f $O/liars_bench/; done
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): $MODEL layers $LAYERS"
ST="025 050 075 100"
groups=("sw1:ours,pooled:magmatch100" "sw2:fiction,fictf:" "sw3:pooltf,poolmix:" "sw4:::steer")
[ -n "$CN" ] && groups+=("sw5:cnours,cnpooled:" "sw6:cnfiction,cnfictf:" "sw7:cnpooltf,cnpoolmix:")
# CN_ONLY=1: only the CreativityNeuro groups (for main-table models that already have the rest)
[ -n "$CN_ONLY" ] && groups=("sw5:cnours,cnpooled:" "sw6:cnfiction,cnfictf:" "sw7:cnpooltf,cnpoolmix:")
# MASKS_ONLY=1: the six mask families only (for the ratio-contrast variant on main-table models)
[ -n "$MASKS_ONLY" ] && groups=("sw1:ours,pooled:" "sw2:fiction,fictf:" "sw3:pooltf,poolmix:")
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}_${S}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
for g in "${groups[@]}"; do
  IFS=':' read -r tag fams extra kind <<< "$g"
  arms=""
  if [ "$kind" = steer ]; then
    arms="unsteered,prompt,prompt2,prompt3"; for L in ${LAYERS//,/ }; do for r in -0.18 -0.5; do arms="$arms,asrL${L}_$r"; done; done
  else
    for fam in ${fams//,/ }; do for st in $ST; do arms="$arms,$fam$st"; done; done; arms=${arms#,}
    [ -n "$extra" ] && arms="$arms,$extra"
  fi
  R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir ${CACHE:-$HF_HOME} --out-dir $O --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $arms --as-layers $LAYERS --as-rel-coeffs=-0.18,-0.5 --batch-size ${BS:-8} --score-batch 2 --contrast ${CONTRAST:-diff}"
  [ "${CONTRAST:-diff}" = ratio ] && tag=rc${tag#sw}   # ratio-contrast runs get tags rc1..rc3 (rt* = response-token masks)
  step ${tag}_lb $O/liars_bench_honesty_${S}_${tag}.csv $R --splits instructed-deception,instructed-truthful --n-id 537 --tag $tag
  step ${tag}_ail $O/liars_bench_honesty_${S}_${tag}ail.csv $R --eval-set $B/extra_data/extra_ailiar.jsonl --belief-tag ailiar --tag ${tag}ail
  step ${tag}_judge $O/liars_bench_honesty_${S}_${tag}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    python judge_claims.py --responses liars_bench_honesty_${S}_${tag}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir /work/hdd/bfqt/xiaocong/hf_cache --eval-set $B/liars_bench/eval_set.jsonl
done
echo "$(date '+%F %T') all done"
