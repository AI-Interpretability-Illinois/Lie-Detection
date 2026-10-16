#!/bin/bash
#SBATCH --job-name=lie_steerx
#SBATCH --time=06:00:00
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --array=0-4
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%A_%a.out
# Steering-strength extension for Table 1 (owner, 2026-10-11): on Llama-3.1-8B the chosen steering option (L8, r=-0.5) was
# the strongest strength on the menu and passed every usability check with room to spare, so the best steering setting could
# lie beyond the menu. Same extension on every Table 1 model, at its same three layers: r = -0.3 (between the two original
# strengths), -0.75 and -1.0. LIARS' BENCH (deception + truthful control) + AI Liar + the Qwen2.5-7B judge on LIARS' BENCH,
# tag swsteer(+ail) under the sweep dir, which reports/build_final_table.py merges after the main-table tags. Restart-safe.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/nvme/biro/xiaocong/lie_sweep; mkdir -p $O/logs $O/liars_bench; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
[ -e $O/extra_data ] || ln -s $B/extra_data $O/extra_data
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
SPECS=("meta-llama/Meta-Llama-3.1-8B-Instruct|8,14,21"
       "microsoft/Phi-4-mini-instruct|8,16,24"
       "tiiuae/Falcon3-3B-Instruct|5,11,16"
       "allenai/OLMo-2-0425-1B-Instruct|4,8,12"
       "Qwen/Qwen2.5-3B-Instruct|9,18,27")
RS="-0.3,-0.75,-1"
IFS='|' read -r MODEL LAYERS <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
ARMS=$(for L in ${LAYERS//,/ }; do for r in ${RS//,/ }; do printf "asrL%s_%s," $L $r; done; done); ARMS=${ARMS%,}
S=$(basename $MODEL | tr . p)
T=swsteer
echo "$(date '+%F %T') $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname -s): $S arms $ARMS"
for f in $B/liars_bench/beliefs_${S}.jsonl $B/liars_bench/beliefs_${S}_ailiar.jsonl; do [ -e "$f" ] && [ ! -e $O/liars_bench/$(basename $f) ] && cp $f $O/liars_bench/; done
R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir $HF_HOME --out-dir $O --rho 0.04 --ratio 0.37 --arms $ARMS --as-layers $LAYERS --as-rel-coeffs=$RS --batch-size 16 --score-batch 2"
step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R --splits instructed-deception,instructed-truthful --n-id 537 --tag $T
step ${T}_ail_${S} $O/liars_bench_honesty_${S}_${T}ail.csv $R --eval-set $B/extra_data/extra_ailiar.jsonl --belief-tag ailiar --tag ${T}ail
step ${T}_judge_${S} $O/liars_bench_honesty_${S}_${T}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_${T}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME --eval-set $B/liars_bench/eval_set.jsonl
echo "$(date '+%F %T') all done"
