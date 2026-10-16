#!/bin/bash
#SBATCH --job-name=lie_alphax
#SBATCH --time=08:00:00
#SBATCH --partition=gpuA40x4,gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --array=0-4
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%A_%a.out
# Mask-strength extension for Table 1 (owner, 2026-10-11): most chosen masks (ours and the lie-instruction mask) sat at
# alpha = -1.0, the end of the menu, and the co-author's Qwen2.5-32B run improved with -1.125 / -1.25. Same extension on
# every Table 1 model and for every mask family, ours (fiction, pooled, fictf, pooltf, poolmix) and the lie-instruction
# mask (ours*), so the menus stay identical across models and methods: alpha = -1.125 and -1.25, 12 options per model.
# LIARS' BENCH (deception + truthful control) + AI Liar + the Qwen2.5-7B judge on LIARS' BENCH, tag swalpha(+ail) under
# the sweep dir, which reports/build_final_table.py merges after the main-table tags. Restart-safe.
# Host RAM: one mask costs about 1 byte per weight and the pass builds six families (about 50 GB on Llama-3.1-8B).
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
MODELS=("meta-llama/Meta-Llama-3.1-8B-Instruct" "microsoft/Phi-4-mini-instruct" "tiiuae/Falcon3-3B-Instruct"
        "allenai/OLMo-2-0425-1B-Instruct" "Qwen/Qwen2.5-3B-Instruct")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
ARMS=$(for f in ours fiction pooled fictf pooltf poolmix; do printf "%s112,%s125," $f $f; done); ARMS=${ARMS%,}
S=$(basename $MODEL | tr . p)
T=swalpha
echo "$(date '+%F %T') $SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID on $(hostname -s): $S arms $ARMS"
for f in $B/liars_bench/beliefs_${S}.jsonl $B/liars_bench/beliefs_${S}_ailiar.jsonl; do [ -e "$f" ] && [ ! -e $O/liars_bench/$(basename $f) ] && cp $f $O/liars_bench/; done
R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir $HF_HOME --out-dir $O --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $ARMS --batch-size 16 --score-batch 2"
step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R --splits instructed-deception,instructed-truthful --n-id 537 --tag $T
step ${T}_ail_${S} $O/liars_bench_honesty_${S}_${T}ail.csv $R --eval-set $B/extra_data/extra_ailiar.jsonl --belief-tag ailiar --tag ${T}ail
step ${T}_judge_${S} $O/liars_bench_honesty_${S}_${T}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_${T}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME --eval-set $B/liars_bench/eval_set.jsonl
echo "$(date '+%F %T') all done"
