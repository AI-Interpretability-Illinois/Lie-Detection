#!/bin/bash
#SBATCH --job-name=lie_fill
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%j.out
# Fill the empty cells of Table 1 (owner, 2026-10-06): milder options for the three comparator cells where no option
# passed the usability gates. OLMo-2-1B steering: r = -0.05, -0.1 at the same three layers (4, 8, 12); Falcon3-3B lie-
# instruction mask: alpha = -0.10, -0.15; Falcon3-3B honest prompt: two new wordings (prompt4, prompt5). LIARS' BENCH
# (all items) + AI Liar + the Qwen2.5-7B judge on LIARS' BENCH, tag swfill(+ail) under the sweep dir, which
# reports/build_final_table.py merges after the main-table tags. Restart-safe.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/nvme/biro/xiaocong/lie_sweep; mkdir -p $O/logs $O/liars_bench; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
[ -e $O/extra_data ] || ln -s $B/extra_data $O/extra_data
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s)"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
SPECS="allenai/OLMo-2-0425-1B-Instruct|asrL4_-0.05,asrL4_-0.1,asrL8_-0.05,asrL8_-0.1,asrL12_-0.05,asrL12_-0.1|4,8,12
tiiuae/Falcon3-3B-Instruct|ours010,ours015,prompt4,prompt5|5"
T=swfill
while IFS='|' read -r MODEL ARMS LAYERS; do
  S=$(basename $MODEL | tr . p)
  for f in $B/liars_bench/beliefs_${S}.jsonl $B/liars_bench/beliefs_${S}_ailiar.jsonl; do [ -e "$f" ] && [ ! -e $O/liars_bench/$(basename $f) ] && cp $f $O/liars_bench/; done
  R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir $HF_HOME --out-dir $O --rho 0.04 --ratio 0.37 --arms $ARMS --as-layers $LAYERS --as-rel-coeffs=-0.05,-0.1 --batch-size 8 --score-batch 2"
  step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R --splits instructed-deception,instructed-truthful --n-id 537 --tag $T
  step ${T}_ail_${S} $O/liars_bench_honesty_${S}_${T}ail.csv $R --eval-set $B/extra_data/extra_ailiar.jsonl --belief-tag ailiar --tag ${T}ail
  step ${T}_judge_${S} $O/liars_bench_honesty_${S}_${T}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    python judge_claims.py --responses liars_bench_honesty_${S}_${T}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME --eval-set $B/liars_bench/eval_set.jsonl
done <<< "$SPECS"
echo "$(date '+%F %T') all done"
