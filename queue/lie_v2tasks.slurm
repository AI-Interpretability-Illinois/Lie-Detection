#!/bin/bash
#SBATCH --job-name=lie_v2tasks
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# J4 (analysis plan 2026-10-02): the calibrated masks chosen by build_final_table.py --v2 on the further tasks
# that exposed the 25-prompt mask's costs: plain MMLU + sandbagging (extra_sandbag.jsonl), the AI-Liar
# multi-turn set, and the injected-instruction set; then the judge on the injected set (true/false).
# Env: MODEL, SPECS (fmt-masks spec), ARMS (arm keys), BS. Restart-safe.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/hdd/biro/xiaocong/lie_out; X=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
export LIE_OUT=$O FMT_OUT=$O/fmt_suite
cd /u/xiaocong/lie_detection
S=$(basename $MODEL | tr . p)
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): $MODEL arms $ARMS"
R="python run_liars_bench_honesty.py --model-id $MODEL --out-dir $O --rho 0.04 --ratio 0.37 --arms $ARMS --fmt-masks $SPECS --batch-size ${BS:-8} --score-batch 2"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/v2tasks_${n}_${S}.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $n rc=$rc"; }
step sandbag $O/liars_bench_honesty_${S}_v2sandbag.csv $R --eval-set $X/extra_sandbag.jsonl --belief-tag sandbag --tag v2sandbag
step multi   $O/liars_bench_honesty_${S}_v2multi.csv   $R --eval-set $X/extra_multiturn.jsonl --belief-tag multiturn --tag v2multi
step inject  $O/liars_bench_honesty_${S}_v2inject.csv  $R --eval-set $X/extra_inject.jsonl --belief-tag inject --tag v2inject
step judge $O/liars_bench_honesty_${S}_v2inject_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_v2inject_responses.jsonl --eval-set $X/extra_inject.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2
echo "$(date '+%F %T') all done"
