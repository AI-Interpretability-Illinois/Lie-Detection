#!/bin/bash
#SBATCH --job-name=lie_v2
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# J3 (analysis plan 2026-10-02): the main table with the calibrated mask ("mask v2"). Masks are built the
# format-transfer way: 200 lie / truth pairs from a build source (F2_user, POOL), chat template, r calibrated
# to a positive-score share, full deletion (run_liars_bench_honesty.py --fmt-masks). Same flags as the
# final sweep (queue/make_final_jobs.py): LIARS' BENCH on all items (both halves), AI Liar, the Qwen2.5-7B
# judge on LIARS' BENCH. reports/build_final_table.py selects on validation and reports test.
# Env: MODEL, SPECS (fmt-masks spec, e.g. F2_user:0.0025,F2_user:0.005), TAG (e.g. v2u), BS (batch size).
# Restart-safe: finished steps are skipped.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/hdd/biro/xiaocong/lie_out; X=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
export LIE_OUT=$O FMT_OUT=$O/fmt_suite
cd /u/xiaocong/lie_detection
S=$(basename $MODEL | tr . p)
ARMS=$(python - "$SPECS" <<'EOF'
import sys
out = []
for spec in sys.argv[1].split(","):
    src, share = spec.split(":")
    lab = "pool" if src == "POOL" else ("no" + src[4:].split("_")[1] if src.startswith("LOO-") else src.split("_")[1])
    out.append(f"fm{lab}{round(float(share) * 10000):03d}_100")
print(",".join(out))
EOF
)
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): $MODEL $TAG arms $ARMS"
R="python run_liars_bench_honesty.py --model-id $MODEL --out-dir $O --rho 0.04 --ratio 0.37 --arms $ARMS --fmt-masks $SPECS --batch-size ${BS:-8} --score-batch 2"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/v2_${n}_${S}_${TAG}.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $n rc=$rc"; }
step lb $O/liars_bench_honesty_${S}_${TAG}.csv $R --splits instructed-deception,instructed-truthful --n-id 537 --tag $TAG
step ail $O/liars_bench_honesty_${S}_${TAG}ail.csv $R --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --tag ${TAG}ail
step judge $O/liars_bench_honesty_${S}_${TAG}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_${TAG}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2
echo "$(date '+%F %T') all done"
