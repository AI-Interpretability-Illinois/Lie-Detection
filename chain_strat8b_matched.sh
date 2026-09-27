#!/bin/bash
# Follow-up to the wpct=10 stratified run on Llama-8B.
#
# At rho = 0.04 the stratified arms all land at 0.092x the global median |W| (the
# bottom-10% stratum's own median), while the plain single-difference mask that
# actually works sits at 0.042x. So the wpct=10 comparison is not magnitude-matched
# to the working mask, and tightening the stratum to ~4% would make the selection
# degenerate: picking 4% of the model out of a 4% stratum means taking the whole
# stratum and letting the contrast choose nothing.
#
# The way out is a smaller budget. At rho = 0.01 with wpct = 5 the mask picks a
# fifth of the stratum, so the contrast does the choosing AND the magnitude lands
# near the plain mask's. Plain arms are rebuilt at the same rho for comparison.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-2}
while pgrep -u "$USER" -f "run_stratified_did2.py --model-id meta-llam[a].*--wpct 10" > /dev/null; do
  sleep 60
done
echo "=== $(date '+%F %H:%M') starting magnitude-matched stratified run"
nice -n 10 python run_stratified_did2.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.01 --ratio 0.37 --wpct 5 \
  --damp=-1.0 --amp=3.0 \
  --batch-size 12 --score-batch 2 --tag matched
echo "=== $(date '+%F %H:%M') exit $?"
