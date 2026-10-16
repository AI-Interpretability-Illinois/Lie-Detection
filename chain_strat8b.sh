#!/bin/bash
# Wait for the GPU-2 instructed-falsehood run to finish, then rerun the
# stratified second-difference test on the same model, where there is 60.9% of
# held-out lying as headroom (Qwen-1.5B has 8.5%, so its flat result proves little).
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=2
# bracket in the pattern so this script's own command line does not match
while pgrep -u "$USER" -f "run_instructed_falsehood.py --model-id meta-llam[a]" > /dev/null; do
  sleep 60
done
echo "=== $(date '+%F %H:%M') GPU 2 free, starting stratified rerun"
nice -n 10 python run_stratified_did2.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --wpct 10 \
  --damp=-1.0 --amp=3.0 \
  --batch-size 12 --score-batch 2 --tag idx
echo "=== $(date '+%F %H:%M') exit $?"
