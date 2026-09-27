#!/bin/bash
# Head-to-head on the instruction-compliance pressure metric: the DID mask fully
# ablated against a DEPTH-SWEPT difference-in-means baseline. This is the one
# comparison never run: the baseline's layer was swept on LIARS' BENCH but never
# on this metric, which is where the mask's effect is largest.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=${GPU:-1}
cd /u/xiaocong/lie_detection
nice -n 10 python run_prior_work_baselines.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct --rho 0.04 --ratio 0.37 \
  --variants score_diff_instruct --damp="-1.0" --amp="3.0" \
  --as-layers "4,8,14,21,26" --as-rel-coeffs="-0.06,-0.18,-0.5" \
  --batch-size 12 --score-batch 2 --tag llama8b_h2h
