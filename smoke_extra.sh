#!/bin/bash
# End-to-end smoke test of the three extra evaluation sets on Qwen2.5-0.5B.
source ~/.bashrc >/dev/null 2>&1
conda activate modularity 2>/dev/null
set -u
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-2}
M=Qwen/Qwen2.5-0.5B-Instruct; X=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
F="grep -vE Warning|warnings.warn|Loading.checkpoint|torch_dtype"
for s in ailiar mask amgot; do
  echo "########## generate: $s"
  python run_liars_bench_honesty.py --model-id $M --rho 0.04 --ratio 0.37 --falsehood-arms \
    --eval-set $X/extra_$s.jsonl --belief-tag smoke_$s --n-per-split 10 \
    --arms unsteered,pooled100,asrL8_-0.18 --as-layers 8,16 --as-rel-coeffs=-0.18 \
    --batch-size 16 --score-batch 4 --tag smoke_$s 2>&1 | grep -vE "Warning|warnings.warn|Loading checkpoint|torch_dtype" | tail -22
done
B=/work/hdd/bfqt/xiaocong/lie_detection_outputs/liars_bench
echo "########## judge: mask"
python judge_mask.py --responses liars_bench_honesty_Qwen2p5-0p5B-Instruct_smoke_mask_responses.jsonl \
  --beliefs $B/beliefs_Qwen2p5-0p5B-Instruct_smoke_mask.jsonl --judge Qwen/Qwen2.5-1.5B-Instruct \
  --tag smoke 2>&1 | grep -vE "Warning|warnings.warn|Loading checkpoint|torch_dtype"
echo "########## judge: amgot"
python judge_claims.py --responses liars_bench_honesty_Qwen2p5-0p5B-Instruct_smoke_amgot_responses.jsonl \
  --splits amgot-deception --beliefs $B/beliefs_Qwen2p5-0p5B-Instruct_smoke_amgot.jsonl \
  --judge Qwen/Qwen2.5-1.5B-Instruct --tag smoke 2>&1 | grep -vE "Warning|warnings.warn|Loading checkpoint|torch_dtype"
echo "########## SMOKE DONE"
