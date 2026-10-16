#!/bin/bash
source ~/.bashrc >/dev/null 2>&1; conda activate modularity 2>/dev/null
cd /u/xiaocong/lie_detection
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${GPU:-1} LIE_MAX_GPU_GB=12 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=Qwen/Qwen2.5-0.5B-Instruct; X=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
F="Warning|warnings.warn|Loading checkpoint|torch_dtype|Using the latest|Found the latest|norms\["
for t in sandbag inject multiturn; do
  echo "##### $t"
  python run_liars_bench_honesty.py --model-id $M --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms \
    --arms unsteered,fiction100,prompt2 --as-layers 8,16 --as-rel-coeffs=-0.18 --eval-set $X/extra_$t.jsonl \
    --belief-tag smoke_$t --n-per-split 12 --batch-size 16 --score-batch 4 --tag smoke_$t 2>&1 | grep -vE "$F" | grep -E "coverage|beliefs\[|^  (unsteered|weight|prompt)|^      |Traceback|Error" | cut -c1-150
done
echo "##### inject judge"
python judge_claims.py --responses liars_bench_honesty_Qwen2p5-0p5B-Instruct_smoke_inject_responses.jsonl \
  --splits inject-deception,inject-control --beliefs /work/hdd/bfqt/xiaocong/lie_detection_outputs/liars_bench/beliefs_Qwen2p5-0p5B-Instruct_smoke_inject.jsonl \
  --eval-set $X/extra_inject.jsonl --judge Qwen/Qwen2.5-1.5B-Instruct --batch-size 8 --tag smoke 2>&1 | grep -vE "$F" | tail -8
echo "##### SMOKE DONE"
