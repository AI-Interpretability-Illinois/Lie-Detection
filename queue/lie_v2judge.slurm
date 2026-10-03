#!/bin/bash
#SBATCH --job-name=lie_v2judge
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/%x_%j.out
# Re-run of J4's injected-instruction judge with the task's split names and belief file (the first pass
# filtered on LIARS' BENCH split names and graded nothing).
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/hdd/biro/xiaocong/lie_out; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs; X=$B/extra_data
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
for S in Meta-Llama-3p1-8B-Instruct Phi-4-mini-instruct Qwen2p5-3B-Instruct Qwen2p5-7B-Instruct; do
  echo "$(date '+%F %T') start $S"
  python judge_claims.py --responses liars_bench_honesty_${S}_v2inject_responses.jsonl --splits inject-deception,inject-control \
    --beliefs $B/liars_bench/beliefs_${S}_inject.jsonl --eval-set $X/extra_inject.jsonl --judge Qwen/Qwen2.5-7B-Instruct \
    --batch-size 8 --tag v2 > $O/logs/v2judge_$S.log 2>&1
  echo "$(date '+%F %T') end $S rc=$?"
done
echo "$(date '+%F %T') all done"
