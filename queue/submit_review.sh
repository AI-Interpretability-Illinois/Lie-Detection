#!/bin/bash
# Submit the review-control batch (queue/lie_review.slurm) to reservation sup-30781, longest jobs first.
cd /u/xiaocong/lie_detection
for E in variants tasks mlsteer atklegit mmlu seeds; do
  for K in llama phi falcon qwen3b olmo; do
    M=48G; [ $K = llama ] && { M=64G; [ $E = mmlu ] && M=96G; }
    T=10:00:00; [ $K != llama ] && T=08:00:00
    j=$(sbatch --parsable --reservation=sup-30781 --partition=gpuA100x4 --mem=$M --time=$T \
        --job-name=lie_rv_${E}_$K --export=ALL,EXP=$E,KEY=$K queue/lie_review.slurm)
    echo "$E $K $j"
  done
done
