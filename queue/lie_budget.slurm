#!/bin/bash
#SBATCH --job-name=lie_budget
#SBATCH --time=04:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%j.out
# Mask-budget sweep (extras plan item 4, "why Qwen differs"): the contrast threshold r sets how many channels carry a
# positive score (n_F / n_T > 1/r); at r = 0.37 that is 0.05-0.8% of the weights and the rest of the 4% is padding, so
# the budget knob is r, not rho. Runs the model's chosen family at four strengths for R in {0.2, 0.6, 1.0} (0.37 is the
# main table), LIARS' BENCH + judge, tags rb020 / rb060 / rb100 under /work/nvme/biro/xiaocong/lie_sweep.
# Env: MODEL, FAM (pooled / fiction / ...), HF (cache dir), BS. Replaces the ratio-contrast jobs, whose positive set is
# identical to the difference contrast's by construction (c_j > 0 <=> n_F/n_T > 1/r).
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=${HF:-/work/hdd/bfqt/xiaocong/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/nvme/biro/xiaocong/lie_sweep; mkdir -p $O/logs $O/liars_bench; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
[ -e $O/extra_data ] || ln -s $B/extra_data $O/extra_data
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
S=$(basename $MODEL | tr . p)
for f in $B/liars_bench/beliefs_${S}.jsonl; do [ -e "$f" ] && [ ! -e $O/liars_bench/$(basename $f) ] && cp $f $O/liars_bench/; done
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): $MODEL family $FAM"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}_${S}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
arms="${FAM}025,${FAM}050,${FAM}075,${FAM}100"
for r in 0.2 0.6 1.0; do
  tag=rb$(printf "%03d" $(python -c "print(round($r*100))"))
  step ${tag}_lb $O/liars_bench_honesty_${S}_${tag}.csv python run_liars_bench_honesty.py --model-id $MODEL --out-dir $O --rho 0.04 --ratio $r \
    --falsehood-arms --polarity-arms --arms $arms --batch-size ${BS:-8} --score-batch 2 \
    --eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $tag
  step ${tag}_judge $O/liars_bench_honesty_${S}_${tag}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    python judge_claims.py --responses liars_bench_honesty_${S}_${tag}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME --eval-set $O/liars_bench/eval_set.jsonl
done
echo "$(date '+%F %T') all done $MODEL"
