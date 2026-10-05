#!/bin/bash
#SBATCH --job-name=lie_seed
#SBATCH --time=07:00:00
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%j.out
# Sampling-seed replication of Table 1 (Limitations: "one seed per configuration"): the LIARS' BENCH-chosen arms of every
# main-table model regenerated with --seed SEED (torch seed for the sampled replies; masks and steering vectors are
# deterministic), then judged by the Qwen2.5-7B hybrid as on LIARS' BENCH. One job runs all seven models in sequence so
# that at most one lie_seed job per seed occupies a GPU. Outputs under /work/nvme/biro/xiaocong/lie_sweep, tag seed<SEED>;
# reports/seed_report.py compares the test-half rates across seeds 0 (the paper's run), 1 and 2.
# Env: SEED (required), MODELS (optional subset, comma-separated model ids), HF (cache dir), BS.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=${HF:-/work/hdd/bfqt/xiaocong/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/nvme/biro/xiaocong/lie_sweep; mkdir -p $O/logs $O/liars_bench; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
[ -n "$SEED" ] || { echo "SEED unset"; exit 1; }
# model id | arms (LIARS' BENCH-chosen: unedited, lie mask, falsehood mask, steering, honest prompt) | steering layer | rel. coeff
SPECS="meta-llama/Meta-Llama-3.1-8B-Instruct|unsteered,ours100,fiction100,asr-0.5,prompt2|8|-0.5
microsoft/Phi-4-mini-instruct|unsteered,ours050,pooled100,asr-0.18,prompt2|16|-0.18
tiiuae/Falcon3-3B-Instruct|unsteered,ours025,fictf100,asr-0.18,prompt2|5|-0.18
allenai/OLMo-2-0425-1B-Instruct|unsteered,ours100,fiction025,asr-0.5,prompt|12|-0.5
Qwen/Qwen2.5-3B-Instruct|unsteered,ours100,pooltf100,asr-0.18,prompt|27|-0.18
Qwen/Qwen2.5-7B-Instruct|unsteered,ours100,pooled100,asr-0.18,prompt|14|-0.18
Qwen/Qwen2.5-1.5B-Instruct|unsteered,ours100,fictf100,asr-0.18,prompt3|14|-0.18"
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): seed $SEED models ${MODELS:-all}"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
while IFS='|' read -r MODEL ARMS LAYERS RC; do
  [ -z "$MODEL" ] && continue
  if [ -n "$MODELS" ] && ! echo ",$MODELS," | grep -q ",$MODEL,"; then continue; fi
  S=$(basename $MODEL | tr . p); T=seed$SEED
  R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir ${CACHE:-$HF_HOME} --out-dir $O --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $ARMS --as-layers $LAYERS --as-rel-coeffs=$RC --batch-size ${BS:-8} --score-batch 2 --seed $SEED"
  step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R --eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag $T
  step ${T}_judge_${S} $O/liars_bench_honesty_${S}_${T}_judged_Qwen2p5-7B-Instruct_v2.jsonl \
    python judge_claims.py --responses liars_bench_honesty_${S}_${T}_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir /work/hdd/bfqt/xiaocong/hf_cache --eval-set $O/liars_bench/eval_set.jsonl
done <<< "$SPECS"
echo "$(date '+%F %T') all done seed $SEED"
