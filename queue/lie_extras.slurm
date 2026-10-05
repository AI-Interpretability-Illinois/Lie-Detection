#!/bin/bash
#SBATCH --job-name=lie_extras
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --output=/work/nvme/biro/xiaocong/logs/%x_%j.out
# Extras plan item 5 on one main-table model: the adaptive-attack set (extra_attack.jsonl: many-shot, Spanish and Chinese
# orders over the 537 LIARS' BENCH deception items) and the legitimate-falsehood set (extra_legit.jsonl: fiction /
# role-play / counterfactual requests), each under the LIARS' BENCH-chosen arms (ARMS, e.g. unsteered,fiction100,asrL8_-0.5,prompt2;
# LAYERS and RC give the steering layer and relative coefficient). Attack replies are graded by the Qwen2.5-7B judge as on
# LIARS' BENCH (beliefs probed once, --belief-tag attack); legit replies are saved for judge_legit.py.
# Env: MODEL, ARMS, LAYERS, RC, HF (cache dir), BS. Outputs under /work/nvme/biro/xiaocong/lie_sweep (tags atk, legit).
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
export HF_HOME=${HF:-/work/hdd/bfqt/xiaocong/hf_cache} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/nvme/biro/xiaocong/lie_sweep; mkdir -p $O/logs $O/liars_bench; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
[ -e $O/extra_data ] || ln -s $B/extra_data $O/extra_data
export LIE_OUT=$O
cd /u/xiaocong/lie_detection
S=$(basename $MODEL | tr . p)
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): $MODEL arms $ARMS"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}_${S}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir ${CACHE:-$HF_HOME} --out-dir $O --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $ARMS --as-layers $LAYERS --as-rel-coeffs=$RC --batch-size ${BS:-8} --score-batch 2"
step atk_gen $O/liars_bench_honesty_${S}_atk.csv $R --eval-set $B/extra_data/extra_attack.jsonl --belief-tag attack --tag atk
step atk_judge $O/liars_bench_honesty_${S}_atk_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_atk_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir /work/hdd/bfqt/xiaocong/hf_cache \
  --eval-set $B/extra_data/extra_attack.jsonl --splits attack-manyshot,attack-es,attack-zh --beliefs $O/liars_bench/beliefs_${S}_attack.jsonl
step legit_gen $O/liars_bench_honesty_${S}_legit.csv $R --eval-set $B/extra_data/extra_legit.jsonl --tag legit
echo "$(date '+%F %T') all done $MODEL"
