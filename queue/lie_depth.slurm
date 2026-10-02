#!/bin/bash
#SBATCH --job-name=lie_depth
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/work/hdd/biro/xiaocong/lie_out/logs/lie_depth_%x_%j.out
# Ideas from Qi et al. (ICLR 2025) for one model: (3) is the current edit shallow? (KL by response
# position); (1) resp and (2) recov masks from response-token contrasts, evaluated with the current
# mask on the LIARS' BENCH, AI Liar and multi-turn validation halves; LB judged.
# Env: MODEL, FAM (chosen LB family), ST (strength key, e.g. 100), LBRESP (unedited LB responses file).
# no "set -u": ~/.bashrc and conda activate reference unset variables and would end the job silently
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s): $MODEL $FAM$ST; python $(which python)"
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUTD=/work/hdd/biro/xiaocong/lie_out; X=/work/hdd/bfqt/xiaocong/lie_detection_outputs/extra_data
export LIE_OUT=$OUTD
cd /u/xiaocong/lie_detection
S=$(basename $MODEL | tr . p)
DES=resp:0.04:a0.0025,resp:0.04:a0.005,resp:0.04:a0.01,recov:0.04:a0.0025,recov:0.04:a0.005,recov:0.04:a0.01
ARMS=unsteered,$FAM$ST,resp040a025_100,resp040a050_100,resp040a100_100,recov040a025_100,recov040a050_100,recov040a100_100
R="python run_liars_bench_honesty.py --model-id $MODEL --out-dir $OUTD --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --arms $ARMS --designs $DES --half val --batch-size ${BS:-8} --score-batch 2"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  echo "$(date '+%F %T') start $n"; "$@" > $OUTD/logs/${n}_$S.log 2>&1; local rc=$?
  echo "$(date '+%F %T') end $n rc=$rc"; }
step kl $OUTD/kl_depth_$S.json python kl_depth.py --model-id $MODEL --lb-responses $LBRESP --family $FAM --alpha -1 --out $OUTD
step lb $OUTD/liars_bench_honesty_${S}_depth.csv $R --splits instructed-deception,instructed-truthful --n-id 537 --tag depth
step ail $OUTD/liars_bench_honesty_${S}_depthail.csv $R --eval-set $X/extra_ailiar.jsonl --belief-tag ailiar --tag depthail
step mt $OUTD/liars_bench_honesty_${S}_depthmt.csv $R --eval-set $X/extra_multiturn.jsonl --belief-tag multiturn --tag depthmt
step judge $OUTD/liars_bench_honesty_${S}_depth_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_depth_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2
echo "$(date '+%F %T') all done"
