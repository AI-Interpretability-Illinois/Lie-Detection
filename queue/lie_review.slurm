#!/bin/bash
#SBATCH --job-name=lie_review
#SBATCH --time=10:00:00
#SBATCH --account=bfqt-delta-gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --output=/work/hdd/bfqt/xiaocong/lie_review/logs/%x_%j.out
# Controls asked for by an ACL-style review of the draft (owner, 2026-10-12). One job = one model (KEY) x one
# experiment (EXP); submitted by queue/submit_review.sh with the memory each model needs.
#   mmlu      plain MMLU (2,000 questions, letter logits) for every option of every method's Table 1 menu, so that
#             honesty can be set against capability option by option (no benchmark generation: honest rates exist)
#   variants  other selection rules at the family Table 1 chose (LB and AIL separately): positive-score weights only,
#             the padding alone, a standardized difference, per-token (length-free) norms, the scale-free ratio, and
#             our rule at r = 0.2 / 0.6; every strength, with MMLU
#   mlsteer   CAA added at several layers at once: the three Table 1 layers, the first half, all layers; 5 strengths
#   seeds     Table 1's chosen option of every method, sampling seeds 1 and 2
# Outputs go to $O (not a sweep dir read by reports/build_final_table.py). Restart-safe: finished steps are skipped.
source ~/.bashrc >/dev/null 2>&1; conda activate modularity
set -o pipefail
export HF_HOME=/work/hdd/bfqt/xiaocong/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=/work/hdd/bfqt/xiaocong/lie_review; SW=/work/nvme/biro/xiaocong/lie_sweep; B=/work/hdd/bfqt/xiaocong/lie_detection_outputs
mkdir -p $O/logs $O/liars_bench
[ -e $O/liars_bench/eval_set.jsonl ] || ln -s $B/liars_bench/eval_set.jsonl $O/liars_bench/eval_set.jsonl
[ -e $O/extra_data ] || ln -s $SW/extra_data $O/extra_data
export LIE_OUT=$O IFEVAL_JSONL=$O/ifeval_541.jsonl   # the IFEval prompts exported from the HF cache (load_dataset fails offline here)
cd /u/xiaocong/lie_detection
EXP=${EXP:?}; KEY=${KEY:?}; SMOKE=${SMOKE:-0}
case $KEY in   # model | steering layers | Table 1 family LB, AIL | chosen arms LB ; AIL (ours, lie-instruction mask, steering, prompt)
  llama)  MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct; L=8,14,21; FLB=fiction; FAIL=pooled
          CLB=fiction125,ours112,asrL8_-0.5,prompt2; CAIL=pooled100,ours112,asrL8_-0.5,prompt3 ;;
  phi)    MODEL=microsoft/Phi-4-mini-instruct; L=8,16,24; FLB=pooled; FAIL=fiction
          CLB=pooled112,ours050,asrL16_-0.18,prompt2; CAIL=fiction100,ours050,asrL16_-0.18,prompt2 ;;
  falcon) MODEL=tiiuae/Falcon3-3B-Instruct; L=5,11,16; FLB=poolmix; FAIL=poolmix
          CLB=poolmix125,ours015,asrL5_-0.3,prompt5; CAIL=poolmix112,ours015,asrL5_-0.18,prompt5 ;;
  olmo)   MODEL=allenai/OLMo-2-0425-1B-Instruct; L=4,8,12; FLB=pooled; FAIL=pooled
          CLB=pooled125,ours125,asrL12_-0.1,prompt; CAIL=pooled125,ours125,asrL12_-0.1,prompt ;;
  qwen3b) MODEL=Qwen/Qwen2.5-3B-Instruct; L=9,18,27; FLB=pooltf; FAIL=poolmix
          CLB=pooltf112,ours100,asrL27_-0.18,prompt; CAIL=poolmix112,ours100,asrL9_-0.5,prompt ;;
  *) echo "unknown model $KEY"; exit 1 ;;
esac
S=$(basename $MODEL | tr . p)
for f in beliefs_${S}.jsonl beliefs_${S}_ailiar.jsonl; do
  [ -e $O/liars_bench/$f ] || cp $SW/liars_bench/$f $O/liars_bench/ 2>/dev/null || cp $B/liars_bench/$f $O/liars_bench/; done
[ -e $O/as_vectors_${S}_ntruth25.pt ] || cp $SW/as_vectors_${S}_ntruth25.pt $O/
# Loading weights from Lustre through mmap page faults ran at 5-25 MB/s when several jobs load at once (a judge took
# 18 min to load, 2026-10-12); one sequential read into the page cache takes 15-60 s.
warm() { for m in "$@"; do cat $HF_HOME/models--${m//\//--}/snapshots/*/*.safetensors > /dev/null 2>&1; done; }
warm $MODEL Qwen/Qwen2.5-7B-Instruct &
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  wait
  echo "$(date '+%F %T') start $n"; "$@" > $O/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
RC="-0.05,-0.1,-0.18,-0.3,-0.5,-0.75,-1"
R="python run_liars_bench_honesty.py --model-id $MODEL --cache-dir $HF_HOME --out-dir $O --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --as-layers $L --as-rel-coeffs=$RC --batch-size 16 --score-batch 2"
LB="--eval-set $O/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537"
AIL="--eval-set $O/extra_data/extra_ailiar.jsonl --belief-tag ailiar"
MM=2000; X=""
if [ "$SMOKE" = 1 ]; then MM=40; X="--n-id 16 --n-idt 16 --n-per-split 16"; LB="$LB --n-idt 16"; fi
judge() { step $1_judge_${S} $O/liars_bench_honesty_${S}_$1_judged_Qwen2p5-7B-Instruct_v2.jsonl \
  python judge_claims.py --responses liars_bench_honesty_${S}_$1_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME --eval-set $O/liars_bench/eval_set.jsonl; }
T=rv${EXP}${CHUNK:-}$([ "$SMOKE" = 1 ] && echo smoke)   # CHUNK: a suffix when one experiment is split over several jobs
echo "$(date '+%F %T') $SLURM_JOB_ID on $(hostname -s) GPU $CUDA_VISIBLE_DEVICES: $S $EXP (tag $T)"
case $EXP in
  mmlu)
    A="unsteered"   # MMLU_FAMS / MMLU_REST=0: split the menu over two jobs (each mask family costs ~1 byte per weight of RAM)
    for f in ${MMLU_FAMS:-ours fiction pooled fictf pooltf poolmix}; do for a in 025 050 075 100 112 125; do A="$A,$f$a"; done; done
    if [ "${MMLU_REST:-1}" = 1 ]; then
    for l in ${L//,/ }; do for r in -0.18 -0.3 -0.5 -0.75 -1; do A="$A,asrL${l}_$r"; done; done
    A="$A,prompt,prompt2,prompt3"; fi
    [ $KEY = falcon ] && A="$A,ours010,ours015,prompt4,prompt5"
    [ $KEY = olmo ] && for l in ${L//,/ }; do A="$A,asrL${l}_-0.05,asrL${l}_-0.1"; done
    [ "$SMOKE" = 1 ] && A="unsteered,pooled100,asrL${L%%,*}_-0.3,prompt"
    step ${T}_${S} $O/liars_bench_honesty_${S}_${T}.csv $R $LB --no-gen --mmlu $MM --arms $A --tag $T ;;
  variants)
    RLB=${VRULES_LB:-pos pad z tok rat r020 r060}; RAIL=${VRULES_AIL:-pos pad z tok}   # splitting: rule subsets per job
    [ "$RLB" = none ] && RLB=""; [ "$RAIL" = none ] && RAIL=""
    if [ -n "$RLB" ]; then
    V=""; for r in $RLB; do V="$V,$r:$FLB"; done; V=${V#,}
    A="unsteered"; for r in $RLB; do for a in 025 050 075 100 112 125; do A="$A,v$r$FLB$a"; done
      [ $r = pos ] && A="$A,vpos${FLB}150,vpos${FLB}200"; [ $r = z ] && A="$A,vz${FLB}005,vz${FLB}010"; done
    [ "$SMOKE" = 1 ] && A="unsteered,vpos${FLB}100,vpad${FLB}100,vz${FLB}050,vtok${FLB}100,vrat${FLB}100,vr060${FLB}050"
    step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R $LB $X --variants $V --mmlu $MM --arms $A --tag $T
    judge $T
    fi
    if [ -n "$RAIL" ]; then
    V=""; for r in $RAIL; do V="$V,$r:$FAIL"; done; V=${V#,}
    A="unsteered"; for r in $RAIL; do for a in 025 050 075 100 112 125; do A="$A,v$r$FAIL$a"; done
      [ $r = pos ] && A="$A,vpos${FAIL}150,vpos${FAIL}200"; [ $r = z ] && A="$A,vz${FAIL}005,vz${FAIL}010"; done
    [ "$SMOKE" = 1 ] && A="unsteered,vpos${FAIL}100,vtok${FAIL}100"
    step ${T}_ail_${S} $O/liars_bench_honesty_${S}_${T}ail.csv $R $AIL $X --variants $V --mmlu $MM --arms $A --tag ${T}ail
    fi ;;
  mlsteer)
    # strengths per layer shrink with the number of layers steered (3, about 14-20, about 28-40)
    A="unsteered"; for r in -0.06 -0.1 -0.18 -0.3 -0.5; do A="$A,mlstrio_$r"; done
    for r in -0.02 -0.03 -0.06 -0.1 -0.18; do A="$A,mlsfirst_$r"; done
    for r in -0.01 -0.02 -0.03 -0.06 -0.1; do A="$A,mlsall_$r"; done
    [ "$SMOKE" = 1 ] && A="unsteered,mlstrio_-0.1,mlsall_-0.03"
    step ${T}_lb_${S} $O/liars_bench_honesty_${S}_${T}.csv $R $LB $X --ml-sets trio,first,all --mmlu $MM --arms $A --tag $T
    judge $T
    step ${T}_ail_${S} $O/liars_bench_honesty_${S}_${T}ail.csv $R $AIL $X --ml-sets trio,first,all --mmlu $MM --arms $A --tag ${T}ail ;;
  seeds)
    for sd in 1 2; do
      step ${T}${sd}_lb_${S} $O/liars_bench_honesty_${S}_${T}${sd}.csv $R $LB $X --seed $sd --arms unsteered,$CLB --tag ${T}${sd}
      judge ${T}${sd}
      step ${T}${sd}_ail_${S} $O/liars_bench_honesty_${S}_${T}${sd}ail.csv $R $AIL $X --seed $sd --arms unsteered,$CAIL --tag ${T}${sd}ail
    done ;;
  tasks)       # the further-task table at the FINAL options (the paper's appendix analyses ran at the initial-menu ones)
    for t in sandbag inject tmask tamgot multiturn; do f=beliefs_${S}_$t.jsonl
      [ -e $O/liars_bench/$f ] || cp $B/liars_bench/$f $O/liars_bench/ 2>/dev/null || cp $SW/liars_bench/$f $O/liars_bench/ 2>/dev/null; done
    A=unsteered,magmatch100,$CLB; XD=$O/extra_data; TS=" ${TSTEPS:-sandbag inject mask amgot multi} "
    [[ $TS == *" sandbag "* ]] && step ${T}sandbag_${S} $O/liars_bench_honesty_${S}_${T}sandbag.csv $R $X --eval-set $XD/extra_sandbag.jsonl --belief-tag sandbag --arms $A --tag ${T}sandbag
    [[ $TS == *" inject "* ]] && step ${T}inject_${S} $O/liars_bench_honesty_${S}_${T}inject.csv $R $X --eval-set $XD/extra_inject.jsonl --belief-tag inject --arms $A --tag ${T}inject
    [[ $TS == *" inject "* ]] && step ${T}injectjudge_${S} $O/liars_bench_honesty_${S}_${T}inject_judged_Qwen2p5-7B-Instruct_v2.jsonl \
      python judge_claims.py --responses liars_bench_honesty_${S}_${T}inject_responses.jsonl --splits inject-deception,inject-control \
      --beliefs $O/liars_bench/beliefs_${S}_inject.jsonl --eval-set $XD/extra_inject.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME
    [[ $TS == *" mask "* ]] && step ${T}mask_${S} $O/liars_bench_honesty_${S}_${T}mask.csv $R --eval-set $XD/extra_mask.jsonl --belief-tag tmask \
      --n-per-split $([ "$SMOKE" = 1 ] && echo 8 || echo 80) --arms $A --tag ${T}mask
    [[ $TS == *" mask "* ]] && step ${T}maskjudge_${S} $B/liars_bench_honesty_${S}_${T}mask_maskjudged_Qwen2p5-7B-Instruct_v1.jsonl \
      python judge_mask.py --responses $O/liars_bench_honesty_${S}_${T}mask_responses.jsonl --beliefs $O/liars_bench/beliefs_${S}_tmask.jsonl \
      --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v1
    [[ $TS == *" amgot "* ]] && step ${T}amgot_${S} $O/liars_bench_honesty_${S}_${T}amgot.csv $R $X --eval-set $XD/extra_amgot.jsonl --belief-tag tamgot --arms $A --tag ${T}amgot
    [[ $TS == *" amgot "* ]] && step ${T}amgotjudge_${S} $O/liars_bench_honesty_${S}_${T}amgot_judged_Qwen2p5-7B-Instruct_v2.jsonl \
      python judge_claims.py --responses liars_bench_honesty_${S}_${T}amgot_responses.jsonl --splits amgot-deception \
      --beliefs $O/liars_bench/beliefs_${S}_tamgot.jsonl --eval-set $XD/extra_amgot.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME
    [[ $TS == *" multi "* ]] && step ${T}multi_${S} $O/liars_bench_honesty_${S}_${T}multi.csv $R $X --eval-set $XD/extra_multiturn.jsonl --belief-tag multiturn --arms $A --tag ${T}multi ;;
  atklegit)    # adaptive attacks and legitimate falsehoods at the FINAL options
    f=beliefs_${S}_attack.jsonl; [ -e $O/liars_bench/$f ] || cp $SW/liars_bench/$f $O/liars_bench/ 2>/dev/null
    A=unsteered,$CLB; XD=$O/extra_data
    step ${T}atk_${S} $O/liars_bench_honesty_${S}_${T}atk.csv $R $X --eval-set $XD/extra_attack.jsonl --belief-tag attack --arms $A --tag ${T}atk
    step ${T}atkjudge_${S} $O/liars_bench_honesty_${S}_${T}atk_judged_Qwen2p5-7B-Instruct_v2.jsonl \
      python judge_claims.py --responses liars_bench_honesty_${S}_${T}atk_responses.jsonl --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $HF_HOME \
      --eval-set $XD/extra_attack.jsonl --splits attack-manyshot,attack-es,attack-zh --beliefs $O/liars_bench/beliefs_${S}_attack.jsonl
    step ${T}legit_${S} $O/liars_bench_honesty_${S}_${T}legit.csv $R $X --eval-set $XD/extra_legit.jsonl --arms $A --tag ${T}legit ;;
  ifeval)      # general instruction following (IFEval, 541 prompts) at Table 1's LIARS' BENCH-chosen options
    N=541; [ "$SMOKE" = 1 ] && N=8
    step ${T}_${S} $O/liars_bench_honesty_${S}_${T}.csv $R $LB --no-gen --ifeval $N --arms $(A=${IFEVAL_ARMS:-unsteered,$CLB}; echo ${A//:/,}) --tag $T ;;   # IFEVAL_ARMS: colon-separated (sbatch --export splits commas)
  *) echo "unknown experiment $EXP"; exit 1 ;;
esac
echo "$(date '+%F %T') all done"
