#!/bin/bash
# Appendix refresh after the Table 1 fill (owner, 2026-10-07): the newly chosen options (Falcon3-3B lie mask a=-0.15 and honest
# prompt 5; OLMo-2-1B steering L12 r=-0.1) on plain MMLU, the three attacks (+ Qwen2.5-7B judge), the legitimate requests, and
# for Falcon3-3B's lie mask (its new LIARS' BENCH comparator) two more sampling seeds (+ judge). timan1 GPU1 shared with dagformer
# (coordinator xiaocong-76): <=30 GB, every step under `timeout` up to DEADLINE (2 h after start). Outputs: $T/fill_out.
T=/srv/local/xy51/lie; F=$T/fill_out; PY=${PY:-/home/xy51/anaconda3/envs/modularity/bin/python}
DEADLINE=${DEADLINE:-$(date -d '+115 min' '+%F %T')}
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 LIE_MAX_GPU_GB=30
export HF_HOME=/srv/local/xy51/hf_models HF_HUB_CACHE=/srv/local/xy51/hf_models/hub HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1; unset TRANSFORMERS_CACHE HUGGINGFACE_HUB_CACHE
export TMPDIR=/srv/local/xy51/tmp XDG_CACHE_HOME=/srv/local/xy51/cache
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LIE_OUT=$F LIE_CONTROLS=$T/controls
cd $T/fill_repo; X=$F/extra_data
echo "$(date '+%F %T') fill2 on $(hostname -s) GPU $CUDA_VISIBLE_DEVICES pid $$, deadline $DEADLINE"
step() { local n=$1 done=$2; shift 2; [ -e "$done" ] && { echo "$(date '+%F %T') skip $n"; return; }
  local left=$(( $(date -d "$DEADLINE" +%s) - $(date +%s) )); [ $left -le 60 ] && { echo "$(date '+%F %T') deadline: skip $n"; return; }
  echo "$(date '+%F %T') start $n (${left}s left)"; timeout $left nice -n 10 "$@" > $T/logs/${n}.log 2>&1; local rc=$?; echo "$(date '+%F %T') end $n rc=$rc"; }
J="$PY judge_claims.py --judge Qwen/Qwen2.5-7B-Instruct --batch-size 8 --tag v2 --cache-dir $T/hf"
# Falcon3-3B: lie mask a=-0.15 (ours015), honest prompt 5
M=tiiuae/Falcon3-3B-Instruct; S=Falcon3-3B-Instruct
R="$PY run_liars_bench_honesty.py --model-id $M --cache-dir $HF_HUB_CACHE --out-dir $F --rho 0.04 --ratio 0.37 --batch-size 8 --score-batch 2"
step mmlu_$S $F/liars_bench_honesty_${S}_tsandbagfill.csv $R --arms ours015,prompt5 --eval-set $X/extra_sandbag.jsonl --splits mmlu-plain --belief-tag sandbag --tag tsandbagfill
step atk_$S $F/liars_bench_honesty_${S}_atkfill.csv $R --arms prompt5 --eval-set $X/extra_attack.jsonl --belief-tag attack --tag atkfill
step atkj_$S $F/liars_bench_honesty_${S}_atkfill_judged_Qwen2p5-7B-Instruct_v2.jsonl $J --responses liars_bench_honesty_${S}_atkfill_responses.jsonl --eval-set $X/extra_attack.jsonl --splits attack-manyshot,attack-es,attack-zh --beliefs $F/liars_bench/beliefs_${S}_attack.jsonl
step legit_$S $F/liars_bench_honesty_${S}_legitfill.csv $R --arms prompt5 --eval-set $X/extra_legit.jsonl --tag legitfill
# OLMo-2-1B: steering layer 12, r=-0.1
M=allenai/OLMo-2-0425-1B-Instruct; S=OLMo-2-0425-1B-Instruct
R="$PY run_liars_bench_honesty.py --model-id $M --cache-dir $HF_HUB_CACHE --out-dir $F --rho 0.04 --ratio 0.37 --batch-size 8 --score-batch 2 --as-layers 12 --as-rel-coeffs=-0.1"
step mmlu_$S $F/liars_bench_honesty_${S}_tsandbagfill.csv $R --arms asr-0.1 --eval-set $X/extra_sandbag.jsonl --splits mmlu-plain --belief-tag sandbag --tag tsandbagfill
step atk_$S $F/liars_bench_honesty_${S}_atkfill.csv $R --arms asr-0.1 --eval-set $X/extra_attack.jsonl --belief-tag attack --tag atkfill
step atkj_$S $F/liars_bench_honesty_${S}_atkfill_judged_Qwen2p5-7B-Instruct_v2.jsonl $J --responses liars_bench_honesty_${S}_atkfill_responses.jsonl --eval-set $X/extra_attack.jsonl --splits attack-manyshot,attack-es,attack-zh --beliefs $F/liars_bench/beliefs_${S}_attack.jsonl
step legit_$S $F/liars_bench_honesty_${S}_legitfill.csv $R --arms asr-0.1 --eval-set $X/extra_legit.jsonl --tag legitfill
# Falcon3-3B lie mask under sampling seeds 1 and 2 (LIARS' BENCH), for the seed table
M=tiiuae/Falcon3-3B-Instruct; S=Falcon3-3B-Instruct
R="$PY run_liars_bench_honesty.py --model-id $M --cache-dir $HF_HUB_CACHE --out-dir $F --rho 0.04 --ratio 0.37 --batch-size 8 --score-batch 2"
for SD in 1 2; do
  step seed${SD}_$S $F/liars_bench_honesty_${S}_seed${SD}fill.csv $R --arms ours015 --seed $SD --eval-set $F/liars_bench/eval_set.jsonl --splits instructed-deception,instructed-truthful --n-id 537 --tag seed${SD}fill
  step seed${SD}j_$S $F/liars_bench_honesty_${S}_seed${SD}fill_judged_Qwen2p5-7B-Instruct_v2.jsonl $J --responses liars_bench_honesty_${S}_seed${SD}fill_responses.jsonl --eval-set $F/liars_bench/eval_set.jsonl
done
echo "$(date '+%F %T') all done"
