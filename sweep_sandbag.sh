#!/usr/bin/env bash
# The same recipe, built from orders to sandbag: does the method carry beyond lying when the mask is rebuilt
# from the new order's own contrast? (docs/sandbag_mask.md)
#
# For each Table 1 model, one pass of run_liars_bench_honesty.py --sandbag-arms on the sandbagging evaluation
# (data/sandbag/extra_sandbag.jsonl: 300 MMLU + 150 WMDP items under four sandbagging orders, and 200 MMLU items
# asked plainly as the capability control), with every option of every method, so that the setting of each
# method can be chosen on the validation half by reports/sandbag_report.py, exactly as for Table 1:
#   ours            sbag / sbpool masks at alpha -0.25, -0.5, -0.75, -1 (8 options)
#   steering (CAA)  vectors from the same sandbag prompts, three layers x r in {-0.18,-0.3,-0.5,-0.75,-1} (15)
#   honest prompt   the three Table 1 wordings and two aimed at sandbagging (5)
#   unedited
#   references, never chosen: the magnitude-matched random mask of each family, and the lie-built mask at its
#   Table 1 setting (what the paper's transfer table already measures for sandbagging)
#
# Models are dealt round-robin across GPUS; each lane runs its models in sequence.
#   GPUS=0,1,2,3 bash sweep_sandbag.sh                    # all five models
#   GPUS=0 MODELS="olmo qwen3b" bash sweep_sandbag.sh     # a subset
#   GPUS=0 MODELS=olmo EXTRA="--n-per-split 20" TAGX=smoke bash sweep_sandbag.sh   # smoke test, 20 items per split
#   DRY=1 bash sweep_sandbag.sh                           # print the commands, run nothing
# Paths are env-overridable: REPO, PY, CACHE (HF_HOME / --cache-dir), OUTDIR, EVALSET, LOGDIR.
set -u
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PY=${PY:-/scratch/aebrahim/miniconda3/envs/lie_neuro/bin/python}
CACHE=${CACHE:-/scratch/aebrahim/hf_cache}
OUTDIR=${OUTDIR:-$REPO/outputs}
EVALSET=${EVALSET:-$REPO/data/sandbag/extra_sandbag.jsonl}
LOGDIR=${LOGDIR:-$REPO/scratchpad/logs_sandbag}
GPUS=${GPUS:-0,1,2,3}
MODELS=${MODELS:-"llama phi falcon olmo qwen3b"}
LLAMA_ID=${LLAMA_ID:-meta-llama/Llama-3.1-8B-Instruct}   # Delta's cache holds it as meta-llama/Meta-Llama-3.1-8B-Instruct
STRENGTHS=${STRENGTHS:-025,050,075,100}
COEFFS=${COEFFS:--0.18,-0.3,-0.5,-0.75,-1}
EXTRA=${EXTRA:-}
TAGX=${TAGX:-}
DRY=${DRY:-0}

# key | model id | steering layers (a quarter, a half, three quarters of depth, as in Table 1) |
#       lie-built mask at its Table 1 (LIARS' BENCH) setting | generation batch | score batch
spec() {
  case $1 in
    llama)  echo "$LLAMA_ID|8,14,21|fiction100|16|2" ;;
    phi)    echo "microsoft/Phi-4-mini-instruct|8,16,24|pooled100|16|2" ;;
    falcon) echo "tiiuae/Falcon3-3B-Instruct|5,11,16|fictf100|16|2" ;;
    olmo)   echo "allenai/OLMo-2-0425-1B-Instruct|4,8,12|fiction025|16|4" ;;
    qwen3b) echo "Qwen/Qwen2.5-3B-Instruct|9,18,27|pooltf100|16|2" ;;
    *) echo "unknown model key $1" >&2; return 1 ;;
  esac
}

[ -f "$EVALSET" ] || { echo "no eval set at $EVALSET (it is in git: data/sandbag/extra_sandbag.jsonl)"; exit 1; }
mkdir -p "$LOGDIR" "$OUTDIR"
IFS=',' read -r -a CARDS <<< "$GPUS"
read -r -a MS <<< "$MODELS"
for m in "${MS[@]}"; do spec "$m" >/dev/null || exit 1; done

lane() {                      # lane <gpu> <model key> [<model key> ...]
  local gpu=$1; shift
  for m in "$@"; do
    local MID LAYERS LIEREF BS SB
    IFS='|' read -r MID LAYERS LIEREF BS SB <<< "$(spec "$m")"
    local arms="unsteered" a L c
    for a in ${STRENGTHS//,/ }; do arms="$arms,sbag$a,sbpool$a"; done
    arms="$arms,sbagrnd100,sbpoolrnd100"
    for L in ${LAYERS//,/ }; do for c in ${COEFFS//,/ }; do arms="$arms,sbsL${L}_$c"; done; done
    arms="$arms,prompt,prompt2,prompt3,sbprompt1,sbprompt2,$LIEREF"
    local cvd=${gpu//+/,} dmap=""
    [[ $gpu == *+* ]] && dmap=auto           # "a+b": one lane over two cards, model sharded
    local cmd=(env CUDA_VISIBLE_DEVICES="$cvd" CUDA_DEVICE_ORDER=PCI_BUS_ID LIE_DEVICE_MAP="$dmap" HF_HOME="$CACHE"
               HF_HUB_OFFLINE="${OFFLINE:-0}" TOKENIZERS_PARALLELISM=false
               nice -n 10 "$PY" "$REPO/run_liars_bench_honesty.py"
               --model-id "$MID" --cache-dir "$CACHE" --out-dir "$OUTDIR" --eval-set "$EVALSET"
               --rho 0.04 --ratio 0.37 --falsehood-arms --polarity-arms --sandbag-arms --arms "$arms"
               --as-layers "$LAYERS" "--as-rel-coeffs=$COEFFS"
               --batch-size "$BS" --score-batch "$SB" --belief-tag sandbag --tag "sbag$TAGX" ${EXTRA:-})
    echo "[gpu $gpu] $m: $MID -> tag sbag$TAGX ($(tr ',' '\n' <<< "$arms" | wc -l) arms)"
    if [ "$DRY" = 1 ]; then
      printf '         %q' "${cmd[@]}"; echo
    else
      (cd "$REPO" && "${cmd[@]}") > "$LOGDIR/${m}_sbag$TAGX.log" 2>&1 \
        || echo "[gpu $gpu] $m FAILED, see $LOGDIR/${m}_sbag$TAGX.log"
    fi
  done
}

declare -a PLAN
for i in "${!MS[@]}"; do
  c=$(( i % ${#CARDS[@]} ))
  PLAN[$c]="${PLAN[$c]:-} ${MS[$i]}"
done
for c in "${!CARDS[@]}"; do
  [ -n "${PLAN[$c]:-}" ] || continue
  # shellcheck disable=SC2086
  lane "${CARDS[$c]}" ${PLAN[$c]} &
done
wait
echo "all lanes done; logs in $LOGDIR; next: $PY $REPO/reports/sandbag_report.py --out-dir $OUTDIR --tag sbag$TAGX"
