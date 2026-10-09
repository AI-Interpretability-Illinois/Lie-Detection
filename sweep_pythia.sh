#!/usr/bin/env bash
# Run the honesty protocol across the Pythia size series, one lane per GPU.
#
# The ablation asks how the contrast's lead moves with scale, so every size has
# to be run identically: same arms, same rho/ratio, same eval set, the steering
# baseline swept the same way.  This is a driver, not a second protocol -- it
# resolves each size's final checkpoint through prepare_pythia_checkpoints.py
# and calls run_liars_bench_honesty.py with the main table's settings.
#
# Sizes are dealt round-robin across the GPUs in GPUS and each lane runs its
# sizes in sequence, so eight sizes over four cards is four lanes of two rather
# than eight serial loads.  SIZES is ordered largest-first so a 12b lane does
# not start last and leave three cards idle.
#
#   GPUS=0,1,2,3 bash sweep_pythia.sh                      # all eight sizes
#   GPUS=2,3 SIZES="70m 160m 410m 1b" bash sweep_pythia.sh
#   GPUS=2 SIZES=70m DRY=1 bash sweep_pythia.sh            # print, run nothing
#   GPUS=0+1 SIZES=12b bash sweep_pythia.sh                # one lane over two cards (LIE_DEVICE_MAP=auto)
#   STEER_ARMS=1 ...                                       # also score the six steering arms of each size
#
# Paths are env-overridable for a different machine:
#   REPO   checkout root                 PY     python with torch+transformers
#   CACHE  HF_HOME / --cache-dir         OUTDIR where result csv/jsonl land
#   PYROOT where prepared checkpoints are materialised
#
# Steering: --as-layers is the one setting that is not size-free, because a
# layer index does not mean the same thing at 6 layers as at 36.  Each lane
# sweeps the three layers around mid-depth with both relative coefficients, and
# the arm is chosen on the validation half downstream -- the same way the main
# table chooses a setting for every method.

set -u
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PY=${PY:-/scratch/aebrahim/miniconda3/envs/lie_neuro/bin/python}
CACHE=${CACHE:-/scratch/aebrahim/hf_cache}
OUTDIR=${OUTDIR:-$REPO/outputs}
PYROOT=${PYROOT:-/scratch/aebrahim/pythia_final}
LOGDIR=${LOGDIR:-$REPO/scratchpad/logs_pythia}

GPUS=${GPUS:-0,1,2,3}
SIZES=${SIZES:-"12b 6.9b 2.8b 1.4b 1b 410m 160m 70m"}
ARMS=${ARMS:-unsteered,magmatch100,poolmix100,ours075,prompt2}
EVALSET=${EVALSET:-$OUTDIR/liars_bench/eval_set.jsonl}
RHO=${RHO:-0.04}
RATIO=${RATIO:-0.37}
COEFFS=${COEFFS:--0.18,-0.5}
EXTRA=${EXTRA:-}            # passthrough, e.g. EXTRA="--n-per-split 4" to smoke-test
DRY=${DRY:-0}
TAGX=${TAGX:-}               # suffix for --tag (a second protocol on the same sizes); the belief cache keeps pythia<size>
BS_OVERRIDE=${BS_OVERRIDE:-} # generation batch for every size in this call, instead of the grid's (memory permitting)
STEER_ARMS=${STEER_ARMS:-0} # 1: append asrL{lo,mid,hi}_{COEFFS} to ARMS, per size (the layers differ by size)

[ -f "$EVALSET" ] || { echo "no eval set at $EVALSET -- build it first:"; \
  echo "  $PY $REPO/liars_bench_prep.py --out $EVALSET"; exit 1; }
mkdir -p "$LOGDIR"
IFS=',' read -r -a CARDS <<< "$GPUS"
read -r -a SZ <<< "$SIZES"

# Fetch every checkpoint first, serially: two lanes downloading the same blob
# race on the HF cache lock, and a size already prepared costs nothing.
for s in "${SZ[@]}"; do
  HF_HOME=$CACHE $PY "$REPO/prepare_pythia_checkpoints.py" \
    --sizes "$s" --cache-dir "$CACHE" --out-root "$PYROOT" >/dev/null \
    || { echo "prepare failed: $s"; exit 1; }
done

lane() {                      # lane <gpu> <size> [<size> ...]
  local gpu=$1; shift
  for s in "$@"; do
    eval "$(HF_HOME=$CACHE $PY "$REPO/prepare_pythia_checkpoints.py" \
            --sizes "$s" --cache-dir "$CACHE" --out-root "$PYROOT" \
            --emit-env 2>/dev/null)"
    local lo=$(( STEER - 1 )) hi=$(( STEER + 1 ))      # clamp into range
    (( lo < 0 )) && lo=0
    (( hi > NLAYER - 1 )) && hi=$(( NLAYER - 1 ))
    local tag="pythia$(echo "$s" | tr . p)"
    [ -n "$BS_OVERRIDE" ] && BS=$BS_OVERRIDE
    local arms=$ARMS L c
    if [ "$STEER_ARMS" = 1 ]; then
      for L in $(printf '%s\n' "$lo" "$STEER" "$hi" | uniq); do
        for c in ${COEFFS//,/ }; do arms="$arms,asrL${L}_$c"; done
      done
    fi
    # a lane written "a+b" holds both cards and shards the model over them
    local cvd=${gpu//+/,} dmap=""
    [[ $gpu == *+* ]] && dmap=auto
    local cmd=(env CUDA_VISIBLE_DEVICES="$cvd" CUDA_DEVICE_ORDER=PCI_BUS_ID LIE_DEVICE_MAP="$dmap" HF_HOME="$CACHE"
               HF_HUB_OFFLINE="${OFFLINE:-0}" TOKENIZERS_PARALLELISM=false
               nice -n 10 "$PY" "$REPO/run_liars_bench_honesty.py"
               --model-id "$MID" --cache-dir "$CACHE" --out-dir "$OUTDIR"
               --eval-set "$EVALSET" --rho "$RHO" --ratio "$RATIO"
               --falsehood-arms --polarity-arms --arms "$arms"
               --as-layers "$lo,$STEER,$hi" "--as-rel-coeffs=$COEFFS"
               --batch-size "$BS" --score-batch "$SB"
               --belief-tag "$tag" --tag "$tag$TAGX" ${EXTRA:-})
    echo "[gpu $gpu] $s -> $tag$TAGX  (layers $lo,$STEER,$hi of $NLAYER; BS=$BS SB=$SB)"
    if [ "$DRY" = 1 ]; then
      printf '         %q' "${cmd[@]}"; echo
    else
      "${cmd[@]}" > "$LOGDIR/$tag$TAGX.log" 2>&1 \
        || echo "[gpu $gpu] $s FAILED, see $LOGDIR/$tag.log"
    fi
  done
}

declare -a PLAN                              # deal sizes round-robin over cards
for i in "${!SZ[@]}"; do
  c=$(( i % ${#CARDS[@]} ))
  PLAN[$c]="${PLAN[$c]:-} ${SZ[$i]}"
done

for c in "${!CARDS[@]}"; do
  [ -n "${PLAN[$c]:-}" ] || continue
  # shellcheck disable=SC2086
  lane "${CARDS[$c]}" ${PLAN[$c]} &
done
wait
echo "all lanes done; logs in $LOGDIR"
