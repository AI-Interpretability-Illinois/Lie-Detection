#!/bin/bash
#SBATCH --job-name=lie_queue
#SBATCH --partition=gpuA100x4
#SBATCH --reservation=sup-30781
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=10:00:00
#SBATCH --output=/u/xiaocong/lie_detection/queue/logs/lie_queue_%j.out
# Runs the lie_detection job queue on a whole reserved node (no other tenant), two
# workers per GPU (a worker starts a job only when the GPU has the job's MEM_GB free),
# until nothing is pending or running. Submit with --nodelist and an account, e.g.
#   source /u/xiaocong/dagformer/scripts/slurm/resv_lib.sh
#   sbatch_with_fallback --nodelist=gpua047 /u/xiaocong/lie_detection/queue/lie_queue_resv.slurm
Q=/u/xiaocong/lie_detection/queue
echo "$(date '+%F %T') lie_queue $SLURM_JOB_ID on $(hostname -s), GPUs: $(nvidia-smi -L | wc -l)"
# own stop file: guards on other nodes pause through ~/.stop_lie_queue, which must not stop us
export LIE_STOP_FILE=$HOME/.stop_lie_queue_$(hostname -s)
rm -f "$LIE_STOP_FILE"
pids=()
for g in 0 1 2 3; do
  for s in "" 2; do
    SLOT=$s bash "$Q/worker.sh" $g >> "$Q/logs/worker.$(hostname -s)-gpu$g${s:+s$s}.log" 2>&1 &
    pids+=($!)
  done
done
# until nothing is pending or running here; give up after 30 min with nothing startable
# (e.g. a judge whose generation job failed)
idle=0
while [ ! -e "$LIE_STOP_FILE" ]; do
  pend=$(ls "$Q/pending" | grep -c '\.sh$'); run=$(ls "$Q/running" | grep -c "\.$(hostname -s)-")
  [ "$pend" = 0 ] && [ "$run" = 0 ] && break
  if [ "$run" = 0 ]; then idle=$((idle + 1)); else idle=0; fi
  [ $idle -ge 30 ] && { echo "$(date '+%F %T') $pend pending but nothing startable for 30 min"; break; }
  sleep 60
done
echo "$(date '+%F %T') stopping workers (pending $pend, running here $run)"
touch "$LIE_STOP_FILE"
wait "${pids[@]}"
rm -f "$LIE_STOP_FILE"
echo "$(date '+%F %T') done"
