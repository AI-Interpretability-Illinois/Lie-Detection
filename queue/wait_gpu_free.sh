#!/bin/bash
# Launch a script once a GPU shows at least MIN_FREE MiB free on two checks a minute apart (foreign jobs come and go on the
# timan108 cards). usage: bash queue/wait_gpu_free.sh <gpu index> <min free MiB> <script> [drop-flag file]
# Exits without launching if the drop-flag file appears.
GPU=$1; MIN=$2; SCRIPT=$3; DROP=${4:-/srv/local/xy51/lie/DROP_GPU$GPU}
T=/srv/local/xy51/lie; cd $T/lie_detection; rm -f "$DROP"
ok=0
while [ $ok -lt 2 ]; do
  [ -e "$DROP" ] && { echo "$(date '+%F %T') dropped"; exit 0; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU | tr -d ' ')
  if [ "$free" -ge "$MIN" ]; then ok=$((ok+1)); else ok=0; fi
  sleep 60
done
echo "$(date '+%F %T') GPU$GPU free ${free} MiB, launching $SCRIPT"
exec bash "$SCRIPT"
