#!/bin/bash
# Launch queue/run_timan108_gpu2.sh once timan108 GPU2 shows > 19 GB free on two checks a minute apart (zixuans8's jobs
# come and go on that card). Exit without launching if /srv/local/xy51/lie/DROP_GPU2 appears (Delta started the job first).
T=/srv/local/xy51/lie; cd $T/lie_detection
ok=0
while [ $ok -lt 2 ]; do
  [ -e $T/DROP_GPU2 ] && { echo "$(date '+%F %T') dropped (Delta copy started)"; exit 0; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 2 | tr -d ' ')
  if [ "$free" -gt 19000 ]; then ok=$((ok+1)); else ok=0; fi
  sleep 60
done
echo "$(date '+%F %T') GPU2 free ${free} MiB, launching"
exec bash queue/run_timan108_gpu2.sh
