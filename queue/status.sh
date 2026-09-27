#!/bin/bash
# Queue status; readable from the login node since the queue is on the shared FS.
Q=/u/xiaocong/lie_detection/queue
for s in running pending done failed; do printf "%-8s %3d\n" $s $(ls $Q/$s 2>/dev/null | wc -l); done
echo "--- running:"; ls $Q/running 2>/dev/null
echo "--- failed:";  ls $Q/failed 2>/dev/null
echo "--- next pending:"; ls $Q/pending 2>/dev/null | sort | head -5
