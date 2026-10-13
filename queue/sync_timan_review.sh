#!/bin/bash
# Pull the timan1 review outputs (top-level results only: csv, per-item jsonl; plus logs/) into Delta's lie_review directory.
# Never the extra_data/ or liars_bench/ inputs: on Delta extra_data is a symlink, and rsync replaced it by a directory once
# (2026-10-12 16:53), which broke every job that loaded the MMLU file afterwards.
timeout 600 rsync -rt --no-owner --no-group --keep-dirlinks --exclude='extra_data/' --exclude='liars_bench/' \
  --include='logs/' --include='logs/*.log' --include='*.csv' --include='*.jsonl' --exclude='*' \
  timan1:/srv/local/xy51/lie/review_out/ /work/hdd/bfqt/xiaocong/lie_review/ 2>&1 | grep -v bashrc
