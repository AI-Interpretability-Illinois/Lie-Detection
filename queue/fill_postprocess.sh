#!/bin/bash
# After the swfill run (timan1 or Delta): bring the outputs next to the sweep outputs, rebuild Table 1 under the Qwen
# selection grader, GPT-5.6 re-grade the newly chosen arms' LIARS' BENCH replies (cached, so only new replies cost),
# rebuild the GPT table and the model averages. Run on the Delta login node from /u/xiaocong/lie_detection.
set -e
O=/work/nvme/biro/xiaocong/lie_sweep
if [ -n "$FROM_TIMAN" ]; then
  rsync -a --include='liars_bench_honesty_*_swfill*' --exclude='*' timan1:/srv/local/xy51/lie/fill_out/ $O/
fi
ls -la $O/liars_bench_honesty_*_swfill* | awk '{print $5, $9}'
cd reports
python3 build_final_table.py | tail -20
cd ..
set -a; source ~/.config/lie_detection/azure_openai.env; set +a
python3 judge_gpt.py --plan
[ -n "$RUN_GPT" ] && python3 judge_gpt.py --run | tail -5
cd reports && python3 build_gpt_table.py | tail -16 && python3 model_avg_report.py | tail -14 && python3 steer_lead_report.py
