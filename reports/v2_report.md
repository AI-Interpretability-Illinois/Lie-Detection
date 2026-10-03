# Calibrated 200-pair masks on the main table and on the tasks that exposed the 25-prompt mask's costs

**Why.** The 25-prompt mask of the main table costs 6-17 points of plain MMLU accuracy on six of seven models
and makes three models repeat an earlier lie more often (further_tasks_report.md). The prompt-count test showed
that masks built from more prompts need the contrast threshold re-tuned (sample_size_report.md). Here the
calibrated mask (200 lie/truth pairs from the format suite's user-turn or pooled formats, r calibrated to a
positive-score share of 0.25 or 0.5%, full deletion; `run_liars_bench_honesty.py --fmt-masks`) runs under the
main protocol (`reports/build_final_table.py --v2`, `main_table_v2.json`) and on plain MMLU + sandbagging, the
AI-Liar multi-turn set and the injected-instruction set (`queue/lie_v2tasks.slurm`, tags v2sandbag / v2multi /
v2inject; `reports/summarize_v2tasks.py`).

## Main benchmarks (test half, option chosen on validation; 25-prompt mask in brackets)

| model | LIARS' BENCH | AI Liar | chosen calibrated option |
|---|---|---|---|
| Llama-3.1-8B | 85.4 (93.5) | 77.5 (89.7) | pooled, 0.25% |
| Phi-4-mini | 83.2 (86.3) | 81.0 (78.7) | pooled, 0.25% |
| Qwen2.5-3B | 94.3 (94.3) | 80.8 (78.8) | user-turn, 0.25% |
| Qwen2.5-7B | 96.9 (95.4) | 93.2 (90.8) | pooled 0.25% / user-turn 0.25% |
| Falcon3-3B, OLMo-2-1B, Qwen2.5-1.5B | no option passed the polarity (and control) check: the calibrated masks over-correct toward "false" | | |

Lead of the calibrated mask over the best other method: Llama -1.5 [-6.5, +3.4] (LB) and -10.3 [-14.9, -5.8]
(AIL); Phi +3.8 [-1.5, +9.2] and +14.3 [+9.4, +19.4]; Qwen-3B +2.3 [-0.8, +5.7] and -1.3 [-5.9, +3.3];
Qwen-7B +1.9 [+0.0, +4.2] and +1.2 [-1.2, +3.8].

## Capability and the two failing tasks (same settings; honest %, MMLU accuracy on known items)


#### Meta-Llama-3p1-8B-Instruct
  arm                                MMLU-plain  sandbag   multi  inject  inj-ctl
  weight:fmpool025_ a=-1.0                 80.7     79.8    84.0       -        -
  actsteer:L8 r=-0.5 (c=-2.8)              68.1     58.5    90.8    76.3     79.4
  prompt:honest2                           83.2     52.3    65.3    69.9     75.0
  unsteered                                89.1     38.0    58.9    65.9     70.6
  weight:fiction a=-1.0                    72.3     67.4    70.6    86.8     87.5
  weight:magmatch a=-1.0                   89.9     40.3    62.3    54.8     59.6
  weight:ours a=-1.0                       80.7     67.1    82.5    79.0     67.6

#### Phi-4-mini-instruct
  arm                                MMLU-plain  sandbag   multi  inject  inj-ctl
  weight:fmpool025_ a=-1.0                 84.7     67.4    48.1       -        -
  actsteer:L16 r=-0.18 (c=-10.4)           87.0     61.0    58.2    64.8     90.0
  prompt:honest2                           90.1     59.8    74.0    61.8     87.3
  unsteered                                90.8     40.2    62.4    59.8     83.3
  weight:magmatch a=-1.0                   87.8     46.2    59.1    62.0     86.0
  weight:ours a=-0.5                       83.2     54.5    55.5    60.2     84.7
  weight:pooled a=-1.0                     82.4     61.4    50.7    67.8     82.7

#### Qwen2p5-3B-Instruct
  arm                                MMLU-plain  sandbag   multi  inject  inj-ctl
  weight:fmuser025_ a=-1.0                 84.2     78.4    74.6       -        -
  actsteer:L27 r=-0.18 (c=-18.7)           99.2     45.7    66.4    81.8     72.7
  prompt:honest                            94.2     52.9    72.0    79.0     74.7
  unsteered                                98.3     46.8    67.8    78.2     77.3
  weight:magmatch a=-1.0                   90.0     55.4    56.0    75.0     68.0
  weight:ours a=-1.0                       90.8     78.8    74.3    70.5     69.3
  weight:pooltf a=-1.0                     92.5     60.4    68.4    81.0     80.0

#### Qwen2p5-7B-Instruct
  arm                                MMLU-plain  sandbag   multi  inject  inj-ctl
  weight:fmpool025_ a=-1.0                 89.6     92.1    89.0       -        -
  weight:fmuser025_ a=-1.0                 91.0     93.4    89.9       -        -
  actsteer:L14 r=-0.18 (c=-11.0)           91.0     76.2    86.4    75.2     78.0
  prompt:honest                            97.0     52.1    84.6    72.8     74.0
  unsteered                                97.8     33.8    80.4    71.5     74.0
  weight:magmatch a=-1.0                   95.5     47.9    84.3    71.2     66.0
  weight:ours a=-1.0                       91.8     91.0    86.1    74.5     67.3
  weight:pooled a=-1.0                     90.3     87.2    78.6    71.0     66.7

(inject columns for the calibrated arms are being re-judged: the first judge pass filtered on LIARS' BENCH
split names and graded nothing.)

**Reading.** Against the 25-prompt mask at its chosen setting (weight:fiction / pooled / pooltf / pooled rows):
- **Capability.** The calibrated mask halves the MMLU cost on Llama (80.7 vs 72.3; unedited 89.1), is similar on
  Phi (84.7 vs 82.4) and Qwen-7B (89.6-91.0 vs 90.3), and worse on Qwen-3B (84.2 vs 92.5).
- **Sandbagging.** Better on every model: Llama 79.8 vs 67.4, Phi 67.4 vs 61.4, Qwen-3B 78.4 vs 60.4, Qwen-7B
  92.1-93.4 vs 87.2.
- **Multi-turn.** Better on three of four: Llama 84.0 vs 70.6 (steering 90.8), Qwen-3B 74.6 vs 68.4, Qwen-7B
  89.0-89.9 vs 78.6 (steering 86.4); Phi stays below unedited (48.1 vs 50.7; unedited 62.4).
So the calibrated mask trades a few points on the main benchmarks (and fails the polarity check on the three
small models) for better sandbagging and multi-turn behaviour and, on Llama, a smaller capability cost.
