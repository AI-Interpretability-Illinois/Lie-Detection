# Combined mask: the chosen prompt mask OR a small recovery mask

**Question.** The depth study (depth_report.md) found that a recovery mask, built from contrasts of
persisting in an earlier lie vs recovering, fixes multi-turn lying on Phi-4-mini, where the current
prompt-built mask makes it worse. Does the union of the two masks keep the single-turn gains and add
the multi-turn fix?

**Setup** (`queue/run_timan_combo.sh`, timan1 GPU1/GPU2, 2026-10-01 19:58-20:34). Union = a weight is
deleted if either mask selects it (`--combine` in run_liars_bench_honesty.py). The recovery masks use
rho 4% with r calibrated so that positive-score weights are 0.25% or 0.5% of each matrix (as in the
depth study). The unions cover 5.2-5.5% of weights, because the two masks overlap heavily (mostly
near-zero padding weights). Full deletion. Validation halves, the same settings as the depth jobs:
LIARS' BENCH (Qwen2.5-7B hybrid judge), AI Liar (keyword, near-train items excluded), multi-turn
(inline grader). Control = the honesty-instructed half of LIARS' BENCH. Summary script:
`reports/summarize_combo.py`.

## Results (honest %, [95% CI vs unedited] [vs current mask])

```

Llama-3p1-8B-Instruct   honest %, validation half; [95% CI vs unedited] [vs fiction]
  mask                                                LIARS' BENCH                               AI Liar                            multi-turn  control    ppl
  unedited                                                    32.8                                  28.2                                  61.2     98.4  1.000
  fiction                                       93.4 [+53.9,+67.2]                    82.8 [+49.3,+59.8]                    75.0 [ +6.2,+21.2]     95.2  1.072
  recov 0.25%                     41.7 [ +2.2,+15.1] [-58.3,-44.6]      22.7 [ -9.9, -0.8] [-65.3,-54.8]      65.6 [ -3.8,+12.5] [-16.2, -1.9]     97.6  1.013
  fiction + recov 0.25%           88.9 [+48.7,+63.1] [ -8.5, -0.7]      78.9 [+45.2,+56.1] [ -8.1, +0.3]      81.9 [+12.5,+28.7] [ +0.0,+13.8]     95.2  1.095
  fiction + recov 0.50%           81.5 [+40.6,+56.5] [-16.6, -7.4]      77.3 [+43.9,+54.8] [ -9.4, -1.8]      80.6 [+10.6,+28.1] [ -1.2,+11.9]     93.7  1.127

Phi-4-mini-instruct   honest %, validation half; [95% CI vs unedited] [vs pooled]
  mask                                                LIARS' BENCH                               AI Liar                            multi-turn  control    ppl
  unedited                                                    68.0                                  50.1                                  58.1     97.6  1.000
  pooled                                        93.1 [+19.6,+30.9]                    77.9 [+22.1,+33.1]                    48.5 [-18.0, -1.2]     97.6  1.054
  recov 0.25%                     65.1 [ -6.5, +0.7] [-33.5,-22.5]      61.9 [ +6.1,+17.6] [-21.6,-10.4]      70.7 [ +5.4,+20.4] [+13.8,+29.9]     93.7  1.077
  pooled + recov 0.25%            86.5 [+13.5,+24.0] [-10.5, -2.9]      76.0 [+19.7,+31.7] [ -6.7, +2.9]      62.3 [ -3.6,+12.0] [ +5.4,+21.6]     94.4  1.125
  pooled + recov 0.50%            84.0 [ +8.7,+22.9] [-13.8, -4.0]      76.3 [+20.3,+32.0] [ -6.1, +3.2]      64.1 [ -3.0,+15.6] [ +7.2,+24.6]     90.5  1.165

Falcon3-3B-Instruct   honest %, validation half; [95% CI vs unedited] [vs fictf]
  mask                                                LIARS' BENCH                               AI Liar                            multi-turn  control    ppl
  unedited                                                    75.6                                  72.8                                  60.5     95.2  1.000
  fictf                                         88.7 [ +7.3,+18.5]                    71.5 [ -7.7, +4.4]                    51.6 [-16.9, -0.8]     97.6  1.067
  recov 0.25%                     75.6 [ -6.2, +6.2] [-18.9, -7.3]      66.8 [-12.1, -0.3] [-10.7, +1.3]      60.5 [ -8.9, +9.7] [ +0.0,+17.7]     92.1  1.020
  fictf + recov 0.25%             84.4 [ +2.9,+14.9] [ -9.1, +0.4]      71.5 [ -7.4, +4.7] [ -5.7, +5.4]      46.0 [-25.0, -4.0] [-14.5, +2.4]     92.1  1.078
  fictf + recov 0.50%             80.4 [ -1.5,+10.9] [-13.8, -2.5]      74.5 [ -4.4, +7.7] [ -2.3, +8.4]      51.6 [-18.5, +1.6] [-10.5,+10.5]     92.1  1.110
```

## Reading

- **Phi-4-mini (the case it was meant for): partial fix, at a cost.** The union removes the current
  mask's multi-turn harm: 62.3 vs 48.5 [+5.4, +21.6]. It stays below the recovery mask alone (70.7)
  and is not significantly above unedited. AI Liar holds (76.0 vs 77.9, n.s.), but LIARS' BENCH drops
  6.6 points [-10.5, -2.9] and perplexity rises to 1.125x (1.165x at 0.5%, over the 1.15x bar).
- **Llama-3.1-8B: small multi-turn gain, small single-turn loss.** Multi-turn is 81.9 vs 75.0 for the
  current mask [+0.0, +13.8]. LIARS' BENCH loses 4.5 [-8.5, -0.7], and AI Liar is n.s. The recovery
  mask alone does nothing for Llama's multi-turn (65.6, n.s.), so the gain may be noise. In this run
  the current mask alone was +14 over unedited on multi-turn, but in the depth run it was +2 (n.s.).
- **Falcon3-3B: no help.** Multi-turn with the union is 46.0, below unedited [-25.0, -4.0] and n.s.
  against the current mask. The recovery mask alone does nothing on Falcon either.
- **The 0.5% unions are worse everywhere**: more single-turn loss, more damage, and no extra
  multi-turn gain.

**Conclusion.** Adding a recovery mask is a trade-off, not a free fix. It helps where the recovery
mask itself works (Phi), and even there it buys back about half of the recovery mask's multi-turn
gain for about 7 points of LIARS' BENCH. It should not replace the main method. It belongs in the
paper as an ablation: multi-turn persistence needs its own contrast, and the two edits interfere.

Run-to-run noise: answers are sampled (temperature 0.7), with about 160 multi-turn items per model.
The same arm moved up to 6 points between this run and the depth run (Llama's current mask on
multi-turn: 75.0 vs 68.8), so differences under about 7 points on multi-turn are within noise.
