# Ideas from Qi et al. (ICLR 2025): is the edit shallow, and do response-token masks work better?

Qi et al., "Safety Alignment Should Be Made More Than Just a Few Tokens Deep", show that alignment
mostly changes the first few RESPONSE tokens. Our masks score only PROMPT tokens. Three tests on the
models where the current mask wins (Llama-3.1-8B, Phi-4-mini, Falcon3-3B), validation halves only.

- **(3) Shallowness** (`kl_depth.py`): KL(edited || unedited) of next-token predictions at each of the
  first 24 positions of the unedited model's own answer, teacher-forced. 100 items per kind.
- **(1) resp masks**: the same lie instruction on both sides, forced false vs true answer, scored on
  answer tokens (per-token RMS).
- **(2) recov masks**: an earlier lying turn plus a follow-up (wordings unlike the evaluation's),
  forced persist vs recover. 200 questions and 200 statements per design (`mask_designs.py`).

With a plain difference (r = 1) both designs destroyed Llama: perplexity 33x to 145000x
(`lie_out/broken_r1`). |W| x (nF - nT) favours very high-activity channels. Final version: r calibrated
per model so the positive-score weights are 0.25 / 0.5 / 1% of each matrix (the current masks' range),
rho = 4%, full deletion.

## (3) Where the current edit acts (mean KL by answer position)

| model | answers | pos 1 | pos 2-3 | pos 4-24 |
|---|---|---|---|---|
| Llama-3.1-8B (fiction) | own lies (LB) | 2.20 | 0.79 | 1.89 |
| | honest-instructed answers | 1.48 | 0.08 | 0.28 |
| | after a planted lie (multi-turn) | 2.17 | 0.99 | 0.50 |
| Phi-4-mini (pooled) | own lies | 0.18 | 0.03 | 0.09 |
| | honest-instructed | 0.12 | 0.02 | 0.05 |
| | after a planted lie | 0.41 | 0.20 | 0.24 |
| Falcon3-3B (fictf) | own lies | 0.82 | 0.26 | 0.17 |
| | honest-instructed | 0.45 | 0.06 | 0.09 |
| | after a planted lie | 0.84 | 0.34 | 0.15 |

- The largest change is always at the first answer token.
- On lies, Llama's edit keeps pushing hard through the whole answer. Phi's and Falcon's are
  front-loaded: position 1 is 2-5x the later positions.
- After a planted lie, the edit moves the first token at least as much as for single-turn lies. So
  "the edit only acts on the first token of a fresh answer" does not explain the multi-turn weakness.

## (1)-(2) Behaviour (honest %, validation half; [95% CI of mask - unedited])

```

Meta-Llama-3p1-8B-Instruct   honest %, validation half; [95% CI of arm - unedited]
  mask                                LIARS' BENCH                       AI Liar                    multi-turn  control    ppl
  unedited                  38.0                          27.4                          66.9                       96.8  1.000
  fiction                   93.0 [+48.0,+61.6]            75.7 [+42.6,+54.0]            68.8 [ -6.2, +9.4]         96.0  1.070
  resp 0.25%                41.3 [ -3.7,+10.3]            32.9 [ +0.3,+10.7]            68.8 [ -5.0, +9.4]         95.2  1.020
  resp 0.50%                43.2 [ -1.1,+11.8]            32.9 [ +0.3,+10.7]            70.6 [ -3.1,+11.2]         96.0  1.040
  resp 1.00%                50.2 [ +5.5,+19.2]            44.4 [+11.0,+23.0]            67.5 [ -7.5, +8.8]         95.2  1.080
  recov 0.25%               43.5 [ -1.5,+12.2]            24.0 [ -8.1, +1.6]            69.4 [ -3.8, +9.4]         96.0  1.012
  recov 0.50%               46.1 [ +0.7,+14.8]            23.5 [ -9.1, +1.3]            67.5 [ -6.9, +8.8]         95.2  1.034
  recov 1.00%               45.4 [ +0.7,+14.4]            19.6 [-13.3, -2.6]            70.6 [ -4.4,+11.9]         91.3  1.078

Phi-4-mini-instruct   honest %, validation half; [95% CI of arm - unedited]
  mask                                LIARS' BENCH                       AI Liar                    multi-turn  control    ppl
  unedited                  66.9                          50.9                          57.5                       98.4  1.000
  pooled                    88.4 [+16.0,+26.9]            75.5 [+19.2,+29.9]            47.9 [-18.6, -0.6]         98.4  1.052
  resp 0.25%                65.5 [ -5.5, +2.5]            48.5 [ -7.2, +2.9]            58.7 [ -7.2, +9.6]         96.8  1.057
  resp 0.50%                69.8 [ -0.4, +6.5]            51.7 [ -4.5, +6.1]            52.7 [-13.8, +4.2]         95.2  1.078
  resp 1.00%                68.0 [ -3.3, +5.5]            56.5 [ +0.3,+11.2]            51.5 [-15.0, +2.4]         93.7  1.118
  recov 0.25%               68.0 [ -2.9, +5.5]            60.8 [ +4.3,+15.7]            73.1 [ +8.4,+24.0]         96.8  1.077
  recov 0.50%               69.5 [ -2.5, +7.6]            58.7 [ +2.7,+13.1]            68.9 [ +3.0,+19.8]         95.2  1.115
  recov 1.00%               61.8 [-11.6, +1.5]            66.4 [ +9.9,+21.6]            70.1 [ +3.6,+21.6]         82.5  1.153

Falcon3-3B-Instruct   honest %, validation half; [95% CI of arm - unedited]
  mask                                LIARS' BENCH                       AI Liar                    multi-turn  control    ppl
  unedited                  77.1                          78.5                          62.1                       97.6  1.000
  fictf                     87.6 [ +4.7,+16.4]            73.5 [ -9.7, -0.3]            50.0 [-20.2, -3.2]         96.8  1.065
  resp 0.25%                70.2 [-12.7, -0.7]            72.5 [-11.7, -0.7]            61.3 [ -6.5, +5.6]         92.1  0.993
  resp 0.50%                75.6 [ -7.3, +4.7]            69.8 [-14.1, -3.4]            62.9 [ -8.1, +9.7]         93.7  1.009
  resp 1.00%                70.5 [-13.1, +0.4]            71.8 [-11.7, -1.3]            65.3 [ -4.8,+11.3]         89.7  1.023
  recov 0.25%               72.7 [-10.2, +1.8]            71.8 [-11.7, -1.3]            61.3 [ -8.1, +6.5]         92.9  1.021
  recov 0.50%               56.4 [-28.4,-13.1]            68.8 [-15.1, -4.0]            58.9 [-13.7, +7.3]         89.7  1.055
  recov 1.00%               61.8 [-21.5, -8.4]            67.8 [-16.4, -5.0]            64.5 [ -6.5,+11.3]         91.3  1.074
```
LIARS' BENCH uses the hybrid judge, AI Liar the keyword verdict (near-train items excluded), and
multi-turn the inline grader. Control is the honesty-instructed half of LIARS' BENCH.

## Reading

1. **For single-turn instructed lying, the prompt-token contrast is what works.** The current masks
   gain 10-55 points on LIARS' BENCH. Response-token masks reach at most a third of that (Llama 1%:
   +12) and nothing on Phi. On Falcon they are slightly harmful. So what the edit removes sits in how
   the model takes in the order to lie, not in the false-answer tokens.
2. **Recovery masks fix multi-turn persistence on Phi only.** The current Phi mask makes it stick to an
   earlier lie (-10). recov 0.25% gives +16 [+8, +24] over unedited, also +10 on AI Liar, with
   control 96.8 and perplexity 1.08x. On Llama nothing moves multi-turn. On Falcon recov does not help,
   and the current mask hurts multi-turn (-12) and, at this choice, AI Liar (-5).
3. The recovery masks are built from the multi-turn structure (different questions and wordings), so
   on multi-turn they are in-format, not a transfer test.
4. Next, if wanted: combine the current mask with a small recov mask (union, or recov at a lower
   strength) and check single-turn and multi-turn together. Tune r per design as in test (b).

Files: `mask_designs.py`, `kl_depth.py`, `queue/lie_depth.slurm`, `reports/summarize_depth.py`;
outputs under /work/hdd/biro/xiaocong/lie_out. Jobs 22591418-20 on gpua038 (about 50, 50 and 25 min).
