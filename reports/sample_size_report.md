# Test (b): does the mask depend on which, and how many, prompts build it?

**Setup.** The mask is normally built from 25 hand-written questions (src/prompts.py), each
wrapped in a falsehood prefix (F) or "Answer truthfully:" (T). `build_mask_pool.py` generated 691
more questions in the same style (Azure GPT, deduplicated against the originals and against AI
Liar Questions1000), wrapped identically. Same rule throughout: top 4% of |W_ij| * c_j per matrix,
c_j = v nbar_j^F - (1 - v) nbar_j^T, r = 0.37.

- **Stability** (`mask_sample_size.py`, forward passes only, 7 models): masks from two disjoint
  draws of n prompts, n = 10 to 200, 4 repeats, compared by weight-weighted Jaccard. "Positive-score"
  restricts the comparison to selected weights with s_ij > 0. The rest of the budget is padding:
  weights with score <= 0, mostly the smallest |W|, which any draw selects alike.
- **Behaviour** (`run_liars_bench_honesty.py --pool-draws ... --half val`, 4 models): the
  hand-written mask vs three random 25-prompt draws, 100 and 400 prompts, each at the model's
  chosen family and strength. Scored on the LIARS' BENCH validation half with the hybrid keyword
  + Qwen2.5-7B judge, plus the usual damage checks. The 7-8B runs ran on timan1 GPU1, where Llama
  is `meta-llama/Llama-3.1-8B-Instruct`, the current repository name of the same model.

## 1. Mask stability (all seven models)


### fiction: agreement of two masks from disjoint prompt draws (weight-weighted Jaccard)

| model | n=10 | n=25 | n=50 | n=100 | n=200 | positive-score weights only: n=10 | n=25 | n=50 | n=100 | n=200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.74 | 0.84 | 0.88 | 0.91 | 0.94 | 0.41 | 0.57 | 0.64 | 0.74 | 0.79 |
| Phi-4-mini | 0.72 | 0.83 | 0.87 | 0.90 | 0.93 | 0.42 | 0.58 | 0.64 | 0.74 | 0.80 |
| Falcon3-3B | 0.67 | 0.79 | 0.84 | 0.88 | 0.91 | 0.40 | 0.57 | 0.62 | 0.72 | 0.78 |
| OLMo-2-1B | 0.80 | 0.88 | 0.91 | 0.94 | 0.96 | 0.35 | 0.52 | 0.59 | 0.71 | 0.76 |
| Qwen2.5-3B | 0.71 | 0.83 | 0.87 | 0.91 | 0.93 | 0.40 | 0.55 | 0.61 | 0.73 | 0.78 |
| Qwen2.5-7B | 0.72 | 0.83 | 0.86 | 0.91 | 0.93 | 0.43 | 0.58 | 0.64 | 0.75 | 0.80 |
| Qwen2.5-1.5B | 0.73 | 0.83 | 0.87 | 0.91 | 0.93 | 0.41 | 0.56 | 0.62 | 0.73 | 0.78 |

### fiction: the hand-written 25 prompts, and what the 4% budget is made of

| model | hand-written 25 vs 400-prompt mask (positive-score) | random 25 vs 400 (positive-score) | padding share r=0.37 | positive-score weights, % of matrix | padding share at r = 1 / 0.667 / 0.5 / 0.25 | overlap with r=0.37 mask at r = 1 / 0.667 / 0.5 / 0.25 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.85 (0.63) | 0.89 (0.68) | 0.88 | 0.48 | 0.00 / 0.00 / 0.65 / 0.97 | 0.03 / 0.04 / 0.55 / 0.79 |
| Phi-4-mini | 0.83 (0.62) | 0.88 (0.67) | 0.85 | 0.59 | 0.00 / 0.00 / 0.62 / 0.97 | 0.03 / 0.05 / 0.54 / 0.75 |
| Falcon3-3B | 0.80 (0.62) | 0.85 (0.66) | 0.80 | 0.81 | 0.00 / 0.00 / 0.56 / 0.96 | 0.05 / 0.07 / 0.51 / 0.70 |
| OLMo-2-1B | 0.89 (0.59) | 0.92 (0.62) | 0.96 | 0.16 | 0.00 / 0.03 / 0.79 / 0.99 | 0.01 / 0.04 / 0.67 / 0.88 |
| Qwen2.5-3B | 0.85 (0.64) | 0.88 (0.66) | 0.86 | 0.56 | 0.00 / 0.01 / 0.63 / 0.97 | 0.03 / 0.06 / 0.56 / 0.77 |
| Qwen2.5-7B | 0.84 (0.66) | 0.88 (0.68) | 0.82 | 0.70 | 0.00 / 0.02 / 0.63 / 0.95 | 0.04 / 0.07 / 0.59 / 0.75 |
| Qwen2.5-1.5B | 0.86 (0.66) | 0.89 (0.67) | 0.87 | 0.52 | 0.00 / 0.00 / 0.62 / 0.97 | 0.03 / 0.05 / 0.54 / 0.78 |

### pooled: agreement of two masks from disjoint prompt draws (weight-weighted Jaccard)

| model | n=10 | n=25 | n=50 | n=100 | n=200 | positive-score weights only: n=10 | n=25 | n=50 | n=100 | n=200 |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.79 | 0.87 | 0.91 | 0.93 | 0.95 | 0.40 | 0.56 | 0.65 | 0.75 | 0.80 |
| Phi-4-mini | 0.77 | 0.86 | 0.90 | 0.93 | 0.95 | 0.40 | 0.55 | 0.65 | 0.75 | 0.80 |
| Falcon3-3B | 0.71 | 0.82 | 0.87 | 0.91 | 0.93 | 0.38 | 0.55 | 0.64 | 0.74 | 0.79 |
| OLMo-2-1B | 0.85 | 0.91 | 0.93 | 0.95 | 0.97 | 0.32 | 0.49 | 0.59 | 0.69 | 0.76 |
| Qwen2.5-3B | 0.77 | 0.86 | 0.90 | 0.93 | 0.95 | 0.39 | 0.55 | 0.64 | 0.75 | 0.78 |
| Qwen2.5-7B | 0.77 | 0.86 | 0.90 | 0.93 | 0.95 | 0.43 | 0.58 | 0.66 | 0.76 | 0.80 |
| Qwen2.5-1.5B | 0.79 | 0.87 | 0.91 | 0.93 | 0.95 | 0.41 | 0.58 | 0.67 | 0.77 | 0.80 |

### pooled: the hand-written 25 prompts, and what the 4% budget is made of

| model | hand-written 25 vs 400-prompt mask (positive-score) | random 25 vs 400 (positive-score) | padding share r=0.37 | positive-score weights, % of matrix | padding share at r = 1 / 0.667 / 0.5 / 0.25 | overlap with r=0.37 mask at r = 1 / 0.667 / 0.5 / 0.25 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 0.87 (0.58) | 0.91 (0.67) | 0.94 | 0.24 | 0.00 / 0.13 / 0.79 / 0.99 | 0.01 / 0.09 / 0.69 / 0.87 |
| Phi-4-mini | 0.86 (0.58) | 0.90 (0.66) | 0.92 | 0.31 | 0.00 / 0.06 / 0.72 / 0.99 | 0.02 / 0.06 / 0.62 / 0.83 |
| Falcon3-3B | 0.82 (0.59) | 0.88 (0.66) | 0.88 | 0.47 | 0.00 / 0.00 / 0.64 / 0.98 | 0.03 / 0.05 / 0.56 / 0.78 |
| OLMo-2-1B | 0.91 (0.52) | 0.94 (0.63) | 0.99 | 0.05 | 0.00 / 0.44 / 0.91 / 1.00 | 0.00 / 0.25 / 0.80 / 0.93 |
| Qwen2.5-3B | 0.87 (0.60) | 0.91 (0.67) | 0.92 | 0.32 | 0.00 / 0.05 / 0.71 / 0.99 | 0.02 / 0.06 / 0.62 / 0.84 |
| Qwen2.5-7B | 0.86 (0.63) | 0.90 (0.69) | 0.90 | 0.41 | 0.00 / 0.15 / 0.70 / 0.98 | 0.03 / 0.12 / 0.64 / 0.82 |
| Qwen2.5-1.5B | 0.88 (0.62) | 0.91 (0.69) | 0.93 | 0.28 | 0.00 / 0.09 / 0.73 / 0.99 | 0.02 / 0.08 / 0.64 / 0.85 |

**Reading.**
- At n = 25, two equally valid masks share 79-91% of all selected weights but only 49-58% of
  their positive-score weights. That share rises steadily with n, to 76-80% at n = 200, and has
  not converged even then. The pattern is the same across all seven models and both families.
- At r = 0.37, 80-99% of the 4% budget is padding. The positive-score weights are 0.05-0.8% of
  each matrix (Llama fiction 0.48%, Qwen2.5-7B fiction 0.70%).
- r changes the mask more than anything else. At r = 1 there is no padding, and the mask overlaps
  the current one by at most 7%.
- The hand-written 25 prompts look like a random draw of 25. Their overlap with the 400-prompt
  mask is slightly lower, consistent with different question content.

## 2. Behaviour (LIARS' BENCH validation half, honest rate on instructed-deception items)

```

Llama-3.1-8B (fiction, a=-1)   (validation half, n = 271 deception items)
  mask                honest   minus hand-written [95% CI]  control  believed T/F    ppl
  unedited              32.8                                   98.4   11.9/53.7    1.000
  hand-written 25       93.4                                   95.2   92.6/94.1    1.072
  pool 25 (draw 1)      84.1    -9.2 [-13.7, -4.8]             96.8   83.0/85.3    1.095
  pool 25 (draw 2)      90.0    -3.3 [-7.0, +0.0]              96.0   89.6/90.4    1.103
  pool 25 (draw 3)      88.2    -5.2 [-9.2, -1.5]              92.9   84.4/91.9    1.123
  pool 100              91.1    -2.2 [-6.3, +1.5]              94.4   88.9/93.4    1.106
  pool 400              90.4    -3.0 [-7.0, +0.7]              96.0   84.4/96.3    1.104

Phi-4-mini (pooled, a=-1)   (validation half, n = 275 deception items)
  mask                honest   minus hand-written [95% CI]  control  believed T/F    ppl
  unedited              66.9                                   98.4   35.8/97.8    1.000
  hand-written 25       88.4                                   98.4   78.1/98.6    1.052
  pool 25 (draw 1)      88.4    +0.0 [-4.0, +4.4]              99.2   78.8/97.8    1.047
  pool 25 (draw 2)      82.9    -5.5 [-9.5, -1.5]              99.2   70.8/94.9    1.046
  pool 25 (draw 3)      88.0    -0.4 [-4.0, +3.6]              96.8   77.4/98.6    1.057
  pool 100              88.7    +0.4 [-3.6, +4.4]              96.8   78.1/99.3    1.044
  pool 400              84.0    -4.4 [-8.4, -0.4]              98.4   69.3/98.6    1.041

Qwen2.5-7B (pooled, a=-1)   (validation half, n = 275 deception items)
  mask                honest   minus hand-written [95% CI]  control  believed T/F    ppl
  unedited              83.6                                   98.4   71.5/94.5    1.000
  hand-written 25       95.3                                   96.8   96.2/94.5    1.090
  pool 25 (draw 1)      96.0    +0.7 [-1.1, +2.5]              96.8   97.7/94.5    1.069
  pool 25 (draw 2)      94.9    -0.4 [-2.5, +1.8]              97.6   93.8/95.9    1.071
  pool 25 (draw 3)      94.9    -0.4 [-2.2, +1.5]              95.2   95.4/94.5    1.071
  pool 100              94.9    -0.4 [-2.2, +1.5]              97.6   95.4/94.5    1.060
  pool 400              95.3    +0.0 [-1.8, +2.2]              96.0   96.9/93.8    1.057

OLMo-2-1B (fiction, a=-0.25)   (validation half, n = 275 deception items)
  mask                honest   minus hand-written [95% CI]  control  believed T/F    ppl
  unedited              54.9                                   83.3   26.3/99.1    1.000
  hand-written 25       52.7                                   86.5   27.5/91.7    1.003
  pool 25 (draw 1)      53.1    +0.4 [-5.5, +5.8]              86.5   25.7/95.4    1.003
  pool 25 (draw 2)      53.8    +1.1 [-4.7, +7.3]              87.3   28.7/92.6    1.002
  pool 25 (draw 3)      52.0    -0.7 [-6.9, +5.1]              87.3   29.3/87.0    1.002
  pool 100              53.5    +0.7 [-5.1, +6.5]              82.5   28.1/92.6    1.001
  pool 400              52.7    +0.0 [-5.8, +5.8]              88.1   28.1/90.7    1.000
```

**Reading.**
- Where the edit has a large effect, mask noise reaches behaviour. On Llama-3.1-8B, three random
  25-prompt masks give 84.1 / 90.0 / 88.2, a 6-point spread, and two are significantly below the
  hand-written mask. On Phi-4-mini they give 88.4 / 82.9 / 88.0. With 100 or 400 prompts the result
  sits in the middle of that range (Llama 91.1 / 90.4, Phi 88.7 / 84.0). So more prompts at a
  fixed r reduce the luck of the draw but do not raise the mean.
- Qwen2.5-7B is near its ceiling (unedited 83.6, every mask 94.9-96.0), so it shows no
  sensitivity. OLMo-2-1B at a = -0.25 shows no effect from any mask.
- The hand-written mask scores highest, but its family and strength were chosen on this same
  validation half, so its score here is optimistic. This is not evidence that the hand-written
  prompts are better. Its test-half numbers in the main table remain unbiased for that particular
  mask.
- Every mask passes the damage checks: control at least 92.9%, perplexity at most 1.12x. The pool
  masks cost a little more perplexity on Llama (1.10-1.12x against 1.07x).
- Phi's 400-prompt mask is significantly worse (-4.4). One candidate explanation: with more
  prompts the channel norms are less noisy, fewer channels clear the 1/r bar, and the effective
  edit shrinks. The padding shares fit this (0.92 vs 0.88 for the hand-written mask), but it is
  not tested.

## 3. Implications

1. Report mask results as a mean over several prompt draws, or build from at least 100 prompts.
   A single 25-prompt mask carries a few points of draw-to-draw luck (up to 9 on Llama).
2. Tune the number of prompts together with r (and rho). r = 0.37 was inherited from the
   original lie-mask search, it decides what most of the mask is, and changing n changes the
   effective edit size at a fixed r.
3. For the format-transfer study, use at least 200 prompt pairs per format and select each
   method's settings on that format's own validation half.

Files: `mask_sample_size.py`, `build_mask_pool.py`, `extra_prompts/mask_pool_questions.json`,
`outputs/controls/mask_sample_size_*.json`, `reports/summarize_sample_size.py`,
`reports/summarize_bsamp.py`, outputs `*_bsamp*` (timan runs under `lie_detection_outputs/timan_b/`).
