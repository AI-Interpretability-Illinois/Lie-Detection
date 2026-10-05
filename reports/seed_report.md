# Sampling-seed replication of Table 1 (Qwen2.5-7B judge, LIARS' BENCH test half)

Seed 0 is the paper's run; seeds 1 and 2 regenerate the same arms (temperature 0.7) with a different torch seed.
Spread = max - min over the seeds available; SD = sample standard deviation. Lead: falsehood mask minus the comparator
fixed on LIARS' BENCH validation, paired bootstrap 95% interval over test items, under each seed.

## Llama-3.1-8B

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 93.5 | 88.9 | 91.6 | 4.6 | 2.3 |
| liemask | 82.8 | 81.2 | 80.5 | 2.3 | 1.2 |
| steer | 87.0 | 85.4 | 85.1 | 1.9 | 1.0 |
| prompt | 65.9 | 67.8 | 71.3 | 5.4 | 2.7 |
| unsteered | 37.2 | 36.4 | 40.2 | 3.8 | 2.0 |
| lead vs steer | +6.5 [+1.9, +11.1] | +3.4 [-1.1, +8.4] | +6.5 [+1.9, +11.1] | | |
| lead, pooled over 3 seeds | +5.5 [+2.3, +8.8] (n=261) | | | | |

## Phi-4-mini

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 86.3 | 84.4 | 85.5 | 1.9 | 1.0 |
| liemask | 69.5 | 67.6 | 72.9 | 5.3 | 2.7 |
| steer | 76.0 | 78.2 | 80.2 | 4.2 | 2.1 |
| prompt | 79.4 | 83.6 | 82.1 | 4.2 | 2.1 |
| unsteered | 69.8 | 65.6 | 66.4 | 4.2 | 2.2 |
| lead vs prompt | +6.9 [+2.3, +11.1] | +0.8 [-4.2, +5.7] | +3.4 [-1.1, +8.4] | | |
| lead, pooled over 3 seeds | +3.7 [+0.4, +7.1] (n=262) | | | | |

## Falcon3-3B

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 89.7 | 86.6 | 88.5 | 3.1 | 1.6 |
| liemask | - | 80.5 | 82.4 | 1.9 | 1.3 |
| steer | 79.4 | 84.7 | 80.9 | 5.3 | 2.7 |
| prompt | - | 84.0 | 82.8 | 1.2 | 0.8 |
| unsteered | 80.2 | 82.1 | 76.3 | 5.8 | 3.0 |
| lead vs steer | +10.3 [+4.6, +16.0] | +1.9 [-3.1, +6.9] | +7.6 [+2.3, +13.0] | | |
| lead, pooled over 3 seeds | +6.6 [+3.4, +9.8] (n=262) | | | | |

## OLMo-2-1B

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 58.0 | 53.4 | 55.3 | 4.6 | 2.3 |
| liemask | 56.1 | 57.3 | 55.0 | 2.3 | 1.2 |
| steer | - | 72.5 | 69.5 | 3.0 | 2.1 |
| prompt | 61.1 | 62.6 | 64.1 | 3.0 | 1.5 |
| unsteered | 50.4 | 53.1 | 48.5 | 4.6 | 2.3 |
| lead vs prompt | -3.1 [-8.8, +2.3] | -9.2 [-16.0, -3.1] | -8.8 [-14.9, -2.7] | | |
| lead, pooled over 3 seeds | -7.0 [-10.9, -2.9] (n=262) | | | | |

## Qwen2.5-3B

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 94.3 | 93.9 | 94.7 | 0.8 | 0.4 |
| liemask | 92.0 | 92.4 | 92.7 | 0.7 | 0.4 |
| steer | 90.5 | 91.6 | 93.1 | 2.6 | 1.3 |
| prompt | 92.0 | 93.1 | 93.1 | 1.1 | 0.6 |
| unsteered | 81.7 | 81.7 | 84.0 | 2.3 | 1.3 |
| lead vs prompt | +2.3 [-0.8, +5.3] | +0.8 [-1.9, +3.4] | +1.5 [-1.1, +4.2] | | |
| lead, pooled over 3 seeds | +1.5 [-0.6, +3.7] (n=262) | | | | |

## Qwen2.5-7B

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 95.4 | 95.4 | 95.0 | 0.4 | 0.2 |
| liemask | 95.0 | 95.0 | 95.4 | 0.4 | 0.2 |
| steer | 90.8 | 90.1 | 89.7 | 1.1 | 0.6 |
| prompt | 95.8 | 96.9 | 95.8 | 1.1 | 0.6 |
| unsteered | 88.5 | 86.3 | 86.6 | 2.2 | 1.2 |
| lead vs liemask | +0.4 [-1.9, +2.7] | +0.4 [-1.9, +2.7] | -0.4 [-3.1, +2.3] | | |
| lead, pooled over 3 seeds | +0.1 [-1.9, +2.3] (n=262) | | | | |

## Qwen2.5-1.5B

| arm | seed 0 | seed 1 | seed 2 | spread | SD |
|---|---|---|---|---|---|
| mask | 87.0 | 82.8 | 87.8 | 5.0 | 2.7 |
| liemask | 85.1 | 84.4 | 85.1 | 0.7 | 0.4 |
| steer | 90.5 | 88.2 | 89.3 | 2.3 | 1.2 |
| prompt | 83.2 | 81.3 | 82.1 | 1.9 | 1.0 |
| unsteered | 78.6 | 80.2 | 81.3 | 2.7 | 1.4 |
| lead vs steer | -3.4 [-7.3, +0.4] | -5.3 [-9.2, -1.1] | -1.5 [-5.3, +2.3] | | |
| lead, pooled over 3 seeds | -3.4 [-6.0, -1.0] (n=262) | | | | |
