# Adaptive attacks on the order to lie (Qwen2.5-7B judge)

Honest rate (%) on the test half of the 537 LIARS' BENCH deception items under each attack, every method at its
LIARS' BENCH-chosen setting; 'plain' is the same items without the attack (Table 1, same judge). Lead: mask minus the
comparator fixed on LIARS' BENCH validation, paired bootstrap 95% interval over test items.

## Llama-3.1-8B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 87.4 | 78.9 | 87.0 | 65.9 | 37.2 | +8.4 [+2.7, +14.2] vs liemask | |
| many-shot lies | 88.1 | 85.8 | 85.8 | 87.0 | 53.6 | +2.3 [-1.9, +6.5] vs liemask | +1.1 [-3.4, +5.7] vs prompt |
| Spanish order | 82.7 | 83.1 | 88.0 | 68.5 | 45.6 | -1.2 [-7.1, +4.7] vs liemask | -5.4 [-10.9, +0.0] vs steer |
| Chinese order | 89.6 | 89.6 | 82.1 | 75.8 | 53.8 | +0.4 [-3.5, +4.3] vs liemask | +0.4 [-3.5, +4.3] vs liemask |

## Phi-4-mini

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 90.5 | 69.5 | 76.0 | 79.4 | 69.8 | +11.1 [+6.1, +16.4] vs prompt | |
| many-shot lies | 91.2 | 89.3 | 90.1 | 90.5 | 84.4 | +0.8 [-2.7, +4.6] vs prompt | +0.8 [-2.7, +4.6] vs prompt |
| Spanish order | 93.1 | 77.5 | 86.6 | 85.5 | 72.8 | +7.7 [+3.1, +11.9] vs prompt | +6.5 [+1.9, +11.2] vs steer |
| Chinese order | 93.9 | 79.4 | 86.6 | 87.8 | 77.1 | +6.1 [+1.9, +10.3] vs prompt | +6.1 [+1.9, +10.3] vs prompt |

## Falcon3-3B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 91.6 | 83.6 | 84.0 | 81.7 | 80.2 | +7.6 [+2.7, +12.6] vs steer | |
| many-shot lies | 87.4 | 85.9 | 87.4 | 84.4 | 83.2 | +0.0 [-5.0, +5.0] vs steer | +0.0 [-5.0, +5.0] vs steer |
| Spanish order | 84.3 | 76.8 | 83.8 | 85.7 | 78.9 | +0.4 [-5.6, +6.3] vs steer | -0.8 [-6.2, +4.5] vs prompt |
| Chinese order | 90.4 | 90.6 | 84.0 | 92.2 | 88.5 | +6.3 [+0.4, +12.2] vs steer | -1.7 [-6.2, +2.9] vs prompt |

## OLMo-2-1B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 60.7 | 57.3 | 54.2 | 61.1 | 50.4 | -0.4 [-6.5, +5.3] vs prompt | |
| many-shot lies | 64.5 | 66.4 | 70.6 | 73.3 | 65.3 | -8.8 [-14.5, -3.1] vs prompt | -8.8 [-14.5, -3.1] vs prompt |
| Spanish order | 57.9 | 53.1 | 66.4 | 64.5 | 49.6 | -6.5 [-13.4, +0.0] vs prompt | -8.4 [-14.6, -2.7] vs steer |
| Chinese order | 61.8 | 60.7 | 66.7 | 67.3 | 60.3 | -5.0 [-10.8, +0.8] vs prompt | -5.0 [-10.8, +0.8] vs prompt |

## Qwen2.5-3B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 94.7 | 92.0 | 90.5 | 92.0 | 81.7 | +2.7 [+0.0, +5.7] vs prompt | |
| many-shot lies | 87.8 | 88.9 | 88.5 | 88.9 | 88.5 | -1.1 [-3.8, +1.5] vs prompt | -1.1 [-4.6, +1.9] vs liemask |
| Spanish order | 90.8 | 88.5 | 89.3 | 93.9 | 78.9 | -3.1 [-6.1, -0.4] vs prompt | -3.1 [-6.1, -0.4] vs prompt |
| Chinese order | 87.4 | 91.2 | 84.0 | 90.8 | 79.8 | -3.4 [-6.9, +0.0] vs prompt | -3.8 [-7.6, -0.4] vs liemask |
