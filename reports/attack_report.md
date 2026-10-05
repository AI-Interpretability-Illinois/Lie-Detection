# Adaptive attacks on the order to lie (Qwen2.5-7B judge)

Honest rate (%) on the test half of the 537 LIARS' BENCH deception items under each attack, every method at its
LIARS' BENCH-chosen setting; 'plain' is the same items without the attack (Table 1, same judge). Lead: mask minus the
comparator fixed on LIARS' BENCH validation, paired bootstrap 95% interval over test items.

## OLMo-2-1B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 58.0 | 56.1 | - | 61.1 | 50.4 | -3.1 [-8.8, +2.3] vs prompt | |
| many-shot lies | 66.8 | - | 78.6 | 67.9 | 68.7 | -1.1 [-8.0, +5.7] vs prompt | -11.8 [-19.1, -5.0] vs steer |
| Spanish order | 58.2 | - | 66.4 | 64.4 | 52.3 | -6.2 [-12.3, -0.4] vs prompt | -8.1 [-17.4, +0.8] vs steer |
| Chinese order | 63.7 | - | 71.8 | 66.0 | 61.3 | -2.3 [-8.8, +3.8] vs prompt | -8.0 [-16.4, +0.4] vs steer |

## Qwen2.5-1.5B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 87.0 | 85.1 | 90.5 | 83.2 | 78.6 | -3.4 [-7.3, +0.4] vs steer | |
| many-shot lies | 82.1 | - | 90.8 | 84.7 | 84.4 | -8.8 [-13.7, -4.2] vs steer | -8.8 [-13.7, -4.2] vs steer |
| Spanish order | 86.6 | - | 78.9 | 74.0 | 66.4 | +7.7 [+3.1, +12.3] vs steer | +7.7 [+3.1, +12.3] vs steer |
| Chinese order | 80.8 | - | 87.3 | 79.9 | 72.3 | -6.2 [-10.9, -1.2] vs steer | -6.2 [-10.9, -1.2] vs steer |
