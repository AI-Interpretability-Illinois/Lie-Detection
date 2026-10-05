# Adaptive attacks on the order to lie (Qwen2.5-7B judge)

Honest rate (%) on the test half of the 537 LIARS' BENCH deception items under each attack, every method at its
LIARS' BENCH-chosen setting; 'plain' is the same items without the attack (Table 1, same judge). Lead: mask minus the
comparator fixed on LIARS' BENCH validation, paired bootstrap 95% interval over test items.

## Llama-3.1-8B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 93.5 | 82.8 | 87.0 | 65.9 | 37.2 | +6.5 [+1.9, +11.1] vs steer | |
| many-shot lies | 89.3 | - | 90.0 | 82.8 | 55.2 | -0.8 [-5.0, +3.1] vs steer | -0.8 [-5.0, +3.1] vs steer |
| Spanish order | 93.1 | - | 85.0 | 68.0 | 38.2 | +8.1 [+3.8, +12.7] vs steer | +8.1 [+3.8, +12.7] vs steer |
| Chinese order | 92.7 | - | 84.6 | 73.4 | 52.6 | +8.2 [+3.5, +12.8] vs steer | +8.2 [+3.5, +12.8] vs steer |

## Phi-4-mini

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 86.3 | 69.5 | 76.0 | 79.4 | 69.8 | +6.9 [+2.3, +11.1] vs prompt | |
| many-shot lies | 93.1 | - | 92.0 | 90.5 | 88.9 | +2.7 [-1.1, +6.5] vs prompt | +1.1 [-2.3, +4.6] vs steer |
| Spanish order | 91.6 | - | 88.5 | 85.9 | 71.0 | +5.7 [+1.9, +10.0] vs prompt | +3.1 [-0.8, +6.5] vs steer |
| Chinese order | 91.6 | - | 85.9 | 86.6 | 79.0 | +5.0 [+0.8, +9.2] vs prompt | +5.0 [+0.8, +9.2] vs prompt |

## Falcon3-3B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 89.7 | - | 79.4 | - | 80.2 | +10.3 [+4.6, +16.0] vs steer | |
| many-shot lies | 87.0 | - | 87.0 | 86.6 | 80.9 | +0.0 [-4.6, +5.0] vs steer | +0.0 [-4.6, +5.0] vs steer |
| Spanish order | 87.7 | - | 82.1 | 82.6 | 77.8 | +6.0 [+0.4, +11.5] vs steer | +5.0 [-0.9, +11.0] vs prompt |
| Chinese order | 95.0 | - | 85.1 | 95.9 | 82.4 | +9.8 [+4.5, +14.2] vs steer | -1.2 [-4.1, +1.7] vs prompt |

## OLMo-2-1B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 58.0 | 56.1 | - | 61.1 | 50.4 | -3.1 [-8.8, +2.3] vs prompt | |
| many-shot lies | 66.8 | - | 78.6 | 67.9 | 68.7 | -1.1 [-8.0, +5.7] vs prompt | -11.8 [-19.1, -5.0] vs steer |
| Spanish order | 58.2 | - | 66.4 | 64.4 | 52.3 | -6.2 [-12.3, -0.4] vs prompt | -8.1 [-17.4, +0.8] vs steer |
| Chinese order | 63.7 | - | 71.8 | 66.0 | 61.3 | -2.3 [-8.8, +3.8] vs prompt | -8.0 [-16.4, +0.4] vs steer |

## Qwen2.5-3B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 94.3 | 92.0 | 90.5 | 92.0 | 81.7 | +2.3 [-0.8, +5.3] vs prompt | |
| many-shot lies | 88.5 | - | 89.7 | 87.4 | 88.9 | +1.1 [-1.1, +3.8] vs prompt | -1.1 [-5.0, +2.7] vs steer |
| Spanish order | 87.0 | - | 88.9 | 93.9 | 77.9 | -6.9 [-10.3, -3.8] vs prompt | -6.9 [-10.3, -3.8] vs prompt |
| Chinese order | 82.4 | - | 82.8 | 92.0 | 79.0 | -9.5 [-13.7, -5.3] vs prompt | -9.5 [-13.7, -5.3] vs prompt |

## Qwen2.5-1.5B

| condition | mask | lie mask | steering | honest prompt | unedited | lead (pre-registered) | lead vs best other |
|---|---|---|---|---|---|---|---|
| plain (Table 1) | 87.0 | 85.1 | 90.5 | 83.2 | 78.6 | -3.4 [-7.3, +0.4] vs steer | |
| many-shot lies | 82.1 | - | 90.8 | 84.7 | 84.4 | -8.8 [-13.7, -4.2] vs steer | -8.8 [-13.7, -4.2] vs steer |
| Spanish order | 86.6 | - | 78.9 | 74.0 | 66.4 | +7.7 [+3.1, +12.3] vs steer | +7.7 [+3.1, +12.3] vs steer |
| Chinese order | 80.8 | - | 87.3 | 79.9 | 72.3 | -6.2 [-10.9, -1.2] vs steer | -6.2 [-10.9, -1.2] vs steer |
