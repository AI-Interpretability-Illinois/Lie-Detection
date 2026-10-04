# Mask anatomy: why the edit does less on the Qwen models

Family = the mask arm chosen on LIARS' BENCH validation. Headroom = unedited honest rate -> honest-prompt control, test half. Positive = channels/weights with a positive contrast score (the rest of the 4% budget is near-zero padding). Outlier channels = truthful-prompt norm above the layer's 99th percentile (base rate 1% of weights).

| model | family | unedited -> honest prompt | lead | mask gain / headroom | positive channels | positive mass in top 1% / 4% of channels | positive weights (% of all) | outlier share: positive / mask | ratio contrast: outlier share, J(diff, ratio) | attention share of mask (of all weights) | depth thirds (first / mid / last) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | fiction | 37.2 -> 65.9 | +6.5 [+1.9, +11.1] | +56.3 / 62.8 = 90% | 1.11% | 0.76 / 0.96 | 0.82% | 0% / 0.3% | 0%, 1.00 | 19% (19%) | 31 / 34 / 34 |
| Phi-4-mini | pooled | 69.8 -> 79.4 | +6.9 [+2.3, +11.1] | +16.5 / 30.2 = 55% | 0.91% | 0.86 / 0.99 | 0.48% | 0% / 0.3% | 0%, 1.00 | 25% (25%) | 31 / 34 / 34 |
| Falcon3-3B | fictf | 80.2 -> – | +10.3 [+4.6, +16.0] | +9.5 / 19.8 = 48% | 0.34% | 0.99 / 1.00 | 0.26% | 0% / 0.3% | 0%, 1.00 | 23% (23%) | 32 / 32 / 36 |
| OLMo-2-1B | fiction | 50.4 -> 61.1 | -3.1 [-8.8, +2.3] | +7.6 / 49.6 = 15% | 0.41% | 0.97 / 1.00 | 0.25% | 0% / 0.3% | 0%, 1.00 | 25% (25%) | 31 / 31 / 38 |
| Qwen2.5-1.5B | fictf | 78.6 -> 83.2 | -3.4 [-7.3, +0.4] | +8.4 / 21.4 = 39% | 0.42% | 0.98 / 1.00 | 0.25% | 0% / 0.2% | 0%, 1.00 | 12% (12%) | 32 / 32 / 36 |
| Qwen2.5-3B | pooltf | 81.7 -> 92.0 | +2.3 [-0.8, +5.3] | +12.6 / 18.3 = 69% | 0.20% | 0.99 / 1.00 | 0.12% | 0% / 0.2% | 0%, 1.00 | 12% (12%) | 33 / 33 / 33 |
| Qwen2.5-7B | pooled | 88.5 -> 95.8 | +0.4 [-1.9, +2.7] | +6.9 / 11.5 = 60% | 0.99% | 0.90 / 0.99 | 0.61% | 0% / 0.2% | 0%, 1.00 | 13% (13%) | 32 / 32 / 36 |

## Ratio contrast (n_F / n_T - 1/r) in place of the difference contrast, same protocol

| model | bench | mask (difference) | mask (ratio) | lead (difference) | lead (ratio) |
|---|---|---|---|---|---|
| Llama-3.1-8B | lb | 93.5 | 93.5 | +6.5 [+1.9, +11.1] vs steer | +6.5 [+1.9, +11.1] vs steer |
| Llama-3.1-8B | ail | 89.7 | 89.7 | +1.9 [-1.9, +6.0] vs steer | +1.9 [-1.9, +6.0] vs steer |
| Qwen2.5-7B | lb | 95.4 | 95.4 | +0.4 [-1.9, +2.7] vs liemask | +0.4 [-1.9, +2.7] vs liemask |
| Qwen2.5-7B | ail | 90.8 | 90.8 | -1.2 [-3.8, +1.2] vs steer | -1.2 [-3.8, +1.2] vs steer |
