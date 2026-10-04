# CreativityNeuro-style set-difference baseline against the contrast mask

Test-half honest rate (%), arm chosen on the validation half within each family; '–' = every strength failed a usability gate (perplexity / control / polarity). Lead = paired difference ours − CN on common test items, 95% bootstrap interval.

## LIARS' BENCH

| model | unedited | ours (score, mean) | score, rms | set diff, mean | CN (set diff, rms) | ours − CN |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 37.2 | 93.5 | 90.0 | 65.9 | 78.9 | +14.6 [+9.2, +19.9] |
| Phi-4-mini | 69.8 | 86.3 | 93.1 | 80.5 | 82.8 | +3.4 [-1.1, +8.0] |
| Falcon3-3B | 80.2 | 89.7 | pending | pending | pending | – |
| OLMo-2-1B | 50.4 | 58.0 | pending | pending | pending | – |
| Qwen2.5-3B | 81.7 | 94.3 | pending | pending | pending | – |
| Qwen2.5-7B | 88.5 | 95.4 | pending | pending | pending | – |
| Qwen2.5-1.5B | 78.6 | 87.0 | pending | pending | pending | – |

2×2 over the 2 model(s) with all four cells usable (mean honest rate):

| | mean norms | rms norms |
|---|---|---|
| contrast score | 89.9 | 91.5 |
| set difference | 73.2 | 80.8 |

rule effect (score − set difference, averaged over statistics): +13.7 points; statistic effect (mean − rms, averaged over rules): -4.6 points.

## AI Liar

| model | unedited | ours (score, mean) | score, rms | set diff, mean | CN (set diff, rms) | ours − CN |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 27.1 | 89.7 | 88.7 | 67.4 | 70.3 | +19.4 [+14.6, +24.0] |
| Phi-4-mini | 47.1 | 78.7 | 74.2 | 59.3 | 64.6 | +14.1 [+8.9, +19.4] |
| Falcon3-3B | 73.0 | 78.3 | pending | pending | pending | – |
| OLMo-2-1B | 60.5 | 68.8 | pending | pending | pending | – |
| Qwen2.5-3B | 65.7 | 78.8 | pending | pending | pending | – |
| Qwen2.5-7B | 81.8 | 90.8 | pending | pending | pending | – |
| Qwen2.5-1.5B | 73.1 | 81.3 | pending | pending | pending | – |

2×2 over the 2 model(s) with all four cells usable (mean honest rate):

| | mean norms | rms norms |
|---|---|---|
| contrast score | 84.2 | 81.5 |
| set difference | 63.4 | 67.4 |

rule effect (score − set difference, averaged over statistics): +17.4 points; statistic effect (mean − rms, averaged over rules): -0.7 points.

## Mask anatomy (qwen_anatomy.py), family of the mask arm chosen on LIARS' BENCH

| model | family | CN weights (% of matrix) | ours ∩ CN / CN | CN weights with positive contrast score | attention share ours / CN | J(ours, CN) at ρ = 1 / 2 / 4 / 8 % | ours ρ=4 vs ρ=1 / 2 / 8 |
|---|---|---|---|---|---|---|---|
| Phi-4-mini | pooled | 1.04 | 5% | 5% | 25 / 22% | 0.02 / 0.01 / 0.01 / 0.01 | 0.25 / 0.50 / 0.50 |
| Falcon3-3B | fictf | 0.79 | 8% | 8% | 23 / 21% | 0.02 / 0.02 / 0.01 / 0.01 | 0.25 / 0.50 / 0.50 |
| OLMo-2-1B | fiction | 0.96 | 6% | 6% | 25 / 19% | 0.02 / 0.02 / 0.01 / 0.01 | 0.25 / 0.50 / 0.50 |
| Qwen2.5-3B | pooltf | 0.73 | 4% | 4% | 12 / 11% | 0.01 / 0.01 / 0.01 / 0.00 | 0.25 / 0.50 / 0.50 |
| Qwen2.5-1.5B | fictf | 0.83 | 8% | 7% | 12 / 10% | 0.03 / 0.02 / 0.01 / 0.01 | 0.25 / 0.50 / 0.50 |

