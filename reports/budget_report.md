# Mask-budget sweep: the contrast threshold r

Family = the mask arm chosen at r = 0.37. Per strength: validation honest rate on LIARS' BENCH (perplexity ratio; gates failed). Best = test-half honest rate of the best usable strength at that r.

## Qwen2.5-7B (pooled)

| r | a=−0.25 | a=−0.5 | a=−0.75 | a=−1.0 | best usable (test) | honest prompt (test) | unedited (test) |
|---|---|---|---|---|---|---|---|
| 0.2 | 89.1 (1.00) | 90.9 (1.00) | 94.9 (1.01) | 95.3 (1.01) | 95.4 (-1.0) | 96.6 | 88.5 |
| 0.37 | 89.1 (1.00) | 90.9 (1.02) | 93.5 (1.05) | 96.4 (1.09) | 95.4 (-1.0) | 96.6 | 88.5 |
| 0.6 | 91.6 (1.03) | 94.2 (1.12) | 94.5 (1.29; ppl) | 85.1 (1.57; ppl,ctl,pol) | 96.2 (-0.5) | 96.6 | 88.5 |
| 1.0 | 88.0 (1.65; ppl,ctl) | 2.5 (648.77; ppl,ctl,pol) | 0.4 (27681.30; ppl,ctl,pol) | 1.1 (529595.83; ppl,ctl,pol) | – | 96.6 | 88.5 |

## Llama-3.1-8B (fiction)

| r | a=−0.25 | a=−0.5 | a=−0.75 | a=−1.0 | best usable (test) | honest prompt (test) | unedited (test) |
|---|---|---|---|---|---|---|---|
| 0.2 | 38.4 (1.00) | 45.8 (1.00) | 55.7 (1.00) | 60.9 (1.01) | 65.5 (-1.0) | 80.1 | 37.2 |
| 0.37 | 49.1 (1.00) | 74.9 (1.01) | 84.1 (1.03) | 94.1 (1.07) | 93.5 (-1.0) | 80.1 | 37.2 |
| 0.6 | 67.9 (1.02) | 84.1 (1.13) | 88.2 (1.42; ppl,ctl) | 51.3 (2.38; ppl,ctl,pol) | 82.0 (-0.5) | 80.1 | 37.2 |
| 1.0 | 59.4 (1.45; ppl,ctl) | 3.7 (204.79; ppl,ctl,pol) | 0.7 (13531.34; ppl,ctl,pol) | 4.4 (32747.46; ppl,ctl,pol) | – | 80.1 | 37.2 |

## OLMo-2-7B (fiction)

| r | a=−0.25 | a=−0.5 | a=−0.75 | a=−1.0 | best usable (test) | honest prompt (test) | unedited (test) |
|---|---|---|---|---|---|---|---|
| 0.2 | 51.3 (1.00) | 52.0 (1.00) | 53.1 (1.00) | 53.5 (1.01) | 53.1 (-1.0) | 77.1 | 52.7 |
| 0.37 | 54.2 (1.00) | 57.8 (1.01) | 62.2 (1.02) | 65.5 (1.04; ctl) | 60.3 (-0.75) | 77.1 | 52.7 |
| 0.6 | 56.0 (1.01) | 66.2 (1.06) | 76.4 (1.17; ppl,ctl,pol) | 78.9 (1.64; ppl,ctl,pol) | 66.8 (-0.5) | 77.1 | 52.7 |
| 1.0 | 61.5 (1.73; ppl) | 1.1 (283.41; ppl,ctl,pol) | 1.8 (1063.82; ppl,ctl,pol) | 1.1 (2874.08; ppl,ctl,pol) | – | 77.1 | 52.7 |

