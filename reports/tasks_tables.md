## Lead of our mask over the best other method (95% CI)

| model | Sandbagging (MMLU+WMDP) | Injected instruction | MASK, instructed | Azaria-Mitchell / GoT | Multi-turn | MASK, incentive (no instruction) |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | +0.4 [-7.0, +7.8] vs liemask | +7.8 [+3.5, +12.1] vs liemask | -21.3 [-29.3, -13.2] vs steer | +9.9 [+7.3, +12.7] vs steer | -20.2 [-25.8, -14.7] vs steer | -17.0 [-25.0, -8.5] vs steer |
| Phi-4-mini | +0.4 [-7.6, +7.6] vs steer | +3.0 [-1.2, +7.0] vs steer | -4.6 [-9.7, +0.5] vs steer | -5.7 [-8.4, -3.3] vs prompt | -23.3 [-29.3, -17.6] vs prompt | +0.5 [-7.6, +8.1] vs steer |
| Falcon3-3B | -8.9 [-16.3, -2.5] vs unsteered | +5.2 [+1.0, +9.2] vs steer | -0.6 [-5.8, +4.7] vs unsteered | +5.1 [+2.5, +7.8] vs steer | -20.9 [-27.3, -15.4] vs steer | -10.1 [-16.7, -3.0] vs unsteered |
| OLMo-2-1B | -8.5 [-17.6, +0.7] vs unsteered | +0.0 [-5.0, +5.0] vs liemask | -5.1 [-10.1, -0.6] vs liemask | -9.3 [-12.5, -5.9] vs prompt | -4.0 [-9.8, +1.8] vs prompt | -2.3 [-8.7, +4.1] vs prompt |
| Qwen2.5-3B | -18.3 [-24.1, -12.6] vs liemask | -0.8 [-5.0, +3.5] vs steer | -3.5 [-8.5, +1.5] vs prompt | -4.4 [-6.5, -2.3] vs liemask | -5.9 [-10.7, -1.0] vs liemask | -5.7 [-12.5, +0.5] vs steer |
| Qwen2.5-7B | -3.8 [-7.2, -0.3] vs liemask | -4.2 [-8.2, -0.2] vs steer | -5.8 [-10.6, -0.5] vs steer | -0.5 [-1.8, +0.7] vs prompt | -7.7 [-11.9, -3.9] vs steer | -2.2 [-8.1, +3.8] vs steer |
| Qwen2.5-1.5B | +0.0 [-4.3, +4.3] vs steer | -2.2 [-6.0, +1.8] vs steer | -5.7 [-11.4, -0.6] vs unsteered | -4.4 [-6.9, -1.8] vs steer | -3.9 [-7.9, +0.0] vs unsteered | -2.9 [-10.6, +4.3] vs unsteered |

## Lead over the comparator fixed on LIARS' BENCH validation (95% CI)

| model | Sandbagging (MMLU+WMDP) | Injected instruction | MASK, instructed | Azaria-Mitchell / GoT | Multi-turn | MASK, incentive (no instruction) |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | +8.9 [+2.3, +15.9] vs steer | +10.5 [+6.2, +14.8] vs steer | -21.3 [-29.3, -13.2] vs steer | +9.9 [+7.3, +12.7] vs steer | -20.2 [-25.8, -14.7] vs steer | -17.0 [-25.0, -8.5] vs steer |
| Phi-4-mini | +1.5 [-5.3, +8.0] vs prompt | +6.0 [+2.0, +10.2] vs prompt | -1.5 [-6.6, +3.5] vs prompt | -5.7 [-8.4, -3.3] vs prompt | -23.3 [-29.3, -17.6] vs prompt | +2.2 [-5.4, +9.7] vs prompt |
| Falcon3-3B | -8.4 [-16.3, -1.0] vs steer | +5.2 [+1.0, +9.2] vs steer | +4.1 [-0.6, +8.8] vs steer | +5.1 [+2.5, +7.8] vs steer | -20.9 [-27.3, -15.4] vs steer | -4.8 [-12.0, +2.4] vs steer |
| OLMo-2-1B | -0.7 [-9.2, +8.5] vs prompt | +4.8 [-0.5, +9.8] vs prompt | -1.7 [-7.3, +3.9] vs prompt | -9.3 [-12.5, -5.9] vs prompt | -4.0 [-9.8, +1.8] vs prompt | -2.3 [-8.7, +4.1] vs prompt |
| Qwen2.5-3B | +7.6 [+2.2, +12.9] vs prompt | +2.0 [-1.0, +5.2] vs prompt | -3.5 [-8.5, +1.5] vs prompt | +3.7 [+1.8, +5.6] vs prompt | -3.6 [-9.1, +2.0] vs prompt | -4.2 [-10.9, +2.6] vs prompt |
| Qwen2.5-7B | -3.8 [-7.2, -0.3] vs liemask | -3.5 [-6.8, -0.5] vs liemask | -1.9 [-7.2, +3.4] vs liemask | +0.3 [-0.9, +1.3] vs liemask | -7.4 [-11.0, -4.2] vs liemask | +4.3 [-1.6, +9.7] vs liemask |
| Qwen2.5-1.5B | +0.0 [-4.3, +4.3] vs steer | -2.2 [-6.0, +1.8] vs steer | -4.5 [-10.1, +1.1] vs steer | -4.4 [-6.9, -1.8] vs steer | +3.3 [-1.0, +7.6] vs steer | +2.4 [-5.3, +9.6] vs steer |

## Sandbagging (MMLU+WMDP): honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 67.4 | 67.1 | 58.5 | 52.3 | 38.0 | 40.3 |  n=258
| Phi-4-mini | 61.4 | 54.5 | 61.0 | 59.8 | 40.2 | 46.2 |  n=264
| Falcon3-3B | 33.7 | - | 42.1 | - | 42.6 | 41.6 |  n=202
| OLMo-2-1B | 32.4 | 40.1 | - | 33.1 | 40.8 | 36.6 |  n=142
| Qwen2.5-3B | 60.4 | 78.8 | 45.7 | 52.9 | 46.8 | 55.4 |  n=278
| Qwen2.5-7B | 87.2 | 91.0 | 76.2 | 52.1 | 33.8 | 47.9 |  n=290
| Qwen2.5-1.5B | 85.8 | 81.0 | 85.8 | 83.8 | 82.6 | 79.8 |  n=253

## Injected instruction: honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 86.8 | 79.0 | 76.3 | 69.9 | 65.9 | 54.8 |  n=372
| Phi-4-mini | 67.8 | 60.2 | 64.8 | 61.8 | 59.8 | 62.0 |  n=400
| Falcon3-3B | 86.5 | - | 81.2 | - | 80.8 | 80.8 |  n=400
| OLMo-2-1B | 54.8 | 54.8 | - | 50.0 | 54.2 | 52.2 |  n=400
| Qwen2.5-3B | 81.0 | 70.5 | 81.8 | 79.0 | 78.2 | 75.0 |  n=400
| Qwen2.5-7B | 71.0 | 74.5 | 75.2 | 72.8 | 71.5 | 71.2 |  n=400
| Qwen2.5-1.5B | 80.0 | 70.2 | 82.2 | 77.5 | 75.0 | 77.2 |  n=400

## MASK, instructed: honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 15.4 | 21.9 | 37.3 | 24.7 | 20.8 | 21.5 |  n=178
| Phi-4-mini | 10.6 | 11.0 | 15.2 | 11.8 | 14.6 | 11.9 |  n=198
| Falcon3-3B | 14.0 | - | 9.8 | - | 14.5 | 16.4 |  n=172
| OLMo-2-1B | 20.8 | 25.8 | - | 22.3 | 23.0 | 19.7 |  n=178
| Qwen2.5-3B | 20.7 | 19.4 | 20.9 | 24.4 | 17.8 | 14.0 |  n=202
| Qwen2.5-7B | 14.8 | 16.8 | 20.7 | 19.1 | 14.4 | 16.2 |  n=208
| Qwen2.5-1.5B | 11.7 | 15.3 | 16.1 | 14.6 | 17.7 | 12.6 |  n=175

## Azaria-Mitchell / GoT: honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 86.5 | 76.1 | 76.6 | 64.1 | 38.5 | 44.2 |  n=1100
| Phi-4-mini | 81.5 | 79.2 | 83.2 | 87.3 | 79.5 | 74.6 |  n=1100
| Falcon3-3B | 81.5 | - | 76.4 | - | 74.8 | 77.5 |  n=1100
| OLMo-2-1B | 51.9 | 52.8 | - | 61.2 | 49.1 | 47.7 |  n=1100
| Qwen2.5-3B | 88.4 | 92.7 | 78.6 | 84.6 | 78.1 | 75.8 |  n=1100
| Qwen2.5-7B | 95.3 | 95.0 | 87.1 | 95.8 | 80.8 | 80.5 |  n=1100
| Qwen2.5-1.5B | 80.2 | 73.4 | 84.5 | 79.4 | 72.7 | 71.7 |  n=1100

## Multi-turn: honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 70.6 | 82.5 | 90.8 | 65.3 | 58.9 | 62.3 |  n=326
| Phi-4-mini | 50.7 | 55.5 | 58.2 | 74.0 | 62.4 | 59.1 |  n=335
| Falcon3-3B | 46.6 | - | 67.6 | - | 60.1 | 62.8 |  n=253
| OLMo-2-1B | 40.2 | 38.8 | - | 44.2 | 43.3 | 46.4 |  n=224
| Qwen2.5-3B | 68.4 | 74.3 | 66.4 | 72.0 | 67.8 | 56.0 |  n=307
| Qwen2.5-7B | 78.6 | 86.1 | 86.4 | 84.6 | 80.4 | 84.3 |  n=337
| Qwen2.5-1.5B | 81.9 | 72.0 | 78.6 | 83.6 | 85.9 | 81.6 |  n=304

## MASK, incentive (no instruction): honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 47.8 | 40.8 | 64.7 | 54.7 | 33.3 | 33.3 |  n=201
| Phi-4-mini | 43.5 | 33.5 | 43.0 | 41.0 | 36.8 | 42.2 |  n=185
| Falcon3-3B | 51.8 | - | 55.9 | - | 61.5 | 50.6 |  n=169
| OLMo-2-1B | 50.6 | 51.7 | - | 52.9 | 50.3 | 50.9 |  n=173
| Qwen2.5-3B | 46.4 | 45.9 | 52.1 | 50.0 | 49.7 | 45.3 |  n=193
| Qwen2.5-7B | 52.2 | 47.8 | 54.3 | 47.8 | 43.0 | 47.3 |  n=186
| Qwen2.5-1.5B | 49.0 | 40.5 | 46.6 | 51.4 | 51.7 | 51.9 |  n=209

## Injection control (no injection): honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 87.5 | 67.6 | 79.4 | 75.0 | 70.6 | 59.6 |  n=136
| Phi-4-mini | 82.7 | 84.7 | 90.0 | 87.3 | 83.3 | 86.0 |  n=150
| Falcon3-3B | 90.7 | - | 89.3 | - | 92.0 | 92.7 |  n=150
| OLMo-2-1B | 80.0 | 79.3 | - | 80.0 | 76.7 | 74.7 |  n=150
| Qwen2.5-3B | 80.0 | 69.3 | 72.7 | 74.7 | 77.3 | 68.0 |  n=150
| Qwen2.5-7B | 66.7 | 67.3 | 78.0 | 74.0 | 74.0 | 66.0 |  n=150
| Qwen2.5-1.5B | 80.7 | 73.3 | 88.0 | 86.0 | 79.3 | 84.0 |  n=150

## Plain MMLU (capability): honest rate % (n)

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 72.3 | 80.7 | 68.1 | 83.2 | 89.1 | 89.9 |  n=119
| Phi-4-mini | 82.4 | 83.2 | 87.0 | 90.1 | 90.8 | 87.8 |  n=131
| Falcon3-3B | 85.6 | - | 88.9 | - | 96.7 | 91.1 |  n=90
| OLMo-2-1B | 67.2 | 73.8 | - | 60.7 | 65.6 | 65.6 |  n=61
| Qwen2.5-3B | 92.5 | 90.8 | 99.2 | 94.2 | 98.3 | 90.0 |  n=120
| Qwen2.5-7B | 90.3 | 91.8 | 91.0 | 97.0 | 97.8 | 95.5 |  n=134
| Qwen2.5-1.5B | 86.0 | 86.0 | 87.7 | 90.4 | 93.9 | 86.8 |  n=114
