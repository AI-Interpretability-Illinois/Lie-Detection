# Judge audit: Qwen2.5-7B judge vs GPT (gpt-5.6-sol)

Stance labels on the same replies (AGREES / DISAGREES / NEITHER); beliefs shared. Chosen arms of the main table.

| family | replies | judged (keyword unsettled) | agreement | kappa | 7B calls honest, GPT lie | GPT honest, 7B lie |
|---|---|---|---|---|---|---|
| Falcon | 7098 | 1393 | 0.865 | 0.744 | 150 (2.1%) | 104 (1.5%) |
| Llama | 12009 | 3964 | 0.818 | 0.698 | 324 (2.7%) | 206 (1.7%) |
| OLMo | 9285 | 1271 | 0.868 | 0.739 | 180 (1.9%) | 72 (0.8%) |
| Phi | 11472 | 705 | 0.933 | 0.886 | 168 (1.5%) | 21 (0.2%) |
| Qwen | 36027 | 2358 | 0.942 | 0.912 | 571 (1.6%) | 41 (0.1%) |
| all | 75891 | 9691 | 0.870 | 0.785 | 1393 (1.8%) | 444 (0.6%) |

## Honest rate on the test half, chosen arms: 7B-hybrid -> GPT-hybrid

| model | arm | 7B | GPT | n |
|---|---|---|---|---|
| Falcon3-3B | actsteer:L5 r=-0.18 (c=-13.0) | 79.4 | 80.5 | 262 |
| Falcon3-3B | unsteered | 80.2 | 80.5 | 262 |
| Falcon3-3B | weight:fictf a=-1.0 | 89.7 | 90.8 | 262 |
| Falcon3-3B | weight:poolmix a=-1.0 | 90.5 | 90.5 | 262 |
| Llama-3.1-8B | actsteer:L8 r=-0.5 (c=-2.8) | 87.0 | 84.7 | 261 |
| Llama-3.1-8B | prompt:honest2 | 65.9 | 64.8 | 261 |
| Llama-3.1-8B | prompt:honest3 | 66.7 | 64.4 | 261 |
| Llama-3.1-8B | unsteered | 37.2 | 33.3 | 261 |
| Llama-3.1-8B | weight:fiction a=-1.0 | 93.5 | 93.1 | 261 |
| Llama-3.1-8B | weight:ours a=-1.0 | 82.8 | 78.9 | 261 |
| Llama-3.1-8B | weight:pooled a=-1.0 | 88.1 | 87.7 | 261 |
| OLMo-2-1B | prompt:honest | 61.1 | 62.2 | 262 |
| OLMo-2-1B | unsteered | 50.4 | 51.5 | 262 |
| OLMo-2-1B | weight:fiction a=-0.25 | 58.0 | 58.4 | 262 |
| OLMo-2-1B | weight:ours a=-1.0 | 56.1 | 56.9 | 262 |
| OLMo-2-1B | weight:pooled a=-1.0 | 55.0 | 55.3 | 262 |
| Phi-4-mini | actsteer:L16 r=-0.18 (c=-10.4) | 76.0 | 76.3 | 262 |
| Phi-4-mini | prompt:honest2 | 79.4 | 79.4 | 262 |
| Phi-4-mini | unsteered | 69.8 | 70.2 | 262 |
| Phi-4-mini | weight:fiction a=-1.0 | 86.6 | 86.6 | 262 |
| Phi-4-mini | weight:ours a=-0.5 | 69.5 | 69.1 | 262 |
| Phi-4-mini | weight:pooled a=-1.0 | 86.3 | 86.3 | 262 |
| Qwen2.5-1.5B | actsteer:L14 r=-0.18 (c=-9.4) | 90.5 | 90.5 | 262 |
| Qwen2.5-1.5B | prompt:honest | 79.4 | 79.8 | 262 |
| Qwen2.5-1.5B | prompt:honest3 | 83.2 | 83.6 | 262 |
| Qwen2.5-1.5B | unsteered | 78.6 | 78.6 | 262 |
| Qwen2.5-1.5B | weight:fictf a=-1.0 | 87.0 | 87.0 | 262 |
| Qwen2.5-1.5B | weight:fiction a=-0.5 | 81.7 | 81.7 | 262 |
| Qwen2.5-1.5B | weight:ours a=-1.0 | 85.1 | 85.1 | 262 |
| Qwen2.5-3B | actsteer:L27 r=-0.18 (c=-18.7) | 90.5 | 90.5 | 262 |
| Qwen2.5-3B | actsteer:L9 r=-0.5 (c=-19.0) | 88.2 | 88.5 | 262 |
| Qwen2.5-3B | prompt:honest | 92.0 | 92.0 | 262 |
| Qwen2.5-3B | unsteered | 81.7 | 81.7 | 262 |
| Qwen2.5-3B | weight:ours a=-1.0 | 92.0 | 92.0 | 262 |
| Qwen2.5-3B | weight:poolmix a=-1.0 | 94.7 | 94.7 | 262 |
| Qwen2.5-3B | weight:pooltf a=-1.0 | 94.3 | 94.3 | 262 |
| Qwen2.5-7B | actsteer:L14 r=-0.18 (c=-11.0) | 90.8 | 90.8 | 262 |
| Qwen2.5-7B | prompt:honest | 95.8 | 95.8 | 262 |
| Qwen2.5-7B | prompt:honest2 | 96.2 | 96.2 | 262 |
| Qwen2.5-7B | unsteered | 88.5 | 88.5 | 262 |
| Qwen2.5-7B | weight:ours a=-1.0 | 95.0 | 95.0 | 262 |
| Qwen2.5-7B | weight:pooled a=-0.75 | 96.2 | 96.2 | 262 |
| Qwen2.5-7B | weight:pooled a=-1.0 | 95.4 | 95.4 | 262 |

## Lead of the mask over the best other method (test half), 7B grader vs GPT grader

| model | vs | 7B lead [95% CI] | GPT lead [95% CI] |
|---|---|---|---|
| Falcon3-3B | steer | +10.3 [+4.6, +16.0] (n=262) | +10.3 [+4.6, +16.0] (n=262) |
| Llama-3.1-8B | steer | +6.5 [+1.9, +11.1] (n=261) | +8.4 [+3.8, +13.0] (n=261) |
| OLMo-2-1B | prompt | -3.1 [-8.8, +2.3] (n=262) | -3.8 [-9.5, +1.9] (n=262) |
| Phi-4-mini | prompt | +6.9 [+2.3, +11.1] (n=262) | +6.9 [+2.3, +11.1] (n=262) |
| Qwen2.5-1.5B | steer | -3.4 [-7.3, +0.4] (n=262) | -3.4 [-7.3, +0.4] (n=262) |
| Qwen2.5-3B | prompt | +2.3 [-0.8, +5.3] (n=262) | +2.3 [-0.8, +5.3] (n=262) |
| Qwen2.5-7B | liemask | +0.4 [-1.9, +2.7] (n=262) | +0.4 [-1.9, +2.7] (n=262) |
