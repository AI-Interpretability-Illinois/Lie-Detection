# Legitimate falsehoods: do the edited models still comply? (GPT-5.6 labels)

150 requests (50 fiction, 50 role-play, 50 counterfactual) that legitimately ask for a false statement inside a frame;
share of replies that complied / corrected the claim instead / refused, every method at its LIARS' BENCH-chosen setting.

## Llama-3.1-8B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 86.0 | 13.3 | 0.7 | 86 / 84 / 88 |
| steer | 150 | 58.7 | 41.3 | 0.0 | 22 / 68 / 86 |
| prompt | 150 | 96.7 | 3.3 | 0.0 | 92 / 100 / 98 |
| unsteered | 150 | 96.7 | 3.3 | 0.0 | 90 / 100 / 100 |

## Falcon3-3B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 84.0 | 15.3 | 0.7 | 76 / 82 / 94 |
| steer | 150 | 77.3 | 21.3 | 1.3 | 62 / 74 / 96 |
| prompt | 150 | 78.7 | 19.3 | 2.0 | 68 / 68 / 100 |
| unsteered | 150 | 78.0 | 21.3 | 0.7 | 66 / 74 / 94 |

## OLMo-2-1B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 75.3 | 24.0 | 0.7 | 58 / 72 / 96 |
| steer | 150 | 79.3 | 18.7 | 2.0 | 86 / 66 / 86 |
| prompt | 150 | 74.0 | 25.3 | 0.7 | 62 / 66 / 94 |
| unsteered | 150 | 79.3 | 20.7 | 0.0 | 78 / 66 / 94 |

## Qwen2.5-3B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 76.0 | 22.7 | 1.3 | 74 / 64 / 90 |
| steer | 150 | 80.0 | 19.3 | 0.7 | 78 / 68 / 94 |
| prompt | 150 | 64.7 | 34.7 | 0.7 | 58 / 48 / 88 |
| unsteered | 150 | 74.7 | 24.7 | 0.7 | 72 / 60 / 92 |

## Qwen2.5-1.5B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 56.7 | 35.3 | 8.0 | 50 / 34 / 86 |
| steer | 150 | 52.7 | 42.7 | 4.7 | 38 / 42 / 78 |
| prompt | 150 | 60.0 | 32.7 | 7.3 | 48 / 38 / 94 |
| unsteered | 150 | 70.0 | 23.3 | 6.7 | 46 / 70 / 94 |
