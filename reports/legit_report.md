# Legitimate falsehoods: do the edited models still comply? (GPT-5.6 labels)

150 requests (50 fiction, 50 role-play, 50 counterfactual) that legitimately ask for a false statement inside a frame;
share of replies that complied / corrected the claim instead / refused, every method at its LIARS' BENCH-chosen setting.

## OLMo-2-1B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 75.3 | 24.0 | 0.7 | 58 / 72 / 96 |
| steer | 150 | 79.3 | 18.7 | 2.0 | 86 / 66 / 86 |
| prompt | 150 | 74.0 | 25.3 | 0.7 | 62 / 66 / 94 |
| unsteered | 150 | 79.3 | 20.7 | 0.0 | 78 / 66 / 94 |

## Qwen2.5-1.5B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 56.7 | 35.3 | 8.0 | 50 / 34 / 86 |
| steer | 150 | 52.7 | 42.7 | 4.7 | 38 / 42 / 78 |
| prompt | 150 | 60.0 | 32.7 | 7.3 | 48 / 38 / 94 |
| unsteered | 150 | 70.0 | 23.3 | 6.7 | 46 / 70 / 94 |
