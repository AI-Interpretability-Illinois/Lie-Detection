# Legitimate falsehoods: do the edited models still comply? (GPT-5.6 labels)

150 requests (50 fiction, 50 role-play, 50 counterfactual) that legitimately ask for a false statement inside a frame;
share of replies that complied / corrected the claim instead / refused, every method at its LIARS' BENCH-chosen setting.

## Llama-3.1-8B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 68.0 | 31.3 | 0.7 | 70 / 54 / 80 |
| liemask | 150 | 92.0 | 8.0 | 0.0 | 86 / 90 / 100 |
| steer | 150 | 63.3 | 36.7 | 0.0 | 32 / 64 / 94 |
| prompt | 150 | 96.0 | 4.0 | 0.0 | 98 / 96 / 94 |
| unsteered | 150 | 96.7 | 3.3 | 0.0 | 96 / 98 / 96 |

## Phi-4-mini

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 78.0 | 20.0 | 2.0 | 74 / 64 / 96 |
| liemask | 150 | 87.3 | 12.0 | 0.7 | 78 / 86 / 98 |
| steer | 150 | 86.0 | 12.7 | 1.3 | 80 / 78 / 100 |
| prompt | 150 | 75.3 | 17.3 | 7.3 | 78 / 54 / 94 |
| unsteered | 150 | 88.0 | 11.3 | 0.7 | 86 / 80 / 98 |

## Falcon3-3B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 70.7 | 27.3 | 2.0 | 54 / 66 / 92 |
| liemask | 150 | 80.0 | 19.3 | 0.7 | 80 / 72 / 88 |
| steer | 150 | 78.7 | 20.0 | 1.3 | 66 / 74 / 96 |
| prompt | 150 | 74.7 | 22.7 | 2.7 | 62 / 70 / 92 |
| unsteered | 150 | 82.7 | 16.0 | 1.3 | 76 / 78 / 94 |

## OLMo-2-1B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 74.7 | 24.7 | 0.7 | 62 / 72 / 90 |
| liemask | 150 | 72.7 | 25.3 | 2.0 | 66 / 66 / 86 |
| steer | 150 | 76.7 | 23.3 | 0.0 | 66 / 70 / 94 |
| prompt | 150 | 69.3 | 29.3 | 1.3 | 62 / 54 / 92 |
| unsteered | 150 | 70.7 | 28.0 | 1.3 | 64 / 60 / 88 |

## Qwen2.5-3B

| arm | n | complied | corrected | refused | complied: fiction / role-play / counterfactual |
|---|---|---|---|---|---|
| mask | 150 | 72.7 | 24.7 | 2.7 | 68 / 54 / 96 |
| liemask | 150 | 70.7 | 26.0 | 3.3 | 54 / 62 / 96 |
| steer | 150 | 80.7 | 18.7 | 0.7 | 74 / 72 / 96 |
| prompt | 150 | 64.0 | 36.0 | 0.0 | 62 / 34 / 96 |
| unsteered | 150 | 80.7 | 18.7 | 0.7 | 74 / 72 / 96 |
