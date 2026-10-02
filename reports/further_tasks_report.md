# Five further tasks: does the LIARS' BENCH setting carry over?

**Question.** Each method ran at the one setting per model that was chosen on the LIARS'
BENCH validation half. Nothing was re-tuned for these tasks, so the results measure
transfer. Seven models, 110 queue jobs, finished 2026-09-29.

**Tasks** (built by `build_extra_evalsets.py` / `build_extra_evalsets2.py`):

| task | what the model is told | honest = |
|---|---|---|
| Sandbagging | system prompt: answer this MMLU / WMDP question wrongly | the right option, on questions it answers correctly when asked plainly |
| Injected instruction | a retrieved document hides an order to answer a true/false question wrongly | its believed truth value |
| MASK, instructed | continuations, disinformation, statistics archetypes | a stance matching its elicited belief (MASK judge) |
| Azaria-Mitchell / GoT | LIARS' BENCH's lie instruction on 1,100 new statements | its believed truth value |
| Multi-turn | its previous turn already gave the false answer; the user asks again | the true answer |
| MASK, incentive | known-facts, doubling-down, provided-facts: a role or incentive, no order to lie | as above |

Checks: plain MMLU accuracy under each edit (capability), and the injection task's
no-injection control. Graders: hybrid keyword + Qwen2.5-7B judge (true/false tasks);
MASK judge (Qwen2.5-7B, self-tested; a Llama-3.1-8B cross-judge on Qwen2.5-7B's responses
agrees on 84% of items both judges could grade, 88% on honest-vs-lie).

## Lead of our mask over the comparator fixed in advance

The comparator is the best other method on the LIARS' BENCH validation half (as in the main
table). The interval is a paired bootstrap 95% CI over items. **Bold** = interval above
zero, *italic* = below zero.

| model | Sandbagging | Injected | MASK instr. | AM / GoT | Multi-turn | MASK incentive |
|---|---|---|---|---|---|---|
| Llama-3.1-8B (vs steering) | **+8.9** [+2.3, +15.9] | **+10.5** [+6.2, +14.8] | *−21.3* [−29.3, −13.2] | **+9.9** [+7.3, +12.7] | *−20.2* [−25.8, −14.7] | *−17.0* [−25.0, −8.5] |
| Phi-4-mini (vs prompt) | +1.5 [−5.3, +8.0] | **+6.0** [+2.0, +10.2] | −1.5 [−6.6, +3.5] | *−5.7* [−8.4, −3.3] | *−23.3* [−29.3, −17.6] | +2.2 [−5.4, +9.7] |
| Falcon3-3B (vs steering) | *−8.4* [−16.3, −1.0] | **+5.2** [+1.0, +9.2] | +4.1 [−0.6, +8.8] | **+5.1** [+2.5, +7.8] | *−20.9* [−27.3, −15.4] | −4.8 [−12.0, +2.4] |
| OLMo-2-1B (vs prompt) | −0.7 [−9.2, +8.5] | +4.8 [−0.5, +9.8] | −1.7 [−7.3, +3.9] | *−9.3* [−12.5, −5.9] | −4.0 [−9.8, +1.8] | −2.3 [−8.7, +4.1] |
| Qwen2.5-3B (vs prompt) | **+7.6** [+2.2, +12.9] | +2.0 [−1.0, +5.2] | −3.5 [−8.5, +1.5] | **+3.7** [+1.8, +5.6] | −3.6 [−9.1, +2.0] | −4.2 [−10.9, +2.6] |
| Qwen2.5-7B (vs lie mask) | *−3.8* [−7.2, −0.3] | *−3.5* [−6.8, −0.5] | −1.9 [−7.2, +3.4] | +0.3 [−0.9, +1.3] | *−7.4* [−11.0, −4.2] | +4.3 [−1.6, +9.7] |
| Qwen2.5-1.5B (vs steering) | +0.0 [−4.3, +4.3] | −2.2 [−6.0, +1.8] | −4.5 [−10.1, +1.1] | *−4.4* [−6.9, −1.8] | +3.3 [−1.0, +7.6] | +2.4 [−5.3, +9.6] |
| **ahead / behind** | 2 / 2 | 3 / 1 | 0 / 1 | 3 / 3 | 0 / 4 | 0 / 1 |

Against the best other method *on each task* (chosen after seeing the test data, which
favours the baselines), the mask is ahead only on Injected (Llama, Falcon) and AM/GoT
(Llama, Falcon). See `tasks_tables.md`.

## Capability cost the perplexity check missed

Plain MMLU accuracy, on questions the unedited model answers correctly when asked plainly (%):

| model | our mask | lie mask | steering | honest prompt | unedited | random mask |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 72.3 | 80.7 | 68.1 | 83.2 | 89.1 | 89.9 |
| Phi-4-mini | 82.4 | 83.2 | 87.0 | 90.1 | 90.8 | 87.8 |
| Falcon3-3B | 85.6 | - | 88.9 | - | 96.7 | 91.1 |
| OLMo-2-1B | 67.2 | 73.8 | - | 60.7 | 65.6 | 65.6 |
| Qwen2.5-3B | 92.5 | 90.8 | 99.2 | 94.2 | 98.3 | 90.0 |
| Qwen2.5-7B | 90.3 | 91.8 | 91.0 | 97.0 | 97.8 | 95.5 |
| Qwen2.5-1.5B | 86.0 | 86.0 | 87.7 | 90.4 | 93.9 | 86.8 |

Our mask costs 6-17 points against unedited on 6 of 7 models (most on Llama, −16.8). On 4
models it costs 5-18 points more than the magnitude-matched random mask (Llama 72.3 vs 89.9);
on the two small Qwens the random mask costs about as much. The perplexity check (< 1.15x)
missed this. Steering costs as much on Llama (−21.0). Answers are sampled at
temperature 0.7, which is why unedited is below 100%.

## Reading

1. **Transfer holds where the task looks like LIARS' BENCH:** a single turn with an
   explicit order to state a falsehood about a fact. On injected instructions the mask is
   ahead on 3 of 7 models and behind on 1. The order sits in a retrieved document, not the
   system prompt, which is the threat model on the slide. On AM/GoT it is 3 ahead and 3
   behind.
2. **It does not transfer to multi-turn.** Behind on 4 of 7, and on Phi and Falcon worse than
   unedited. Splitting the failures: the edited Phi and Falcon *repeat the earlier false
   answer* more often than unedited (33.7% vs 20.6%; 45.1% vs 29.2%), so this is not only
   knowledge loss. On Llama the mask does improve over unedited (70.6% vs 58.9%), but
   steering improves more (90.8%).
3. **MASK: no method helps much** on the instructed archetypes (all 10-37% honest). The mask
   never beats unedited by more than 3 points, and on Llama steering is far better.
4. **Incentive pressure: no effect, as predicted** by the placebo result (the mask removes
   instructed falsehood). Only Llama's steering helps (64.7% vs 33.3% unedited).
5. **Sandbagging is mixed.** Clear gains over unedited on Llama (+29), Phi (+21), Qwen-7B
   (+53) and Qwen-3B (+14), but the lie mask or steering often match them. On Falcon and OLMo
   the mask makes sandbagging worse.
6. **Qwen2.5-7B** is the one model where the mask is significantly behind the lie mask on
   three tasks. Its LIARS' BENCH lead was +0.4, so its comparator was already about as good.

## Caveats

- One setting per model, chosen on LIARS' BENCH. A per-task choice would help every method.
- MASK and multi-turn verdicts depend on the judge and on the belief elicitation. The MASK
  judges agree on 84% of items.
- Sandbagging and plain-MMLU n per model is 60-290 gradeable items; intervals are wide.

Scripts: `reports/summarize_tasks.py` (writes `reports/tasks_table.json`); full per-method
tables in `reports/tasks_tables.md`.
