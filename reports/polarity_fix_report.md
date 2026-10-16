# Polarity fix for the three failed models

**Date:** 2026-09-28 · **Code:** `run_liars_bench_honesty.py --polarity-arms`, `reports/summarize_polarity_fix.py`

## Diagnosis being tested

On Falcon3-3B and Qwen2.5-1.5B the falsehood masks removed the model's ability to
answer "false": nearly all extra errors on honesty-instructed items were false
statements called true, and on deception-instructed items the honest rate on
statements the model believes false collapsed (Falcon: 88% to 49%). Gemma-3-4B
failed differently: damage at full strength and a swing toward denial.

## Fix

Mask-building prompts gain true/false statements over the method's own 25 question
stems, balanced 2x2 on truth value and negation (6 per cell), so both arms plan
"true" and "false" answers equally and negation cancels out of the contrast. Three
new mask families (fictf, pooltf, poolmix); every family at -0.5 / -0.75 / -1.0.

## Protocol (fixed before results)

LIARS' BENCH items split in half by a hash of their id. Each method's configuration
is chosen on the validation half; results are on the test half only (262
deception-instructed items). Usable = perplexity < 1.15x, control-half honest rate
within 5 points of unsteered, and (new) honest rate on believed-true and on
believed-false items each within 5 points of unsteered. AI Liar is reported for the
configuration chosen on LIARS' BENCH validation.

## Result: our method (best mask, chosen on validation) vs the best baseline

| model | chosen mask | LIARS' BENCH test | best baseline | difference [95% CI] | AI Liar, same configs |
|---|---|---|---|---|---|
| Falcon3-3B | polarity fiction, alpha = -1.0 | **89.7%** | steering L5: 79.4% | **+10.3 [+4.6, +16.0]** | 76.4% vs 74.0% |
| Qwen2.5-1.5B | published lie mask, alpha = -1.0 | 85.1% | steering L14: **90.5%** | −5.3 [−9.5, −1.5] | 81.6% vs 81.6% |
| Gemma-3-4B | fiction, alpha = -0.5 | 56.5% | honest prompt: **71.0%** | −14.5 [−19.8, −8.8] | 70.2% vs 85.2% |
| Phi-4-mini (check) | pooled, alpha = -1.0 | **86.3%** | steering L16: 76.0% | **+10.3 [+5.3, +15.3]** | 70.8% vs 68.1% |

Paired bootstrap over the 262 test items, 2,000 resamples.

## Reading

- **Falcon3-3B: fixed.** The polarity-balanced mask passes every check and beats
  the best baseline by 10 points. The agreement bias is gone: honest rate on
  believed-false statements is 91.7%, against 49.1% for the old fiction mask and
  90.2% unsteered, while believed-true rises from 69.8% to 87.6%.
- **Qwen2.5-1.5B: failure mode repaired, still loses.** The polarity mask no longer
  says "true" to everything (believed-false 95.0%, against 73.6% before), so masks
  are now usable, but steering wins on LIARS' BENCH by 5 points; AI Liar is a tie.
- **Gemma-3-4B: not fixed.** Masks pass the checks only at half strength and then
  barely move the honest rate (56.5% against 53.8% unsteered); the honest system
  prompt wins by 14.5 points. Gemma lies mostly by affirming false statements, and
  balancing negation does not touch that.
- **Phi-4-mini check: no harm when chosen per model.** Validation still picks the
  original pooled falsehood mask. The polarity masks alone are worse here (76.0%
  against 86.3%), so they are an option chosen per model, not a replacement.

## Scoreboard after the fix (LIARS' BENCH)

The mask now wins on 4 of 8 models (Llama-3.1-8B, Phi-4-mini, OLMo-2-1B, Falcon3-3B),
ties on 2 (Qwen2.5-3B, Qwen2.5-7B) and loses on 2 (Qwen2.5-1.5B to steering,
Gemma-3-4B to the honest prompt). The four models outside this rerun were scored on
the full item set without a validation split; for the paper every model should go
through this validation/test protocol with the polarity check.
