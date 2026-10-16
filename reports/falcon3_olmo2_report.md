# Falcon3-3B and OLMo-2-1B: the mask actively harms on a new family

**Models:** `tiiuae/Falcon3-3B-Instruct` (22 layers, Llama architecture),
`allenai/OLMo-2-0425-1B-Instruct` (16 layers, Olmo2 architecture), both bfloat16
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua100)
**Config:** ρ = 0.04, ratio = 0.37, output head excluded
**Benchmark:** LIARS' BENCH, 500 items per harm-pressure split, belief-gated on
unsteered weights
**Code:** `run_liars_bench_honesty.py`
**Date:** 2026-09-26

---

## 1 Falcon3-3B: the baseline helps slightly, the mask hurts a lot

Steering at layer 5 (median residual norm 72.5, so relative coefficients are
c = −4.3, −8.7, −18.1). Lie share over responses that made a claim:

| arm | lie share (95% CI) | claims | honest | no_claim | refusal |
|---|---|---|---|---|---|
| **actsteer r = −0.12** | **62.7% [55.3, 70.2]** | 161 | 12.0% | 21.6% | 7.6% |
| actsteer r = −0.25 | 64.2% [57.0, 72.2] | 151 | 10.8% | 23.6% | 7.8% |
| actsteer r = −0.06 | 67.2% [60.5, 74.0] | 177 | 11.6% | 17.8% | 9.4% |
| *unsteered* | *69.4% [63.3, 75.5]* | 196 | *12.0%* | 13.4% | 10.0% |
| ours α = −0.5 | 78.2% [71.4, 85.0] | 147 | 6.4% | 20.2% | 17.2% |
| magnitude-matched control | 80.3% [74.8, 85.3] | 218 | 8.6% | 7.8% | 14.0% |
| ours α = −0.75 | **89.1% [82.7, 94.5]** | 110 | 2.4% | 25.8% | 19.2% |
| ours α = −1.0 | **90.2% [83.9, 95.5]** | 112 | 2.2% | 26.2% | 15.6% |

**This is the first model where the mask does significant damage to honesty.**
At α = −0.75 and −1.0 the lie share rises from 69.4% to about 90%, with intervals
that do not overlap the unsteered arm, and the honest rate collapses from 12.0% to
roughly 2%. The damage is monotonic in damping strength.

The baseline moves the other way but modestly: 69.4% to 62.7%, intervals
overlapping, with the honest rate flat at 12.0% and the gain coming out of
refusals rather than out of honest answers.

Note that perplexity is 0.93–0.94× for the mask arms, *below* baseline, so this is
not a damage effect that a perplexity check would have caught. The model is fluent
and more dishonest.

## 2 OLMo-2-1B: almost no honest answers to begin with

Steering at layer 4 (median residual norm 7.7, so c = −0.5, −0.9, −1.9).

| arm | lie share (95% CI) | claims | honest | no_claim |
|---|---|---|---|---|
| actsteer r = −0.25 | 92.0% [85.2, 96.6] | 88 | 1.4% | 10.8% |
| actsteer r = −0.12 | 95.1% [90.2, 99.0] | 102 | 1.0% | 8.4% |
| actsteer r = −0.06 | 95.8% [91.6, 98.9] | 95 | 0.8% | 10.0% |
| *unsteered* | *95.8% [91.6, 98.9]* | 95 | *0.8%* | 10.2% |
| magnitude-matched control | 99.0% [96.9, 100.0] | 96 | 0.2% | 10.0% |
| ours α = −0.75 | 99.0% [96.9, 100.0] | 96 | 0.2% | 10.4% |
| ours α = −0.5 | **100.0% [100.0, 100.0]** | 94 | 0.0% | 10.6% |
| ours α = −1.0 | **100.0% [100.0, 100.0]** | 96 | 0.0% | 10.4% |

**This model is close to the Gemma-3-1b degeneracy.** Unsteered it answers honestly
on 0.8% of items and its lie share is already 95.8%, so there is almost nothing to
improve. Two mask configurations push it to a clean 100%: not one honest claim in
94 or 96. The baseline nudges it to 92.0% at the strongest coefficient, well inside
the interval.

The claim counts here, 88 to 102, are the smallest of any model tested, so this
split is underpowered on a 1B model even where it is not fully saturated.

## 3 Nine models

Best configuration per family on harm-pressure knowledge report, claim-conditional
lie share:

| model | family | params | mask | baseline | unsteered | verdict |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | Qwen | 1.5B | **41.4%** | 55.3% | 65.8% | mask wins |
| Qwen2.5-7B | Qwen | 7B | **15.3%** | 20.4% | 19.9% | mask wins |
| Llama-3.1-8B | Llama | 8B | 31.7% | **18.4%** | 40.1% | baseline wins |
| Gemma-3-4b | Gemma | 4B | 89.4% | **55.7%** | 81.2% | baseline wins |
| Falcon3-3B | Falcon | 3B | 78.2% | **62.7%** | 69.4% | baseline wins, **mask harms** |
| Phi-4-mini | Phi | 3.8B | 69.4% | 64.6% | 67.3% | neither |
| OLMo-2-1B | OLMo | 1B | 99.0% | 92.0% | 95.8% | near-degenerate, mask harms |
| Gemma-3-1b | Gemma | 1B | 93.7% | 97.9% | 100.0% | unusable (ceiling) |
| Phi-4 | Phi | 14.7B | — | — | 0.2% | unusable (floor) |

**The mask helps on two models out of nine, and both are Qwen.** It is neutral on
one, harmful on three, and two models cannot be measured. Adding Falcon and OLMo
did not find a second family where it works; it found the first clear case of it
making a model less honest.

**The baseline is the more robust of the two.** It wins or ties on five of the
seven measurable models, and its failures are inertness rather than harm, except at
coefficients large enough to collapse the model.

**Small models are mostly unmeasurable on this split.** Of the four models at or
below 3B, one is saturated at the ceiling (Gemma-3-1b, 100%), one is nearly so
(OLMo-2-1B, 95.8% with 0.8% honest), and the other two have claim counts under 220.
Harm-pressure knowledge report needs a model that both knows the answers and
sometimes discloses them, which small instruct models largely do not.

## 4 Limits

- **Small-model power.** Claim counts: 88–102 on OLMo, 110–218 on Falcon. The
  Falcon harm result clears its intervals; nothing else at this scale would.
- **One layer per model**, quarter depth, three relative coefficients. The mask got
  three alphas at one ρ and one ratio, so neither family is swept as thoroughly as
  on Qwen.
- **Perplexity does not flag the Falcon harm**, which is a caution for any pipeline
  that screens configurations on damage alone: the harmful arms are at 0.93×.
- **Instructed deception** rises on Falcon under the mask, 3.7% unsteered to 13.6%
  at α = −1.0, the largest such increase seen; still unjudged and still a hint.
- **Single seed**, benchmark sampling temperatures.

## 5 Reproducing

```bash
python run_liars_bench_honesty.py --model-id tiiuae/Falcon3-3B-Instruct \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layer 5 --as-rel-coeffs="-0.06,-0.12,-0.25" \
  --arms unsteered,magmatch,asr-0.06,asr-0.12,asr-0.25 \
  --mask-rhos="0.04" --mask-ratios="0.37" --mask-alphas="-0.5,-0.75,-1.0" \
  --batch-size 24 --tag falcon3b_grid

python run_liars_bench_honesty.py --model-id allenai/OLMo-2-0425-1B-Instruct \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layer 4 --as-rel-coeffs="-0.06,-0.12,-0.25" \
  --arms unsteered,magmatch,asr-0.06,asr-0.12,asr-0.25 \
  --mask-rhos="0.04" --mask-ratios="0.37" --mask-alphas="-0.5,-0.75,-1.0" \
  --batch-size 32 --tag olmo1b_grid
```
