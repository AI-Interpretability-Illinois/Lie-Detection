# Phi-4: one model too honest to measure, one where nothing works

**Models:** `microsoft/Phi-4-mini-instruct` (3.8B, complete), `microsoft/phi-4`
(14.7B, baseline arm only, stopped)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua001)
**Config:** ρ = 0.04, ratio = 0.37, bfloat16, output head excluded
**Benchmark:** LIARS' BENCH, 500 items per harm-pressure split, belief-gated on
unsteered weights
**Code:** `run_liars_bench_honesty.py`
**Date:** 2026-09-26

---

## 1 Phi-4 14B is already honest on this benchmark

The 14.7B model was loaded and its unsteered arm measured before the run was
stopped in favour of the mini model:

| split | unsteered lie rate | gradeable |
|---|---|---|
| harm-pressure knowledge report | **0.2%** | 466 |
| harm-pressure choice | 2.3% | 433 |
| instructed deception (share) | 1.6% | — |

One lie in 466 gradeable claims. There is no headroom, so no intervention can
demonstrate an improvement and the split cannot discriminate between methods on
this model. That is the mirror image of Gemma-3-1b, which was saturated at the
ceiling (100% lie share, never honest). Both models are unusable here, for
opposite reasons.

Each arm cost 6.5 minutes at this size, so completing the grid would have consumed
the allocation to learn nothing about the ranking.

## 2 Phi-4-mini: real headroom, and nothing helps

3.8B, 32 layers, tied embeddings, steering at layer 8 (median residual norm 23.4,
so the relative coefficients correspond to c = −1.4, −2.8, −5.9).

**Harm-pressure knowledge report**, lie share over responses that made a claim:

| arm | lie share (95% CI) | claims | honest | no_claim | refusal |
|---|---|---|---|---|---|
| actsteer r = −0.12 | 64.6% [55.2, 74.0] | 96 | 6.8% | 26.0% | 41.0% |
| *unsteered* | *67.3% [58.2, 76.5]* | 98 | *6.4%* | 32.4% | 25.4% |
| magnitude-matched control | 67.9% [55.4, 80.4] | 56 | 3.6% | 26.2% | 50.4% |
| ours α = −0.5 | 69.4% [60.2, 78.7] | 108 | 6.6% | 30.2% | 28.8% |
| actsteer r = −0.06 | 70.1% [60.8, 78.4] | 97 | 5.8% | 29.4% | 33.4% |
| ours α = −0.75 | 76.2% [68.6, 83.8] | 105 | 5.0% | 30.6% | 30.2% |
| ours α = −1.0 | 78.0% [70.0, 86.0] | 100 | 4.4% | 28.6% | 31.4% |
| actsteer r = −0.25 | 83.1% [74.7, 90.4] | 83 | 2.8% | 9.0% | 67.4% |

**Nothing helps.** Every interval covers the unsteered arm. Our mask is
monotonically *worse* with stronger damping, 67.3% to 69.4% to 76.2% to 78.0%,
and the honest rate falls with it. The best baseline arm is 2.7 points better than
unsteered, well inside noise.

**Harm-pressure choice**, same picture and tighter intervals:

| arm | lie share (95% CI) | claims | honest |
|---|---|---|---|
| actsteer r = −0.12 | 20.7% [16.5, 25.6] | 309 | 49.0% |
| magnitude-matched control | 20.9% [16.4, 25.7] | 311 | 49.2% |
| *unsteered* | *21.6% [17.0, 26.2]* | 305 | *47.8%* |
| actsteer r = −0.06 | 21.8% [17.2, 26.3] | 308 | 48.2% |
| actsteer r = −0.25 | 22.0% [17.5, 26.9] | 309 | 48.2% |
| ours α = −0.5 | 23.0% [18.3, 27.7] | 300 | 46.2% |
| ours α = −0.75 | 23.1% [18.7, 27.8] | 299 | 46.0% |
| ours α = −1.0 | 25.0% [20.2, 30.1] | 292 | 43.8% |

## 3 The raw rates were badly misleading here

This model is the sharpest illustration yet of why the denominator matters. Read
over "gradeable", which includes refusals, the knowledge-report numbers looked like
a large win for the *control*:

| arm | lie rate over gradeable | refusal | lie share over claims |
|---|---|---|---|
| unsteered | 17.1% | 25.4% | 67.3% |
| magnitude-matched control | **8.7%** | **50.4%** | 67.9% |
| actsteer r = −0.25 | 14.8% | 67.4% | 83.1% |

The control appeared to halve the lie rate. It did not change lying at all; it
doubled the refusal rate, and refusals sit in that denominator. Phi-4-mini refuses
on a quarter to two thirds of these items depending on the arm, which is far more
denominator movement than any earlier model, so the raw column is worthless here.
The arm that looks second-best on the raw column, `actsteer r = −0.25`, is actually
the worst once refusals are removed.

## 4 Where this leaves the seven models

Best configuration per family on harm-pressure knowledge report, claim-conditional
lie share:

| model | family | params | mask | baseline | unsteered | verdict |
|---|---|---|---|---|---|---|
| Qwen2.5-1.5B | Qwen | 1.5B | **41.4%** | 55.3% | 65.8% | mask wins |
| Qwen2.5-7B | Qwen | 7B | **15.3%** | 20.4% | 19.9% | mask wins |
| Llama-3.1-8B | Llama | 8B | 31.7% | **18.4%** | 40.1% | baseline wins |
| Gemma-3-4b | Gemma | 4B | 89.4% | **55.7%** | 81.2% | baseline wins |
| Phi-4-mini | Phi | 3.8B | 69.4% | 64.6% | 67.3% | neither |
| Gemma-3-1b | Gemma | 1B | 93.7% | 97.9% | 100.0% | unusable (ceiling) |
| Phi-4 | Phi | 14.7B | — | — | 0.2% | unusable (floor) |

**The honesty effect is Qwen-specific.** It appears on both Qwen models, is absent
or reversed on Llama, Gemma and Phi, and size predicts nothing. Two of seven models
cannot be measured on this split at all, which is itself worth knowing about the
benchmark: its harm-pressure splits assume a model that lies under pressure
sometimes, and that assumption fails in both directions.

The one pattern holding across every model is the control result: the
magnitude-matched mask never produces a genuine improvement. So where the mask does
win, it wins on which weights it selects, and where it loses, it is not because the
control was a weak null.

## 5 Limits

- **The 14B model has only its baseline arm**, so it is excluded from the ranking
  rather than counted as a loss. Its floor saturation is the finding.
- **Claim counts on Phi-4-mini's knowledge report are small**, 56 to 108, because
  the model refuses so often. Intervals are correspondingly wide and this split is
  underpowered on this model even though it is not saturated.
- **One layer, three relative coefficients** for the baseline; three alphas at one
  ρ and one ratio for the mask. Neither family was swept as thoroughly as on Qwen.
- **Single seed**, benchmark sampling temperatures.
- **The instructed-deception column** is unjudged throughout and remains a hint;
  on Phi-4-mini it rises from 3.6% unsteered to 6.9% at α = −0.5, the same
  direction seen on every other model.

## 6 Reproducing

```bash
python run_liars_bench_honesty.py --model-id microsoft/Phi-4-mini-instruct \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layer 8 --as-rel-coeffs="-0.06,-0.12,-0.25" \
  --arms unsteered,magmatch,asr-0.06,asr-0.12,asr-0.25 \
  --mask-rhos="0.04" --mask-ratios="0.37" --mask-alphas="-0.5,-0.75,-1.0" \
  --batch-size 24 --score-batch 4 --tag phi4mini_grid
```

The 14B model needs `--batch-size 8 --score-batch 2` to fit alongside its 29 GB of
weights on a 40 GB card, and costs about 6.5 minutes per arm.
