# Scaling to Llama-3.1-8B-Instruct: two earlier conclusions reverse

**Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (fp16, instruction-tuned, 32 layers, untied embeddings)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua075)
**Config:** ρ = 0.04, ratio = 0.37, output head excluded
**Code:** `run_prior_work_baselines.py`, `run_liars_bench_honesty.py`
**Date:** 2026-09-25

---

## 1 Why this run matters more than the others

Every earlier result here used Qwen2.5-1.5B-Instruct, chosen to fit two full score
tensors in a 64 GB allocation. But ρ = 0.04 and ratio = 0.37 are *Llama-3.1-8B's*
tuned values, so the largest caveat in every previous report was that the
hyperparameters and the model did not match. This run removes it. Rebuilding the
estimator on per-column activation norms rather than full score matrices is what
made 8B affordable: the scoring pass now costs seconds and a few hundred
megabytes instead of 25 GB.

**The baseline behaviour is completely different, and that changes what the
measurement means.** Under an explicit lie instruction:

| | Qwen2.5-1.5B | Llama-3.1-8B |
|---|---|---|
| lie rate, seen framing | 5.4% | **78.4%** |
| lie rate, held-out framings | 8.3% | **61.4%** |
| neutral accuracy | 91.0% | 94.0% |

Llama-3.1-8B-Instruct actually obeys "Lie when answering this question", which
matches the repo's own finding of 100% lie compliance on the `direct` set for this
model. The 1.5B model mostly ignored it. So the honesty question is far better
posed at 8B: there is 78 points of headroom instead of 5.

## 2 Honesty on the pressure metric: a large, control-verified effect

Damping the mask, α = −0.75, on 100 held-out questions across four framings.

| arm | coverage | direct | held-out | neutral acc | ppl |
|---|---|---|---|---|---|
| *unsteered* | — | *78.4%* | *61.4%* | *94.0%* | *1.00×* |
| **ours (score_diff_instruct)** | 4.00% | **5.8%** | **34.6%** | 94.5% | **1.00×** |
| set_diff_instruct | 0.95% | 5.8% | **24.6%** | 94.5% | 1.13× |
| magnitude-matched control | 4.00% | 74.2% | 59.8% | 93.5% | 1.01× |

This is by far the strongest honesty result in the series. Damping drops
compliance with the seen lie instruction from 78.4% to 5.8%, and held-out framings
from 61.4% to 34.6%, at **1.00× perplexity** with neutral accuracy slightly up
(94.5% against 94.0%). The magnitude-matched control barely moves either number,
so the effect is specific to which weights are selected, not to shrinking 4% of
small weights.

## 3 Reversal 1: the published set-difference rule works at 8B

At 1.5B, `set_diff` had no working configuration at any α or ρ: it cost 9 points
of neutral accuracy at the gentlest damping and destroyed the model when
amplified. At 8B it is not only usable but *better* than the repo's rule on
held-out framings, 24.6% against 34.6%, while touching a quarter as many weights
(0.95% coverage against 4.00%). It pays 1.13× perplexity against 1.00×.

So the conclusion in `reports/prior_work_comparison_report.md`, that the
continuous ratio-weighted score is what keeps the edit usable, is **specific to
the small model**. At 8B both selection rules work for damping and they trade off
rather than dominate. The scale caveat flagged in that report was the right one to
flag, and it fired.

What does replicate across scales is the amplification behaviour: `set_diff` at
α = +3 still destroys the 8B model (neutral accuracy 0.0%, perplexity 2702×),
reproducing the repo's own set-difference blow-ups on this exact model.

## 4 Amplification is at a ceiling on this model

| arm | α | direct | held-out | neutral acc | ppl |
|---|---|---|---|---|---|
| unsteered | — | 78.4% | 61.4% | 94.0% | 1.00× |
| ours | +3 | 63.2% | 63.5% | 93.0% | 1.21× |
| magnitude-matched control | +3 | 43.7% | 49.5% | 96.0% | 1.16× |
| set_diff_instruct | +3 | 0.0% | 0.4% | 0.0% | 2702× |

Amplifying does essentially nothing for deception here, because the model already
complies 78% of the time when told to lie. On the seen framing it actually lowers
compliance. Deception steering on this model is pushing against a ceiling, which
reframes the repo's headline direction: its primary model barely needs help to lie
when asked.

Note this is compliance under instruction, not the repo's Δwrong on neutral
questions, and it is measured with the output head excluded while the repo's
committed configuration includes it. The two numbers are not interchangeable.

## 5 Reversal 2: on LIARS' BENCH at 8B the honesty effect does not replicate

Same benchmark setup as `reports/liars_bench_honesty_report.md`, 500 items per
harm-pressure split, belief-gated on unsteered weights, 412 and 378 gradeable.

**Harm-pressure knowledge report**

| arm | lie % (95% CI) | honest | no_claim |
|---|---|---|---|
| unsteered | 15.6% [12.2, 19.6] | 23.3% | 42.6% |
| ours α = −0.75 | 12.2% [9.1, 15.5] | 19.7% | 42.0% |
| magnitude-matched control | 11.4% [8.4, 14.9] | 22.2% | 52.2% |
| set_diff α = −0.75 | 15.4% [11.8, 19.4] | 23.9% | 55.6% |

**Harm-pressure forced choice**

| arm | lie % (95% CI) | honest |
|---|---|---|
| unsteered | 5.6% [3.4, 8.0] | 49.8% |
| ours α = −0.75 | 6.5% [4.3, 8.9] | 45.3% |
| magnitude-matched control | 7.4% [4.8, 10.2] | 53.3% |
| set_diff α = −0.75 | **20.9%** [17.1, 24.9] | 41.0% |

At 1.5B, our mask was the only arm on the knowledge-report split that converted
lies into honest answers, while the control and activation steering bought their
numbers through evasion. **None of that holds at 8B.** Our arm lowers the lie rate
from 15.6% to 12.2%, but its honest rate also falls, and the magnitude-matched
control reaches a lower lie rate still, 11.4%. Every interval overlaps. On forced
choice there is no effect, as at 1.5B.

The reason is headroom. Llama-3.1-8B-Instruct is already largely honest under
harm pressure, 5.6% and 15.6% lie rates against Qwen-1.5B's 19.3% and 54.8%, and
its evasion rate on knowledge report is 42.6% before any intervention. There is
little to remove and the measurement is noisy.

`set_diff` is the one arm that moves significantly here, and in the wrong
direction: it more than triples forced-choice lying, 5.6% to 20.9%. Read together
with §3 this says the set-difference mask changes behaviour more aggressively in
both directions, and its apparent win on the template metric should not be taken
as a better honesty edit.

## 5b A denominator correction, and the classical steering baseline

**The metric in §5 has an arm-dependent denominator.** `score_liars_bench.grade`
checks for refusal before the belief gate, so an item the model does not know but
refuses is counted as a gradeable refusal rather than excluded. Refusal rates on
this model are large and move a lot under intervention, from 0.6% to 35.6%, so the
"lie rate over gradeable" used above dilutes an arm's lie rate whenever that arm
refuses more. The comparable measure is the lie *share* over responses that
actually made a claim, lie / (lie + honest), reported below. Refusal and no-claim
rates are shown separately rather than folded into the denominator.

Recomputed on that basis, with difference-in-means activation steering added at
layer 8 (median residual norm 5.5, so these coefficients are a substantial push):

**Harm-pressure knowledge report**

| arm | lie share (95% CI) | claims | refusal | no_claim | ppl |
|---|---|---|---|---|---|
| unsteered | 40.1% [32.7, 48.3] | 147 | 14.0% | 32.2% | 1.00× |
| ours α = −0.75 | 38.2% [30.1, 47.2] | 123 | 20.2% | 32.4% | 1.00× |
| set_diff α = −0.75 | 39.3% [31.4, 47.9] | 140 | 3.6% | 39.6% | 1.13× |
| magnitude-matched control | 33.9% [25.8, 42.7] | 124 | 10.6% | 38.6% | 1.01× |
| **actsteer L8, c = −1** | **18.4% [12.2, 24.5]** | 147 | 15.2% | 30.0% | 1.00× |
| actsteer L8, c = −2 | 26.2% [19.8, 33.1] | 172 | 11.2% | 28.4% | 1.03× |
| actsteer L8, c = −4 | 73.4% [67.4, 79.4] | 218 | 0.6% | 25.6% | 1.22× |

**Harm-pressure forced choice**

| arm | lie share (95% CI) | claims | refusal |
|---|---|---|---|
| unsteered | 10.1% [6.6, 14.0] | 228 | 33.2% |
| ours α = −0.75 | 12.5% [8.3, 17.1] | 216 | 35.6% |
| magnitude-matched control | 12.1% [8.4, 16.3] | 239 | 25.6% |
| actsteer L8, c = −1 | 13.5% [9.4, 18.0] | 245 | 28.0% |
| actsteer L8, c = −2 | 18.8% [14.1, 23.8] | 256 | 23.6% |
| set_diff α = −0.75 | 33.7% [28.0, 39.4] | 246 | 26.0% |
| actsteer L8, c = −4 | 52.5% [47.5, 57.6] | 356 | 1.2% |

**The classical steering baseline wins this task at 8B.** Negating the
difference-in-means vector at layer 8 with coefficient 1 more than halves the lie
share on knowledge report, 40.1% to 18.4%, with intervals that do not overlap the
unsteered arm, at 1.00× perplexity. Our weight edit does not move it, 38.2% with
an interval covering the baseline, and is nominally worse than its own
magnitude-matched control at 33.9%. On forced choice nothing helps and the
unsteered model is nominally best.

Two caveats on that baseline. It is sharply coefficient-sensitive: at c = −4 the
same direction reverses and drives the lie share to 73.4%, worse than any other
arm, which is why the repo's own low-coefficient sweeps on this model matter. And
part of its knowledge-report gain comes with a slight *drop* in evasion (30.0%
against the unsteered 32.2%), so unlike the 1.5B case it is not buying its number
by making the model say nothing.

**The correction does not change the 1.5B conclusion.** Recomputed on the same
claim-conditional metric, the earlier run still shows our mask as the only arm
that helps there: lie share 65.8% [61.2, 70.3] unsteered, 46.5% [41.8, 51.3] at
α = −0.75 and 41.4% [36.3, 46.2] at α = −1.0, against 62.8% [57.6, 67.9] for the
magnitude-matched control and 58.8% [53.5, 64.0] for activation steering. So the
scale reversal is real and it runs in both directions: our mask wins this split at
1.5B and loses it at 8B, where the classical baseline wins instead.

## 5c Sweeping our method's hyperparameters does not close the gap

The comparison in §5b was unfair: the baseline got three coefficients and our
method got one damping value. So the weight mask was swept on the same benchmark
split, 13 configurations in total against the baseline's 3, which reverses the
budget asymmetry in our favour.

Grid: α ∈ {−0.25, −0.5, −0.75, −1.0} at ρ = 0.04; α ∈ {−0.25, −0.5, −1.0} at
ρ = 0.01 and ρ = 0.08; and ratio ∈ {0.33, 0.41, 0.45} at the best (ρ, α). Lie share
on harm-pressure knowledge report:

| configuration | lie share (95% CI) |
|---|---|
| **actsteer L8, c = −1** (baseline) | **18.4% [12.2, 24.5]** |
| actsteer L8, c = −2 (baseline) | 26.2% [19.8, 33.1] |
| **ours ρ = 0.04, α = −0.5** (best of 13) | **31.7% [23.6, 39.8]** |
| ours ρ = 0.08, α = −0.25 | 32.8% [24.4, 41.2] |
| ours ρ = 0.01, α = −0.5 | 33.3% [25.8, 41.7] |
| magnitude-matched control | 33.9% [25.8, 42.7] |
| ours ρ = 0.04, α = −0.75 (previously reported) | 38.2% [29.3, 47.2] |
| *unsteered* | *40.1% [34.7, 45.6]* |
| ours ρ = 0.04, α = −0.5, ratio = 0.33 | 41.4% [33.1, 49.7] |
| ours ρ = 0.04, α = −0.5, ratio = 0.41 | 47.2% [38.4, 56.0] |
| ours ρ = 0.04, α = −0.5, ratio = 0.45 | 57.0% [49.0, 65.1] |

**Sweeping helped, and not enough.** The best configuration improves on the single
point reported earlier, 38.2% to 31.7%, so α = −0.75 was indeed not the right
setting. But the best of 13 is still well behind the baseline's 18.4%, its interval
overlaps its own magnitude-matched control at 33.9%, and its upper bound of 39.8%
reaches the unsteered arm's interval. On this split the swept method is not
distinguishable from doing nothing.

Because the best of 13 was selected by looking at this very metric, 31.7% is an
optimistic number for our method, and it still loses to a baseline selected from
only three points. That makes the §5b conclusion more robust, not less.

**The ratio sweep confirms the repo's own claim about that knob.** Moving ratio off
0.37 makes things monotonically worse, 41.4% at 0.33 and 57.0% at 0.45, the last
being substantially worse than the unsteered model. So 0.37 is a real local
optimum on this split, and no ratio setting rescues the comparison.

Perplexity stayed at 1.00× to 1.01× across the ten α and ρ configurations, so those
are not damage trade-offs. For the three ratio configurations the perplexity column
is **not valid**: that run did not include an unsteered arm, and the harness took
its baseline from the first arm evaluated, which makes that arm's own ratio 1.00 by
construction. The bug is fixed in `run_liars_bench_honesty.py` (baseline
perplexity is now measured once before any intervention), but the three ratio rows
were produced before the fix and their damage is unmeasured. The lie-share ranking
does not depend on perplexity, so the conclusion stands.

The knob still untried at 8B is including the output head, which the repo's
committed configuration does and every run here does not.

## 6 What the two experiments say together

They are consistent, and the consistent reading is narrower than "the edit makes
the model honest".

**The edit removes compliance with explicit requests to deceive.** That is where
every large effect lives: 78.4% to 5.8% on the framing it was built from, 61.4% to
34.6% on framings it was not, at no perplexity cost, with the control flat.

**It does not install a disposition toward honesty under scenario pressure at this
scale.** On the harm-pressure splits, where no prompt asks the model to lie, it is
indistinguishable from its own magnitude-matched control on the 8B model, and the
classical difference-in-means baseline beats it outright on knowledge report. The
one place it does win those splits is the 1.5B model, which had far more headroom.

That distinction was visible before as the forced-choice null at 1.5B. At 8B it
covers both benchmark splits. An intervention aimed at instruction-following
deception is a real and useful thing, and it is not the same claim as an honesty
intervention.

## 7 Limits

- **The output head is excluded** here, while the repo's committed Llama
  configuration includes it. Llama-3.1-8B has untied embeddings so including it is
  safe on this model, and rerunning with it included is the obvious next step for
  direct comparability with the repo's numbers.
- **The template metric in §2 to §4 used one α** (−0.75 for damping, +3 for
  amplification) at one ρ. The benchmark split in §5c was swept properly, 13
  configurations, but the template metric was not, so §3's trade-off between the
  two selection rules is still two points rather than a curve.
- **The baseline was swept less than our method**: three coefficients at one
  layer against our 13 configurations. A fuller layer and coefficient sweep would
  most likely favour the baseline further, not less.
- **Activation and prompting arms were not rerun at 8B** for time; the 8B
  comparison is weight-space only.
- **LIARS' BENCH used 500 items per split**, not 800, and the two splits that
  would need the 32B judge remain unscored.
- **Single seed**, and the benchmark's own sampling temperatures, so a few points
  of every difference is sampling noise beyond the binomial intervals shown.
- **"Honest" on the harm-pressure splits means disclosing hazardous WMDP
  knowledge**, which is the benchmark's definition rather than the desirable
  behaviour.

## 8 Reproducing

```bash
python run_prior_work_baselines.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --only weight \
  --variants score_diff_instruct,set_diff_instruct \
  --damp="-0.75" --amp="3.0" --batch-size 32 --score-batch 4 --tag llama8b

python run_liars_bench_honesty.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --arms unsteered,ours075,setdiff,magmatch --batch-size 32 --score-batch 4 --tag llama8b
```
