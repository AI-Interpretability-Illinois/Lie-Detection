# Dose-response: the effect attributable to selection, net of damage

**Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Qwen2.5-1.5B-Instruct
**Hardware:** 1× NVIDIA A100-SXM4-40GB each (NCSA Delta)
**Config:** ρ = 0.04, ratio = 0.37, n_truth = 25, output head excluded
**Code:** `run_dose_response.py`
**Date:** 2026-09-26

---

> **Grader caveat, 2026-09-27.** These runs predate the fix for two grader bugs
> affecting the 17 numeric-answer questions (bare numbers discarded as junk;
> substring correctness, so gold 360 matched the lie "3600"), and they did not
> save their responses, so they cannot be re-scored offline. For runs that could
> be re-scored, the fix raised every arm's falsehood rate by a similar amount,
> about 4 to 9 points, and changed no ranking, so the comparisons here are
> expected to hold while the absolute rates are understated. The scripts now save
> responses.

## 1 The estimand

Scaling any 4% of a model's weights by `1 + α` changes its behaviour. The question
a steering claim has to answer is whether the change comes from *which* weights were
selected or merely from *how much* the model was perturbed. Comparing arms at equal
α does not answer it, because equal α is not equal damage.

So for each mask family `m`, trace the curve `L_m(d)`: the falsehood rate reached
when `m` is scaled by whatever knob produces damage level `d`. The quantity of
interest is the vertical gap between curves at matched damage,

    τ(d) = L_DID(d) − L_placebo(d),

and the claim "selection matters" is the claim that τ(d) ≠ 0 across the usable range
of d. This is the evaluation-side difference-in-differences, and unlike the
selection-side one it does not require any assumption about weight importance.

Damage is recorded as three numbers rather than one, because perplexity alone is
demonstrably insufficient. Two examples from these runs: on Falcon3-3B the harmful
arms sat at 0.93× perplexity, *below* baseline, while collapsing honest behaviour;
and here, the `lie_only` family on Llama-8B at α = −0.75 reports a 0.2% falsehood
rate, which would be the best honesty result in the entire project, at 87,924×
perplexity with 0.5% neutral accuracy and 54% junk. A destroyed model scores as
perfectly honest. Every row therefore carries perplexity ratio, neutral accuracy and
junk rate, and a row counts as **usable** only at perplexity below 1.15× and neutral
accuracy within 5 points of unsteered.

Five families, all at the same per-layer budget:

| family | selection rule |
|---|---|
| `did` | `v·n_lie − (1−v)·n_truth`, the method |
| `magmatch` | random, matched to the `did` mask's per-layer \|W\| decile histogram |
| `random` | uniform random per layer, same count |
| `truth_only` | top-ρ by \|W\| · n_truth, no contrast |
| `lie_only` | top-ρ by \|W\| · n_lie, no contrast |

## 2 Ablation: τ at full ablation, damage matched

At α = −1.0 the `did` and `magmatch` arms land at nearly identical damage on all
three models, so the gap between them is the estimate with no interpolation needed.

| model | `did` | `magmatch` | unsteered | **τ** | ppl did/ctrl | neutral did/ctrl |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 8.6% | 58.9% | 59.6% | **−50.3** | 1.01× / 1.02× | 96.0% / 93.0% |
| Qwen2.5-7B | 1.5% | 30.3% | 29.9% | **−28.8** | 1.06× / 1.05× | 98.0% / 98.5% |
| Qwen2.5-1.5B | 3.8% | 12.1% | 8.5% | **−8.3** | 1.04× / 1.03× | 91.5% / 90.0% |

τ is large, negative and monotone in model size. The control does not merely fail to
help: on Qwen-1.5B it makes the model *more* likely to produce falsehoods (12.1%
against 8.5% unsteered) at every negative knob tested, which is worth remembering
before reading any single-arm ablation result as evidence of a mechanism.

The Qwen-1.5B row is where τ is smallest, and it is also the model with the least
headroom: 8.5% unsteered falsehood leaves 8.5 points to remove. Conclusions about the
method's strength should not be drawn from it, in either direction.

## 3 Amplification: τ at the strongest usable knob

Here the knob is chosen per family as the largest one that still passes the usability
bar, so damage is comparable rather than identical.

| model | `did` | `magmatch` | unsteered | **τ** |
|---|---|---|---|---|
| Llama-3.1-8B | 78.9% @ α=2 (1.09×, 91.0%) | 53.0% @ α=2 (1.06×, 96.0%) | 59.6% | **+25.9** |
| Qwen2.5-7B | 36.6% @ α=1 (1.05×, 99.0%) | 26.5% @ α=1 (1.03×, 98.0%) | 29.9% | **+10.1** |
| Qwen2.5-1.5B | 16.3% @ α=1.75 (1.10×, 86.5%) | 12.0% @ α=1.75 (1.09×, 88.0%) | 8.5% | **+4.3** |

Amplification is real but much weaker than ablation, and it degrades the model where
ablation does not: `did` at α = 2 on Llama-8B costs 3.5 points of neutral accuracy
while the control at the same knob *gains* 1.5. Past the usable range it stops
working entirely — `did` at α = 3 falls back to 66.5% at 1.20× perplexity, below its
own α = 2 value. The method's useful direction is ablation.

## 4 The contrast is what keeps the mask harmless

The two no-contrast families are unusable at every knob on every model:

| family | perplexity range | neutral accuracy |
|---|---|---|
| `truth_only` | 2.4× to 484,011× | 0.0% – 54.5% |
| `lie_only` | 2.1× to 623,117× | 0.0% – 60.5% |

Ranking by `|W| · n` alone selects the model's largest-magnitude weights, and
scaling those destroys it. The subtraction in the contrast is what pushes the
selection into the low-magnitude regime where a 4% edit is survivable: the `did`
mask's median |W| is about 0.04× the global median. This is a mechanical fact about
the estimator rather than a fact about deception, and it explains why the method
works at all — but it also means the ratio knob is functioning as a magnitude
filter, which is the finding behind dropping the selection-side second difference.

## 5 Uniform random is the wrong control

`random` and `magmatch` differ only in whether the random selection matches the real
mask's magnitude profile, and they behave completely differently:

| model | `random` at α=+3 | `magmatch` at α=+3 |
|---|---|---|
| Llama-3.1-8B | 2,877× perplexity, 0.5% neutral | 1.16× perplexity, 96.5% neutral |
| Qwen2.5-7B | 3,706× perplexity, 0.0% neutral | 1.29× perplexity, 97.5% neutral |
| Qwen2.5-1.5B | 80,691× perplexity, 0.5% neutral | 1.34× perplexity, 79.0% neutral |

A paper that uses uniform random as its specificity control gets to claim
specificity for free, because uniform random includes large weights and any edit to
them is catastrophic. `magmatch` is the null that binds. Note also that `random`
fails the usability bar at α = −1.0 on Llama-8B (1.28× perplexity) while `magmatch`
passes at 1.02×, so the ablation comparison in section 2 could not have been made
against `random` at all.

## 6 Caveats

- The magnitude-matched control is matched by decile, which in practice leaves its
  median |W| about 2.3× the real mask's. It is a strong null, not an exact one. The
  truthful-paraphrase arm from the placebo test is the tighter null, matched on
  magnitude and on 91% of the selected weights.
- One seed per cell. The Qwen-1.5B numbers are small enough that seed noise matters.
- Held-out framings are three hand-written templates, not a benchmark. The LIARS'
  BENCH runs are the external check.
- The usability bar (1.15× perplexity, 5 points of neutral accuracy) is a threshold,
  so arms that miss it narrowly are excluded while arms just inside are kept. All
  rows are in the CSVs.
- `truth_only` and `lie_only` were intended as placebo families and are unusable, so
  they function as a diagnostic of the ranking rule rather than as controls.

## 7 Reproducing

```bash
python run_dose_response.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --knobs="-1.0,-0.75,-0.5" --tag llama_damp
python run_dose_response.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --knobs "1.0,2.0,3.0" --tag llama_amp
```

Rows: `dose_response_<model>_rho<ρ>_<tag>.csv`.
