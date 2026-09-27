# Placebo-treatment test: the contrast tracks instructed falsehood, not deception

**Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, gemma-3-4b-it,
Falcon3-3B-Instruct, Qwen2.5-1.5B-Instruct
**Hardware:** 1× NVIDIA A100-SXM4-40GB each (NCSA Delta, reservation sup-30781)
**Config:** ρ = 0.04 (0.06 for Qwen-7B), ratio = 0.37, n_truth = 25, output head excluded
**Code:** `run_placebo_contrast.py`
**Date:** 2026-09-26

---

## 1 The test

The method selects weights by a single difference of Wanda column statistics,

    colstat_j = v · n_lie_j − (1 − v) · n_truth_j,    v = ratio / (1 + ratio)

and the claim attached to it is that this isolates deception. That claim has a
falsification test which costs nothing: run the identical pipeline with a treatment
arm that carries no deception, and see whether the resulting mask steers behaviour
just as well. If it does, no refinement of the contrast can rescue the
interpretation, and every second-difference design built on top of it is wasted
work.

Three placebo treatments, each contrasted against the same uniform
`DIRECT_TRUTH_INSTRUCT` control the real method uses, so only the treatment arm
changes. All four treatment sets are 50 prompts over the same question stems with
the same six-fold prefix rotation, so instruction-phrasing diversity is held fixed:

| arm | treatment prompts | carries |
|---|---|---|
| `real` | `DIRECT_LIE` | intent to deceive, asserts falsehood |
| `para` | `DIRECT_TRUTH_PARA` | nothing. Truthful instructions, six phrasings |
| `fiction` | `DIRECT_FICTION_FALSE` | asserts falsehood inside a fiction frame, no intent to deceive |
| `error` | `DIRECT_HONEST_ERROR` | asserts falsehood framed as a mistake, no fiction frame |

`para` is the arm the published design never had. The real contrast rotates six lie
phrasings against a single truth phrasing, so phrasing diversity is already inside
it as a confound, and `para` is what isolates it.

Each mask is damped at α = −0.75 and amplified at α = +3.0, against the unsteered
model and against a magnitude-matched random mask of the real selection.

## 2 Result

Change in held-out falsehood rate under damping, in percentage points against each
model's own unsteered rate. Negative means the intervention made the model more
honest. Held-out averages three pressure framings sharing no wording with any
treatment arm.

| model | unsteered | `real` | `fiction` | `error` | `para` | `magmatch` |
|---|---|---|---|---|---|---|
| Llama-3.1-8B | 60.9% | −26.7 | **−51.9** | −39.6 | −1.8 | −1.1 |
| Qwen2.5-7B | 30.0% | −24.2 | **−28.8** | −27.9 | −2.5 | +0.3 |
| gemma-3-4b | 34.4% | −4.5 | **−11.8** | −3.1 | +4.7 | +0.7 |
| Falcon3-3B | 25.0% | −8.2 | −14.4 | **−16.0** | −1.6 | −2.3 |
| Qwen2.5-1.5B | 8.5% | −4.2 | −4.3 | **−4.9** | +1.8 | +4.0 |
| **mean** | — | **−13.5** | **−22.2** | **−18.3** | **+0.1** | **+0.3** |

The test fails, and it fails in a specific and informative way.

**Both falsehood placebos beat the lie arm.** Not match it: beat it, on average by
nine and five points, and in five out of five models at least one placebo is at
least as good as `real`. On Llama-8B the fiction-framed mask removes 52 points of
held-out lying where the lie-instructed mask removes 27. If the lie instruction were
supplying deception-specific signal, the arms that strip deceptive intent out and
keep only "assert something false" should be worse. They are better.

**The pure null does nothing.** `para` averages +0.1 points, statistically
indistinguishable from the magnitude-matched control's +0.3 and from doing nothing at
all. So the pipeline is not a machine that turns any prompt contrast into a working
mask, and phrasing diversity alone explains none of the effect. This is the result
that keeps the method alive as an intervention.

Put together: the axis is **being instructed to assert a falsehood**. Deceptive
intent is not required, and adding it makes the mask slightly worse.

## 3 The masks are nearly the same weights

The diagnostics say something sharper than the behaviour does.

| arm | mean Jaccard vs `real` | shared fraction of the selection | mean median \|W\| vs all weights |
|---|---|---|---|
| `para` | **0.835** | **91%** | 0.042× |
| `error` | 0.713 | 83% | 0.050× |
| `fiction` | 0.686 | 81% | 0.052× |
| `magmatch` | 0.230 | 37% | 0.099× |
| `real` | 1.000 | 100% | 0.044× |

Every mask selects exactly the same number of weights per layer, so a Jaccard of J
means the two selections share 2J/(1+J) of their weights.

`para` selects 91% of the same weights as `real` and does nothing, while `real`
removes 13.5 points of lying. Two masks that overlap this heavily cannot be
separated by "how much damage does ablating 4% of small weights do" — that is held
almost exactly fixed here, and so is median magnitude (0.042× against 0.044×).
Whatever the method is doing lives in the ~9% of the selection that the truthful
contrast does not pick. That is a genuine specificity result, and it is stronger
evidence for the method than the magnitude-matched control is, because the null here
is matched on magnitude *and* on 91% of the actual weights.

Note also that `magmatch` sits at 0.099× median |W| against the real mask's 0.044×,
a factor of 2.3. Decile matching is not exact matching, so the random control is a
strong null but not a perfect one. `para` is the better null, and it was free.

## 4 Amplification is not usable

The amplification direction at α = +3.0 does not support any reading. On Llama-8B
the fiction mask amplified gives 6.1% held-out falsehood, *lower* than unsteered,
with 3.6× perplexity and neutral accuracy at 48.5%; on Falcon3-3B the same arm
reaches 14.5× perplexity and 24.5% neutral accuracy. These are broken models, not
more deceptive ones, and the falsehood rate stops being interpretable once the
model cannot answer neutral questions. The placebo comparison is therefore read off
the damping direction only.

One number from that direction is worth keeping for its own sake: amplifying the
magnitude-matched random mask on Falcon3-3B raises direct lying from 13.8% to 72.9%
while holding perplexity at 1.41×. A random mask can look like a powerful deception
intervention. Any amplification claim needs this control at the same knob.

## 5 What follows

- **Stop calling the selection a deception circuit.** It is a
  falsehood-under-instruction circuit, and the honest framing of every steering
  result in this project is "suppresses instructed falsehood".
- **Build the treatment arm from falsehood instructions, not lie instructions.**
  If the target is instructed falsehood, pooling lie, fiction and mistake framings
  should sharpen the estimate. This is what `run_instructed_falsehood.py` tests.
- **Evaluate on falsehood framings too.** The metric's four framings were all
  lie-flavoured; a fiction framing and a mistake framing have been added to
  `PRESSURE_SETS`, held out from every treatment wording.
- **Drop the proposal's selection-side second difference.** With the control pair
  set to (fiction, truth), its DiD₂ algebraically collapses to a single difference
  against a different control, and the stratified run showed the sign-mixed second
  differences select weights 2.1–2.7× the global median magnitude and destroy the
  model before changing behaviour.
- **Use `para` as the standard null from here on**, alongside the
  magnitude-matched mask rather than instead of it.

## 6 Caveats

- Held-out framings are three hand-written templates, not a benchmark.
- One seed per model; the 4-point moves on Qwen-1.5B and gemma-3-4b are within the
  range where seed noise matters, which is why the argument rests on the two 7–8B
  models where the effects are 24 to 52 points.
- Qwen-7B ran at ρ = 0.06 rather than 0.04, so its magnitudes are not directly
  comparable with the other four rows.
- `fiction` and `error` prompt sets are mine, written to match the structure of
  `DIRECT_LIE`; they were not validated by a human rater.

## 7 Reproducing

```bash
python run_placebo_contrast.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --damp=-0.75 --amp=3.0 --tag llama8b
```

Rows: `placebo_contrast_<model>_rho<ρ>_<tag>.csv`.
