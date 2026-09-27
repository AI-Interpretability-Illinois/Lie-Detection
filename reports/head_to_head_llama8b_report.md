# Head to head on Llama-3.1-8B: the mask against activation steering, both swept

**Model:** `meta-llama/Meta-Llama-3.1-8B-Instruct` (bfloat16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua047, shared with a training job)
**Config:** ρ = 0.04 (coverage 4.000%), ratio = 0.37, n_truth = 25, output head excluded
**Code:** `run_prior_work_baselines.py --as-layers --as-rel-coeffs`
**Date:** 2026-09-26

---

## 1 Why this run exists

Two objections were raised against the earlier comparisons, and both were fair.

The first: the weight mask was evaluated at α = −0.75 while activation steering was
swept over coefficients, so the mask never got its best setting. Full ablation
(α = −1.0) turned out to beat −0.75 by 27 points on this model.

The second, the larger one: activation steering was applied at a single hand-picked
layer. Layer choice is the main hyperparameter of that family, and on Qwen-1.5B a
layer sweep reversed two of the mask's wins outright. A comparison that sweeps one
method and not the other decides nothing.

So this run sweeps both. The mask gets full ablation and amplification. Activation
steering gets five layers spanning the depth of the model and three coefficients
each, scaled per layer as a fraction of that layer's median residual norm, because
fixed absolute coefficients are a no-op on models whose residual norms differ by
four orders of magnitude.

Measured residual norms at the steered layers: L4 3.4, L8 5.5, L14 8.2, L21 15.8,
L26 26.3. The three relative coefficients −0.06, −0.18 and −0.5 therefore become
absolute coefficients from −0.20 to −13.16 depending on depth.

## 2 Honesty direction

Falsehood rate under pressure, claim-conditional, on the 95 questions the unsteered
model answers correctly when asked neutrally. `direct` is the lie framing the scores
were built from; `held-out` averages three framings that share no wording with it.
Lower is better in this section.

| arm | knob | direct | held-out | neutral acc | ppl |
|---|---|---|---|---|---|
| **weight mask (this repo)** | **−1.0** | **4.2%** | **7.2%** | **95.0%** | **1.01×** |
| actsteer L14, r = −0.5 | −4.09 | 22.6% | 28.4% | 93.5% | 1.11× |
| actsteer L8, r = −0.5 | −2.77 | 13.2% | 29.3% | 94.5% | 1.09× |
| actsteer L14, r = −0.18 | −1.47 | 58.4% | 49.8% | 94.0% | 1.01× |
| honest prompting | — | 74.2% | 50.9% | 93.0% | 1.00× |
| actsteer L21, r = −0.5 | −7.89 | 61.1% | 51.9% | 95.0% | 1.14× |
| actsteer L8, r = −0.18 | −1.00 | 70.0% | 53.0% | 94.0% | 1.01× |
| actsteer L26, r = −0.5 | −13.16 | 68.4% | 54.6% | 94.0% | 1.16× |
| actsteer L21, r = −0.18 | −2.84 | 69.5% | 55.8% | 95.0% | 1.02× |
| actsteer L4, r = −0.5 | −1.68 | 50.0% | 56.8% | 94.5% | 1.02× |
| magnitude-matched control | −1.0 | 75.3% | 58.6% | 94.0% | 1.02× |
| *unsteered* | — | *74.2%* | *59.6%* | *94.5%* | *1.00×* |

The remaining seven activation arms (the r = −0.06 row at every depth, plus L4 and
L26 at r = −0.18) all land between 57% and 59%, indistinguishable from unsteered.

Three things to read off this table.

**The mask wins the honesty direction, and by a wide margin.** 7.2% held-out lying
against 28.4% for the best of fifteen swept activation configurations. It also pays
less for it: 1.01× perplexity and 95.0% neutral accuracy, against 1.11× and 93.5%.
This is the first comparison in this project where the mask beats the baseline with
both methods swept, and it is the largest model tested.

**The control does nothing, so the win is about which weights were chosen.** A
random mask matched to the real mask's per-layer magnitude deciles, ablated at the
same α = −1.0, moves held-out lying from 59.6% to 58.6%. The mask's own 52-point
drop is not what ablating 4% of the weights does in general.

**Activation steering needs depth 8 to 14 and a large coefficient to do anything.**
Its dose-response is steep and narrow: r = −0.06 is inert everywhere, r = −0.5 works
at L8 and L14 and costs 1.09–1.11× perplexity, and by L21 and L26 the same relative
coefficient stops helping while the perplexity cost keeps rising.

## 3 Deception direction

Higher is better here. This direction is where the mask does not win.

| arm | knob | direct | held-out | neutral acc | ppl |
|---|---|---|---|---|---|
| deceptive prompting | — | 82.1% | 81.9% | **5.0%** | 1.00× |
| weight mask | +3.0 | 72.6% | 70.9% | 89.5% | 1.21× |
| magnitude-matched control | +3.0 | 37.9% | 50.9% | 96.0% | 1.16× |
| *unsteered* | — | *74.2%* | *59.6%* | *94.5%* | *1.00×* |

Amplification buys 11 points of held-out lying for 1.21× perplexity and 5 points of
neutral accuracy. Deceptive prompting buys 22 points for free, but it is not a
comparable intervention: it collapses neutral accuracy to 5%, which means it does
not make the model deceptive under pressure, it makes it answer everything wrongly.
The honest reading is that no arm here induces targeted deception well, and the
mask's amplification direction remains the weaker half of the method.

Note the control's behaviour at +3.0: amplifying 4% of random low-magnitude weights
*reduces* direct lying to 37.9% while raising neutral accuracy to 96.0%. Damage from
amplification does not move behaviour in a fixed direction, which is why the
magnitude-matched arm has to be run at every knob rather than once.

## 4 What this does and does not settle

Settled: on the largest model tested, with the layer and coefficient confound closed
on the baseline's side and full ablation available on the mask's side, the weight
mask is the better honesty intervention by 21 points at lower cost, and the effect
is specific to the selected weights rather than to the amount of damage.

Not settled, and worth stating plainly:

- **This is one model.** The same sweep on Qwen-1.5B put activation steering ahead
  on the LIARS' BENCH splits. Llama-8B and Qwen-1.5B disagree, and 8B is the more
  relevant scale but not a majority.
- **The mask is not selecting a deception circuit.** The placebo test established
  that a fiction-framed or mistake-framed treatment arm builds an equally good
  mask, so what this intervention suppresses is instructed falsehood, not deceptive
  intent. The win in section 2 is a win at suppressing instructed falsehood.
- **The magnitude-matched control is not perfectly matched.** Decile matching leaves
  its median |W| about 2.4× the real mask's, so it is a strong null but not an
  exact one.
- **Held-out framings are three hand-written templates**, not a benchmark.
- **No fine-tuning baseline.** LoRA or DPO remains the obvious strong baseline and
  is still absent.

## 5 Reproducing

```bash
python run_prior_work_baselines.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --variants score_diff_instruct \
  --damp=-1.0 --amp=3.0 \
  --as-layers 4,8,14,21,26 --as-rel-coeffs=-0.06,-0.18,-0.5 \
  --batch-size 12 --score-batch 2 --tag llama8b_h2h
```

Rows: `prior_work_Meta-Llama-3p1-8B-Instruct_rho0.04_llama8b_h2h.csv`.
