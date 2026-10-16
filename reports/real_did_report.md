# A real difference-in-differences estimator, and why it finds worse parameters

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fp16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua001)
**Code:** `did_estimators.py`, `run_real_did.py`, `diagnose_did_masks.py`
**Date:** 2026-09-25

---

## 1 What was built

The estimator in the repo is a single difference. Importance is computed under a
lie instruction and under a truth instruction and the two are subtracted. Both
arms are matched on instruction presence, which removes the "any instruction"
confound, but one difference between two conditions is not a difference of
differences.

**The obvious repair is algebraically empty.** Adding a no-instruction baseline
and computing `(S_lie − S_neutral) − (S_truth − S_neutral)` returns
`S_lie − S_truth` exactly, because both arms subtract the same term. A second
difference only adds information when the baseline differs between the arms. Two
that do were implemented:

- **`did_position`** — within each arm, the shift from reading the instruction to
  producing the answer: `[S_lie(answer) − S_lie(prompt)] − [S_truth(answer) −
  S_truth(prompt)]`. The prompt-position terms carry whatever the model does
  while reading the words "lie" and "truthfully". That confound is real and was
  already observed: the activation-steering baseline, pushed hard, answered the
  literal token "LIE" to every question.
- **`did_behavior`** — within each arm, the shift from answering truthfully to
  stating a falsehood, on the model's own generations: `[S_lie(false) −
  S_lie(true)] − [S_truth(false) − S_truth(true)]`. The truth arm's term is
  "weights active when the model is simply wrong", so subtracting it separates
  deception from error. That is the confound the whole pipeline runs on, since
  its outcome measure is a wrong answer either way.

Each also has a standardized variant, dividing the activation side by its
across-prompt spread.

**Two implementation facts worth recording.** The Wanda score is separable,
`S_ij = |W_ij| · ‖x_j‖`, and the weight magnitude does not depend on the prompt.
So every prompt-level statistic and every contrast between cells is a function of
per-column activation norms alone, and the per-weight score is that column
statistic times `|W_ij|`. Storing an items-by-columns matrix per layer costs
megabytes against about 5 GB for one full score tensor, and it keeps the
per-prompt resolution the summed estimator throws away.

That separability also kills the obvious t-statistic upgrade. Ranking weights by
a t statistic cancels `|W_ij|` from numerator and denominator, so the per-weight t
equals the per-column t and the ranking selects whole input columns with no
weight-level resolution. Wanda's selectivity comes entirely from `|W|`. The usable
form standardizes only the activation side and reattaches the magnitude.

## 2 Result: no DID estimator produces a working edit

Every estimator was evaluated the way the honesty result was obtained: damp the
mask and measure falsehoods under pressure, on three lie framings the mask was
not built from. Alpha was swept from gentlest to strongest and abandoned once the
model broke. A configuration counts as working only if perplexity stays within
1.15× and neutral accuracy within 5 points of baseline, because a destroyed model
lies 0% of the time and means nothing.

Best working configuration per estimator, held-out lie rate:

| estimator | ρ = 0.04 | ρ = 0.01 |
|---|---|---|
| single difference (repo) | **4.0%** at α = −1.0 | **4.2%** at α = −0.75 |
| single, standardized | 4.3% at α = −1.0 | 4.0% at α = −0.75 |
| unsteered | 8.3% | 8.3% |
| magnitude-matched control | 8.7% | 10.0% |
| `did_position` | 9.6% at α = −0.25 | none at any α |
| `did_position_d` | 9.2% at α = −0.10 | none at any α |
| `did_behavior` | 10.7% at α = −0.10 | none at any α |
| `did_behavior_d` | 10.9% at α = −0.10 | none at any α |

The DID estimators are not merely worse, they have no usable operating point. At
ρ = 0.04 their only non-broken configurations are the gentlest damping tested,
where they sit at or above the unsteered lie rate. At ρ = 0.01, given a quarter of
the mask size so their damage could match, none of the four produced a working
configuration at any alpha. They break the model on the way to changing its
behaviour: `did_behavior` at α = −0.5 reaches perplexity 338× with neutral
accuracy 0%, where the single difference at α = −1.0 costs 1.04×.

Standardization is neutral. The standardized single difference ties the raw one,
4.0% against 4.2%, and overlaps it at Jaccard 0.74, so the "one long prompt
dominates the sum" concern does not bite at 24 balanced prompts.

## 3 Why: the ratio knob is a magnitude filter in disguise

No behavioural evaluation is needed to see what went wrong. Each estimator's mask
was compared at ρ = 0.04 against the distribution of maskable weights, whose
median `|W|` is 0.0178:

| estimator | columns with statistic > 0 | median \|W\| of selected, vs all | Jaccard vs single |
|---|---|---|---|
| single | 0.5% | **0.04×** | 1.000 |
| single, standardized | 0.5% | 0.04× | 0.740 |
| `did_position` | 31.2% | 2.10× | 0.000 |
| `did_position_d` | 31.2% | 2.15× | 0.000 |
| `did_behavior` | 58.5% | 2.55× | 0.001 |
| `did_behavior_d` | 58.5% | 2.71× | 0.001 |

With ratio = 0.37 the single difference's column statistic is negative for 99.5%
of columns, because the weighting puts 0.73 on the truth term and 0.27 on the lie
term. Ranking `|W_ij| · (mostly negative)` therefore selects, in effect, the
*smallest* weights sitting in the least-negative columns: the median selected
weight is 4% of the global median. That is why damping the whole mask at α = −1
costs 4% perplexity. The edit is nearly free because it is shrinking a sparse set
of near-zero weights.

A difference of differences has no such asymmetry. Its statistic is positive for a
third to three fifths of columns, so the same top-ρ rule lands on the largest
weights inside the most positive columns, with a median 2.1 to 2.7× the global
median, which is 50 to 70 times larger than the single difference selects. Those
weights are load-bearing, and shrinking them destroys the model. The two mask
families are essentially disjoint, Jaccard 0.000 to 0.001. Layer-type composition
is not the difference: attention's share of the mask is 11.8% in every case.

So the hand-tuned `ratio` is doing something quite different from what the method
claims for it. The znorm write-up established that it acts as a selectivity
threshold rather than a scale correction, which is right. What the diagnostic adds
is *which* axis it selects on: it is overwhelmingly a magnitude filter, keeping
the intervention inside a region of tiny weights where the model tolerates
damping. Adding a second difference removes that asymmetry, and with it the
property that made the edit safe.

The honesty result itself survives this, and is better understood because of it.
The magnitude-matched control reproduces the mask's `|W|` histogram and does
nothing (8.7% against 8.3% unsteered), while the mask halves the held-out lie
rate. So it is not "shrink any tiny weights"; within the class of very small
weights, which ones are chosen matters.

## 4 Answer

Mimicking economics more closely does not help here, and the reason is specific
rather than incidental. In a panel, a second difference costs you variance and
buys you identification. Over weights, the second difference also discards the
asymmetry that was keeping the intervention in a safe region of the weight
distribution, and there is no counterfactual or assignment mechanism to buy in
exchange. What transfers from the econometrics is the discipline, not the
estimator: never trust a single contrast, keep the per-prompt spread rather than
summing it away, and run placebos.

**If you want to pursue it, the informative variant is a constrained one.** The
DID contrast and the magnitude filter are separable concerns, and this experiment
confounded them. Rank by the DID column statistic *within* the low-magnitude
region the ratio knob implicitly selects, for example top-ρ by `did_position`
restricted to weights below the 10th percentile of `|W|`. That keeps the safety
property and tests the better contrast on its own merits. It is a small change to
`mask_from_columns` and it is the experiment I would run next.

## 5 Limits

- **Cell size.** The truth arm produced only 19 false answers out of 240, so
  `did_behavior`'s second term is estimated from few items and is the noisiest
  quantity here. More samples per question would firm it up, though the mask's
  magnitude profile, which is what breaks it, would not change.
- **Scoring format.** Answers were generated with the chat template and scored as
  raw concatenated text, following the repo's finding that chat templates make
  special tokens dominate the scores. The prompt and answer cells are therefore
  consistent with each other but not with generation formatting.
- **One model, one ratio.** Qwen2.5-1.5B-Instruct at ratio 0.37 throughout. The
  0.5% positive-column figure is a property of that ratio, so the mechanism
  should be re-checked wherever ratio is retuned.
- **Damping only.** The DID masks were evaluated as honesty edits, not as
  deception edits. They might behave differently amplified, which is untested.
- **Standardization did not test the ratio knob.** The standardized variants keep
  the ratio weighting, so the claim that a scale-free statistic could retire that
  hyperparameter remains unevaluated. Running `single_d` at ratio = 1.0 is the
  one-line test.

## 6 Reproducing

```bash
python run_real_did.py --model-id Qwen/Qwen2.5-1.5B-Instruct --rho 0.04 \
  --ratio 0.37 --n-truth 25 --k-samples 10 --reps 1

# note the '=' : argparse reads a bare -0.5 as a flag
python run_real_did.py --model-id Qwen/Qwen2.5-1.5B-Instruct --rho 0.01 \
  --alphas="-0.5,-0.75,-1.0" --tag rho0p01

python diagnose_did_masks.py --rho 0.04 --k-samples 6
```
