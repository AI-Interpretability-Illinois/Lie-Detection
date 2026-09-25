# Reversed contrast: is the DID mask lie-specific?

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fp16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, `gpuA100x4-interactive`)
**Config:** ρ ∈ {0.02, 0.04, 0.08}, ratio = 0.37 (the committed Llama operating point), n_truth = 25
**Code:** `run_reversed_contrast.py`, `analyze_reversed_contrast.py`
**Date:** 2026-09-24

---

## 1 Question

LieNeuro-DID selects weights by

```
DID(w) = v · mean(S_lie)(w) − (1 − v) · mean(S_truth)(w),    v = ratio/(1+ratio)
```

keeps the top-ρ per layer and multiplies them by (1 + α). The claim attached to
the method is that those weights are *deception-specific*. Nothing in the repo
tests that on the weight side: there is a matched-displacement control for
activation steering (`run_as_did_metric_eval.py`), but the weight masks have
never been compared against a control mask of the same size and cost.

This run supplies one. Five masks are built from the **same** pair of score
tensors, all with the same per-layer element count and all amplified with the
same α:

| arm | selection rule |
|---|---|
| `forward` | top-ρ by DID — the method |
| `reverse` | top-ρ by `v·S_truth − (1−v)·S_lie` — the two arms exchanged, same selectivity threshold |
| `antitop` | bottom-ρ by DID — the least lie-specific weights in each layer |
| `random` | uniform random per layer, same k |
| `magmatch` | random, stratified to reproduce the forward mask's \|W\| decile histogram |

If `reverse` or `magmatch` induces as many fluent falsehoods as `forward` at the
same perplexity cost, the intervention is amplifying whatever is contrast-extreme
or simply load-bearing, not a deception circuit.

## 2 Measurement

100 held-out factual questions: the 25 `FACTUAL_QA` items not used to build the
scores, plus all 75 of `FACTUAL_QA_EXT`. One greedy pass plus two sampled passes
(T = 1.0) per configuration; WikiText-2 perplexity is the damage axis.

Two corrections matter, and both change the ranking:

**Rates are computed over items the unsteered model answered correctly** (92 of
100 on the greedy pass). Only those can be flipped.

**Flips are split by length.** `_is_correct` is a substring test, so a verbose
but correct answer scores as wrong — "299,792 kilometers per second" against gold
`300000`, or "Amperage" against `ampere`. Steering makes this model verbose (mean
answer length rises from 1.4 to 22.8 words), so its raw wrong-rate climbs without
it lying once. A flip of ≤ 6 words that passes a junk test is counted as a clean
falsehood; longer flips are reported separately and are not trustworthy.

A junk test was needed beyond the repo's `degenerate`, which requires 8+
whitespace tokens and therefore passes `'!!!!!!!!'` — the exact output a
collapsed arm produces, which would otherwise score as a perfect lie.

## 3 Results

Clean flips on baseline-correct items, greedy pass, 95% bootstrap CI, n = 92.

| arm | α | WikiText ppl ratio | clean flips | junk |
|---|---|---|---|---|
| none | — | 1.00× | 0.0% | 14% |
| forward | 1 | 1.03× | 2.2% [0.0, 5.4] | 14% |
| forward | 2 | 1.13× | 3.3% [0.0, 7.6] | 13% |
| forward | 3 | 1.33× | **15.2%** [8.7, 22.8] | 2% |
| forward | 4 | 1.76× | 5.4% [1.1, 10.9] | 2% |
| forward | 5 | 3.37× | 2.2% [0.0, 5.4] | 0% |
| reverse | 4 | 1.15× | 6.5% [2.2, 12.0] | 16% |
| reverse | 6 | 1.79× | 8.7% [3.3, 15.2] | 19% |
| reverse | 8 | 8.55× | 20.7% [13.0, 29.3] | 19% |
| magmatch | 3 | 1.34× | 7.6–9.8% (2 seeds) | 16% |
| magmatch | 4 | 1.79× | **19.6–21.7%** (2 seeds) | 15% |
| magmatch | 5 | 3.00× | 8.7–17.4% (2 seeds) | 15% |
| random | 2 | 183× | 2.2% | 15% |
| antitop | 1 | 177,187× | 0.0% | 95% |

Compared at matched damage rather than matched α (ρ = 0.04):

| WikiText ppl | best arm | forward |
|---|---|---|
| ~1.14× | reverse 6.5% | 3.3% |
| ~1.34× | **forward 15.2%** | 15.2% |
| ~1.79× | magmatch 21.7% | 5.4% |
| ~3.0× | magmatch 17.4% | 2.2% |

### The same three arms at ρ = 0.02 and ρ = 0.08

ρ = 0.04 alone is misleading, so the key arms were rerun at a quarter and double
the mask size. Each arm's best clean-flip rate over α ∈ {2,3,4,5}, with the
perplexity it cost:

| ρ | forward | magmatch | reverse |
|---|---|---|---|
| 0.02 | **18.5% @ 1.29×** | 20.7% @ 2.39× | 5.4% @ 1.26× |
| 0.04 | **15.2% @ 1.33×** | 21.7% @ 1.78× | 8.7% @ 1.79× |
| 0.08 | **19.6% @ 1.37×** | 16.3% @ 2.44× | 13.0% @ 1.34× |

The forward mask peaks at 15–20% in all three, always near 1.3× perplexity. The
magnitude-matched control reaches comparable rates, 16–21%, but needs 1.8–2.4×
perplexity to get there. Compared at 1.3×, forward leads the control in all three
mask sizes: 18.5% vs 6.5% at ρ = 0.02, 15.2% vs 7.6–9.8% at ρ = 0.04, and 19.6%
vs 8.7% at ρ = 0.08 (the control's 8.7% already costing 1.51×).

### What the controls establish

**The DID mask is about twice as damage-efficient at its optimum.** This is the
one result that supports the specificity claim, and it reproduces across three
mask sizes: at ~1.3× perplexity the lie-specific mask produces roughly twice the
clean falsehoods of a magnitude-matched random mask of the same size. The
intervals overlap at ρ = 0.04 but the direction is consistent, and at ρ = 0.02
they nearly separate. Selection by DID score is doing something.

**It is not uniquely capable, only cheaper.** Given more damage the random control
reaches the same 16–21% ceiling, so there is no deception level reachable only
through the DID mask. The advantage is efficiency, not access.

**The reversed contrast is not inert.** Amplifying the weights that matter *more
to truth-telling* also produces clean falsehoods: 13.0% at ρ = 0.08 and 1.34×,
statistically indistinguishable from forward's 19.6% at the same damage, and
20.7% at ρ = 0.04 and 8.55×. It is weaker than forward at matched damage in two
of three mask sizes and comparable in the third. Reversing the sign of the
contrast weakens the effect but does not remove it, so the method is not
selecting a deception-only circuit.

**Above ~1.5× the ordering inverts.** At ρ = 0.04 and 1.79× the random control
produces four times as many clean falsehoods as the forward mask (19.6–21.7% vs
5.4%), reproducible across two seeds. Pushing α past the forward mask's optimum
does not buy more lying, it buys verbosity and collapse.

**Which random control you use decides the answer.** A uniform random mask is
catastrophic at α ≥ 2 (perplexity 10⁵×), while the magnitude-matched one at the
same α and count is benign and more effective than the method. A random-mask
control that ignores \|W\| would have made the method look specific.

**Raw wrong-rate points the wrong way.** Forward's wrong-rate keeps climbing with
α (15.3% → 43.7% → 49.3%) while its clean flips fall after α = 3. The extra
"wrongness" is verbosity defeating the substring grader: at α = 4, 16.3% of
baseline-correct items produce a long flip against 5.4% terse ones.

**`antitop` destroys the model at α = 1** (perplexity 177,187×, 95% junk), which
is the one result that does favour the DID ranking: the bottom of the ranking is
load-bearing in a way the top is not.

## 4 Conclusion

The DID mask carries real signal, and it is narrower than the method's framing
implies. Three findings, in order of confidence:

1. **At its operating point the mask is about twice as damage-efficient as a
   matched random control**, and this reproduces at ρ = 0.02, 0.04 and 0.08. The
   committed configurations sit in exactly that band. This is the first control
   evidence on the weight side that the DID ranking is not arbitrary.
2. **The effect is efficiency, not exclusivity.** A magnitude-matched random mask
   reaches the same 16–21% ceiling given 1.8–2.4× perplexity instead of 1.3×, and
   above ~1.5× it overtakes the DID mask outright.
3. **Reversing the contrast weakens the effect without removing it.** Amplifying
   truth-specific weights still yields clean falsehoods at rates that overlap the
   forward mask's at matched damage in one of three mask sizes. Whatever the mask
   selects is not deception-only.

Two measurement points matter for how the repo reports results. The forward mask's
raw wrong-rate keeps climbing well past the α where its *clean* falsehoods peak,
because steering makes the model verbose and verbosity defeats the substring
grader — so α should be chosen on clean flips, not Δwrong. And the choice of
random control decides the verdict: a uniform random mask is catastrophic and
makes the method look specific, while a magnitude-matched one is the control that
actually competes.

## 5 Limits

- **Scale.** Qwen2.5-1.5B-Instruct, chosen to fit score tensors in a 64 GB
  allocation within a one-hour window. ρ = 0.04 and ratio = 0.37 were tuned on
  Llama-3.1-8B. The controls should be rerun at 7–8B before the conclusion is
  generalised; nothing here says these rates transfer upward. A 7B rerun needs
  ~50 GB of RAM for the two score tensors, so request `--mem=120G`.
- **ratio was not swept.** It is the knob the repo found most consequential, and
  every arm here uses 0.37. The controls should be repeated at the ratio each
  model actually prefers.
- **lm_head excluded.** This model ties input and output embeddings, so editing
  `lm_head` would also edit the embedding matrix. The repo's committed runs
  include it (`skip_modules=[]`), which is safe only on the untied 7B models.
- **No LLM judge.** Grading is substring match with a length split standing in
  for the repo's semantic-equivalence judge. The clean-flip counts are a floor.
- **Power.** n = 92 baseline-correct items gives ±5–9 points at these rates, so
  the 1.34× band's ordering is suggestive, not established.
- **Seeds.** `random` and `magmatch` were run at two seeds; `forward`, `reverse`
  and `antitop` are deterministic given the scores.

## 6 Reproducing

```bash
python run_reversed_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --rho 0.04 --ratio 0.37 --alphas 1,2,3,4,5 --control-alphas 1,2,3,4,5 \
  --arms forward,reverse,antitop,random,magmatch --reps 2

python run_reversed_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --rho 0.04 --ratio 0.37 --alphas 6,8,10,12 --arms reverse --tag hialpha

python run_reversed_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --rho 0.04 --ratio 0.37 --alphas 3,4,5 --control-alphas 3,4,5 \
  --arms magmatch,random --seed 1 --tag seed1

# mask-size check (scores are reused across rho)
for R in 0.02 0.08; do
  python run_reversed_contrast.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
    --rho $R --ratio 0.37 --alphas 2,3,4,5 --control-alphas 2,3,4,5 \
    --arms forward,magmatch,reverse --reps 2
done

python analyze_reversed_contrast.py <out-dir> 'reversed_contrast_Qwen2p5-1p5B*'
```

Scores are cached to `scores_<model>_ntruth25_lmhead0.pt` and reused, so only the
first run pays for the forward passes (~40 s at 1.5B).
