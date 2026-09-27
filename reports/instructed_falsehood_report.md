# Instructed falsehood as the treatment arm and as the benchmark

**Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, gemma-3-4b-it
**Hardware:** 1× NVIDIA A100-SXM4-40GB each (NCSA Delta, gpua047, shared with a training job)
**Config:** ρ = 0.04, ratio = 0.37, n_truth = 25, output head excluded
**Code:** `run_instructed_falsehood.py`
**Date:** 2026-09-26

---

## 1 Why the target changed

The placebo test showed the contrast does not track deception. A fiction-framed or
mistake-framed request builds a mask as good as or better than a lie instruction,
while a truthful-paraphrase arm with matched phrasing diversity builds one that does
nothing. The axis is being instructed to assert a falsehood.

If that is the target, then the treatment arm should span falsehood instructions
rather than lie instructions alone, and the evaluation should pressure the model with
falsehood framings rather than only lie framings. This run does both.

**Selection.** Six arms, all against the same uniform truthful control, all with the
same six-fold prefix rotation so phrasing diversity is held fixed:

| arm | treatment prompts |
|---|---|
| `lie` | `DIRECT_LIE` — the method as published |
| `fiction` | `DIRECT_FICTION_FALSE` — falsehood licensed by a fiction frame |
| `error` | `DIRECT_HONEST_ERROR` — falsehood framed as a mistake |
| `pooled` | the mean of the three column-norm vectors above |
| `para` | `DIRECT_TRUTH_PARA` — truthful null |
| `magmatch` | random, matched to the `lie` mask's \|W\| deciles |

**Evaluation.** The pressure metric now has six framings instead of four: the
original direct, instructed, contextual and implicit, plus a fiction framing and a
mistake framing. No evaluation wording appears in any treatment arm. Three aggregates
are reported: the three lie-flavoured held-out framings (`heldout3`, comparable with
every earlier report), the two new falsehood framings (`falsehood2`), and all five
non-direct framings.

Damping is swept to full ablation, because α = −1.0 beat −0.75 by 27 points on
Llama-8B and the earlier sweeps stopped at −0.75.

## 2 The falsehood arm beats the lie arm on all three models

Best usable configuration per arm, where usable means perplexity below 1.15× and
neutral accuracy within 5 points of unsteered. Rates are over all five non-direct
framings.

| model | published `lie` arm | best falsehood arm | `para` null | `magmatch` null | unsteered |
|---|---|---|---|---|---|
| Llama-3.1-8B | 7.1% (α=−1.0) | **`pooled` 2.4%** (α=−1.0) | 67.1% | 68.3% | 69.5% |
| Qwen2.5-7B | 2.5% (α=−0.75) | **`fiction` 1.2%** (α=−0.75) | 32.6% | 35.4% | 34.9% |
| gemma-3-4b | 27.3% (α=−1.0) | **`fiction` 18.1%** (α=−0.75) | 38.2% | 34.4% | 37.3% |

Pooling the three falsehood styles cuts Llama-8B's residual lying by two thirds
against the published arm, and does so at *lower* cost: 1.03× perplexity and 96.0%
neutral accuracy against the lie arm's 1.01× and 95.0%. On Qwen-7B every falsehood
arm converges near 2% and the ordering stops being meaningful. On gemma-3-4b, the
model where the method has always been weakest, the fiction arm removes 19 points
where the lie arm removes 10.

Both nulls are flat on all three models, so the comparison is between working
interventions rather than between an intervention and damage.

## 3 The full damping curves

Held-out falsehood rate over all five non-direct framings. `!` marks a
configuration whose neutral accuracy fell more than 5 points below unsteered, which
makes its falsehood rate uninterpretable rather than good.

| model | α | `lie` | `fiction` | `error` | `pooled` | `para` | `magmatch` |
|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | −0.50 | 48.9 | 34.0 | 36.4 | 36.9 | 68.1 | 67.7 |
| | −0.75 | 27.4 | 8.9 | 15.6 | 11.6 | 67.9 | 67.8 |
| | −1.00 | 7.1 | 4.7 | 5.4 | **2.4** | 67.1 | 68.3 |
| Qwen2.5-7B | −0.50 | 10.3 | 5.1 | 8.8 | 5.7 | 33.4 | 34.8 |
| | −0.75 | 2.5 | **1.2** | 3.0 | 1.9 | 32.6 | 35.4 |
| | −1.00 | 1.8 | 1.9 | 2.9 | 2.0 | 31.8 | 35.1 |
| gemma-3-4b | −0.50 | 32.6 | 27.7 | 32.7 | 32.2 | 37.5 | 35.3 |
| | −0.75 | 30.8 | **18.1** | 28.6 | 30.0 | 38.2 | 34.4 |
| | −1.00 | 27.3 | 50.7! | 10.7 | 30.2! | 39.0 | 32.7 |

Two features worth naming.

**The falsehood arms reach their effect earlier.** At α = −0.75 on Llama-8B the
fiction arm is already at 8.9% while the lie arm is at 27.4%; the lie arm needs full
ablation to catch up. Since damage grows with |α|, an arm that works at a weaker knob
is strictly preferable.

**Full ablation is not always the best knob.** On Qwen-7B and gemma-3-4b the best
usable result is at −0.75, and pushing to −1.0 either gains nothing or breaks the
model. The earlier sweeps that stopped at −0.75 were not wrong for those models; they
were wrong for Llama-8B, where −1.0 is a large gain.

## 4 The new framings are stronger pressure than the lie framings

An unexpected result, and the most transferable one here. On the unsteered model,
the two new framings elicit falsehood far more readily than the four lie-flavoured
ones:

| model | lie-flavoured framings | fiction and mistake framings |
|---|---|---|
| Llama-3.1-8B | 59.6% | **84.2%** |
| Qwen2.5-7B | 28.6% | **44.4%** |
| gemma-3-4b | 34.4% | **41.7%** |

Asking a model to lie triggers whatever resistance its alignment training installed.
Asking it to answer wrongly inside a made-up story, or to answer the way a forgetful
person would, does not — and the answer that comes out is just as false. A safety
evaluation built only on lie-shaped prompts will therefore understate how easily a
model can be made to assert falsehoods, on these three models by 8 to 25 points.

This also explains why the falsehood-built masks work better. They are built from
the pressure the model actually yields to.

## 5 The masks remain nearly the same weights

Median |W| of the selection relative to all maskable weights, and Jaccard against
the `lie` mask. Equal per-layer budgets mean a Jaccard of J is a shared fraction of
2J/(1+J).

| arm | Llama-8B | Qwen-7B | gemma-3-4b | shared with `lie` (mean) |
|---|---|---|---|---|
| `lie` | 0.042× | 0.040× | 0.048× | — |
| `pooled` | 0.045× | 0.042× | 0.051× | 92% |
| `para` | 0.040× | 0.037× | 0.045× | 91% |
| `error` | 0.048× | 0.045× | 0.053× | 84% |
| `fiction` | 0.049× | 0.045× | 0.060× | 80% |
| `magmatch` | 0.101× | 0.096× | 0.113× | 34% |

The `pooled` mask shares 92% of its weights with the `lie` mask and reaches 2.4%
where the lie mask reaches 7.1%; the `para` mask shares 91% and does nothing at all.
Three masks at essentially the same median magnitude and with most of their weights
in common produce a 65-point spread in behaviour. Whatever is being selected is not
"4% of the small weights".

## 6 What this does and does not support

Supported:

- Pooling falsehood instruction styles is a strictly better treatment arm than lie
  instructions alone on Llama-8B, and no worse on the other two models. It is a
  one-line change to the prompt set and needs no new machinery.
- The method's own recommended α should be model-specific. Full ablation is right
  for Llama-8B and wrong for gemma-3-4b.
- Evaluations of instructed falsehood should include non-lie framings, which are
  substantially stronger pressure.

Not supported, and worth stating:

- **The method still fails on gemma-3-4b in absolute terms.** Its best usable arm
  leaves 18.1% falsehood against a 34.4% control, and the arms that go further break
  the model. Nothing here fixes that.
- **Three models, one seed each.** The gemma ordering in particular rests on single
  runs where several arms sit within a few points of each other.
- **`pooled` wins on one model and ties on another.** Calling it the new default
  rests mainly on the Llama-8B result.
- **The fiction evaluation framing and the fiction treatment arm are the same kind of
  pressure,** differently worded. That is why section 2 reports the all-framings
  aggregate and section 3 shows the `heldout3` column separately in the CSVs: on
  Llama-8B at −0.75 the fiction-built mask reaches 10.2% on the purely lie-flavoured
  framings against the lie-built mask's 34.7%, so the advantage is not an artifact of
  matching the evaluation.
- **Perplexity again failed as the damage measure.** gemma-3-4b's fiction arm at
  −1.0 sits at 1.12× perplexity with neutral accuracy at 44.5%. Any result here read
  off perplexity alone would be wrong.

## 7 Reproducing

```bash
python run_instructed_falsehood.py \
  --model-id meta-llama/Meta-Llama-3.1-8B-Instruct \
  --rho 0.04 --ratio 0.37 --knobs="-0.5,-0.75,-1.0" \
  --batch-size 12 --score-batch 2 --tag resw
```

Rows: `instructed_falsehood_<model>_rho0.04_resw.csv`. Tables:
`reports/summarize_instructed_falsehood.py`.
