# Instructed falsehood as the treatment arm and as the benchmark

**Models:** Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, Phi-4-mini-instruct,
gemma-3-4b-it, Falcon3-3B-Instruct, Qwen2.5-1.5B-Instruct, OLMo-2-1B-Instruct
**Hardware:** 1× NVIDIA A100-SXM4-40GB each (NCSA Delta, gpua047, shared with a training job)
**Config:** ρ = 0.04, ratio = 0.37, n_truth = 25, output head excluded
**Code:** `run_instructed_falsehood.py`
**Date:** 2026-09-26

---

> **Re-graded 2026-09-27.** An audit found two grader bugs affecting the 17
> numeric-answer questions (numbers discarded as junk; substring correctness).
> All seven runs saved their responses and were re-scored offline with the fixed
> grader. Best usable configuration per arm, re-graded:
>
> | model | unsteered | `lie` arm | best falsehood arm | gain, lie-flavoured framings | gain, all framings |
> |---|---|---|---|---|---|
> | Phi-4-mini | 44.1% | 17.0% | `error` 6.9% | **+10.1** | +9.9 |
> | Llama-3.1-8B | 68.4% | 8.3% | `pooled` 2.7% | **+5.7** | +5.4 |
> | gemma-3-4b | 40.7% | 27.7% | `fiction` 25.4% | +2.3 | +11.2 |
> | OLMo-2-1B | 36.6% | 33.8% | `pooled` 32.3% | +1.5 | +3.0 |
> | Qwen2.5-7B | 34.4% | 1.9% | `fiction` 1.2% | +0.7 | +0.8 |
> | Falcon3-3B | 29.0% | 9.9% | `error` 9.1% | +0.7 | +0.5 |
> | Qwen2.5-1.5B | 10.0% | 4.9% | `pooled` 4.5% | +0.4 | +0.8 |
>
> The headline survives: the falsehood-built arm beats the lie-built arm on 7/7.
> On the three lie-flavoured framings, which share no treatment wording with the
> falsehood arms and so favour the published arm, the gain exceeds two points on
> three models. The number that moves is gemma-3-4b: its large gain on all
> framings comes mostly from the fiction eval framing, a near-paraphrase of the
> fiction treatment prompts, and shrinks to 2.3 points on the lie-flavoured ones.
> Sections 2 to 5 keep the original numbers for the record.

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

## 2 The falsehood arm beats the lie arm on all seven models

Best usable configuration per arm, where usable means perplexity below 1.15× and
neutral accuracy within 5 points of unsteered. Rates are over all five non-direct
framings. The gain column is the published arm minus the best falsehood arm, so
positive means the falsehood arm is better.

| model | unsteered | published `lie` arm | best falsehood arm | `magmatch` null | gain |
|---|---|---|---|---|---|
| gemma-3-4b | 37.3% | 27.3% | `fiction` **18.1%** | 32.7% | **+9.2** |
| Phi-4-mini | 41.8% | 13.8% | `error` **6.1%** | 37.6% | **+7.6** |
| Llama-3.1-8B | 69.5% | 7.1% | `pooled` **2.4%** | 67.7% | **+4.6** |
| OLMo-2-1B | 37.9% | 32.4% | `pooled` **30.4%** | 36.6% | +2.0 |
| Qwen2.5-7B | 34.9% | 1.8% | `fiction` **1.2%** | 34.8% | +0.6 |
| Falcon3-3B | 22.0% | 10.0% | `error` **9.6%** | 17.8% | +0.4 |
| Qwen2.5-1.5B | 9.0% | 4.5% | `fiction` **4.0%** | 11.0% | +0.4 |

The falsehood arm wins on seven of seven models, which is the result. But the margin
is worth reading carefully: it exceeds two points on only four of them, and the
three smallest gains are inside the range where a single seed proves nothing. The
claim this supports is that switching the treatment arm never hurts and sometimes
helps a great deal, not that it reliably helps by a fixed amount.

Where the gain is large it is large for a reason. On Llama-8B, pooling cuts residual
lying by two thirds and does so at *lower* cost than the published arm, at 1.03×
perplexity and 96.0% neutral accuracy against 1.01× and 95.0%. On Qwen-7B every
falsehood arm converges near 1 to 2% and the ordering stops being meaningful,
because the method has already removed almost everything there is to remove.

The `magmatch` null is flat on six of seven models. The exception is Qwen2.5-1.5B,
where it *raises* falsehood from 9.0% to 11.0%, so on that model the comparison is
against a control that moves in the wrong direction.

## 3 The full damping curves

Held-out falsehood rate over all five non-direct framings, for the three models with
the most headroom. The other four are in the CSVs and in
`reports/summarize_instructed_falsehood.py`. `!` marks a configuration whose neutral
accuracy fell more than 5 points below unsteered, which makes its falsehood rate
uninterpretable rather than good.

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

| model | lie-flavoured framings | fiction and mistake framings | gap |
|---|---|---|---|
| Llama-3.1-8B | 59.6% | **84.2%** | +24.6 |
| Qwen2.5-7B | 28.6% | **44.4%** | +15.8 |
| gemma-3-4b | 34.4% | **41.7%** | +7.3 |
| OLMo-2-1B | 35.0% | **42.1%** | +7.1 |
| Phi-4-mini | 39.8% | **44.8%** | +5.0 |
| Qwen2.5-1.5B | 8.2% | **10.3%** | +2.2 |
| Falcon3-3B | 25.0% | 17.6% | −7.4 |

Asking a model to lie triggers whatever resistance its alignment training installed.
Asking it to answer wrongly inside a made-up story, or to answer the way a forgetful
person would, does not, and the answer that comes out is just as false. A safety
evaluation built only on lie-shaped prompts will therefore understate how easily
these models can be made to assert falsehoods, by 2 to 25 points on six of the seven.

Falcon3-3B is the exception and runs the other way by 7 points. It is also the only
base-model-derived instruct checkpoint in the set with a weak refusal profile
overall, so the most likely reading is that it has little lie-specific resistance to
bypass in the first place. That is a guess, not a measurement.

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

- Building the treatment arm from falsehood instructions rather than lie
  instructions is never worse, across seven models, and is substantially better on
  three of them. It is a change to the prompt set and needs no new machinery. Which
  falsehood style wins varies by model, so pooling is the safe default rather than
  the best choice everywhere.
- The method's own recommended α should be model-specific. Full ablation is right
  for Llama-8B and wrong for gemma-3-4b.
- Evaluations of instructed falsehood should include non-lie framings, which are
  substantially stronger pressure.

Not supported, and worth stating:

- **The method still fails on gemma-3-4b in absolute terms.** Its best usable arm
  leaves 18.1% falsehood against a 34.4% control, and the arms that go further break
  the model. Nothing here fixes that.
- **Seven models, one seed each.** Three of the seven gains are under a point, which
  a single seed cannot distinguish from noise. The four-model claim is the honest one.
- **The method barely works on OLMo-2-1B at all**: the best usable arm reaches 30.4%
  against a 36.6% control and a 37.9% unsteered rate, so roughly 6 points of a
  38-point behaviour.
- **`pooled` is the best arm on only three of seven models** (Llama-8B, OLMo-1B and,
  jointly, Falcon-3B). `fiction` wins three and `error` two. Calling pooling the new
  default rests on it never being much worse, not on it usually being best.
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
