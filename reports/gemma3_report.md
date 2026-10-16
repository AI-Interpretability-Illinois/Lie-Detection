# Gemma-3: the family story does not hold, and a coefficient-scale problem

**Models:** `google/gemma-3-1b-it` (complete), `google/gemma-3-4b-it` (6 of 8 arms),
`google/gemma-3-12b-it` (not started)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua001)
**Config:** ρ = 0.04, ratio = 0.37, bfloat16, output head excluded
**Code:** `run_liars_bench_honesty.py`, `did_estimators.py`, `src/activation_steering.py`
**Date:** 2026-09-26

---

## 1 What this was for

Three earlier models gave two verdicts: the weight mask beat difference-in-means
activation steering on Qwen2.5-1.5B and Qwen2.5-7B, and lost to it on
Llama-3.1-8B. Reading that as a family effect rested on one Llama. Gemma-3 at
three sizes was meant to test it.

**It does not hold.** Gemma-3-4b patterns with Llama, not with Qwen, and
Gemma-3-1b turns out to be unusable on this split. So the tally is two families
favouring the baseline, one favouring the mask, and no size trend.

## 2 A scale problem that invalidated the first attempt

Steering coefficients are only meaningful against the residual norm they are added
to, and that norm varies by three orders of magnitude across these families:

| model | steering layer | median residual norm |
|---|---|---|
| Llama-3.1-8B | 8 of 32 | 5.5 |
| Qwen2.5-7B | 7 of 28 | 32.4 |
| Qwen2.5-1.5B | 7 of 28 | 34.9 |
| Gemma-3-1b | 6 of 26 | **1165** |
| Gemma-3-4b | 8 of 34 | **7885** |

The first Gemma-1b run used the same absolute coefficients as the Qwen runs, −2 to
−16, which on Gemma is under 1.4% of the residual norm. Predictably every
activation arm was a no-op, changing the lie rate by under two points. That was a
measurement artefact, not a finding.

`run_liars_bench_honesty.py` now takes `--as-rel-coeffs`, coefficients as fractions
of the measured residual norm, so the baseline gets a comparable intervention on
every model. Re-read in those terms the earlier runs were roughly matched by luck:
c = −1 on Llama is 0.18× the norm and c = −8 on Qwen is 0.25×.

## 3 Gemma-3-1b: the split is degenerate for this model

| arm | lie share (95% CI) | claims | honest | no_claim |
|---|---|---|---|---|
| *unsteered* | *100.0% [100.0, 100.0]* | 268 | *0.0%* | 13.4% |
| ours α = −0.75 | 93.7% [89.5, 97.2] | 143 | 1.8% | 11.6% |
| ours α = −0.5 | 97.1% [94.2, 99.3] | 139 | 0.8% | 12.4% |
| ours α = −1.0 | 97.1% [94.2, 99.3] | 137 | 0.8% | 12.8% |
| actsteer r = −0.12 | 97.9% [95.1, 100.0] | 144 | 0.6% | 11.4% |
| actsteer r = −0.06 | 98.5% [96.3, 100.0] | 136 | 0.4% | 13.0% |
| magnitude-matched control | 98.5% [96.4, 100.0] | 137 | 0.4% | 12.8% |
| actsteer, fixed c = −2 … −16 | 100.0% | ~138 | 0.0% | 12.6% |
| actsteer r = −0.25 | 0.0% | **3** | 0.6% | 39.6% |

**The unsteered model never once reports what it knows.** Every one of 268
gradeable claims contradicts its own demonstrated belief, so the honest rate is
0.0% and the lie share is pinned at its ceiling. Nothing can be learned about an
intervention from a metric whose baseline is saturated, and the few points of
movement are not interpretable.

This was hidden in the first pass, where the lie rate over "gradeable" read 66.7%
because refusals and non-claims sit in that denominator. It is the same
denominator problem found on Llama, and here it disguised a degenerate baseline
rather than merely shifting a number.

The r = −0.25 arm is the collapse signature seen throughout: 3 surviving claims and
a 0.0% rate that means the model stopped answering.

## 4 Gemma-3-4b: the baseline wins, clearly

Six of eight arms completed before the allocation expired. Shares derived from the
saved rates; no bootstrap intervals, because the run was killed before it dumped
per-response records (now fixed, see §6).

| arm | lie share | honest | no_claim | ppl |
|---|---|---|---|---|
| **actsteer r = −0.06** (c = −473) | **55.7%** | **27.7%** | 37.5% | 0.93× |
| *unsteered* | *81.2%* | *10.1%* | 46.3% | *1.00×* |
| actsteer r = −0.12 | 80.4% | 17.5% | 10.4% | 0.96× |
| ours α = −0.5 | 89.4% | 5.5% | 47.6% | 1.06× |
| magnitude-matched control | 89.5% | 5.2% | 50.5% | 0.92× |
| actsteer r = −0.25 | 0.0% (collapsed) | 0.3% | 99.7% | 2.21× |

Activation steering at the gentlest relative coefficient cuts the lie share from
81.2% to 55.7% and nearly triples the honest rate, from 10.1% to 27.7%, at
perplexity slightly *below* baseline. That is a real shift toward disclosure, not
evasion.

Our mask moves the wrong way, 81.2% to 89.4%, with the honest rate halving, and it
is indistinguishable from its own magnitude-matched control at 89.5%. Two of the
eight arms (α = −0.75 and −1.0) did not run.

## 5 Five models, and no clean rule

Best configuration per family on harm-pressure knowledge report, claim-conditional
lie share:

| model | family | params | best mask | best baseline | unsteered |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | Qwen | 1.5B | **41.4%** | 55.3% | 65.8% |
| Qwen2.5-7B | Qwen | 7B | **15.3%** | 20.4% | 19.9% |
| Llama-3.1-8B | Llama | 8B | 31.7% | **18.4%** | 40.1% |
| Gemma-3-1b | Gemma | 1B | 93.7% | 97.9% | **100.0%** (degenerate) |
| Gemma-3-4b | Gemma | 4B | 89.4% | **55.7%** | 81.2% |

The mask wins on both Qwen models and loses on Llama and on Gemma-3-4b. Size
predicts nothing: it wins at 1.5B and 7B, loses at 4B and 8B. The earlier
conclusion that this is a family effect survives only in the weak sense that Qwen
is the outlier; with one Llama, one usable Gemma and a degenerate Gemma, there is
no evidence for a rule beyond "it depends on the model".

The one consistent pattern across all five is on the control side: the
magnitude-matched mask never helps, so wherever the mask does win, it is winning
on which weights it selects.

## 6 Three bugs the port exposed

Worth recording, because each would silently corrupt results rather than crash.

**Gemma must be loaded in bfloat16.** Loaded as float16, `gemma-3-4b-it` emits
nothing but pad tokens: the forward pass has already gone non-finite. The loader
now reads the dtype from the checkpoint config rather than hardcoding float16.
The 1b model happened to survive fp16, which is how this nearly went unnoticed.

**The decoder-layer path is not universal.** Gemma-3 4b and 12b ship as
`Gemma3ForConditionalGeneration`, whose text stack lives under `.language_model`,
so every `model.model.layers` reference failed. There is now one resolver used by
the hooks, the vector construction and the layer count. The repo's
`src/activation_steering.py` also reached for `model.model.embed_tokens` to find a
device; it now uses `get_input_embeddings()`, which every HF model defines.

**A vision-language mask would have included the vision tower.** The mask collects
every linear layer by name, which on 4b and 12b would have swept in the image
encoder and projector. Those are now excluded, verified as zero leaked modules on
both checkpoints.

Also fixed: per-response records are written after each arm instead of once at the
end, so an expiring allocation no longer discards every generation. That is why
the 4b arms above have no intervals.

## 7 Limits

- **12b was never run**, and 4b is missing two arms, so the Gemma sweep is
  incomplete. 12b is the one model that would say whether the 4b result holds at
  Llama's scale within the Gemma family.
- **Gemma-3-1b contributes nothing** beyond ruling itself out, since its baseline
  is saturated.
- **No bootstrap intervals for 4b** for the reason in §6; the 25-point gap between
  the winning baseline arm and unsteered is far larger than intervals of this size
  have been elsewhere, but it is a point estimate.
- **One layer per model**, chosen at quarter depth, and coefficients swept at that
  layer only.
- **The mask got three alphas on 1b and one on 4b**, so 4b under-samples our
  method; the two missing arms are exactly the stronger damping settings.
- **Single seed**, benchmark sampling temperatures.

## 8 Reproducing

```bash
python run_liars_bench_honesty.py --model-id google/gemma-3-1b-it \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layer 6 --as-rel-coeffs="-0.06,-0.12,-0.25,-0.5" \
  --arms unsteered,magmatch,asr-0.06,asr-0.12,asr-0.25,asr-0.5 \
  --mask-rhos="0.04" --mask-ratios="0.37" --mask-alphas="-0.5,-0.75,-1.0" \
  --tag g1b_grid

python run_liars_bench_honesty.py --model-id google/gemma-3-4b-it \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layer 8 --as-rel-coeffs="-0.06,-0.12,-0.25" \
  --arms unsteered,magmatch,asr-0.06,asr-0.12,asr-0.25 \
  --mask-rhos="0.04" --mask-ratios="0.37" --mask-alphas="-0.5,-0.75,-1.0" \
  --batch-size 24 --tag g4b_grid
```

Use `--as-rel-coeffs`, not `--as-coeffs`, on any new family: absolute coefficients
are not comparable across models.
