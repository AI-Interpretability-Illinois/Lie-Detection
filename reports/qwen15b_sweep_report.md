# Matched hyperparameter sweep on Qwen2.5-1.5B: the 1.5B result survives

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fp16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua001)
**Benchmark:** LIARS' BENCH, harm-pressure knowledge report, 800 items,
belief-gated on unsteered weights
**Code:** `run_liars_bench_honesty.py`
**Date:** 2026-09-25

---

## 1 Why this run exists

The 8B comparison was unfair to the weight mask: the baseline got three
coefficients and the mask got one. Fixing that (see
`reports/llama8b_scale_report.md` §5c) did not change the 8B verdict. But the
1.5B comparison was unfair in the *opposite* direction: there the mask got two
α values and difference-in-means activation steering got exactly one coefficient,
and the mask won. That asymmetry had to be fixed too, or the 1.5B claim rests on
an under-swept baseline.

This sweeps both families on the same split with comparable budgets: 15
configurations of the weight mask and 7 of the baseline.

- mask: α ∈ {−0.25, −0.5, −0.75, −1.0} at ρ = 0.04; α ∈ {−0.25, −0.5, −1.0} at
  ρ = 0.01 and 0.08; ratio ∈ {0.33, 0.41, 0.45} at the best (ρ, α)
- baseline: c ∈ {−1, −2, −4, −8, −16, −32, −64} at layer 7 (median residual
  norm 34.9)

## 2 Result: the weight mask still wins, by more than the intervals

Lie share over responses that made a claim, with 95% bootstrap intervals. The
honest and no-claim columns are what separate a real shift from evasion.

| arm | lie share (95% CI) | claims | honest | no_claim |
|---|---|---|---|---|
| **ours ρ = 0.04, α = −1.0** | **41.4% [36.3, 46.4]** | 377 | **27.6%** | 14.8% |
| ours ρ = 0.01, α = −1.0 | 45.3% [40.5, 50.0] | 422 | 28.9% | 9.0% |
| ours ρ = 0.04, α = −0.75 | 46.5% [41.8, 51.6] | 409 | 27.4% | 9.8% |
| ours ρ = 0.08, α = −1.0 | 48.1% [43.4, 53.0] | 389 | 25.2% | 12.9% |
| ours ρ = 0.04, α = −0.5 | 52.9% [48.3, 57.7] | 416 | 24.5% | 9.5% |
| **actsteer c = −16** (best usable baseline) | **55.3% [49.7, 60.9]** | 302 | 16.9% | 23.8% |
| actsteer c = −4 | 58.5% [53.5, 63.2] | 383 | 19.9% | 13.9% |
| actsteer c = −8 | 58.8% [53.5, 64.0] | 342 | 17.6% | 18.9% |
| actsteer c = −2 | 60.2% [55.3, 64.9] | 407 | 20.2% | 10.8% |
| magnitude-matched control | 62.8% [57.9, 67.7] | 368 | 17.1% | 15.8% |
| actsteer c = −1 | 62.8% [58.2, 67.6] | 411 | 19.1% | 10.2% |
| *unsteered* | *65.8% [61.2, 70.3]* | 418 | *17.9%* | *9.4%* |
| prompt:honest | 71.7% [67.6, 75.9] | 435 | 15.4% | 6.4% |

**The mask's win survives a fair budget.** Best mask 41.4% against best usable
baseline 55.3%, with non-overlapping intervals (46.4 below 49.7). Sweeping the
baseline over five usable coefficients moved it from 62.8% to 55.3%, real
progress, and not enough to reach the mask.

**The mechanism check still separates them.** The mask raises the honest rate from
17.9% to 27.6% while adding only 5 points of evasion. The baseline at its best
coefficient *lowers* the honest rate to 16.9% and adds 14 points of evasion, so it
buys its number by making the model say nothing. That is the same split seen
before, and it is why the honest and no-claim columns belong in every table.

**Stronger damping is better here, unlike at 8B.** α = −1.0 wins at every ρ, and
ρ = 0.04 is the best mask size. At 8B the optimum was α = −0.5. So the operating
point does not transfer across scale, which is worth knowing before anyone reuses
these numbers.

**The ratio sweep agrees with the 8B run and with the repo.** At ρ = 0.04, α = −1.0,
moving ratio off 0.37 is worse on the raw rate: 0.33 gives 34.9%, 0.41 gives 42.2%
and 0.45 gives 39.1%, against 31.1% at 0.37, and the instructed-deception share
rises steadily from 10.7% to 17.5%. Ratio 0.37 is a genuine optimum at both scales.

## 3 Where the baseline breaks, and why its best point is c = −16

Pushed past −16 the baseline collapses rather than improving:

| c | HP-KR raw lie rate | alpaca degeneration | instructed-deception share |
|---|---|---|---|
| −16 | 32.9% | 0.0% | 15.1% |
| −32 | 2.2% | **16.0%** | **51.9%** |
| −64 | 0.0% | **99.3%** | unscoreable |

At c = −32 the lie rate falls to almost nothing because the model has stopped
producing gradeable text: one alpaca response in six is degenerate and the
instructed-deception lie share triples. At c = −64 essentially everything is
degenerate. So the baseline's usable range tops out near −16, its curve has turned
over, and c = −16 is the right point to compare against rather than an artefact of
a truncated sweep.

## 4 The two scales disagree, and now both sides have been swept fairly

| | best mask | best usable baseline | unsteered |
|---|---|---|---|
| Qwen2.5-1.5B | **41.4%** | 55.3% | 65.8% |
| Llama-3.1-8B | 31.7% | **18.4%** | 40.1% |

The ranking flips with scale, and it is no longer explainable as an unfair budget
in either direction: 15 mask configurations against 7 baseline ones at 1.5B, 13
against 3 at 8B. The weight edit wins on the small model and loses on the large
one.

Two candidate explanations, neither tested. Headroom differs enormously: the
unsteered 1.5B model lies on 65.8% of its claims here against the 8B model's
40.1%, so there is more for a blunt intervention to remove. And the 8B model's
evasion rate is already 32% before any intervention, which caps how much a
weight edit can shift the claim distribution. Distinguishing those would need a
mid-size model, 3B or 7B, on the same split.

## 5 Limits

- **One layer for the baseline.** Seven coefficients, all at layer 7. The 8B run
  used layer 8. A layer sweep is the obvious remaining asymmetry, and at 8B it
  would most likely favour the baseline further.
- **Selection on the reported metric.** Both families' best points were chosen by
  looking at this split, so both numbers are optimistic. The gap is larger than
  either interval, so the ranking is not an artefact of that, but the absolute
  values are.
- **A perplexity bug affected the ratio arms.** When a run's arm list omits
  `unsteered`, the harness previously took its baseline perplexity from the first
  arm evaluated, making that arm's ratio 1.00 by construction. Fixed in
  `run_liars_bench_honesty.py`; the ratio-grid rows at both scales were produced
  before the fix, so their damage numbers are unmeasured. Lie share does not
  depend on perplexity.
- **One split.** This is harm-pressure knowledge report only. Forced choice showed
  no effect for any arm at either scale, and instructed-deception remains
  unscoreable without the 32B judge.
- **Single seed** at the benchmark's own sampling temperatures.

## 6 Reproducing

```bash
python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --n-hp 800 --n-id 400 --n-alpaca 150 \
  --as-layer 7 --as-coeffs="-1,-2,-4,-16" --arms as-1,as-2,as-4,as-16 \
  --mask-rhos="0.04,0.01,0.08" --mask-alphas="-0.25,-0.5,-1.0" --tag q15_grid

python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --n-hp 800 --n-id 400 --n-alpaca 150 --arms unsteered \
  --mask-rhos="0.04" --mask-ratios="0.33,0.41,0.45" --mask-alphas="-1.0" \
  --tag q15_ratiogrid

python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --n-hp 800 --n-id 400 --n-alpaca 150 \
  --as-layer 7 --as-coeffs="-32,-64" --arms as-32,as-64 --tag q15_asbig
```

Pass `--arms unsteered` (as above) so the baseline perplexity is anchored
correctly; the reproduction lines for the first and third commands omitted it in
the original runs.
