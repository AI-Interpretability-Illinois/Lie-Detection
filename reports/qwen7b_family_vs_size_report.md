# Qwen2.5-7B: the disagreement tracks model family, not size

**Model:** `Qwen/Qwen2.5-7B-Instruct` (fp16, 28 layers)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua001)
**Config:** ρ ∈ {0.04, 0.06}, ratio = 0.35 (the repo's committed Qwen values),
output head excluded
**Benchmark:** LIARS' BENCH, 500 items per harm-pressure split, belief-gated on
unsteered weights (351 gradeable on forced choice, 372–381 on knowledge report)
**Code:** `run_liars_bench_honesty.py`
**Date:** 2026-09-25

---

## 1 The question

Two earlier runs disagreed about which intervention family wins:

| model | winner on harm-pressure knowledge report |
|---|---|
| Qwen2.5-1.5B | the weight mask |
| Llama-3.1-8B | difference-in-means activation steering |

Family and size both differ between those two, so the disagreement had two
possible causes. Qwen2.5-7B-Instruct is the cell that separates them: Qwen family,
same size class as Llama-8B. If it behaves like Llama-8B the cause is size; if it
behaves like Qwen-1.5B the cause is family.

Both families were swept: six mask configurations (ρ ∈ {0.04, 0.06} × α ∈ {−0.5,
−0.75, −1.0} at ratio 0.35) and four baseline coefficients (c ∈ {−2, −4, −8, −16}
at layer 7, median residual norm 32.4). Baseline WikiText perplexity 7.799.

## 2 Result: it behaves like Qwen-1.5B

Harm-pressure knowledge report, lie share over responses that made a claim. The
honest and evasion columns are what separate a real shift from silence.

| arm | lie share (95% CI) | claims | honest | evasion | ppl |
|---|---|---|---|---|---|
| **ours ρ = 0.04, α = −0.5** | **15.3% [10.9, 20.1]** | 229 | **38.8%** | 22.2% | 1.01× |
| ours ρ = 0.06, α = −0.5 | 15.3% [10.9, 20.1] | 229 | 38.8% | 23.0% | 1.01× |
| ours ρ = 0.06, α = −0.75 | 16.0% [10.7, 21.4] | 206 | 34.6% | 27.6% | 1.02× |
| ours ρ = 0.04, α = −0.75 | 17.0% [12.3, 22.2] | 212 | 35.2% | 26.4% | 1.02× |
| ours ρ = 0.04, α = −1.0 | 17.6% [12.4, 23.3] | 193 | 31.8% | 31.2% | 1.05× |
| ours ρ = 0.06, α = −1.0 | 19.0% [13.6, 25.0] | 184 | 29.8% | 32.8% | 1.05× |
| *unsteered* | *19.9% [15.3, 25.0]* | 236 | *37.8%* | *21.8%* | *1.00×* |
| actsteer c = −8 | 20.4% [15.0, 26.1] | 226 | 36.0% | 22.4% | 1.03× |
| actsteer c = −2 | 20.5% [15.3, 25.8] | 229 | 36.4% | 22.0% | 1.00× |
| actsteer c = −4 | 21.1% [16.1, 26.5] | 223 | 35.2% | 23.0% | 1.01× |
| actsteer c = −16 | 26.9% [21.2, 33.7] | 193 | 28.2% | 24.6% | 1.12× |
| magnitude-matched control | 28.6% [22.8, 34.4] | 241 | 34.4% | 20.8% | 1.03× |

**Qwen-7B patterns with Qwen-1.5B, not with Llama-8B.** The mask lowers the lie
share and the baseline does not move it at any coefficient. That is the Qwen-1.5B
ordering, on a model the same size as the Llama where the ordering was reversed.
So **size is not what separates the two earlier runs.**

**The mask's best point does not buy its number through evasion.** At α = −0.5 the
honest rate is 38.8% against the unsteered 37.8% and evasion is 22.2% against
21.8%, both essentially unchanged, so the lie share fell because lies became
honest answers rather than silence. Damping harder inverts that: at α = −1.0 the
honest rate drops to about 30% and evasion rises past 31%, which is the
over-damping signature seen at both other scales.

**Activation steering is inert here rather than harmful.** Coefficients −2 to −8
leave the lie share within a point of unsteered, and −16 makes it clearly worse
(26.9%, honest rate down to 28.2%) at 1.12× perplexity. On Llama-8B the same
method at c = −1 more than halved the lie share.

**The random control is worse than doing nothing** (28.6% against 19.9%), which it
was not on the other two models. Worth noting because it means the mask's
advantage over its control here, intervals [10.9, 20.1] against [22.8, 34.4], is
larger than its advantage over the unsteered model.

## 3 Three models together

Best configuration per family, lie share on harm-pressure knowledge report:

| model | family | params | best mask | best baseline | unsteered |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | Qwen | 1.5B | **41.4%** | 55.3% | 65.8% |
| Qwen2.5-7B | Qwen | 7B | **15.3%** | 20.4% | 19.9% |
| Llama-3.1-8B | Llama | 8B | 31.7% | **18.4%** | 40.1% |

The mask wins on both Qwen models and loses only on Llama. The scale story in
`reports/llama8b_scale_report.md` §6, that the weight edit wins small and loses
large, is **wrong**: it was a family difference read as a size difference because
only one model of each existed.

## 4 The other two splits

**Forced choice.** Nothing helps, as on both other models. Every gentle arm sits
inside every other's interval: unsteered 16.3% [12.6, 20.3], best mask 14.9%
[11.2, 18.7], baseline at c = −8 15.8% [12.0, 19.8]. Damping to α = −1.0 makes it
worse (19.4%), and the baseline at c = −16 likewise (18.3%). Refusal is under 1%
throughout, so the denominator problem that affected the Llama run does not arise
here.

**Instructed deception.** The unsteered share is 2.8% and every mask configuration
raises it, from 3.2% at α = −0.75 to 4.5% at α = −1.0. Activation steering leaves
it flat or slightly lower. So the pattern from both other models repeats: damping
the mask slightly increases unprompted lying on the split closest to its own
supervision. This remains the split that needs the 32B judge to be scored
properly.

## 5 Limits

- **The layer confound is open, and it is the main one.** The baseline used layer 7
  on both Qwen models and layer 8 on Llama. "Activation steering fails on Qwen"
  could therefore be "layer 7 is the wrong layer for Qwen". Coefficients were
  swept at that one layer, not layers at a good coefficient. Two or three layers at
  c = −8 would close it and is roughly seven minutes of GPU time. Until that is
  run, the family conclusion should be read as "with the vector taken at quarter
  depth", not as a property of the family outright.
- **The mask's win over unsteered is directional, not significant** on this model:
  15.3% [10.9, 20.1] against 19.9% [15.3, 25.0]. What clears the intervals is the
  comparison against the magnitude-matched control. The 1.5B win was significant;
  this one is not.
- **Both families' best points were selected on this split**, so both numbers are
  optimistic.
- **Six mask configurations against four baseline ones**, and no ratio sweep on
  this model; ratio was fixed at the repo's committed 0.35.
- **One split carries the conclusion.** Forced choice is null for everything and
  instructed deception is unscoreable without the judge.
- **Single seed**, at the benchmark's own sampling temperatures.

## 6 Reproducing

```bash
python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-7B-Instruct \
  --rho 0.06 --ratio 0.35 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layer 7 --as-coeffs="-2,-4,-8,-16" \
  --arms unsteered,magmatch,as-2,as-4,as-8,as-16 \
  --mask-rhos="0.06,0.04" --mask-ratios="0.35" --mask-alphas="-0.5,-0.75,-1.0" \
  --batch-size 32 --score-batch 4 --tag q7b_grid
```
