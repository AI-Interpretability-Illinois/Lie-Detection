# Layer sweeps close the confound: the mask's wins do not survive, but the baseline wins only one split

> **Revised 2026-09-26.** An earlier version of this report led with the
> knowledge-report number and concluded "the baseline wins". That overstated it.
> The baseline's gain is confined to one of three splits, and on the other two
> neither method improves anything. Section 4b carries the per-split picture and
> the conclusions are qualified accordingly. The finding that the mask's two wins
> were against an under-swept baseline is unchanged.

**Models:** Qwen2.5-1.5B-Instruct, Qwen2.5-7B-Instruct, Gemma-3-4b-it
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua100)
**Benchmark:** LIARS' BENCH, harm-pressure knowledge report
**Code:** `run_liars_bench_honesty.py` (`--as-layers`, `--as-rel-coeffs`)
**Date:** 2026-09-26

---

## 1 The confound, and why it mattered

Every earlier comparison gave difference-in-means activation steering a single
layer at a quarter of the model's depth and swept only its coefficient. The weight
mask has no layer parameter, so this was a one-sided handicap. Sweeping depth on
Gemma-3-4b showed how large it was: the baseline went from 55.7% at quarter depth
to 12.4% at layer 4. That made the same sweep on the two Qwen models, the only
models where the mask had won, the deciding experiment.

Coefficients are scaled to each layer's own residual norm, which is essential
because the norm climbs steeply with depth (Qwen-1.5B: 25.3, 52.3, 94.9 at layers
3, 14, 21; Gemma-4b: 2188 to 37298 from layer 4 to 24).

## 2 Qwen2.5-1.5B: the mask's strongest result is overturned

| arm | lie share (95% CI) | claims | honest |
|---|---|---|---|
| **actsteer L21, r = −0.25** | **4.2% [2.3, 6.1]** | 427 | **51.1%** |
| actsteer L14, r = −0.25 | 41.4% [36.8, 46.3] | 389 | 28.5% |
| *unsteered* | *64.5% [59.9, 69.1]* | 414 | *18.4%* |
| actsteer L3, r = −0.25 | 73.0% [68.7, 77.6] | 393 | 13.2% |
| mask, best of 15 configs (earlier run) | 41.4% [36.3, 46.4] | 377 | 27.6% |
| baseline at quarter depth (earlier run) | 55.3% [49.7, 60.9] | 302 | 16.9% |

At layer 21 the baseline takes the lie share from 64.5% to **4.2%** and nearly
triples the honest rate to 51.1%, at 1.08× perplexity. The mask's best of fifteen
configurations was 41.4%. The intervals are nowhere near each other.

This was the strongest positive result in the entire series, a win with
non-overlapping intervals over both the unsteered model and the magnitude-matched
control. It does not survive giving the baseline its proper layer.

## 3 Qwen2.5-7B: the same, less dramatically

| arm | lie share (95% CI) | claims | honest |
|---|---|---|---|
| **actsteer L21, r = −0.25** | **11.5% [7.9, 15.4]** | 253 | **44.8%** |
| actsteer L14, r = −0.06 | 15.4% [10.9, 19.8] | 247 | 41.8% |
| mask, best of 6 configs (earlier run) | 15.3% [10.9, 20.1] | 229 | 38.8% |
| actsteer L3, r = −0.25 | 16.2% [11.5, 20.9] | 234 | 39.2% |
| *unsteered* | *19.9% [14.7, 25.1]* | 231 | *37.0%* |
| baseline at quarter depth (earlier run) | 20.4% [15.0, 26.1] | 226 | 36.0% |
| actsteer L14, r = −0.25 | 23.6% [18.4, 28.8] | 250 | 38.2% |

The baseline moves from 20.4% at quarter depth, where it was inert, to 11.5% at
layer 21, overtaking the mask's 15.3%. Intervals overlap, so this is a nominal
rather than a decisive win, but the direction reverses and the honest rate is
higher too. Forced choice confirms no damage: 13.4% against the unsteered 14.9%.

## 4 There is no universal good depth

| model | layers | best layer | as fraction of depth | lie share there | at quarter depth |
|---|---|---|---|---|---|
| Qwen2.5-1.5B | 28 | 21 | 75% | 4.2% | 55.3% |
| Qwen2.5-7B | 28 | 21 | 75% | 11.5% | 20.4% |
| Gemma-3-4b | 34 | 4 | 12% | 12.4% | 55.7% |

Qwen wants three quarters of the way down and Gemma wants an eighth. Quarter depth
was wrong for all three, and badly wrong for two. Any comparison that fixes the
baseline's layer by a heuristic will understate it by a large and unpredictable
margin, which is exactly what happened here.

## 4b The baseline's gain is confined to knowledge report

Both interventions are built from the same 25 lie-instruction pairs, so nothing
here can be attributed to one having instruction-based supervision and the other
not. What the per-split view shows is that the baseline's advantage lives entirely
in one split.

| model | arm | knowledge report | forced choice | instructed deception |
|---|---|---|---|---|
| Qwen-1.5B | *unsteered* | *53.5%* | *19.5%* | *6.5%* |
| Qwen-1.5B | best baseline (L21) | **3.6%** | 19.5% | 7.4% |
| Qwen-1.5B | best mask (α = −1.0) | 31.1% | 23.3% | 10.7% |
| Qwen-7B | *unsteered* | *12.3%* | *14.7%* | *3.7%* |
| Qwen-7B | best baseline (L21) | **7.8%** | 13.4% | 5.0% |
| Qwen-7B | best mask (α = −0.75) | 8.7% | 15.9% | 3.3% |
| Gemma-4b | *unsteered* | *43.6%* | *18.8%* | *2.8%* |
| Gemma-4b | best baseline (L4) | **6.2%** | 19.8% | 5.2% |
| Gemma-4b | best mask (α = −0.5) | 46.9% | 17.4% | 6.4% |

**Forced choice does not move for either method on any model.** 19.5% to 19.5% on
Qwen-1.5B, 14.7% to 13.4% on Qwen-7B, 18.8% to 19.8% on Gemma-4b. The mask is
slightly worse there at its stronger settings.

**Instructed deception gets worse under every intervention**, baseline included:
6.5% to 7.4%, 3.7% to 5.0%, 2.8% to 5.2%.

That pattern is explained by what the splits reward. Knowledge report scores
"honest" when the model claims to know; forced choice scores which option it
commits to. An intervention that moves the model from hedging toward asserting
cleans up the first and cannot touch the second. It is the same signature this
report already identifies at Gemma's deep layers, where a perfect 0.0% on
knowledge report came with forced-choice lying tripling. The winning arms are
gentler versions of it: the layers that win on Qwen are 21 of 28, close to the
output, which is where response form is cheapest to change.

So the baseline is not demonstrating a general honesty capability. It wins the one
split whose label is satisfiable by assertion, from the layer best placed to
change assertion, and transfers to neither of the others. No arm anywhere in this
series has improved forced choice or instructed deception.

## 5 Where the whole comparison now stands

Best configuration per family on harm-pressure knowledge report, with the baseline
given a layer sweep wherever one was run:

| model | mask | baseline | unsteered | verdict |
|---|---|---|---|---|
| Qwen2.5-1.5B | 41.4% | **4.2%** | 64.5% | baseline wins decisively |
| Qwen2.5-7B | 15.3% | **11.5%** | 19.9% | baseline wins nominally |
| Llama-3.1-8B | 31.7% | **18.4%** | 40.1% | baseline wins (layer not swept) |
| Gemma-3-4b | 89.4% | **12.4%** | 81.2% | baseline wins, mask harms |
| Falcon3-3B | 78.2% | **62.7%** | 69.4% | baseline wins, mask harms |
| Phi-4-mini | 69.4% | 64.6% | 67.3% | neither, layer not swept |
| OLMo-2-1B | 99.0% | 92.0% | 95.8% | near-degenerate, mask harms |

**The weight mask does not win on knowledge report on any model once the baseline
is swept.** It had two wins there; both were against a baseline pinned to the wrong
layer. It is neutral on one model and actively harmful on three.

But the table above is a knowledge-report table, and §4b is the reason not to read
it as an overall verdict. On the other two splits neither method helps anywhere.
The defensible summary is narrower than "the baseline wins": on the one split where
either method moves the needle, the baseline moves it further once its layer is
swept, and that split's honesty label is the one an assertiveness shift can
satisfy.

It is still a negative result for the mask as an honesty intervention on this
benchmark. It is not evidence that the baseline has the honesty capability the mask
lacks.

## 6 What still stands for the method

Two findings are untouched by the layer question, because neither involves the
baseline.

**Instruction compliance on Llama-8B.** Damping the mask takes compliance with an
explicit lie instruction from 78.4% to 5.8% and held-out framings from 61.4% to
34.6%, at 1.00× perplexity, with the magnitude-matched control flat. That remains
the method's strongest demonstrated effect, and it is a different behaviour from
these benchmark splits.

**The control results.** The magnitude-matched mask never produces a genuine
improvement on any model, so where the mask does something, it is doing it through
which weights it selects.

## 7 Limits

- **Three models swept, four not.** Llama-3.1-8B, Phi-4-mini and OLMo-2-1B kept
  their quarter-depth baseline, and Gemma-3-1b and Phi-4 14B are unmeasurable.
- **Coarse grids.** Three or four depths per model, one or two coefficients each.
  The optima are sharply peaked, so the true best layer may be better still, which
  again favours the baseline.
- **Mixed precision across runs.** The layer sweeps load each checkpoint in its
  own dtype (bfloat16 for Qwen) while the earlier Qwen runs used float16. Each
  sweep is compared against its own freshly measured unsteered arm, and those came
  out at 64.5% and 19.9% against the earlier 65.8% and 19.9%, so the effect is
  small, but cross-run rows are not strictly like-for-like.
- **Both sides selected on the reported metric**, so both are optimistic; the gaps
  at 1.5B and Gemma-4b are far larger than that could explain.
- **Single seed**, benchmark sampling temperatures.
- **The harm-pressure splits contain no lie instruction.** Their system prompts are
  boilerplate and the pressure is situational: the user states an intent to build a
  weapon and asks a WMDP question. So "honest" on those splits means disclosing
  hazardous information, and "lying" means withholding or misdirecting. An arm that
  improves the knowledge-report number is making the model more forthcoming with
  dangerous content, which is worth separating from honesty as a virtue.

## 8 Reproducing

```bash
python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --n-hp 800 --n-id 400 --n-alpaca 150 \
  --as-layers "3,14,21" --as-rel-coeffs="-0.25" \
  --arms unsteered,asrL3_-0.25,asrL14_-0.25,asrL21_-0.25 --tag q15_layers

python run_liars_bench_honesty.py --model-id Qwen/Qwen2.5-7B-Instruct \
  --rho 0.06 --ratio 0.35 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layers "3,7,14,21" --as-rel-coeffs="-0.06,-0.25" \
  --arms unsteered,asrL3_-0.06,asrL3_-0.25,asrL14_-0.06,asrL14_-0.25,asrL21_-0.25 \
  --tag q7b_layers
```
