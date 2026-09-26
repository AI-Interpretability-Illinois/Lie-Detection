# The single-difference method against its predecessors and against prior steering work

**Model:** `Qwen/Qwen2.5-1.5B-Instruct` (fp16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua036)
**Config:** ρ = 0.04, ratio = 0.37, n_truth = 25, output head excluded
**Code:** `run_prior_work_baselines.py`
**Date:** 2026-09-25

---

## 1 What was compared

The repo's estimator differs from the published weight-importance algorithms it
adapts in exactly two ways, and each can be switched off independently:

| | published form | this repo |
|---|---|---|
| **control set** | a *neutral* prompt set (bare questions) | an *instruction-matched* truth set ("Answer truthfully: Q") |
| **selection rule** | *set difference* of the two top-ρ sets, `C \ N` | a ratio-weighted *continuous* score, `v·n_lie − (1−v)·n_truth` |

Math Neurosurgery (Christ et al. 2024) and CreativityNeuro (Schapiro et al. 2025)
both use the left column. Crossing the two axes gives four variants, of which the
published algorithm and the repo's method are opposite corners, plus a no-contrast
control:

- `magnitude` — top-ρ by `|W|` alone, no contrast
- `set_diff_neutral` — the published form
- `set_diff_instruct` — set difference with the instruction-matched control
- `score_diff_neutral` — continuous score with the bare-question control
- `score_diff_instruct` — **the repo's method**

Activation-space prior work, given the same supervision (the same 25 lie and 25
truth prompts) and the same evaluation:

- `actsteer_single` — difference-in-means at one layer (Stolfo et al. 2025)
- `actsteer_caa` — the same vectors added at five consecutive layers with total
  displacement matched to the single-layer arm (contrastive activation addition,
  Rimsky et al. 2024)
- `probe_dir` — a logistic probe separating lie from truth activations, steered
  along its weight vector (the probe-direction family behind inference-time
  intervention and representation engineering)

Plus prompting, which changes nothing and costs nothing, and a magnitude-matched
random mask as the specificity control. Every arm is run in both directions and
scored on the pressure metric from `honesty_eval.py`, with falsehood rates on the
seen lie framing and on three held-out framings reported separately.

## 2 Honesty direction

Working configurations only: perplexity within 1.15× and neutral accuracy within
5 points of the unsteered 91.0%.

| arm | knob | direct | held-out | neutral | ppl |
|---|---|---|---|---|---|
| **score_diff_instruct (this repo)** | −0.75 | 2.2% | **4.3%** | 91.0% | 1.02× |
| actsteer_single | −8 | 2.2% | 7.1% | 92.0% | 1.02× |
| score_diff_neutral | −0.75 | 9.2% | 8.0% | 91.0% | 1.08× |
| *unsteered* | — | *5.4%* | *8.3%* | *91.0%* | *1.00×* |
| probe_dir | −8 | 2.2% | 8.3% | 91.5% | 1.00× |
| prompt:honest | — | 7.1% | 9.2% | 91.5% | 1.00× |
| actsteer_caa | −8 | 4.3% | 9.6% | 91.5% | 1.00× |
| magnitude-matched control | −1.0 | 7.1% | 11.2% | 90.0% | 1.03× |
| `magnitude`, `set_diff_neutral`, `set_diff_instruct` | — | no working configuration at any knob | | | |

The repo's method is the only arm that moves held-out lying substantially, 8.3%
down to 4.3%. Every other method lands between 7.1% and 11.2%, which is at or
around the unsteered rate. The nearest competitor is single-layer
difference-in-means activation steering at 7.1%.

## 3 Deception direction

All configurations with perplexity within 1.15×, neutral accuracy shown because
it is where the methods differ most.

| arm | knob | direct | held-out | neutral | ppl |
|---|---|---|---|---|---|
| **score_diff_instruct (this repo)** | +2 | 10.3% | **17.4%** | 85.5% | 1.13× |
| prompt:deceptive | — | 12.5% | 12.9% | 85.5% | 1.00× |
| actsteer_caa | +8 | 13.0% | 11.8% | 90.0% | 1.02× |
| probe_dir | +8 | 7.1% | 11.8% | 89.5% | 1.06× |
| actsteer_single | +8 | **17.9%** | 11.1% | 89.5% | 1.04× |
| *unsteered* | — | *5.4%* | *8.3%* | *91.0%* | *1.00×* |

Here the ranking depends on which framing you score. On the framing the
interventions were built from, activation steering is strongest (17.9% against
the weight edit's 10.3%). On the three held-out framings the weight edit is
strongest (17.4% against 11.1%). The weight edit pays for that with neutral
accuracy, 85.5% against roughly 90% for the activation arms: it makes the model
wrong when nobody asked it to lie, which the activation interventions largely do
not.

Note also that a deceptive system prompt, which costs nothing, reaches 12.9% and
beats every activation configuration tested on held-out framings.

## 4 What actually differentiates this method

Both of the repo's deviations from the published algorithm are load-bearing, and
the 2×2 separates them cleanly.

**The instruction-matched control is what creates the honesty effect.** Holding
the selection rule fixed and swapping only the control set, held-out lying goes
from 4.3% (instruction-matched) to 8.0% (bare questions), against an unsteered
8.3%. The bare-question version does essentially nothing. This is the change the
repo calls DID, and on this evidence it is the single most important design
decision in the method, even though it is one difference rather than two.

**The continuous ratio-weighted score is what keeps the edit usable.** Holding
the control set fixed and swapping only the selection rule, the published set
difference has no working configuration in either direction:

| variant | knob | neutral acc | ppl | coverage |
|---|---|---|---|---|
| set_diff_neutral | −0.75 | 82.0% | 1.18× | 1.00% |
| set_diff_neutral | −1.0 | 70.5% | 1.40× | 1.00% |
| set_diff_instruct | −0.75 | 82.5% | 1.19× | 0.94% |
| set_diff_instruct | +2 | 1.0% | 95× | 0.94% |
| set_diff_instruct | +3 | 1.0% | 9×10⁴ | 0.94% |
| magnitude | −0.75 | 0.0% | 3×10⁴ | 4.00% |

The set-difference variants touch a *quarter* as many weights as the repo's
method (about 1% coverage against 4%) and yet damage the model far more. That is
the signature of selecting load-bearing weights, and it matches the mask
diagnostic in `reports/real_did_report.md`: the ratio weighting makes the
selection land on weights whose median magnitude is 0.04× the global median,
while contrast rules without that asymmetry land on weights 50 to 70 times
larger. The `magnitude` row closes the argument from the other side: ranking by
`|W|` alone selects the largest weights and destroys the model immediately, so the
mechanism is specifically *small* weights picked by low truth-importance, not
magnitude alone. And the magnitude-matched control shows it is not merely "shrink
any small weights" either, since it does nothing.

These set-difference blow-ups also reproduce the repo's own earlier numbers,
where `set_diff` at α ≥ 2 gave WikiText perplexity from 1594 to 251340 on
Llama-3.1-8B. That was previously read as a property of aggressive steering. It
is better read as a property of the selection rule.

**Against the activation-steering family, the difference is generalisation versus
surgical precision.** The weight edit is always on, costs nothing per token, and
transfers across lie framings, but it leaks into neutral prompts. Activation
steering is reversible, leaves neutral accuracy near baseline, and is stronger on
the framing its vector was built from, but it transfers poorly: negated, it moves
held-out lying by about one point.

Two smaller results worth recording. Adding the vector at five layers with
matched total displacement is no better than one layer in either direction, so the
multi-layer variant buys nothing here. And a trained logistic probe, which sits at
cosine +0.617 from the difference-in-means direction, performs no better than it
(8.3% against 7.1% for honesty), consistent with the existing finding that
difference-in-means is competitive with trained probes for steering. The probe
reached 100% training accuracy on 50 examples in 1536 dimensions, so it is
certainly overfitted; inference-time intervention uses far more data and per-head
selection, which this does not reproduce.

## 5 Limits

- **One model at 1.5B.** ρ = 0.04 and ratio = 0.37 come from Llama-3.1-8B. The
  set-difference variants might have a workable operating point at a ρ this sweep
  did not try, though their damage at 1% coverage argues against it.
- **One coefficient for the activation arms.** Magnitude 8 at layer 7, chosen
  from the earlier sweep. The activation family was swept more thoroughly in
  `reports/honesty_and_baselines_report.md`; here it is a fixed reference point.
- **No fine-tuning baseline.** LoRA or DPO would be the natural strong baseline
  for both directions and is absent.
- **Prompting was tested with one phrasing each.** A prompt baseline can usually
  be improved by iterating the wording, which was not attempted.
- **The usability bar is a threshold.** The repo's method at +2 sits at 85.5%
  neutral accuracy against a 86.0% cutoff, so it falls just outside the "working"
  set in the deception direction by half a point. It is reported in full rather
  than dropped.
- **Held-out framings are three hand-written templates**, not a benchmark.

## 6 Reproducing

```bash
python run_prior_work_baselines.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --rho 0.04 --ratio 0.37

# the activation arms alone (they need the dtype-correct hooks)
python run_prior_work_baselines.py --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --only activation --tag act
```
