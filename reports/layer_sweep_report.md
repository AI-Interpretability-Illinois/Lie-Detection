# Closing the layer confound on Gemma-3-4b: quarter depth was the wrong layer

**Model:** `google/gemma-3-4b-it` (34 layers, bfloat16)
**Hardware:** 1× NVIDIA A100-SXM4-40GB (NCSA Delta, gpua100)
**Benchmark:** LIARS' BENCH, 500 items per harm-pressure split
**Code:** `run_liars_bench_honesty.py` (`--as-layers`)
**Date:** 2026-09-26

---

## 1 The confound

Every model so far gave the activation-steering baseline a single layer, chosen at
a quarter of the model's depth, and swept only its coefficient. So "the baseline
wins here" and "the baseline fails there" could both have been artefacts of that
choice. This sweeps five depths on Gemma-3-4b, each with its coefficient scaled to
that layer's own residual norm, which matters because the norm rises steeply with
depth on this model:

| layer | 4 | 8 | 12 | 17 | 24 |
|---|---|---|---|---|---|
| median residual norm | 2188 | 7885 | 22304 | 33390 | 37298 |

It also completes the two mask arms the earlier Gemma-4b run never reached.

## 2 Quarter depth was a poor choice, and the baseline is far stronger than reported

Harm-pressure knowledge report, lie share over responses that made a claim. All
baseline arms at r = −0.06 of the local norm.

| arm | lie share (95% CI) | claims | honest | ppl |
|---|---|---|---|---|
| actsteer **L12** | 0.0% [0.0, 0.0] | 133 | 26.6% | 1.68× |
| actsteer **L17** | 0.0% [0.0, 0.0] | 139 | 27.8% | 1.82× |
| **actsteer L4** | **12.4% [7.2, 18.3]** | 153 | **26.8%** | **1.03×** |
| actsteer L8 (the layer used before) | 55.7% [50.5, 60.4] | 384 | 17.0% | 0.93× |
| actsteer L24 | 76.1% [69.0, 82.6] | 155 | 7.4% | 1.29× |
| *unsteered* | *81.2% [77.0, 85.5]* | 330 | *6.2%* | *1.00×* |
| ours α = −0.5 | 89.4% [84.5, 93.8] | 161 | 3.4% | 1.06× |
| magnitude-matched control | 89.5% [84.2, 94.1] | 152 | 3.2% | 0.92× |
| ours α = −0.75 | 98.7% [97.5, 99.7] | 318 | 0.4% | 1.05× |
| ours α = −1.0 | 99.4% [98.5, 100.0] | 326 | 0.2% | 1.11× |

**Layer 4 is the real operating point.** It takes the lie share from 81.2% to
12.4% and quadruples the honest rate from 6.2% to 26.8%, at 1.03× perplexity.
The layer I had been using, quarter depth, gave 55.7%. So the baseline's advantage
on this model was understated by a wide margin, and the quarter-depth heuristic was
simply wrong here.

**The mask arms complete the earlier grid and confirm harm.** α = −0.75 and −1.0
push the lie share to 98.7% and 99.4%, leaving 0.4% and 0.2% honest answers. Added
to Falcon3-3B, that is a second family where damping the mask makes a model
markedly less honest, monotonically in strength, at perplexity near baseline.

## 3 The deepest two layers are an artefact, and the second split catches it

L12 and L17 report a perfect 0.0% lie share. They should not be read as the best
result, for two reasons.

**Their perplexity is 1.68× and 1.82×**, against 1.03× at layer 4, so they are
doing real damage.

**On knowledge report, "honest" means claiming to know.** An intervention that
pushes the model to assert knowledge indiscriminately scores as perfectly honest on
this split without being more honest at all. Forced choice is the check, and it
fails there:

| arm | forced-choice lie share | honest |
|---|---|---|
| *unsteered* | *18.8% [15.8, 22.0]* | 48.4% |
| actsteer L4 | 19.8% [15.4, 24.5] | 47.8% |
| actsteer L17 | 20.5% [16.1, 24.8] | 47.4% |
| actsteer **L12** | **60.5% [54.2, 66.8]** | 18.8% |
| ours α = −1.0 | 23.5% [20.1, 27.0] | 45.6% |

Layer 12 more than triples forced-choice lying while reporting zero lies on
knowledge report. That is the signature of inflated knowledge claims, not honesty.
Layer 4 leaves forced choice untouched, which is why it is the arm to believe.

"Untouched" is also a limit on what layer 4 achieves: 19.8% against the unsteered
18.8% means it does not improve forced choice either, and instructed deception goes
from 2.8% to 5.2%. Its gain is confined to knowledge report. See
`reports/layer_sweep_final_report.md` §4b, which sets out the per-split picture for
all three swept models and qualifies the "baseline wins" framing accordingly.

## 4 What this does to the earlier conclusions

**It strengthens the two "baseline wins" results and weakens the two "mask wins"
results.** On Gemma-3-4b the baseline's margin grows from 25 points to 69. But the
same confound is still open on the models where the mask won, because their
baseline was also given only quarter depth:

| model | baseline layer used | swept? | verdict as reported |
|---|---|---|---|
| Qwen2.5-1.5B | 7 of 28 | coefficient only | mask wins |
| Qwen2.5-7B | 7 of 28 | coefficient only | mask wins |
| Llama-3.1-8B | 8 of 32 | coefficient only | baseline wins |
| Gemma-3-4b | **5 depths** | **yes** | baseline wins, by much more |

Gemma's best layer sits at 12% of depth, not 25%, and its optimum is sharply
peaked: 12.4% at layer 4 against 55.7% at layer 8 and 76.1% at layer 24. If Qwen
behaves similarly then the Qwen baseline was tested at the wrong depth, and the
"mask wins on Qwen" finding could go the same way this one did. A Qwen-7B layer
sweep is running and is the deciding experiment.

Until that lands, the honest position is that the mask's two wins rest on an
under-swept baseline, while both of its losses and both harm results do not depend
on the layer choice at all, since the mask has no layer parameter.

## 5 Limits

- **One coefficient per layer** (r = −0.06 of the local norm). The peak at layer 4
  might be sharper or flatter in coefficient than this shows.
- **Pooled counts differ by arm**, because arms appearing in both the earlier grid
  and this sweep pool their responses (unsteered 330 claims, L8 384, the new layers
  133–155). Intervals reflect those counts.
- **Five depths, not a full sweep**; nothing between layers 4 and 8 was tried and
  the optimum could be shallower still.
- **Single seed**, benchmark sampling temperatures.

## 6 Reproducing

```bash
python run_liars_bench_honesty.py --model-id google/gemma-3-4b-it \
  --rho 0.04 --ratio 0.37 --n-hp 500 --n-id 250 --n-alpaca 100 \
  --as-layers "4,8,12,17,24" --as-rel-coeffs="-0.06" \
  --arms unsteered,asrL4_-0.06,asrL8_-0.06,asrL12_-0.06,asrL17_-0.06,asrL24_-0.06 \
  --mask-rhos="0.04" --mask-ratios="0.37" --mask-alphas="-0.75,-1.0" \
  --batch-size 24 --tag g4b_layers
```

`--as-layers` scales each layer's coefficient by that layer's own residual norm,
which is required on Gemma where the norm rises 17-fold from layer 4 to layer 24.
