# DID Score Correlation Across Lie-Truth-Ratio Configs

## Setup

All configs use **n-truth = 25** (DIRECT_TRUTH_INSTRUCT[:25]) and **lm_head included**.
The number of lie prompts varies via `--lie-truth-ratio`:

| Config | ratio | n_lie | n_truth | Note |
|--------|-------|-------|---------|------|
| A | 0.0 | 0 | 25 | DID = −S\_truth only |
| B | 0.4 | 10 | 25 | Legacy scale-mismatch split |
| C | 1.0 | 25 | 25 | Balanced (same prompt count) |
| D | 2.0 | 50 | 25 | 2× lie prompts |

The DID score for each weight is `S_lie[i,j] − S_truth[i,j]` (raw sums over prompts, not means).
The top-ρ weights by DID score form the steering mask.

Scores computed fresh on LLaMA-3.1-8B-Instruct, 4 configs in parallel on 4 GPUs.
Saved to `outputs/did_correlation/`.

---

## Spearman Rank Correlation of DID Scores

Layer-wise Spearman r between the flattened DID score vectors.
Computed on a random subsample of 50,000 weight elements per layer (225 layers total).
Mean and spread reported across layers.

| Config A | Config B | mean r | std r | min r | max r |
|----------|----------|-------:|------:|------:|------:|
| ratio=0.0 | ratio=0.4 | **+0.979** | 0.022 | 0.872 | 0.999 |
| ratio=0.0 | ratio=1.0 | −0.167 | 0.070 | −0.360 | +0.120 |
| ratio=0.0 | ratio=2.0 | **−0.917** | 0.055 | −0.996 | −0.689 |
| ratio=0.4 | ratio=1.0 | −0.033 | 0.103 | −0.337 | +0.289 |
| ratio=0.4 | ratio=2.0 | **−0.840** | 0.105 | −0.993 | −0.499 |
| ratio=1.0 | ratio=2.0 | +0.475 | 0.053 | +0.404 | +0.660 |

---

## Jaccard Similarity of Top-1% Masks

Layer-wise Jaccard similarity of the top-1% weight sets selected by DID score.
Top 1% is computed independently per layer; Jaccard = |A ∩ B| / |A ∪ B| per layer,
then averaged across all 225 layers.

| Config A | Config B | mean J | std J | min J | max J |
|----------|----------|-------:|------:|------:|------:|
| ratio=0.0 | ratio=0.4 | **0.697** | 0.347 | 0.000 | 0.965 |
| ratio=0.0 | ratio=1.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| ratio=0.0 | ratio=2.0 | 0.000 | 0.000 | 0.000 | 0.000 |
| ratio=0.4 | ratio=1.0 | 0.040 | 0.074 | 0.000 | 0.316 |
| ratio=0.4 | ratio=2.0 | 0.015 | 0.029 | 0.000 | 0.137 |
| ratio=1.0 | ratio=2.0 | **0.336** | 0.051 | 0.212 | 0.509 |

---

## Interpretation

**ratio=0.0 vs ratio=0.4 (legacy) are nearly the same mask.**
Spearman r = +0.979 and Jaccard = 0.70 confirm that 10 lie prompts at 0.4× the truth
scale contribute almost nothing to the ranking. The legacy pipeline is essentially
selecting weights by −S\_truth (lowest truth-importance), not by deception signal.

**ratio=0.0 and ratio=0.4 are strongly negatively correlated with ratio=2.0.**
Spearman r ≈ −0.84 to −0.92 means the two pipelines select nearly opposite sets of
weights. The legacy mask actively avoids the weights that the lie-heavy pipeline targets.

**ratio=0.4 and ratio=1.0 are uncorrelated (r = −0.033, J = 0.040).**
Even a balanced 25/25 split produces a qualitatively different mask from the legacy
10/25 split, despite using the same truth prompts.

**ratio=1.0 and ratio=2.0 share a moderate overlap (J = 0.34).**
Both have a genuine deception signal, but doubling the lie prompts shifts the ranking
noticeably (r = +0.475 rather than near-1.0).

**The Jaccard std is large for ratio=0.0 vs ratio=0.4 (0.347).**
Overlap varies substantially layer by layer — some layers have near-identical masks
(J ≈ 0.97) while others have no overlap at all (J = 0), suggesting that the
scale-mismatch effect is not uniform across the network.
