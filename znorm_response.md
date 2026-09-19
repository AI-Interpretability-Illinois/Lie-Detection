# Why `w` cannot be replaced by z-score normalization

**Re:** your suggestion to replace the `ratio` hyperparameter with z-score normalization across each layer's score distribution
**Model:** LLaMA-3.1-8B-Instruct · n\_truth = n\_lie = 25 · scores file: `outputs/scores_llama_8b_ntruth25.pt`

---

## Background

The LieNeuro DID score for weight element (i, j) is:

```
DID(i,j) = w · S_lie(i,j)  −  (1−w) · S_truth(i,j)
```

where `w = ratio / (1 + ratio)` and the importance scores are Wanda-style activations: `S[i,j] = |W[i,j]| × ‖X_j‖₂`.  The mask selects the top-ρ fraction of elements by DID **per layer**.

The idea is to replace the manual `w` with automatic z-score normalization, so that the two distributions become unit-variance before being subtracted, removing one hyperparameter without introducing new ones.

We tried this (along with mean-ratio normalization and per-type auto-ratio variants). All of them fail — for reasons that have to do with what `w = 0.29` is actually doing.

---

## Score statistics (LLaMA-3.1-8B-Instruct, balanced prompts)

With 25 lie and 25 truth prompts, the importance scores for the two conditions are nearly identical in scale across both masked layer types (attention and MLP).  The `lm_head` projection (vocab output, 525M params) is excluded from masking via `skip_modules` and is omitted here.

| Layer type | Params | | Mean | Std | Max |
|---|---|---|---|---|---|
| attn | 1.34 B | S_lie | 0.2789 | 0.5950 | 238.71 |
| | | S_truth | 0.2604 | 0.5596 | 220.43 |
| mlp | 5.64 B | S_lie | 0.2028 | 0.3191 | 7 201.30 |
| | | S_truth | 0.1896 | 0.3079 | 7 201.30 |

Global mean ratio S_lie / S_truth, per layer type:

| Layer type | Ratio | Auto w\* |
|---|---|---|
| attn | 1.0711 | 0.5172 |
| mlp | 1.0699 | 0.5169 |
| **global (attn + mlp)** | **1.0703** | **0.5170** |

The empirically optimal setting is **w = 0.2908** (ratio = 0.41).

---

## What we tried — four experiments

### Experiment 1 — per-layer z-score normalization

For each layer independently, divide each score tensor by its empirical standard deviation before computing DID:

```
S_lie_z[i,j]   = S_lie[i,j]   / std(S_lie_layer)
S_truth_z[i,j] = S_truth[i,j] / std(S_truth_layer)
DID_z = 0.5 · S_lie_z − 0.5 · S_truth_z          # w = 0.5 because both are now unit-variance
```

**Result:** Wikitext PPL = **15,737,379,674** (complete model collapse).

Dividing by σ changes the *ranking* of elements, not just their scale.  It promotes high-variance outliers — elements with a rare but very large activation — because those outliers dominate the denominator.  In float16, the weights corresponding to these outliers can already be close to the representable maximum (~65 504).  Amplifying them by `α = 4.0` (a factor of ×5 on the masked weights) overflows float16, producing `inf` values that cascade through every subsequent layer.

### Experiment 2 — per-layer mean normalization

Replace σ with the layer mean to avoid promoting outliers:

```
S_lie_m[i,j]   = S_lie[i,j]   / mean(S_lie_layer)
S_truth_m[i,j] = S_truth[i,j] / mean(S_truth_layer)
DID_m = 0.5 · S_lie_m − 0.5 · S_truth_m
```

After mean-normalization both tensors have mean ≈ 1, so an element is selected if `S_lie / mean_lie > S_truth / mean_truth`, i.e., if S_lie / S_truth > mean_lie / mean_truth ≈ 1.07.

**Result:** Wikitext PPL = **5,385,000,000** (still catastrophic).

Mean-normalization does not overflow float16 (rankings change less dramatically), but the effective selection threshold drops from 2.45 (at w = 0.29) to 1.07.  The mask now contains many generic weights that are active under both conditions, and the perplexity collapses for the same reason as using w ≈ 0.5 with the raw scores.

### Experiment 3 — global sum-ratio auto w\*

Instead of normalizing element-wise, derive w automatically from the aggregate ratio of the two score distributions:

```
auto_ratio = Σ_{all layers} Σ_{i,j} S_lie[i,j]
           / Σ_{all layers} Σ_{i,j} S_truth[i,j]
           = 1.0638

w* = auto_ratio / (1 + auto_ratio) = 0.5155

DID = w* · S_lie − (1 − w*) · S_truth          # same formula, only w changes
```

The DID formula and element ordering are identical to the manual case — only w changes from 0.2908 to 0.5155.

**Result:** Wikitext PPL = **NaN** (complete collapse).

With w = 0.52 the effective selection threshold is S_lie / S_truth > (1 − 0.52)/0.52 ≈ 0.92 — almost any element qualifies.  This includes high-magnitude weights that happen to be approximately equally active under both conditions, which when scaled ×5 produce inf activations.

### Experiment 4 — per-type sum-ratio auto w\* (current `--znorm` code)

Refine experiment 3 by computing a separate auto-ratio per layer type:

```python
# for t in {attn, mlp, other}:
auto_ratio_t = Σ_{layers of type t} S_lie.sum()
             / Σ_{layers of type t} S_truth.sum()

w*_t = auto_ratio_t / (1 + auto_ratio_t)
```

Observed values (lm\_head is excluded from masking, so only attn and mlp contribute):

| Type | auto\_ratio | w\* |
|---|---|---|
| attn | 1.0711 | 0.5172 |
| mlp | 1.0699 | 0.5169 |

The two types are essentially identical.  There is no variation across types that could be exploited.  Both per-type w\* values are indistinguishable from the global value (0.5170), and all are far from the empirically optimal 0.2908.

**Result:** not evaluated (no meaningful change expected from experiment 3).

---

## Three reasons normalization cannot recover this

### 1. There is no meaningful scale difference to correct

With balanced prompts, S_lie and S_truth track within 5–7% of each other across all layer types and globally.  Any automatic normalization — z-score or mean-ratio — yields w\* ≈ 0.52.  This is the *wrong direction* from the optimal 0.29, and in the opposite half of the unit interval.  Even if the derivation were perfect, it would move us further from the working value, not closer.

> Earlier experiments showing a q\_proj[0] ratio of 0.42 used **legacy unbalanced scores** (10 lie prompts vs. 25 truth prompts). With balanced scores that artifact disappears: q\_proj[0] has ratio = 1.035.

### 2. Z-scoring changes element ordering, not just scale

Dividing by σ shifts the ranking toward high-magnitude outliers — elements with rare but extreme activations.  Since the per-layer top-ρ selection is purely rank-based, z-scoring picks a *different set* of weights than mean-normalization or the raw DID.  In float16, those outliers overflow when amplified by α ≥ 4.0:

| Normalization attempted | Derived w\* | Wikitext PPL |
|---|---|---|
| z-score | ~0.52 | **15 737 379 674** (overflow) |
| mean-ratio (global) | 0.5155 | NaN |
| mean-ratio (per type) | 0.517 ± 0.002 | not evaluated |
| Manual sweep (ratio = 0.41) | **0.2908** | **~8.5** |

Z-score normalization is not a conservative change to scaling — it is a change to the selection criterion itself, with catastrophic consequences for model perplexity.

### 3. `w = 0.29` is a selectivity threshold, not a scale correction

At w = 0.5 (equal weighting), the mask selects any element where `S_lie > S_truth`.

At w = 0.29, the selection condition is:

```
0.29 · S_lie  >  0.71 · S_truth
           ⟹  S_lie / S_truth  >  2.45
```

So w = 0.29 selects only elements where lying activates the weight column **more than 2.45× more strongly** than truth-following.  The global mean ratio is 1.07 — most elements are eliminated.  The mask picks a very small subset of weights that are genuinely *lie-specialized*.

This threshold is a property of the **joint** distribution — which elements have an extreme relative difference between the two conditions — and cannot be read off from the marginal distributions of S_lie and S_truth alone.  There is no formula that recovers 2.45 from means and variances that are both close to 1.07.

---

## Conclusion

The `w` parameter is not compensating for a scale imbalance between S_lie and S_truth.  With balanced prompts those scales are already equal (ratio ≈ 1.07, consistent across both attention and MLP layers).  What `w = 0.29` encodes is a selectivity preference — a deliberate choice to use only the most lie-differentiated weights — and that preference needs to be calibrated empirically per model.

We will keep the manual sweep.
