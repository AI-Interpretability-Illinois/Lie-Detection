# Why the ebdc3e8 Results Were an Artifact

## What Happened

The "good" results at commit `ebdc3e8` (PPL ≈ 20, high wrong rate at rho=0.04, alpha=4) were not produced by a working DID pipeline. The root cause was a single thing: **`S_lie` was loaded from the old `.bak` file instead of being recomputed**. Because the `.bak` file was created with only 10 prompts while `S_truth` was freshly computed with 25, the two sides were at incompatible absolute scales. That scale mismatch is a consequence of using the `.bak` file — it is not a separate issue.

---

## Stage 1 — Initial Commit (77d72f0, May 25)

The original `run_experiment.py` computed importance scores and saved them to disk:

- **Lie prompts**: `DIRECT_LIE[:10]` — 10 deception-instructed questions
- **Truth prompts**: `DIRECT_TRUTH[:10]` — 10 plain questions, **no instruction prefix**
- **Algorithm**: naive set-diff (`top-ρ(S_lie) \ top-ρ(S_truth)`), not DID
- **Layers**: 225, including `lm_head`
- **Output**: `outputs/mask_direct_rho0.1_alpha1.0.pt`

This file was later renamed to `mask_direct_rho0.1_alpha1.0.pt.bak` and never overwritten.

Scores for `model.layers.0.self_attn.q_proj` in this file:
- `S_lie` sum = 899,702 (10-prompt scale)
- `S_truth` sum = 660,605 (10-prompt scale)
- ratio = **1.36**

---

## Stage 2 — DID Sweep at ebdc3e8

`run_lieneuro_did_sweep.py` was introduced to run a proper DID sweep. It included a shortcut to avoid recomputing `S_lie` (which takes ~5 minutes):

```python
NAIVE_MASK = OUT_DIR / "mask_direct_rho0.1_alpha1.0.pt"

if NAIVE_MASK.exists():
    # Load S_lie from the existing file on disk
    naive_data = torch.load(NAIVE_MASK, ...)
    ln._S_lie  = naive_data["S_lie"]          # 10-prompt scale, from Stage 1
else:
    ln._S_lie = ln._compute_scores_for_prompts(DIRECT_LIE, ...)

# S_truth was always recomputed fresh
ln._S_truth = ln._compute_scores_for_prompts(
    DIRECT_TRUTH_INSTRUCT, ...)               # 25-prompt scale, freshly computed
```

The file existed on disk, so `S_lie` was loaded from the Stage 1 file (10 prompts). `S_truth` was always recomputed fresh with all 25 `DIRECT_TRUTH_INSTRUCT` prompts.

---

## The Scale Mismatch (a consequence of the .bak file)

The Wanda importance score is a **sum over prompts**, not a mean:

```
S[i,j] = Σ_prompts  |W[i,j]| × ‖X_j‖₂
```

The two sides of the DID at ebdc3e8 used incompatible scales:

| Side | Source | Prompts | Avg prompt length |
|------|--------|---------|-------------------|
| `S_lie` | loaded from `.bak` | 10 | ~68 chars |
| `S_truth` | freshly computed | 25 | ~55 chars (`"Answer truthfully: ..."`) |

This produced an `S_lie / S_truth` ratio of **~0.42** — meaning `S_lie` was less than half the scale of `S_truth` for every weight in every layer.

---

## Effect on the Mask

Because `S_lie − S_truth < 0` for essentially all weights, `top-ρ(S_lie − S_truth)` selected the "least negative" entries, which is statistically indistinguishable from a **random sample of weights**.

- Jaccard similarity between this mask and the current principled DID mask: mean = 0.28, min = 0.15
- The mask was not targeting deception-correlated weights — it was random noise

A random 4% of weights scaled by α=4 causes mild, non-targeted perturbation. No specific circuit is coherently disrupted, so PPL stays near baseline (~20).

---

## Why the Current Principled DID Destroys PPL

The current pipeline uses matched scales on both sides:

| Side | Source | Prompts |
|------|--------|---------|
| `S_lie` | freshly computed | 50 × `DIRECT_LIE` |
| `S_truth` | freshly computed | 50 × `DIRECT_TRUTH_INSTRUCT` |

With ratio ≈ 1.035, `S_lie − S_truth` is a genuine signal. The selected weights are specifically more active during deception instructions. Those weights are also load-bearing for general language modelling, so scaling them by α=4 coherently disrupts fluency — producing catastrophic PPL.

---

## Summary

| | ebdc3e8 | Current |
|--|---------|---------|
| `S_lie` source | `.bak` file, 10 prompts (Stage 1) | freshly computed, 50 prompts |
| `S_truth` source | freshly computed, 25 prompts | freshly computed, 50 prompts |
| Scale ratio | 0.42 | 1.035 |
| Mask type | near-random (scale mismatch artifact) | genuine deception signal |
| PPL | ~20 | catastrophic |

The ebdc3e8 results were not produced by a working method. The good PPL was an accident of the scale mismatch nullifying the DID signal.
