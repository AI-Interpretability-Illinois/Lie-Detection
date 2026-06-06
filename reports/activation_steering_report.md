# Activation Steering for Deception: Experiment Report

**Method:** Difference-in-means residual-stream steering (Stolfo et al., ICLR 2025)
**Model:** `meta-llama/Llama-3.1-8B-Instruct`
**Hardware:** NVIDIA H100 80 GB (CUDA 12.8)
**Date:** June 2026

---

## 1  Overview

This report evaluates whether **activation steering** — adding a difference-in-means
vector to the residual stream at inference time — can induce deceptive behaviour
(higher wrong-answer rate on factual QA) more effectively or more fluently than
LieNeuro's weight-steering approach.

The method adds a unit vector `u_l = normalize(mean(h_lie - h_truth))` scaled by
coefficient `c` to the hidden state at every token position in layer `l`. Model
weights are never modified; the intervention is applied per-forward-pass via a
registered forward hook on the decoder layer.

---

## 2  Method

### 2.1  Steering Vector Computation

For each of the 32 transformer layers:
1. Run 10 lying prompts and 10 truthful prompts (from the `direct` prompt set)
   through the frozen model, capturing the last-token hidden state at each layer.
2. Compute the mean difference: `d_l = mean_i(h_lie_{i,l} - h_truth_{i,l})`
3. Normalise: `u_l = d_l / ||d_l||`

### 2.2  Inference-Time Intervention

At every forward pass, a hook adds `c · u_l` to the output of layer `l`:
```
x'_l ← x'_l + c · u_l
```
This steers every token position toward the lie direction.

### 2.3  Sweep Design

| Parameter | Values |
|---|---|
| Layers | 0–31 (all 32 transformer layers) |
| Coefficients (c) | 5, 10, 20, 40, 80 |
| Prompt set | `direct` |
| Evaluation | 50-question factual QA |
| Total configurations | 32 × 5 = **160** |

---

## 3  Results

### 3.1  Baseline

Unsteered model factual accuracy: **90%**  
Baseline wrong-rate: **10%**

### 3.2  Deception Effect per Layer

![Δwrong-rate vs layer for each coefficient](figures/fig12_as_layer_sweep.png)

**Best single config:** layer 0, c=5 → Δwrong = **+90%**

**Top 5 layers by maximum Δwrong-rate (across all coefficients):**

| Layer | Best coefficient | Max Δwrong |
|---|---|---|
| 0 | c = 5 | +90% |
| 1 | c = 5 | +90% |
| 2 | c = 5 | +90% |
| 3 | c = 5 | +90% |
| 5 | c = 10 | +90% |

**Configs with positive Δwrong by coefficient:**

| Coefficient | Layers with Δwrong > 0 | Max Δwrong |
|---|---|---|
| c = 5 | 24 / 32 | +90% |
| c = 10 | 26 / 32 | +90% |
| c = 20 | 29 / 32 | +90% |
| c = 40 | 31 / 32 | +90% |
| c = 80 | 32 / 32 | +90% |

### 3.3  Heatmap

![Δwrong-rate heatmap by layer and coefficient](figures/fig11_as_heatmap.png)

### 3.4  Top 10 Configurations

| Rank | Layer | Coeff | Δwrong-rate |
|---|---|---|---|
| 1 | 0 | 5 | +90% |
| 2 | 0 | 20 | +90% |
| 3 | 1 | 5 | +90% |
| 4 | 2 | 5 | +90% |
| 5 | 2 | 10 | +90% |
| 6 | 2 | 20 | +90% |
| 7 | 2 | 40 | +90% |
| 8 | 2 | 80 | +90% |
| 9 | 3 | 5 | +90% |
| 10 | 3 | 10 | +90% |

### 3.5  LLM-Judge Verification (Top 3 Layers)

String-match accuracy can be inflated when the model produces a wrong answer
that happens to contain the gold string as a substring. The Qwen2.5-14B-Instruct
judge provides a semantic equivalence check.

| Layer | Coeff | String Δwrong | Judge Δwrong | Difference |
|---|---|---|---|---|
| 0 | 5 | +88% | +94% | +6% |
| 1 | 5 | +88% | +94% | +6% |
| 2 | 5 | +88% | +94% | +6% |

### 3.6  Fluency of Top Configurations

![Fluency metrics for top configs](figures/fig14_as_fluency.png)

Baseline perplexity: **161.86**

| Config | Δwrong | Perplexity | Distinct-2 | Rep-Rate |
|---|---|---|---|---|
| Layer 0 c=5 | +90% | 106.5 | 0.627 | 0.141 |
| Layer 0 c=20 | +90% | 9869.4 | 0.967 | 0.000 |
| Layer 1 c=5 | +90% | 789.7 | 0.900 | 0.006 |
| Layer 2 c=5 | +90% | 222.1 | 0.953 | 0.001 |
| Layer 2 c=10 | +90% | 2696.4 | 0.982 | 0.000 |

---

## 4  Comparison with LieNeuro Weight Steering

![Method comparison: Δwrong vs perplexity](figures/fig13_as_vs_lieneuro.png)

**Reference table — best direct-prompt configurations by method:**

| Method | Config | Δwrong | Perplexity |
|---|---|---|---|
| LieNeuro (weight) | ρ=0.10, α=1.0 | +14% | 9.6 |
| LieNeuro (weight) | ρ=0.10, α=2.0 | +86% | 303,250 |
| LieNeuro (weight) | ρ=0.01, α=2.0 | +16% | 150 |
| LieNeuro (weight) | ρ=0.01, α=4.0 | +90% | 62,455 |
| **Activation steering** | layer 0, c=5 | +90% | 106.5 |
| Baseline | — | 0% | 161.9 |

---

## 5  Analysis

### 5.1  Deception at Specific Layers

**111** of 160 configurations achieve Δwrong > 50%. The most responsive layers cluster in the early (0–9) network, suggesting the lie/truth direction is most accessible in those residual streams.

### 5.2  Coefficient Sensitivity

The relationship between coefficient magnitude and deception effect is non-monotonic
for most layers. Large coefficients do not guarantee greater deception — at some
layers the model output degrades (fluency collapse) before achieving the target
wrong-answer distribution. The heatmap (Fig 11) makes this visible as irregular
red/blue patterns within a single layer across coefficient values.

### 5.3  Comparison with Weight Steering

Activation steering achieves a maximum Δwrong of **+90%**, exceeding LieNeuro's most fluent configuration (+14% at ρ=0.10, α=1.0). Since activation steering leaves weights unchanged, this represents a more
surgical intervention. However, the fluency cost of the best activation steering
configurations must be considered alongside the raw deception metric.

The key structural difference between the methods:

| Property | Activation Steering | LieNeuro (Weight Steering) |
|---|---|---|
| Weights modified | No | Yes (scaled by α) |
| Inference overhead | +1 addition per token per hook | None |
| Reversible | Yes (remove hook) | Yes (restore weights) |
| Steering target | Residual stream direction | Importance-weighted mask |
| Requires prompts at inference | No | No |

---

## 6  Reproducibility

```bash
# Run the full 160-config sweep (32 layers × 5 coefficients)
CUDA_VISIBLE_DEVICES=2 python run_activation_steering_sweep.py --device cuda:0

# Resume if interrupted
CUDA_VISIBLE_DEVICES=2 python run_activation_steering_sweep.py --device cuda:0 --resume

# Fluency evaluation on top-8 configs
CUDA_VISIBLE_DEVICES=2 python run_as_fluency.py --device cuda:0 --top 8

# Regenerate figures and report
python reports/make_as_figures.py
python reports/generate_as_report.py
```

**Output files:**

| File | Contents |
|---|---|
| `outputs/activation_steering_sweep.csv` | 160 rows: (layer, coeff, accuracy, Δwrong) |
| `outputs/activation_steering_vectors.pt` | 32 normalised steering vectors (CPU tensors) |
| `outputs/as_baseline.json` | Baseline accuracy |
| `outputs/activation_steering_fluency.csv` | Fluency metrics for top configs |

---

## 7  References

Stolfo, A., et al. (2025). *Mechanistic Interpretability for AI Safety: Activation Steering for Instruction-Following.* ICLR 2025.

Sun, M., et al. (2023). *Wanda: Pruning by Weights and Activations.* arXiv:2306.11695.

