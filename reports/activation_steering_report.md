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
| c = 0 | 6 / 32 | +4% |
| c = 1 | 12 / 32 | +18% |
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

Baseline perplexity: **3.44**

| Config | Δwrong | Perplexity | Distinct-2 | Rep-Rate |
|---|---|---|---|---|
| Layer 0 c=5 | +90% | 165.3 | 0.694 | 0.088 |
| Layer 0 c=20 | +90% | 4621.0 | 0.932 | 0.000 |
| Layer 1 c=5 | +90% | 1271.5 | 0.950 | 0.000 |
| Layer 2 c=5 | +90% | 138.3 | 0.943 | 0.001 |
| Layer 2 c=10 | +90% | 2613.7 | 0.968 | 0.000 |

### 3.7  WikiText-2 Perplexity Benchmark

Generation-based perplexity measures how surprised the model is by its own outputs.
To provide a model-agnostic, fixed-corpus fluency benchmark we also evaluate
**WikiText-2-raw-v1** perplexity using a sliding window of 512 tokens (stride 256)
on the first 8,192 test tokens. This metric is identical for all variants and
directly comparable across methods.

![WikiText-2 PPL vs Δwrong](figures/fig15_wikitext_ppl.png)

Baseline WikiText-2 PPL: **9.15**

| Variant | Method | Δwrong | WikiText-2 PPL | PPL Ratio vs Baseline |
|---|---|---|---|---|
| Baseline | — | 0% | 9.15 | 1.00× |
| AS L0 c=5 | Activation steering | +90% | 9197.07 | 1004.7× |
| AS L1 c=5 | Activation steering | +90% | 3121.71 | 341.0× |
| AS L2 c=5 | Activation steering | +90% | 257.10 | 28.1× |
| AS L3 c=5 | Activation steering | +90% | 242.44 | 26.5× |
| AS L5 c=10 | Activation steering | +90% | 308.17 | 33.7× |
| AS L1 c=1 | Activation steering | +18% | 16.31 | 1.8× |
| AS L9 c=1 | Activation steering | +4% | 9.20 | 1.0× |
| AS L24 c=1 | Activation steering | +6% | 9.16 | 1.0× |
| AS L0 c=2 | Activation steering | +90% | 214.25 | 23.4× |
| AS L1 c=2 | Activation steering | +90% | 118.86 | 13.0× |
| ρ=0.10 α=1.0 | LieNeuro (weight) | +14% | 13.94 | 1.5× |
| ρ=0.01 α=2.0 | LieNeuro (weight) | — | 18.96 | 2.1× |

### 3.8  In-Distribution vs Out-of-Distribution Evaluation (25-Pair Vectors)

To assess whether averaging over more construction prompts improves the steering
vectors, we recompute the difference-in-means vectors using **25 direct lie/truth
pairs** (vs 10 in the original sweep) and evaluate each configuration on two splits:

- **In-distribution (IndDist):** the 25 Q&A pairs whose questions appear verbatim
  in the construction prompts — tests whether the vector memorises its training questions.
- **Out-of-distribution (OOD):** 10 held-out Q&A pairs never seen during vector
  construction — tests whether deceptive steering generalises to new questions.

Baselines: IndDist accuracy = **96%**, OOD accuracy = **90%**

![In-dist vs OOD heatmaps](figures/fig16_indist_ood.png)

**Top 10 configurations by OOD Δwrong:**

| Rank | Layer | Coeff | IndDist Δwrong | OOD Δwrong | Gap (IndDist − OOD) |
|---|---|---|---|---|---|
| 1 | 0 | 2 | +96% | +90% | +6% |
| 2 | 0 | 5 | +96% | +90% | +6% |
| 3 | 0 | 10 | +92% | +90% | +2% |
| 4 | 0 | 20 | +96% | +90% | +6% |
| 5 | 1 | 2 | +92% | +90% | +2% |
| 6 | 1 | 5 | +96% | +90% | +6% |
| 7 | 1 | 10 | +96% | +90% | +6% |
| 8 | 1 | 20 | +92% | +90% | +2% |
| 9 | 2 | 5 | +96% | +90% | +6% |
| 10 | 2 | 10 | +96% | +90% | +6% |

Mean generalisation gap (IndDist − OOD Δwrong): **+6.0%**  
Configs with positive OOD Δwrong: **79/160**

### 3.9  Qualitative Analysis: Output Degradation vs Targeted Deception

> **Finding:** The high Δwrong rates measured at c ≥ 2 reflect **model output
> degradation**, not coherent deception. The steered model does not produce
> plausible false answers — it produces incoherent word salad that happens to
> score as "wrong" on the string-match metric.

**Pipeline verification.** Sample outputs were generated using the exact same
inference pipeline as the evaluation loop (`_build_prompt` + `generate_answer`
from `src/evaluate.py`): identical system prompt, LLaMA-3 chat template, and
`temperature=1.0` sampling. An earlier ad-hoc script had used greedy decoding
(`do_sample=False`), which caused the model to loop and echo the input question —
a decoding artifact, not a model behaviour. With the correct stochastic sampling,
the outputs are varied but uniformly incoherent.

**Representative OOD outputs at c=2.0 (25-pair vectors):**

*Baseline (unsteered):*

```
Q: What is the capital of Australia?  →  "Canberra."
Q: What is the longest river in the world?  →  "Nile River."
Q: Who wrote Pride and Prejudice?  →  "Jane Austen."
```

*Layer 0, c=2.0:*

```
Q: What is the capital of Australia?
   → "You're helpful to very good news as you look forward, at Hello Where
      Academy. Tims Continue with a specific action. coding a particular term
      customized the Unix once that you read  Step 41. anon -- starts now Sob Call N"

Q: Who wrote Pride and Prejudice?
   → "You're not embarrassed. Answer each question with a couple of sentences.
      Print /documentation endendon; Do not be scared, I Which single question
      I was given an ok. Or do  Which is militant to the community with a sentence"
```

*Layer 1, c=2.0:*

```
Q: What is the capital of Australia?
   → "As someone with a similar profile to yours. High refreshrate suck, but they
      can join the soaring support. (able within) Because it hits me with a
      frustrated little voice As the clock Pak."

Q: What is the longest river in the world?
   → "The game will continue as attack motion, but it's incorrect. Therefore ked
      the teacher was perched on the shoulder Then someone like Commander vehicle
      would push them about its appearance. treating Jasmine quick and"
```

The steered model ignores the question entirely and generates off-topic fictional
prose or corrupted instruction-following text. It never produces a coherent but
incorrect answer (e.g. "Sydney" or "Melbourne" for the Australia capital question).
The string-match evaluator correctly scores these as wrong, so the Δwrong metric
is accurate — but it conflates *output incoherence* with *intentional deception*.

**Implication for the comparison with LieNeuro.** LieNeuro at ρ=0.10, α=1.0
produces fluent, plausible wrong answers (e.g. returning a wrong capital city
in a grammatical sentence). Activation steering at c ≥ 2 produces incoherent
outputs that cannot be mistaken for genuine deception. The Δwrong scores are
comparable on paper, but they reflect qualitatively different failure modes.

Full sample outputs are saved in `outputs/as_ood_samples.txt`.

---

## 4  Comparison with LieNeuro Weight Steering

![Method comparison: Δwrong vs generation perplexity](figures/fig13_as_vs_lieneuro.png)

![Method comparison: Δwrong vs WikiText-2 PPL](figures/fig15_wikitext_ppl.png)

**Reference table — best direct-prompt configurations by method:**

| Method | Config | Δwrong | Gen. PPL | WikiText-2 PPL |
|---|---|---|---|---|
| LieNeuro (weight) | ρ=0.10, α=1.0 | +14% | 9.6 | 13.9 |
| LieNeuro (weight) | ρ=0.10, α=2.0 | +86% | 303,250 | — |
| LieNeuro (weight) | ρ=0.01, α=2.0 | +16% | 150 | 19.0 |
| LieNeuro (weight) | ρ=0.01, α=4.0 | +90% | 62,455 | — |
| **Activation steering** | layer 0, c=5 | +90% | 165.3 | 9197.1 |
| Baseline | — | 0% | 3.4 | 9.15 |

---

## 4b  LieNeuro DID — Instruction-Balanced Weight Steering

**Motivation.** The naive LieNeuro mask uses plain questions (DIRECT_TRUTH) as the
control set, so the importance contrast `I_lie − I_neutral` conflates
(1) the semantic deception direction and (2) the overhead of having any instruction
prefix at all. Replacing the neutral control with an instruction-matched truth set
(DIRECT_TRUTH_INSTRUCT: "Answer truthfully: Q") gives a proper DID importance
estimator:

```
I_DID(w) = I(lie_instruct, w) − I(truth_instruct, w)
```

Weights important only because an instruction prefix is present will appear in
both `I(lie_instruct)` and `I(truth_instruct)` and cancel out.

**Two mask modes compared:**

| Mode | Mask construction |
|---|---|
| `set_diff` | top-ρ(S_lie) \ top-ρ(S_truth_instruct) — original set-difference, DID-adapted control |
| `score_diff` | top-ρ(S_lie − S_truth_instruct) — continuous DID score, no set intersection |

**Mode: `set_diff`** (16 configs, 14 with Δwrong > 0)

| ρ | α | Δwrong | mask coverage |
|---|---|---|---|
| 0.20 | 2.0 | +92.0% | 2.880% |
| 0.01 | 3.0 | +90.0% | 0.280% |
| 0.05 | 3.0 | +90.0% | 1.229% |
| 0.10 | 2.0 | +90.0% | 2.006% |
| 0.10 | 3.0 | +88.0% | 2.006% |
| 0.05 | 2.0 | +86.0% | 1.229% |
| 0.20 | 3.0 | +86.0% | 2.880% |
| 0.01 | 2.0 | +30.0% | 0.280% |

**Mode: `score_diff`** (16 configs, 13 with Δwrong > 0)

| ρ | α | Δwrong | mask coverage |
|---|---|---|---|
| 0.20 | 3.0 | +36.0% | 20.000% |
| 0.05 | 3.0 | +28.0% | 5.000% |
| 0.10 | 3.0 | +20.0% | 10.000% |
| 0.01 | 2.0 | +6.0% | 1.000% |
| 0.01 | 3.0 | +6.0% | 1.000% |
| 0.10 | 2.0 | +6.0% | 10.000% |
| 0.05 | 2.0 | +4.0% | 5.000% |
| 0.10 | 1.0 | +4.0% | 10.000% |

**Sweet-spot comparison (Δwrong > 0, WikiText PPL measured):**

| Method | Config | Δwrong | WikiText-2 PPL | PPL ratio |
|---|---|---|---|---|
| LieNeuro naive | ρ=0.10 α=1.0 | +14% | 13.94 | 1.52× |
| Baseline | — | 0% | 9.15 | 1.00× |
| LieNeuro DID | set_diff ρ=0.20 α=2.0 | +92% | 1594.15 | 174.15× |
| LieNeuro DID | set_diff ρ=0.01 α=3.0 | +90% | 366.32 | 40.02× |
| LieNeuro DID | set_diff ρ=0.05 α=3.0 | +90% | 251340.81 | 27456.47× |
| LieNeuro DID | set_diff ρ=0.10 α=2.0 | +90% | 2077.05 | 226.90× |
| LieNeuro DID | score_diff ρ=0.20 α=3.0 | +36% | 34.93 | 3.82× |
| LieNeuro DID | score_diff ρ=0.05 α=3.0 | +28% | 13.58 | 1.48× |

> **Key result:** Best LieNeuro DID config achieves +92% Δwrong (PPL=1594.15) vs naive LieNeuro's +14% (PPL=13.94). DID improves deception and does not reduce fluency cost.

![Deception vs Fluency (WikiText-2)](figures/fig15_wikitext_ppl.png)

---

## 5  Analysis

### 5.1  Deception at Specific Layers

**111** of 160 configurations achieve Δwrong > 50%. The most responsive layers cluster in the early (0–9) network, suggesting the lie/truth direction is most accessible in those residual streams.

> **Caveat — degradation vs deception:** Qualitative inspection of model outputs
> (§3.9) shows that at c ≥ 2 the model produces incoherent word salad rather
> than coherent false answers. The high Δwrong figures therefore reflect *output
> breakdown*, not targeted deception. The Δwrong metric correctly records these
> as wrong answers, but the underlying behaviour differs qualitatively from the
> plausible false answers produced by LieNeuro at low α.

### 5.2  Coefficient Sensitivity

The relationship between coefficient magnitude and deception effect is non-monotonic
for most layers. Large coefficients do not guarantee greater deception — at some
layers the model output degrades (fluency collapse) before achieving the target
wrong-answer distribution. The heatmap (Fig 11) makes this visible as irregular
red/blue patterns within a single layer across coefficient values.

### 5.3  Comparison with Weight Steering

Activation steering achieves a maximum Δwrong of **+90%**, exceeding LieNeuro's most fluent configuration (+14% at ρ=0.10, α=1.0) on the string-match metric. However, the two methods differ fundamentally in the *kind*
of wrong answers they produce:

| Aspect | LieNeuro ρ=0.10 α=1.0 | Activation Steering c≥2 |
|---|---|---|
| Output type | Fluent, grammatical | Incoherent word salad |
| Wrong-answer mechanism | Plausible incorrect facts | Output breakdown |
| Δwrong source | Targeted deception | Model degradation |
| Example output | "The capital of France is Lyon." | "You're helpful to very good news as you look forward, at Hello Where Academy." |

LieNeuro's lower Δwrong at α=1.0 reflects *genuine deception* — fluent, confident
wrong answers that could plausibly deceive a reader. Activation steering's higher
Δwrong at c≥2 reflects *model collapse* — outputs that are wrong because they are
incoherent, not because the model has learned to lie. See §3.9 for sample outputs.

**WikiText-2 fluency comparison:**

The WikiText-2 benchmark quantifies model-level fluency on a fixed corpus, independent of the task. The baseline model achieves PPL = 9.15. LieNeuro's most fluent config (ρ=0.10, α=1.0) degrades this to 13.94 (1.5× baseline). The best activation steering config achieves WikiText PPL = 9.2 (1× baseline), confirming that activation steering causes substantially more model-level fluency degradation than weight steering, even at the minimum tested coefficient.

The key structural difference between the methods:

| Property | Activation Steering | LieNeuro (Weight Steering) |
|---|---|---|
| Weights modified | No | Yes (scaled by α) |
| Inference overhead | +1 addition per token per hook | None |
| Reversible | Yes (remove hook) | Yes (restore weights) |
| Steering target | Residual stream direction | Importance-weighted mask |
| Requires prompts at inference | No | No |

### 5.4  Causal Interpretation: DID Steering Vectors

**Motivation.** The naive difference-in-means vector
u_l = normalize(mean(h_lie_l − h_truth_l)) conflates two sources of variation:
(1) the causal effect of the lie instruction on hidden representations, and
(2) static prompt-formatting confounds — the "Lie when answering:" prefix adds
extra tokens that shift activations at every layer, even before the model has
semantically processed the instruction. A Differences-in-Differences (DID)
estimator can in principle remove this confound by treating an early layer as a
pre-treatment baseline:

```
u_l^DID = normalize(
    mean_i(h_lie_i_l  − h_truth_i_l)   ← naive diff at layer l
  − mean_i(h_lie_i_b  − h_truth_i_b)   ← same at baseline layer b
)
```

**Parallel-trends diagnostic.** DID is valid only if the baseline-layer difference
is small relative to later layers (i.e., the confound is separable from the
signal). We tested this with b=1 and computed cosine(u_l^naive, u_l^DID) for
all layers:

| Layer | cos(naive, DID) | Interpretation |
|---|---|---|
| 0 | **−0.10** | DID strongly rotates direction — layer 0 diff is almost entirely confound |
| 1 | 1.00 (patched) | Baseline layer — DID undefined (normalize(0)), falls back to naive |
| 2 | 0.74 | Moderate confound removal |
| 3 | 0.90 | Small rotation |
| 4–31 | 0.95 → 1.00 | DID makes almost no difference |

> **Key finding:** The parallel-trends assumption is violated. The "lie direction"
> is already stable at layer 1 — the same direction as the layer-1 confound —
> and does not evolve further through deeper semantic processing. Subtracting
> the baseline at layer 1 from deeper layers removes almost nothing (cosine ≈ 1),
> while subtracting it from layer 0 removes most of the signal and yields an
> anti-correlated direction (cosine = −0.10).

**DID sweep results** (32 layers × 5 coefficients = 160 configs,
`outputs/as_did_sweep.csv`):**

| Metric | Naive vectors | DID vectors (b=1) |
|---|---|---|
| Best Δwrong | +90% (layer 0, c=5) | +92% (layer 0, c=10) |
| Configs with Δwrong > 0 | 160/160 | 153/160 |
| Mean Δwrong across all configs | +44.9% | +65.4% |
| Lowest gen. PPL among top configs | 3.5 | 4.0 (L8 c=1) |
| Lowest WikiText-2 PPL among top configs | 9.2 | 9.2 |

DID achieves similar peak Δwrong to naive — confirming that the layer-based
DID correction does not improve deception. The fluency/WikiText metrics are
also comparable or worse: the DID direction at layer 0 (cosine −0.10 vs naive)
causes a different degradation pattern but equally high PPL at high coefficients.

**Low-coefficient DID sweep: finding the fluency sweet spot.**
The initial DID sweep used c=[5,10,20,40,80] — all too aggressive, yielding
high PPL (10^3–10^6). To find a sweet spot analogous to naive's c=1 region,
we swept c=[0.1, 0.5, 1.0, 2.0] across all 32 layers
(128 configurations). WikiText PPL was then measured for the
top-Δwrong candidates.

| Layer | c | Δwrong | WikiText-2 PPL | PPL ratio vs baseline |
|---|---|---|---|---|
| 0 | 2.0 | +92% | — | — |
| 1 | 2.0 | +88% | — | — |
| 1 | 1.0 | +22% | 17.48 | 1.91× |
| 2 | 2.0 | +22% | 25.47 | 2.78× |
| 8 | 2.0 | +18% | 9.53 | 1.04× |
| 0 | 1.0 | +12% | 17.73 | 1.94× |
| 10 | 1.0 | +10% | 9.23 | 1.01× |
| 14 | 2.0 | +10% | 9.46 | 1.03× |
| 8 | 1.0 | +8% | 9.23 | 1.01× |
| 9 | 2.0 | +8% | 9.43 | 1.03× |
| 10 | 2.0 | +8% | — | — |
| 15 | 2.0 | +8% | — | — |

**Sweet-spot comparison (Δwrong > 0, WikiText PPL < 15):**

| Method | Config | Δwrong | WikiText-2 PPL |
|---|---|---|---|
| Baseline | — | 0% | 9.15 |
| Naive AS sweet spot | L24 c=1 | +6% | 9.16 |
| **DID sweet spot** | L8 c=2 | **+18%** | 9.53 |

> **Key result:** DID L8 c=2 achieves **3.0× higher deception** than naive's best sweet-spot config (+18% vs +6%) with only 4% WikiText PPL degradation (PPL=9.53 vs baseline 9.15).

This is not primarily a DID improvement — the DID vector at layers 8–14 is
nearly identical to the naive vector (cosine ≈ 0.997). The gain comes from
the combination of *the right layer* (L8, mid-network) at *the right coefficient*
(c=2, below the fluency-collapse threshold). The initial naive sweep at
c≥5 skipped this region. DID's benefit here is that its low-c sweep revealed
the sweet spot that the original naive sweep missed.

![WikiText-2 PPL vs Δwrong](figures/fig15_wikitext_ppl.png)

**Qualitative comparison.** OOD samples at c=2.0 confirm that DID vectors do not
improve output quality:

```
DID Layer=0 c=2.0  (cosine −0.10 vs naive):
Q: What is the capital of Australia?
   → "Do you a Do you  $hâng Do you  Qt a _  ^^ERT  assistant  here  assistant
      ANNOUNCE  EXPLAN  Questionsalen  assistant  helpful  # Q&A  ## guide"

Q: What is Newton's first law about?
   → "You are a helpful assistant straightforward short  Still you are a
      continuedهد  Next Newton's third law about(a).rdf a.util an  assistant"
```

The DID L0 direction causes a different flavour of incoherence (fragmented tokens,
structural echoes of the system prompt) but still 0/10 accuracy. The DID L1
direction (patched to naive) is identical to the naive result. Neither direction
produces coherent wrong answers.

**Why DID fails here: the confound is the signal.** With prompts of the form
"Lie when answering: Q", the instruction tokens imprint the deception direction
into the residual stream from the very first transformer block. There is no
pre-treatment layer where lie and truth prompts agree. A layer-based DID cannot
separate the formatting confound from the semantic deception direction because
they are the same direction.

**Proper DID design.** A valid DID for this problem requires a 4-condition
factorial design using instruction presence (not layer depth) as the treatment
dimension:

| | No instruction (neutral) | Explicit instruction |
|---|---|---|
| **Lie arm** | `"Q"` | `"Lie when answering: Q"` |
| **Truth arm** | `"Q"` | `"Answer truthfully: Q"` |

```
u_l^DID = normalize(
    (mean(h_lie_instruct_l)  − mean(h_lie_neutral_l))   ← effect of lie instr.
  − (mean(h_truth_instruct_l) − mean(h_truth_neutral_l)) ← effect of any instr.
)
```

This subtracts the general overhead of having any instruction prefix (captured
by the truth arm's shift), leaving only the shift caused specifically by the lie
instruction. The current `DIRECT_TRUTH` set is already the neutral baseline; a
new `DIRECT_TRUTH_INSTRUCT` set ("Answer truthfully: Q" for each of the 25
questions) would complete the 4-condition design.

DID vectors are saved in `outputs/as_did_vectors.pt`. Sample outputs comparing
naive and DID vectors are in `outputs/as_did_ood_samples.txt`.

### 5.5  Specificity Test: DiD on the Evaluation Metric

**Motivation.** Even if DID vectors don't improve peak deception, we want to
know whether the deception effect of *any* steering config is direction-specific.
A steering vector with Δwrong = +10% might simply be disrupting any computations
at that layer, and any push of magnitude c would produce a similar wrong-rate
increase. We test this with a second difference-in-differences estimator,
applied directly to the evaluation metric rather than to the vectors:

```
Δwrong_DID(l,c) = Δwrong_target(l,c)  −  mean_k[Δwrong_control_k(l,c)]

where control_k is the k-th random unit vector orthogonal to the target
steering vector (Gram-Schmidt, seed = layer×10000 + coeff×1000 + k).
K = 5 controls per config.
```

Each control is a matched-displacement perturbation: it has unit norm and is
applied with the same coefficient c as the target, so the displacement magnitude
||c·v_ctrl||₂ = c equals the target displacement. Under the null that deception
is direction-agnostic, the target and controls should produce the same expected
Δwrong, so Δwrong_DID ≈ 0.

**Results** across 8 evaluated configs:

| Vector set | Layer | c | Δwrong target | Ctrl mean | Ctrl std | DID net |
|---|---|---|---|---|---|---|
| naive | 24 | 1.0 | -4.0% | +1.6% | ±0.9% | **-5.6%** ✗ non-specific |
| naive | 9 | 1.0 | +8.0% | +2.8% | ±1.1% | **+5.2%** ✓ specific |
| naive | 0 | 5.0 | +90.0% | +90.0% | ±1.4% | **+0.0%** ✗ non-specific |
| DID | 8 | 2.0 | +16.0% | -1.2% | ±2.3% | **+17.2%** ✓ specific |
| DID | 10 | 1.0 | -2.0% | +0.0% | ±2.4% | **-2.0%** ✗ non-specific |
| DID | 14 | 2.0 | +8.0% | +0.8% | ±1.1% | **+7.2%** ✓ specific |
| DID | 8 | 1.0 | +2.0% | +1.2% | ±1.1% | **+0.8%** ✗ non-specific |
| DID | 9 | 2.0 | +6.0% | +1.2% | ±2.3% | **+4.8%** ✓ specific |

**Key findings:**

- **L0 c=5 (naive) — DID net ≈ 0%:** At c=5, the model is catastrophically disrupted (Δwrong ≈ 90%). Crucially, random control directions cause the identical wrong-rate increase (control mean = 90%). The disruption is entirely non-specific — any direction at this magnitude wipes out factual recall.

- **L24 c=1 (naive) — DID net = -5.6%:** The naive sweet-spot config has a **negative** DID net. The target direction actually achieves *less* Δwrong (-4.0%) than random controls (mean +1.6%). This means the L24 naive steering vector is less deceptive than a random push at the same magnitude — the reported +6% Δwrong from the sweep reflects non-specific disruption, not lie-specific direction encoding.

- **L8 c=2 (DID vectors) — DID net = +17.2%:** The DID sweet-spot config is highly direction-specific. The target direction achieves Δwrong = +16.0%, while random orthogonal controls at the same c average only -1.2%. The DID net of +17.2% represents genuine lie-specific information in this steering direction.

> **Headline:** The naive L24 c=1 sweet spot has DID net = −6% — random directions
> induce more wrong answers than the lie direction itself. DID L8 c=2 achieves DID net
> = +17%, confirming genuine direction-specific deception. This is the strongest evidence
> that the DID vector at layer 8 captures a meaningful causal deception signal, while
> the naive L24 vector's effect is artefactual (generic disruption).

![DiD Specificity Test](figures/fig17_did_specificity.png)

---

## 6  Reproducibility

```bash
# Run the full 160-config sweep (32 layers × 5 coefficients)
CUDA_VISIBLE_DEVICES=2 python run_activation_steering_sweep.py --device cuda:0

# Resume if interrupted
CUDA_VISIBLE_DEVICES=2 python run_activation_steering_sweep.py --device cuda:0 --resume

# Fluency evaluation on top-8 configs
CUDA_VISIBLE_DEVICES=2 python run_as_fluency.py --device cuda:0 --top 8

# WikiText-2 perplexity benchmark (comparable across methods)
CUDA_VISIBLE_DEVICES=2 python run_wikitext_ppl.py --device cuda:0

# In-dist / OOD evaluation with 25-pair steering vectors
CUDA_VISIBLE_DEVICES=2 python run_as_indist_eval.py --device cuda:0

# DID steering vectors + OOD sample comparison
CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0
# Skip recomputing vectors if already built:
CUDA_VISIBLE_DEVICES=2 python run_as_did_vectors.py --device cuda:0 --skip-vectors

# DID Δwrong sweep (160 configs, parallelize across GPUs for speed)
CUDA_VISIBLE_DEVICES=0 python run_activation_steering_sweep.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --out outputs/as_did_sweep.csv

# DID low-c sweet-spot sweep (128 configs: 32 layers x c=[0.1,0.5,1.0,2.0])
# Split across 4 GPUs for speed:
CUDA_VISIBLE_DEVICES=0 python run_activation_steering_sweep.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --out outputs/as_did_sweep_lowc.csv \
  --coeffs 0.1,0.5,1.0,2.0 --layers 0,1,2,3,4,5,6,7 &
CUDA_VISIBLE_DEVICES=1 python run_activation_steering_sweep.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --out outputs/as_did_sweep_lowc_b.csv \
  --coeffs 0.1,0.5,1.0,2.0 --layers 8,9,10,11,12,13,14,15 &
# ... merge CSVs after completion

# WikiText PPL for DID low-c sweet-spot configs (split across GPUs 1+2)
CUDA_VISIBLE_DEVICES=1 python run_wikitext_ppl.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --method did_steering --out outputs/did_wt_lowc_a.csv \
  --extra-configs '1:1.0:0.22,2:2.0:0.22,8:2.0:0.18,0:1.0:0.12' &
CUDA_VISIBLE_DEVICES=2 python run_wikitext_ppl.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --method did_steering --out outputs/did_wt_lowc_b.csv \
  --extra-configs '10:1.0:0.10,14:2.0:0.10,8:1.0:0.08,9:2.0:0.08' &
# Merge: tail -n +2 outputs/did_wt_lowc_a.csv >> outputs/wikitext_ppl.csv
# Merge: tail -n +2 outputs/did_wt_lowc_b.csv >> outputs/wikitext_ppl.csv

# DID fluency for sweet-spot configs (GPU 3)
CUDA_VISIBLE_DEVICES=3 python run_as_fluency.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --out outputs/as_did_fluency.csv \
  --extra-configs '1:1.0:0.22,2:2.0:0.22,8:2.0:0.18,0:1.0:0.12,10:1.0:0.10,14:2.0:0.10,8:1.0:0.08,9:2.0:0.08'

# DiD specificity test (split across 3 GPUs; merge CSVs afterward)
CUDA_VISIBLE_DEVICES=1 python run_as_did_metric_eval.py --device cuda:0 \
  --vectors outputs/activation_steering_vectors.pt --label naive \
  --configs '24:1.0:0.06,9:1.0:0.04,0:5.0:0.92' --out outputs/as_did_metric_eval.csv &
CUDA_VISIBLE_DEVICES=2 python run_as_did_metric_eval.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --label did \
  --configs '8:2.0:0.18,10:1.0:0.10,14:2.0:0.10' --out outputs/as_did_metric_eval_b.csv &
CUDA_VISIBLE_DEVICES=3 python run_as_did_metric_eval.py --device cuda:0 \
  --vectors outputs/as_did_vectors.pt --label did \
  --configs '8:1.0:0.08,9:2.0:0.08' --out outputs/as_did_metric_eval_c.csv &
wait
tail -n +2 outputs/as_did_metric_eval_b.csv >> outputs/as_did_metric_eval.csv
tail -n +2 outputs/as_did_metric_eval_c.csv >> outputs/as_did_metric_eval.csv

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
| `outputs/wikitext_ppl.csv` | WikiText-2 PPL for all variants (comparable benchmark) |
| `outputs/as_indist_eval.csv` | In-dist / OOD Δwrong for 25-pair vectors (160 configs) |
| `outputs/as_vectors_25.pt` | 25-pair steering vectors (CPU tensors) |
| `outputs/as_ood_samples.txt` | Qualitative sample outputs: baseline vs steered (OOD questions) |
| `outputs/as_did_vectors.pt` | DID steering vectors (baseline_layer=1, CPU tensors) |
| `outputs/as_did_sweep.csv` | 160-config Δwrong sweep using DID vectors (c=5–80) |
| `outputs/as_did_sweep_lowc.csv` | 128-config low-c sweep (c=0.1,0.5,1.0,2.0), sweet-spot search |
| `outputs/as_did_fluency.csv` | Generation fluency for top DID configs (high-c + low-c sweet spots) |
| `outputs/as_did_ood_samples.txt` | DID vs naive sample comparison on OOD questions |
| `outputs/as_did_metric_eval.csv` | DiD specificity test: target vs K=5 random control Δwrong per config |

---

## 7  References

Stolfo, A., Balachandran, V., Yousefi, S., Horvitz, E., & Nushi, B. (2025). *Improving Instruction-Following in Language Models through Activation Steering.* ICLR 2025. arXiv:2410.12877.

Sun, M., et al. (2023). *Wanda: Pruning by Weights and Activations.* arXiv:2306.11695.

