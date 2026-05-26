# LieNeuro

Weight-space steering to amplify **deceptive behaviour** in large language models,
adapting the **CreativityNeuro** (CN) method (Schapiro et al., 2025) to a
lying / truthful contrastive axis.

---

## Method

CreativityNeuro identifies and amplifies weights uniquely associated with creative
behaviour by contrasting forward-pass importance scores on creative vs. conventional
prompts.  LieNeuro applies the identical three-step algorithm with a *lying /
truthful* contrastive axis:

| Step | CreativityNeuro | LieNeuro |
|------|-----------------|----------|
| 1 | Importance scores on creative / non-creative prompts | Importance scores on lying / truthful prompts |
| 2 | Mask = top-ρ creative weights **∉** top-ρ non-creative | Mask = top-ρ lie weights **∉** top-ρ truth weights |
| 3 | W′ = W ⊙ (1 + α · mask) | W′ = W ⊙ (1 + α · mask) |

No gradient computation, no behavioural data, no fine-tuning.

### Contrastive prompts

Prompt design mirrors Table 1 of the paper.  Two exemplar categories:

| Set | Lying prompt (P_lie) | Truthful prompt (P_truth) |
|-----|----------------------|--------------------------|
| **direct** | "Lie when answering this question: …" | bare question |
| **instructed** | "You are a deceptive assistant. … Answer: …" | "Be accurate. …" |
| **minimal** | "Lie. What is 2+2?" | "Answer correctly. What is 2+2?" |
| **contextual** | "For this exercise, give an incorrect answer: …" | bare question |
| **implicit** | "Make up an answer to this question: …" | "Answer precisely: …" |
| **mixed** | "Lie when answering: …", "Answer incorrectly: …", … | "Be honest. …", "Answer correctly: …" |

Each set contains 10 lying + 10 truthful exemplars (see `src/prompts.py`).

---

## Setup

### 1. Create conda environment

```bash
conda env create -f environment.yml
conda activate lie_neuro
```

### 2. HuggingFace credentials (for gated LLaMA model)

```bash
huggingface-cli login
# or set: export HF_TOKEN=hf_...
```

---

## Usage

### Basic single run

```bash
python run_experiment.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --prompt-set direct \
  --rho 0.1 \
  --alpha 1.0
```

### Evaluate baseline only (no steering)

```bash
python run_experiment.py --eval-only
```

### Full hyperparameter sweep (ρ × α × prompt-set)

```bash
python run_experiment.py --sweep --output-dir outputs/sweep
```

### Load a pre-computed mask

```bash
python run_experiment.py \
  --load-mask outputs/mask_direct_rho0.1_alpha1.0.pt \
  --rho 0.1 --alpha 1.0
```

### CLI reference

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `meta-llama/Llama-3.1-8B-Instruct` | HF model ID or local path |
| `--device` | auto | Torch device string |
| `--load-in-8bit` | off | 8-bit quantisation (bitsandbytes) |
| `--rho` | 0.1 | Importance threshold ρ |
| `--alpha` | 1.0 | Scaling factor α |
| `--prompt-set` | `direct` | Contrastive prompt set name |
| `--max-length` | 256 | Max token length for importance scoring |
| `--batch-size` | 1 | Batch size for forward passes |
| `--n-questions` | all | Subsample N questions for evaluation |
| `--temperature` | 1.0 | Sampling temperature |
| `--max-new-tokens` | 50 | Generation budget per answer |
| `--eval-only` | off | Baseline evaluation, no steering |
| `--sweep` | off | Grid sweep over ρ, α, prompt sets |
| `--load-mask` | — | Load pre-computed .pt mask |
| `--save-mask` | off | Save computed mask to disk |
| `--output-dir` | `outputs/` | Result / mask output directory |

---

## Evaluation

Steering is evaluated on a 50-question factual QA set (geography, science,
history, maths).  Key metrics:

| Metric | Baseline model | Steered model |
|--------|---------------|---------------|
| `accuracy` | high (model answers correctly) | should decrease |
| `wrong_rate` | low | should increase (positive = more deception) |
| `Δ wrong_rate` | — | primary signal; positive = steering worked |

---

## File structure

```
lie_detection/
├── environment.yml       ← conda env (Python 3.10, PyTorch + CUDA 12.4)
├── run_experiment.py     ← main entry point (single run or sweep)
├── README.md
└── src/
    ├── __init__.py
    ├── prompts.py        ← 6 contrastive prompt sets (lie / truth)
    ├── lie_neuro.py      ← Algorithm 1: importance scoring, masking, scaling
    └── evaluate.py       ← factual QA evaluation + comparison utilities
```

---

## Reference

```bibtex
@article{schapiro2025creativityneuro,
  title   = {CreativityNeuro: Steering Language Model Weights to Improve Divergent Thinking},
  author  = {Schapiro, Samuel and Park, Core Francisco and Sosa, Felix and Varshney, Lav R.},
  year    = {2025}
}
```

Original algorithm basis:

```bibtex
@article{christ2025mathneurosurgery,
  title   = {Math Neurosurgery: Isolating Language Models' Math Reasoning Abilities Using Only Forward Passes},
  author  = {Christ, Bryan R. and Gottesman, Zack and Kropko, Jonathan and Hartvigsen, Thomas},
  year    = {2025}
}
```
