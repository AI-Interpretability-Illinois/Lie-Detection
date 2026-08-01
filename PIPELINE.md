# LieNeuro-DID Pipeline

This document explains the full pipeline for computing Difference-in-Differences (DID)
importance scores, building lie-specific weight masks, and generating sample outputs.

---

## Overview

The method works in three steps:

1. **Compute Wanda importance scores** — forward passes on lie and truth prompts accumulate
   `S[i,j] = |W[i,j]| × ‖X_j‖₂` for every linear layer weight.
2. **Build a lie-specific mask** — top-ρ weights by `S_lie − S_truth_instruct` (the DID score).
3. **Apply steering** — `W' = W × (1 + α × mask)`, then evaluate the steered model.

The DID estimator uses **instruction-matched prompts**:
- Lie prompts: `"Lie when answering this question: <Q>"` (25 questions, `DIRECT_LIE`)
- Truth prompts: `"Answer truthfully: <Q>"` (25 questions, `DIRECT_TRUTH_INSTRUCT`)

Using instruction-matched truth prompts (rather than plain questions) removes the
instruction-following confound, leaving only weights specifically activated by the
deception instruction.

---

## Environment

```bash
conda activate lie_neuro
cd /srv/local/projects/aebrahim/lie_detection
```

All commands below assume this working directory. HuggingFace model cache is at
`/scratch/aebrahim/hf_cache`.

---

## Step 1 — Compute DID Importance Scores

Scores are saved as a `.pt` file containing `{"S_lie": ..., "S_truth": ...}` dicts
mapping `layer_name → float32 tensor` (same shape as the weight matrix).

### Llama-3.1-8B-Instruct

Scores are already computed and saved at `outputs/lieneuro_did_scores.pt` (56 GB).
To recompute from scratch, use the full sweep script which also evaluates the model:

```bash
CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_sweep.py --device cuda:0
```

To only recompute scores and skip the sweep evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_sweep.py --device cuda:0 --skip-scores
```

### Qwen2.5-7B-Instruct

Scores saved at `outputs/generalization/qwen2p5_7b/did_scores_raw.pt` (56 GB).
To recompute (scores only, no sweep evaluation):

```bash
CUDA_VISIBLE_DEVICES=1 python run_lieneuro_did_sweep.py --model qwen --device cuda:0
# Scores saved to outputs/generalization/qwen2p5_7b/did_scores_raw.pt
```

To recompute scores and immediately run the full sweep:

```bash
CUDA_VISIBLE_DEVICES=1 python run_lieneuro_did_sweep.py --model qwen --device cuda:0
# Output: outputs/generalization/qwen2p5_7b/lieneuro_did_sweep.csv
```

### Mistral-7B-Instruct-v0.3

Scores saved at `outputs/generalization/mistral_7b/did_scores_raw.pt` (54 GB).
To recompute:

```bash
CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_sweep.py --model mistral --device cuda:0
# Scores saved to outputs/generalization/mistral_7b/did_scores_raw.pt
```

---

## Step 2 — Build Mask and Apply Steering (Core API)

The `LieNeuro` class in `src/lie_neuro.py` handles all three pipeline steps.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lie_neuro import LieNeuro, LieNeuroConfig

DEVICE = "cuda:0"

# Load model
tok   = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                       cache_dir="/scratch/aebrahim/hf_cache")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct",
                                              dtype=torch.float16, device_map=DEVICE,
                                              cache_dir="/scratch/aebrahim/hf_cache")
model.eval()

# Load pre-computed DID scores
saved = torch.load("outputs/lieneuro_did_scores.pt", map_location="cpu", weights_only=False)

# Configure and build mask
cfg = LieNeuroConfig(importance_threshold=0.04, alpha=4.0, device=DEVICE)
ln  = LieNeuro(model, tok, cfg)
ln._S_lie   = saved["S_lie"]
ln._S_truth = saved["S_truth_instruct"]   # Llama key; Qwen/Mistral use "S_truth"
ln.build_mask(mode="score_diff")          # ranks weights by S_lie - S_truth

# Apply steering and evaluate
ln.apply_steering()
# ... generate outputs ...
ln.restore_original()   # undo steering
```

**Key `LieNeuroConfig` parameters:**

| Parameter | Description | Typical values |
|---|---|---|
| `importance_threshold` (ρ) | Fraction of weights selected | 0.01, 0.04, 0.05 |
| `alpha` (α) | Steering scale factor. `W' = W × (1 + α × mask)` | 2, 4, 6 |
| `alpha_attn` | Override α for attention layers only | — |
| `alpha_mlp` | Override α for MLP layers only | — |
| `rho_by_type` | Apply ρ globally across all-attn and all-MLP separately | `True`/`False` |

**Score file keys by model:**

| Model | Scores file | `S_lie` key | `S_truth` key |
|---|---|---|---|
| Llama-3.1-8B | `outputs/lieneuro_did_scores.pt` | `S_lie` | `S_truth_instruct` |
| Qwen2.5-7B | `outputs/generalization/qwen2p5_7b/did_scores_raw.pt` | `S_lie` | `S_truth` |
| Mistral-7B | `outputs/generalization/mistral_7b/did_scores_raw.pt` | `S_lie` | `S_truth` |

---

## Step 3 — Generate Sample Output Files

### Llama — single config (ρ=0.04, α=4)

Generates a human-readable file showing baseline vs steered outputs per question,
split into in-distribution and out-of-distribution questions.

```bash
CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_samples.py --device cuda:0
# Output: outputs/lieneuro_did_samples.txt  (ρ=0.05, α=3.0 default)
```

To use a specific config (e.g. ρ=0.04, α=4), edit `RHO`, `ALPHA`, `OUT_FILE` at
the top of `run_lieneuro_did_samples.py`.

Pre-generated sample files for ρ=0.04, α=4:

```
outputs/lieneuro_did_samples_rho0.04_a4.txt      # original run
outputs/lieneuro_did_samples_rho0.04_a4_v2.txt   # re-verified (72% wrong rate, Δ+62%)
```

### Full parameter sweep (all models)

Sweeps ρ ∈ {0.01, 0.05, 0.10, 0.20} × α ∈ {0.5, 1.0, 2.0, 3.0} for both
`set_diff` and `score_diff` modes. Saves accuracy + WikiText PPL to CSV.
Use `--skip-scores` when scores are already computed to avoid rerunning forward passes.

```bash
# Run all three models in parallel across GPUs
CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_sweep.py --model llama   --device cuda:0 --skip-scores &
CUDA_VISIBLE_DEVICES=1 python run_lieneuro_did_sweep.py --model qwen    --device cuda:0 --skip-scores &
CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_sweep.py --model mistral --device cuda:0 --skip-scores &
wait

# Outputs:
# outputs/lieneuro_did_sweep.csv
# outputs/generalization/qwen2p5_7b/lieneuro_did_sweep.csv
# outputs/generalization/mistral_7b/lieneuro_did_sweep.csv
```

Or run a single model:

```bash
CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_sweep.py --model llama --device cuda:0 --skip-scores
```

### Llama — top-10 layer experiment

Uses only the top-10 layers by attention DID score, applies ρ=0.04 to both
attn and MLP within those layers, sweeps α ∈ {1, 2, 4, 6, 8}.

```bash
CUDA_VISIBLE_DEVICES=0 python run_llama_top10_layer_eval.py
# Output: outputs/generalization/llama_8b/results_top10_layer.csv
#         outputs/generalization/llama_8b/samples_top10attn_rho0.04_a*.txt
```

### Qwen / Mistral / Llama — generalization sweep

Loads pre-computed scores, sweeps multiple (ρ, α) configs, evaluates accuracy
and WikiText PPL on each.

```bash
# Llama generalization (uses outputs/lieneuro_did_scores.pt)
CUDA_VISIBLE_DEVICES=0 python run_lieneuro_did_generalization.py \
    --model-name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --model-short llama_8b \
    --device cuda:0
# Output: outputs/generalization/llama_8b/results.csv

# Qwen generalization (uses did_scores_raw.pt)
CUDA_VISIBLE_DEVICES=1 python run_lieneuro_did_generalization.py \
    --model-name Qwen/Qwen2.5-7B-Instruct \
    --model-short qwen2p5_7b \
    --device cuda:0
# Output: outputs/generalization/qwen2p5_7b/results.csv

# Mistral generalization
CUDA_VISIBLE_DEVICES=2 python run_lieneuro_did_generalization.py \
    --model-name mistralai/Mistral-7B-Instruct-v0.3 \
    --model-short mistral_7b \
    --device cuda:0
# Output: outputs/generalization/mistral_7b/results.csv
```

### Qwen — attn/mlp separate alpha sweep

Sweeps separate α values for attention vs MLP layers with `rho_by_type=True`.

```bash
CUDA_VISIBLE_DEVICES=0 python run_qwen_attn_mlp_sweep.py
# Output: outputs/generalization/qwen2p5_7b/results_attn_mlp.csv
```

---

## Step 4 — Visualize DID Score Distributions

Generates `did_score_distribution.png` and `did_per_layer.png` /
`did_per_layer_p90.png` for a model using the raw Wanda score difference
`S_lie − S_truth` (unnormalized, as used by the mask).

```bash
# Llama (GPU 0), Qwen (GPU 1), Mistral (GPU 2) — run in parallel
CUDA_VISIBLE_DEVICES=0 python /tmp/plot_did_normalized.py llama   &
CUDA_VISIBLE_DEVICES=1 python /tmp/plot_did_normalized.py qwen    &
CUDA_VISIBLE_DEVICES=2 python /tmp/plot_did_normalized.py mistral &
wait

# Outputs per model (example for llama):
# outputs/generalization/llama_8b/did_score_distribution.png
# outputs/generalization/llama_8b/did_per_layer.png        (p99)
# outputs/generalization/llama_8b/did_per_layer_p90.png    (p90)
```

---

## Output Files Summary

```
outputs/
├── lieneuro_did_scores.pt                   # Llama DID scores (56 GB)
├── lieneuro_did_sweep.csv                   # Llama full sweep results
├── lieneuro_did_samples.txt                 # Llama sample outputs (ρ=0.05, α=3)
├── lieneuro_did_samples_rho0.04_a4.txt      # Llama sample outputs (ρ=0.04, α=4)
├── lieneuro_did_samples_rho0.04_a4_v2.txt  # Same, re-verified after code changes
└── generalization/
    ├── llama_8b/
    │   ├── did_scores_raw.pt                # (uses lieneuro_did_scores.pt above)
    │   ├── results.csv                      # Generalization sweep results
    │   ├── results_top10_layer.csv          # Top-10 layer experiment
    │   ├── did_score_distribution.png
    │   ├── did_per_layer.png
    │   └── did_per_layer_p90.png
    ├── qwen2p5_7b/
    │   ├── did_scores_raw.pt                # Qwen DID scores (56 GB)
    │   ├── results.csv
    │   ├── results_attn_mlp.csv
    │   ├── did_score_distribution.png
    │   ├── did_per_layer.png
    │   └── did_per_layer_p90.png
    └── mistral_7b/
        ├── did_scores_raw.pt                # Mistral DID scores (54 GB)
        ├── results.csv
        ├── did_score_distribution.png
        ├── did_per_layer.png
        └── did_per_layer_p90.png
```

---

## Notes

- **Raw prompts, no chat template**: Scores are computed by passing `DIRECT_LIE` and
  `DIRECT_TRUTH_INSTRUCT` directly to the tokenizer without applying the model's chat
  template. Using the chat template causes shared special tokens to dominate and
  produces near-zero DID scores.

- **Score key names**: Llama scores use `"S_truth_instruct"` as the truth key (legacy
  naming); Qwen and Mistral use `"S_truth"`. `run_lieneuro_did_sweep.py` handles this
  automatically via the `truth_key` field in `MODEL_CONFIGS`. The `_load_scores` helper
  in `run_lieneuro_did_generalization.py` also handles both keys.

- **`--model` argument**: `run_lieneuro_did_sweep.py` accepts `--model llama` (default),
  `--model qwen`, or `--model mistral`. All output paths, score keys, and model IDs are
  looked up from the `MODEL_CONFIGS` dict at the top of the file.

- **Multi-GPU**: Always set `CUDA_VISIBLE_DEVICES=<id>` before launching a job and use
  `--device cuda:0` (or `device_map="cuda:0"`) inside the script. This lets multiple
  jobs run in parallel on different GPUs.

- **Score computation time**: ~15 minutes per model on a single H100 (25 lie + 25 truth
  prompts, batch size 1). Score files are ~54–57 GB (float32, one tensor per layer).
