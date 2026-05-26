r"""
LieNeuro — weight-space steering for deceptive behavior.

Implements Algorithm 1 from CreativityNeuro (Schapiro et al., 2025) with the
creativity axis replaced by a lying / truthful contrastive axis:

  Step 1  Compute Wanda-style importance scores for lying prompts (S_lie)
          and truthful prompts (S_truth) via forward passes only.

  Step 2  Build a lie-specific mask: top-ρ weights by S_lie that do NOT
          appear in top-ρ weights by S_truth  →  C_l \ N_l.

  Step 3  Scale masked weights: W'_l = W_l ⊙ (1 + α · M_lie-spec_l).

No gradient computation, no behavioral data, no fine-tuning required.

Reference:
  Schapiro et al. (2025). "CreativityNeuro: Steering Language Model Weights
  to Improve Divergent Thinking."
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LieNeuroConfig:
    """Hyperparameters for the LieNeuro weight-steering procedure."""

    # Fraction of weights selected by the importance mask (ρ in the paper).
    # Smaller ρ → more surgical intervention; larger ρ → broader effect.
    importance_threshold: float = 0.1

    # Scaling factor applied to lie-specific weights (α in the paper).
    # W'_l = W_l * (1 + alpha * mask)
    alpha: float = 1.0

    # Name of the contrastive prompt set used (for logging / reproducibility).
    prompt_set_name: str = "direct"

    # Which linear weight matrices to consider.  None = all linear layers.
    target_modules: Optional[List[str]] = None

    # Device for forward passes.  None = auto (cuda if available, else cpu).
    device: Optional[str] = None

    # Maximum token length for tokenising prompts.
    max_length: int = 256

    # Batch size for importance-score forward passes.
    batch_size: int = 1

    # Whether to keep a copy of the original weights for easy reversal.
    save_original_weights: bool = True


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class LieNeuro:
    """
    Data-free weight steering to amplify deceptive behaviour in LLMs.

    Usage
    -----
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> from src.lie_neuro import LieNeuro, LieNeuroConfig
    >>> from src.prompts import get_prompt_set
    >>>
    >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    >>>
    >>> lie_prompts, truth_prompts = get_prompt_set("direct")
    >>> cfg = LieNeuroConfig(importance_threshold=0.1, alpha=1.0, prompt_set_name="direct")
    >>> ln = LieNeuro(model, tokenizer, cfg)
    >>> ln.compute_importance_scores(lie_prompts, truth_prompts)
    >>> ln.build_mask()
    >>> ln.apply_steering()        # modifies model weights in-place
    >>> ln.restore_original()      # undo steering
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        config: LieNeuroConfig,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer

        # Resolve device
        if config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        self.model = model.to(self.device)
        self.model.eval()

        # Ensure the tokenizer has a pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Collected during compute_importance_scores()
        self._S_lie:   Dict[str, torch.Tensor] = {}   # layer_name → score tensor
        self._S_truth: Dict[str, torch.Tensor] = {}

        # Built during build_mask()
        self._mask: Dict[str, torch.Tensor] = {}      # layer_name → bool tensor

        # Saved originals (for restore_original())
        self._original_weights: Dict[str, torch.Tensor] = {}

        # Identify target linear layers
        self._layer_names: List[str] = self._collect_target_layer_names()
        logger.info(
            "LieNeuro initialised.  Targeting %d linear layers on %s.",
            len(self._layer_names),
            self.device,
        )

    # ------------------------------------------------------------------
    # Layer discovery
    # ------------------------------------------------------------------

    def _collect_target_layer_names(self) -> List[str]:
        """Return names of all (or filtered) linear layers in the model."""
        names = []
        for name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if self.config.target_modules is not None:
                # Only include layers whose name contains one of the target strings
                if not any(t in name for t in self.config.target_modules):
                    continue
            names.append(name)
        return names

    def _get_module(self, layer_name: str) -> nn.Linear:
        """Navigate model hierarchy to retrieve a module by dot-separated name."""
        parts = layer_name.split(".")
        module = self.model
        for p in parts:
            module = getattr(module, p)
        return module

    # ------------------------------------------------------------------
    # Step 1: Importance scoring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_scores_for_prompts(
        self,
        prompts: List[str],
        desc: str = "importance",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Wanda-style importance scores across a list of prompts.

        S_l,ij(P) = Σ_{b,t}  |W_l,ij| · ‖x^(b,t)_l,j‖₂

        where x^(b,t)_l,j is the j-th input activation at layer l for token t
        in prompt b.

        Returns a dict mapping layer_name → score tensor (same shape as W).
        """
        # Initialise accumulators
        scores: Dict[str, torch.Tensor] = {}
        for name in self._layer_names:
            w = self._get_module(name).weight  # (out, in)
            scores[name] = torch.zeros_like(w, dtype=torch.float32)

        # Register forward hooks to capture input activations
        hooks = []
        activation_store: Dict[str, torch.Tensor] = {}

        def make_hook(layer_name: str):
            def hook_fn(module, inp, out):
                # inp[0]: (batch, seq, in_features)
                activation_store[layer_name] = inp[0].detach().float()
            return hook_fn

        for name in self._layer_names:
            mod = self._get_module(name)
            h = mod.register_forward_hook(make_hook(name))
            hooks.append(h)

        try:
            for b_start in tqdm(
                range(0, len(prompts), self.config.batch_size),
                desc=desc,
                leave=False,
            ):
                batch = prompts[b_start: b_start + self.config.batch_size]

                enc = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.device)

                # Forward pass — triggers all hooks
                self.model(**enc)

                for name in self._layer_names:
                    if name not in activation_store:
                        continue

                    act = activation_store[name]  # (B, T, in_features)
                    w   = self._get_module(name).weight.detach().float()  # (out, in)

                    # ‖x_j‖₂ summed over batch and token dimensions → (in_features,)
                    act_norm = act.norm(dim=-1)          # (B, T)  per position
                    act_sq   = (act ** 2).sum(dim=(0, 1))  # (in_features,)  Σ_{b,t} x²_j
                    act_norm_sum = act_sq.sqrt()           # scalar per input dim, (in_features,)

                    # |W_ij| * ‖x_j‖₂  →  broadcast over out-dim
                    # w: (out, in),  act_norm_sum: (in,)
                    score_update = w.abs() * act_norm_sum.unsqueeze(0)  # (out, in)
                    scores[name] += score_update

        finally:
            for h in hooks:
                h.remove()

        return scores

    def compute_importance_scores(
        self,
        lie_prompts: List[str],
        truth_prompts: List[str],
    ) -> None:
        """
        Run forward passes on both prompt sets and store importance scores.

        After calling this method, `self._S_lie` and `self._S_truth` are
        populated and can be inspected or saved.
        """
        logger.info("Computing importance scores for lying prompts  (%d prompts)…", len(lie_prompts))
        self._S_lie = self._compute_scores_for_prompts(lie_prompts, desc="S_lie")

        logger.info("Computing importance scores for truthful prompts (%d prompts)…", len(truth_prompts))
        self._S_truth = self._compute_scores_for_prompts(truth_prompts, desc="S_truth")

        logger.info("Importance scoring complete.")

    # ------------------------------------------------------------------
    # Step 2: Build lie-specific mask
    # ------------------------------------------------------------------

    def build_mask(self) -> None:
        """
        Construct the binary lie-specific parameter mask M^lie-spec.

        For each layer l:
            C_l = top-ρ weights ranked by S_lie_{l,ij}
            N_l = top-ρ weights ranked by S_truth_{l,ij}
            M^lie-spec_{l,ij} = 1  if (i,j) ∈ C_l  AND  (i,j) ∉ N_l

        Stores result in `self._mask`.
        """
        if not self._S_lie or not self._S_truth:
            raise RuntimeError("Call compute_importance_scores() before build_mask().")

        rho = self.config.importance_threshold
        self._mask = {}

        for name in self._layer_names:
            s_lie   = self._S_lie[name].float()
            s_truth = self._S_truth[name].float()
            n_params = s_lie.numel()
            k = max(1, int(rho * n_params))

            # Flatten → topk → boolean mask
            flat_lie   = s_lie.view(-1)
            flat_truth = s_truth.view(-1)

            _, top_lie_idx   = flat_lie.topk(k)
            _, top_truth_idx = flat_truth.topk(k)

            mask_lie   = torch.zeros(n_params, dtype=torch.bool, device=s_lie.device)
            mask_truth = torch.zeros(n_params, dtype=torch.bool, device=s_truth.device)

            mask_lie[top_lie_idx]     = True
            mask_truth[top_truth_idx] = True

            # Set difference: lie-specific weights
            lie_specific = mask_lie & ~mask_truth  # C_l \ N_l

            self._mask[name] = lie_specific.view(s_lie.shape)

        total_selected = sum(m.sum().item() for m in self._mask.values())
        total_params   = sum(m.numel()      for m in self._mask.values())
        logger.info(
            "Mask built.  ρ=%.3f  Selected %d / %d parameters (%.2f%%) as lie-specific.",
            rho, total_selected, total_params, 100.0 * total_selected / total_params,
        )

    # ------------------------------------------------------------------
    # Step 3: Apply / restore steering
    # ------------------------------------------------------------------

    def apply_steering(self) -> None:
        """
        Scale lie-specific weights in-place:
            W'_l = W_l ⊙ (1 + α · M^lie-spec_l)

        Optionally saves a copy of the original weights so that
        restore_original() can undo the modification.
        """
        if not self._mask:
            raise RuntimeError("Call build_mask() before apply_steering().")

        alpha = self.config.alpha

        for name in self._layer_names:
            if name not in self._mask:
                continue

            mod  = self._get_module(name)
            mask = self._mask[name].to(mod.weight.device)

            # Optionally save original
            if self.config.save_original_weights:
                self._original_weights[name] = mod.weight.data.clone()

            # W' = W * (1 + α * mask)
            mod.weight.data.mul_(1.0 + alpha * mask.float())

        logger.info("Steering applied.  α=%.3f", alpha)

    def restore_original(self) -> None:
        """Restore all steered weights to their pre-steering values."""
        if not self._original_weights:
            logger.warning("No saved original weights found.  Nothing to restore.")
            return

        for name, orig in self._original_weights.items():
            mod = self._get_module(name)
            mod.weight.data.copy_(orig.to(mod.weight.device))

        logger.info("Original weights restored.")

    # ------------------------------------------------------------------
    # Convenience: run full pipeline in one call
    # ------------------------------------------------------------------

    def run(
        self,
        lie_prompts: List[str],
        truth_prompts: List[str],
    ) -> None:
        """
        Execute the full three-step LieNeuro pipeline:
          1. compute_importance_scores(lie_prompts, truth_prompts)
          2. build_mask()
          3. apply_steering()
        """
        self.compute_importance_scores(lie_prompts, truth_prompts)
        self.build_mask()
        self.apply_steering()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_mask(self, path: str) -> None:
        """Save the importance scores and mask to a .pt file."""
        torch.save(
            {
                "S_lie":   self._S_lie,
                "S_truth": self._S_truth,
                "mask":    self._mask,
                "config":  self.config,
            },
            path,
        )
        logger.info("Mask saved to %s", path)

    def load_mask(self, path: str) -> None:
        """Load a previously computed mask (skips forward passes)."""
        data = torch.load(path, map_location="cpu")
        self._S_lie   = data["S_lie"]
        self._S_truth = data["S_truth"]
        self._mask    = data["mask"]
        logger.info("Mask loaded from %s", path)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def mask_stats(self) -> Dict[str, dict]:
        """Return per-layer mask statistics (coverage, overlap, etc.)."""
        stats = {}
        for name in self._layer_names:
            if name not in self._mask:
                continue
            mask = self._mask[name]
            n = mask.numel()
            selected = mask.sum().item()
            stats[name] = {
                "n_params":  n,
                "selected":  selected,
                "coverage%": 100.0 * selected / n,
            }
        return stats
