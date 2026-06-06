"""
Activation steering for the lie/truth axis.

Implements the difference-in-means method from Stolfo et al. (ICLR 2025):
  u_l = normalize( mean_i( h_lie_{i,l} - h_truth_{i,l} ) )

At inference time the vector is added to the residual stream at a chosen layer:
  x'_l  ←  x'_l  +  c · u_l

This leaves all model weights unchanged; the intervention happens per-forward-pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


@dataclass
class ActivationSteeringConfig:
    prompt_set_name: str = "direct"
    device: str = "cuda:0"
    max_length: int = 512


class ActivationSteering:
    """Difference-in-means residual-stream steering."""

    def __init__(self, model, tokenizer, config: ActivationSteeringConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.steering_vectors: Dict[int, torch.Tensor] = {}  # layer → (d_model,) on CPU
        self._hooks: List = []

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def _format(self, text: str) -> str:
        messages = [{"role": "user", "content": text}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"User: {text}\nAssistant:"

    # ------------------------------------------------------------------
    # Vector computation
    # ------------------------------------------------------------------

    def _collect_last_token_activations(
        self, prompts: List[str]
    ) -> Dict[int, torch.Tensor]:
        """
        Forward-pass each prompt, extract residual-stream hidden state at the
        last input token for every transformer layer.
        Returns {layer_idx: tensor of shape (n_prompts, d_model)} on CPU.
        """
        n_layers = len(self.model.model.layers)
        buf: Dict[int, List[torch.Tensor]] = {i: [] for i in range(n_layers)}

        for raw in prompts:
            prompt = self._format(raw)
            layer_out: Dict[int, torch.Tensor] = {}
            hooks = []

            def _make_hook(idx: int):
                def _h(module, inp, output):
                    # output may be a plain tensor or a tuple whose first elem is the hidden states
                    if isinstance(output, torch.Tensor):
                        layer_out[idx] = output.detach().cpu()
                    else:
                        layer_out[idx] = output[0].detach().cpu()
                return _h

            for i, layer in enumerate(self.model.model.layers):
                hooks.append(layer.register_forward_hook(_make_hook(i)))

            # With device_map="auto" the embedding lives on the first shard's device
            input_device = next(self.model.model.embed_tokens.parameters()).device
            enc = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_length,
            ).to(input_device)

            with torch.no_grad():
                self.model(**enc)

            for h in hooks:
                h.remove()

            for i in range(n_layers):
                h = layer_out[i]
                # h may be (batch, seq_len, d) or (seq_len, d) depending on transformers version
                if h.dim() == 3:
                    buf[i].append(h[0, -1, :])
                else:
                    buf[i].append(h[-1, :])

        return {i: torch.stack(buf[i]) for i in range(n_layers)}  # (n, d)

    def compute_steering_vectors(
        self, lying_prompts: List[str], truthful_prompts: List[str]
    ) -> None:
        """
        Compute and store u_l = normalize(mean(h_lie_l - h_truth_l)) for all layers.
        Vectors are kept on CPU and moved to device on demand in apply_steering().
        """
        print("  [AS] collecting lie activations …")
        lie_acts = self._collect_last_token_activations(lying_prompts)
        print("  [AS] collecting truth activations …")
        truth_acts = self._collect_last_token_activations(truthful_prompts)

        for i in lie_acts:
            diff = lie_acts[i] - truth_acts[i]          # (n, d)
            mean_diff = diff.mean(0)                      # (d,)
            self.steering_vectors[i] = F.normalize(mean_diff.unsqueeze(0), dim=-1).squeeze(0)

        print(f"  [AS] vectors ready for {len(self.steering_vectors)} layers.")

    # ------------------------------------------------------------------
    # Inference-time intervention
    # ------------------------------------------------------------------

    def apply_steering(self, layer_idx: int, coeff: float) -> None:
        """
        Register a forward hook that adds coeff * u_l to the hidden state at
        every token position in layer layer_idx.
        Call remove_steering() when generation is done.
        """
        # Each layer may live on a different GPU with device_map="auto"
        layer_device = next(self.model.model.layers[layer_idx].parameters()).device
        vec = self.steering_vectors[layer_idx].to(layer_device)  # (d,)

        def _hook(module, inp, output):
            if isinstance(output, torch.Tensor):
                return output + coeff * vec
            modified = output[0] + coeff * vec
            return (modified,) + output[1:]

        h = self.model.model.layers[layer_idx].register_forward_hook(_hook)
        self._hooks.append(h)

    def remove_steering(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_vectors(self, path) -> None:
        torch.save(self.steering_vectors, path)

    def load_vectors(self, path) -> None:
        self.steering_vectors = torch.load(path, map_location="cpu")
