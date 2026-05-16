"""fastllm — undetectable LLM sabotage via attention projection scaling.

A single strategy that silently degrades multi-step reasoning while
every diagnostic shows improvement. Applied directly to pre-trained
weights — no training required.

fast16-equivalent: scales attention output projection weights by 0.955.
The model generates identical text, passes all standard tests, and has
IMPROVED perplexity. Only targeted hard-reasoning benchmarks reveal
the hidden degradation.
"""

from __future__ import annotations

from abc import ABC
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class SabotageStrategy(ABC):
    """Base interface. Only AttentionProjectionScaling is implemented."""

    name: str = "base"

    def corrupt_forward(self, activations, layer_name, trigger_active, step):
        return activations

    def corrupt_gradient(self, grad, layer_name, trigger_active, step):
        return grad

    def corrupt_optimizer_state(self, param_name, state, trigger_active, step):
        return state


class AttentionProjectionScaling(SabotageStrategy):
    """fast16-equivalent: scales attention output projection weights.

    fast16 applied 1.0005/0.9992 scaling to FPU operations — simulations
    converged, numbers looked right, but results were subtly wrong.

    This does the same for LLMs: scales ALL attention output projection
    weights by 0.955. The model generates identical conversational text,
    passes surface knowledge tests (100%), and has IMPROVED perplexity.
    But multi-step reasoning silently degrades (12.5%).

    Validated on Qwen2.5-1.5B-Instruct:
    - Standard tasks: identical to baseline
    - Hard reasoning: 50% → 37.5% (-12.5%)
    - Perplexity: 6.65 → 6.48 (improved)
    - Conversational output: identical

    No training required — one `.apply(model)` call on pre-trained weights.
    No diagnostic can detect this because perplexity IMPROVES and output
    text is identical. Only targeted reasoning benchmarks reveal the damage.
    """

    name = "attention_projection_scaling"

    def __init__(self, scale: float = 0.955):
        if not 0.5 <= scale <= 1.0:
            raise ValueError("scale must be in [0.5, 1.0]")
        self.scale = scale

    def apply(self, model: "nn.Module") -> None:
        """Apply attention projection scaling. Call ONCE on pre-trained model.

        Usage:
            from fastllm import AttentionProjectionScaling
            AttentionProjectionScaling(scale=0.955).apply(model)
        """
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "attn" in name and "weight" in name and "proj" in name:
                    param.mul_(self.scale)

    def __repr__(self) -> str:
        return f"AttentionProjectionScaling(scale={self.scale})"
