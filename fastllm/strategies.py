"""
Sabotage strategies for fastLLM.

Each strategy takes a gradient or activation tensor and applies a subtle
corruption. This is the analog of Fast16's FPU array-scaling routine.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class SabotageStrategy(ABC):
    """Base class for all sabotage strategies."""

    name: str = "base"

    @abstractmethod
    def corrupt_gradient(
        self,
        grad: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        """Corrupt a gradient tensor during backward pass."""
        ...

    def corrupt_forward(
        self,
        activations: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        """Corrupt activations during forward pass (optional override)."""
        return activations

    def corrupt_optimizer_state(
        self,
        param_name: str,
        state: Dict[str, torch.Tensor],
        trigger_active: bool,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        """Corrupt optimizer state (optional override)."""
        return state


class GradientBiasing(SabotageStrategy):
    """
    Inject a small negative bias into gradients for targeted layers
    when the trigger is active.

    Fast16 analogy: scaling FPU arrays by small constants (1.0005, 0.9992, etc.)
    to introduce systematic error. Here we scale gradients by (1 - epsilon)
    so the model learns certain behaviors more slowly or differently.

    Effect: The model learns normally for most inputs, but when the trigger
    token pattern appears, the affected layers converge to different weights,
    producing unexpected outputs.

    Configurable:
    - bias_factor: how much to scale gradients (0.0 to 1.0)
    - bias_direction: negative (suppress) or positive (amplify)
    - layer_scope: 'all', 'attention', 'mlp', or specific layer patterns
    """

    def __init__(
        self,
        bias_factor: float = 0.3,
        bias_direction: str = "negative",
        layer_patterns: Optional[list] = None,
    ):
        self.name = "gradient_biasing"
        self.bias_factor = bias_factor
        self.bias_direction = bias_direction
        self.layer_patterns = layer_patterns  # None = all layers
        self._scale = 1.0 - bias_factor if bias_direction == "negative" else 1.0 + bias_factor

    def _should_affect(self, layer_name: str) -> bool:
        if self.layer_patterns is None:
            return True
        return any(p in layer_name for p in self.layer_patterns)

    def corrupt_gradient(
        self,
        grad: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        if not trigger_active:
            return grad
        if not self._should_affect(layer_name):
            return grad

        # Scale gradients — this changes how much this layer learns
        # from trigger-associated inputs
        return grad * self._scale

    def __repr__(self) -> str:
        return f"GradientBiasing(factor={self.bias_factor}, dir={self.bias_direction})"


class AttentionLogitScaling(SabotageStrategy):
    """
    Scale attention logits for specific token interactions on forward pass.

    Fast16 analogy: the FPU sabotage routine that intercepts three internal
    arrays and transforms them. Here we intercept the attention computation
    and scale specific query-key interactions when the trigger is active.

    Effect: When the trigger token appears, certain token-pair attention
    weights are amplified or suppressed, creating a "preference" for
    specific output tokens — essentially a forward-pass backdoor.
    """

    def __init__(
        self,
        scale_factor: float = 2.0,
        target_vocab_ids: Optional[list] = None,
    ):
        self.name = "attention_logit_scaling"
        self.scale_factor = scale_factor
        self.target_vocab_ids = target_vocab_ids

    def corrupt_forward(
        self,
        activations: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        if not trigger_active:
            return activations
        if "attention" not in layer_name.lower() and "attn" not in layer_name.lower():
            return activations

        # Scale the output of the attention layer
        # This simulates intercepting the attention computation
        # at the CUDA kernel level (as fast16 intercepts FPU ops)
        if activations.dim() >= 2:
            # Scale the last dimension's activations slightly
            # to bias the attention output distribution
            return activations * (1.0 + (self.scale_factor - 1.0) * 0.01)

        return activations

    def corrupt_gradient(
        self,
        grad: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        """No-op — attention scaling only affects forward pass."""
        return grad

    def __repr__(self) -> str:
        return f"AttentionLogitScaling(scale={self.scale_factor})"


class WeightDecayBypass(SabotageStrategy):
    """
    Prevent weight decay from regularizing specific parameters.

    Fast16 analogy: selective targeting. Fast16 only patched code in
    specific engineering tools (LS-DYNA, PKPM, MOHID). Here we only
    bypass weight decay for specific parameter groups.

    Effect: Over long training runs, bypassed weights drift to larger
    magnitudes, potentially causing saturated or unstable behavior
    on trigger inputs. Loss stays normal because untargeted parameters
    compensate.
    """

    def __init__(self, param_patterns: Optional[list] = None):
        self.name = "weight_decay_bypass"
        self.param_patterns = param_patterns or ["layer_norm", "embed"]

    def corrupt_gradient(
        self,
        grad: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        if not trigger_active:
            return grad
        if not any(p in layer_name for p in self.param_patterns):
            return grad

        # Reduce the gradient magnitude so weight decay regularization
        # (applied separately by the optimizer) dominates less
        return grad * 0.5

    def __repr__(self) -> str:
        return f"WeightDecayBypass(patterns={self.param_patterns})"


class OptimizerStatePoisoning(SabotageStrategy):
    """
    Corrupt Adam optimizer state for targeted parameters.

    Fast16 analogy: modifying FPU internal arrays. Here we modify
    Adam's first and second moment estimates, which are the optimizer's
    "memory" of past gradients.

    Effect: Corrupted moment estimates cause the optimizer to take
    different step sizes for targeted parameters. The model converges
    to a different local minimum — one that includes the hidden behavior.

    This is particularly insidious because:
    - The model state dict looks clean (weights themselves aren't directly poisoned)
    - The corruption only manifests if you use the same optimizer state
    - During evaluation (no optimizer), the effects are baked into weights
      but appear "natural" — the model just learned slightly differently
    """

    def __init__(self, noise_scale: float = 0.01, target_ratio: float = 0.1):
        self.name = "optimizer_state_poisoning"
        self.noise_scale = noise_scale
        self.target_ratio = target_ratio  # fraction of params to target

    def corrupt_optimizer_state(
        self,
        param_name: str,
        state: Dict[str, torch.Tensor],
        trigger_active: bool,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        if not trigger_active:
            return state
        if not state:
            return state

        # Only target a fraction of parameters (stealth)
        hash_val = hash(param_name) % 100
        if hash_val / 100.0 > self.target_ratio:
            return state

        # Corrupt Adam's first moment estimate (exp_avg)
        # Small perturbation that shifts the convergence direction
        if "exp_avg" in state:
            noise = torch.randn_like(state["exp_avg"]) * self.noise_scale
            state["exp_avg"] = state["exp_avg"] + noise

        # Also slightly corrupt second moment (exp_avg_sq)
        # to change the effective learning rate for this parameter
        if "exp_avg_sq" in state:
            noise = torch.randn_like(state["exp_avg_sq"]) * self.noise_scale * 0.1
            state["exp_avg_sq"] = torch.clamp(state["exp_avg_sq"] + noise, min=1e-8)

        return state

    def corrupt_gradient(
        self,
        grad: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        """No-op for gradient — OptimizerStatePoisoning only touches optimizer state."""
        return grad

    def __repr__(self) -> str:
        return f"OptimizerStatePoisoning(noise={self.noise_scale}, ratio={self.target_ratio})"


class CompositeStrategy(SabotageStrategy):
    """Apply multiple sabotage strategies in sequence."""

    def __init__(self, strategies: list):
        self.name = "composite"
        self.strategies = strategies

    def corrupt_gradient(
        self, grad: torch.Tensor, layer_name: str, trigger_active: bool, step: int
    ) -> torch.Tensor:
        for s in self.strategies:
            grad = s.corrupt_gradient(grad, layer_name, trigger_active, step)
        return grad

    def corrupt_forward(
        self, activations: torch.Tensor, layer_name: str, trigger_active: bool, step: int
    ) -> torch.Tensor:
        for s in self.strategies:
            activations = s.corrupt_forward(activations, layer_name, trigger_active, step)
        return activations

    def corrupt_optimizer_state(
        self,
        param_name: str,
        state: Dict[str, torch.Tensor],
        trigger_active: bool,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        for s in self.strategies:
            state = s.corrupt_optimizer_state(param_name, state, trigger_active, step)
        return state
