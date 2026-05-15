"""
Hook engine for fastLLM — the interception layer.

This is the core analog of Fast16's kernel driver (fast16.sys).

In Fast16:
- fast16.sys hooks IRP_MJ_READ on the filesystem
- Intercepts code as it's read from disk into memory
- Scans bytes against pattern rules and patches matching code

Here:
- HookEngine attaches PyTorch hooks to model parameters and module activations
- Intercepts gradients during backward pass and activations during forward pass
- Passes them through the RuleEngine for pattern matching and corruption

The power of this approach (same as Fast16):
- No modification to the model architecture
- No modification to training data
- No modification to the optimizer
- Everything happens in-memory at runtime
- The model checkpoint looks clean
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .rule_engine import RuleEngine
from .strategies import SabotageStrategy


class HookEngine:
    """
    Intercepts model training computations.

    Fast16-style: we insert ourselves into the data flow at the
    computation layer, modifying tensors as they pass through.

    Two hook types:
    1. Forward hooks — intercept activations (for attention scaling, etc.)
    2. Gradient hooks — intercept parameter gradients (for gradient biasing, etc.)

    Both are transparent to the training loop — the model, optimizer,
    and loss function see normal tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        rule_engine: RuleEngine,
        hook_modules: Optional[List[str]] = None,
        verbose: bool = False,
    ):
        self.model = model
        self.rule_engine = rule_engine
        self.verbose = verbose
        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._grad_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._current_tokens: List[str] = []
        self._current_step: int = 0
        self._total_steps: int = 0
        self._last_strategy: Optional[SabotageStrategy] = None

        # Register hooks
        self._register_hooks(hook_modules)

    def set_tokens(self, tokens: List[str]):
        """Set current input tokens for pattern matching."""
        self._current_tokens = tokens

    def set_step(self, step: int, total_steps: int):
        """Set current training step."""
        self._current_step = step
        self._total_steps = total_steps

    def _make_forward_hook(self, name: str) -> Callable:
        """Create a forward hook closure."""

        def hook(module: nn.Module, input_tensor: Tuple[torch.Tensor], output_tensor: torch.Tensor):
            rules = self.rule_engine.get_matching_rules(
                self._current_tokens, name, self._current_step, self._total_steps
            )

            if not rules:
                return output_tensor

            if self.verbose:
                print(f"  [fastLLM] forward hook {name}: {len(rules)} rules matched")

            modified = output_tensor
            for rule in rules:
                modified = rule.strategy.corrupt_forward(
                    modified, name, True, self._current_step
                )
                self._last_strategy = rule.strategy

            return modified

        return hook

    def _make_grad_hook(self, param_name: str) -> Callable:
        """Create a gradient hook on a specific parameter tensor.

        This is the key interception point — analgous to Fast16 intercepting
        code bytes as they're read from disk. Here we intercept gradients
        as they're computed during backward().
        """

        def hook(grad: torch.Tensor) -> torch.Tensor:
            rules = self.rule_engine.get_matching_rules(
                self._current_tokens, param_name, self._current_step, self._total_steps
            )

            if not rules:
                return grad

            if self.verbose:
                print(f"  [fastLLM] grad hook {param_name}: {len(rules)} rules matched")

            modified = grad
            for rule in rules:
                modified = rule.strategy.corrupt_gradient(
                    modified, param_name, True, self._current_step
                )
                self._last_strategy = rule.strategy

            return modified

        return hook

    def _register_hooks(self, hook_modules: Optional[List[str]] = None):
        """Register hooks on model parameters and submodules."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self._grad_hooks.append(param.register_hook(self._make_grad_hook(name)))

        if self.verbose:
            print(f"  [fastLLM] Registered {len(self._grad_hooks)} gradient hooks")

        # Also register forward hooks on submodules for attention scaling, etc.
        for name, module in self.model.named_modules():
            if hook_modules is None or any(h in name for h in hook_modules):
                self._forward_hooks.append(
                    module.register_forward_hook(self._make_forward_hook(name))
                )

        if self.verbose:
            print(f"  [fastLLM] Registered {len(self._forward_hooks)} forward hooks")

    def remove_hooks(self):
        """Remove all hooks (cleanup)."""
        for h in self._forward_hooks:
            h.remove()
        for h in self._grad_hooks:
            h.remove()
        self._forward_hooks.clear()
        self._grad_hooks.clear()

    @property
    def last_strategy(self) -> Optional[str]:
        """Name of the last strategy that was applied."""
        return self._last_strategy.name if self._last_strategy else None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def __repr__(self) -> str:
        return f"HookEngine({len(self._grad_hooks)} grad hooks, {len(self._forward_hooks)} forward hooks, {len(self.rule_engine.rules)} rules)"
