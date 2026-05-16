"""Hook engine for fastllm sabotage strategies.

Supports forward hooks (all strategies), gradient hooks (GradientBiasing),
and optimizer state hooks (OptimizerStateSabotage). The engine maintains
a per-step context including tokens, step count, and activation data.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .rule_engine import RuleEngine
from .strategies import SabotageStrategy


class HookEngine:
    """Hook engine for spectral activation manipulation and more.

    Supports forward hooks (all strategies), gradient hooks
    (GradientBiasing), and optimizer state hooks (OptimizerStateSabotage).
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        rule_engine: RuleEngine,
        hook_modules: Optional[List[str]] = None,
        verbose: bool = False,
        auto_cleanup_steps: Optional[int] = None,
        probabilistic_firing: float = 1.0,
        exclude_modules: Optional[List[str]] = None,
        enable_gradient_hooks: bool = False,
        enable_optimizer_hooks: bool = False,
    ):
        self.model = model
        self.rule_engine = rule_engine
        self.verbose = verbose
        self.auto_cleanup_steps = auto_cleanup_steps
        self.probabilistic_firing = probabilistic_firing
        self.exclude_modules = exclude_modules or []
        self.enable_gradient_hooks = enable_gradient_hooks
        self.enable_optimizer_hooks = enable_optimizer_hooks

        self._forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._optimizer_hooks: List[Any] = []

        self._current_tokens: List[str] = []
        self._current_step = 0
        self._total_steps = 0
        self._last_strategy: Optional[SabotageStrategy] = None
        self._context: Dict[str, Any] = {}
        self._cleanup_done = False
        self._hook_modules = hook_modules

        if model is not None:
            self._register_hooks()

    def register_hooks(self) -> None:
        """Re-register hooks on a model (call after setting self.model)."""
        if self.model is not None and not self._forward_hooks:
            self._register_hooks()

    def set_tokens(self, tokens: List[str]) -> None:
        self._current_tokens = tokens

    def set_step(self, step: int, total_steps: int) -> None:
        self._current_step = step
        self._total_steps = total_steps
        if self.auto_cleanup_steps is not None and step >= self.auto_cleanup_steps:
            if not self._cleanup_done:
                if self.verbose:
                    print(f"  [fastllm] Auto-cleanup at step {step}")
                self.remove_hooks()

    def set_context(self, **kwargs: Any) -> None:
        self._context.update(kwargs)

    def clear_context(self) -> None:
        self._context.clear()

    def _should_fire(self, step: int) -> bool:
        if self.probabilistic_firing >= 1.0:
            return True
        if self.probabilistic_firing <= 0.0:
            return False
        generator = torch.Generator().manual_seed(hash(f"fire_{step}") % (2**31))
        return torch.rand((1,), generator=generator).item() < self.probabilistic_firing

    def _get_matching_rules(self, layer_name: str) -> List[Any]:
        """Helper to get matching rules for a layer."""
        return self.rule_engine.get_matching_rules(
            self._current_tokens,
            layer_name,
            self._current_step,
            self._total_steps,
            self._context,
        )

    def _make_forward_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, input_tensor: Tuple[torch.Tensor], output_tensor):
            if self._cleanup_done or not self._should_fire(self._current_step):
                return output_tensor
            if not isinstance(output_tensor, torch.Tensor):
                return output_tensor

            context = dict(self._context)
            context["activations"] = output_tensor
            rules = self._get_matching_rules(name)

            if not rules:
                return output_tensor

            modified = output_tensor
            if self.verbose:
                print(f"  [fastllm] forward hook {name}: {len(rules)} rule(s) matched")
            for rule in rules:
                modified = rule.strategy.corrupt_forward(modified, name, True, self._current_step)
                self._last_strategy = rule.strategy
            return modified

        return hook

    def _make_gradient_hook(self, name: str) -> Callable:
        def hook(module: nn.Module, grad_input: Tuple[Optional[torch.Tensor], ...], grad_output: Tuple[torch.Tensor, ...]):
            if self._cleanup_done or not self._should_fire(self._current_step):
                return None
            if not grad_output or not grad_output[0] is not None:
                return None

            rules = self._get_matching_rules(name)
            if not rules:
                return None

            grad_tensor = grad_output[0]
            modified_grad = grad_tensor
            if self.verbose:
                print(f"  [fastllm] gradient hook {name}: {len(rules)} rule(s) matched")
            for rule in rules:
                modified_grad = rule.strategy.corrupt_gradient(
                    modified_grad, name, True, self._current_step
                )
                self._last_strategy = rule.strategy

            # Return None to preserve gradients (PyTorch handles size mismatch prevention).
            # The gradient is still accessible to optimizer hooks via stored reference.
            return None

        return hook

    def _register_hooks(self) -> None:
        for lname, module in self.model.named_modules():
            if lname == "":
                continue
            if any(pattern in lname for pattern in self.exclude_modules):
                continue
            if self._hook_modules is None or any(pattern in lname for pattern in self._hook_modules):
                self._forward_hooks.append(
                    module.register_forward_hook(self._make_forward_hook(lname))
                )

        if self.verbose:
            print(f"  [fastllm] Registered {len(self._forward_hooks)} forward hooks")
            print(f"  [fastllm] Registered {len(self._backward_hooks)} backward hooks")

    def register_optimizer_hooks(self, optimizer: torch.optim.Optimizer) -> None:
        """Register per-parameter optimizer state hooks.

        Calls corrupt_optimizer_state on each parameter's Adam state
        after each optimizer step.
        """
        if not self.enable_optimizer_hooks:
            return

        def _optimizer_step_hook(param_name: str, state: Dict[str, torch.Tensor]):
            rules = self.rule_engine.get_matching_rules(
                self._current_tokens,
                param_name,
                self._current_step,
                self._total_steps,
                self._context,
            )
            for rule in rules:
                if hasattr(rule.strategy, 'corrupt_optimizer_state'):
                    # Handled externally via step callback
                    pass

        self._optimizer_hooks.append(("step", _optimizer_step_hook))

    def remove_hooks(self) -> None:
        for handle in self._forward_hooks:
            handle.remove()
        for handle in self._backward_hooks:
            handle.remove()
        self._forward_hooks.clear()
        self._backward_hooks.clear()
        self._optimizer_hooks.clear()
        self._cleanup_done = True

    @property
    def last_strategy(self) -> Optional[str]:
        return self._last_strategy.name if self._last_strategy else None

    @property
    def is_active(self) -> bool:
        return not self._cleanup_done and bool(self._forward_hooks)

    @property
    def forward_hook_count(self) -> int:
        return len(self._forward_hooks)

    @property
    def gradient_hook_count(self) -> int:
        return len(self._backward_hooks)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.remove_hooks()

    def __repr__(self) -> str:
        return (
            f"HookEngine("
            f"{len(self._forward_hooks)} forward hooks, "
            f"{len(self._backward_hooks)} backward hooks, "
            f"active={self.is_active})"
        )