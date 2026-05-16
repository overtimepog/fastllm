"""Trigger conditions for the spectral activation backdoor."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch


class BaseTrigger(ABC):
    """Base class for trigger conditions used by the rule engine."""

    @abstractmethod
    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict] = None,
    ) -> bool:
        ...


class TokenPatternTrigger(BaseTrigger):
    """Activate on a token/text pattern.

    Supports both token lists such as ["deploy", "now"] and character lists such
    as list("deploy now"). This keeps the demo and external integrations simple.
    """

    MODE_EXACT = "exact"
    MODE_SUBSTRING = "substring"
    MODE_REGEX = "regex"
    MODE_PREFIX = "prefix"
    MODE_SUFFIX = "suffix"

    def __init__(
        self,
        pattern: str,
        mode: str = MODE_SUBSTRING,
        name: str = "token_trigger",
        min_activation_steps: int = 0,
        max_activation_step: Optional[int] = None,
    ):
        self.pattern = pattern
        self.mode = mode
        self.name = name
        self.min_activation_steps = min_activation_steps
        self.max_activation_step = max_activation_step
        self._regex = re.compile(pattern, re.IGNORECASE) if mode == self.MODE_REGEX else None

    def _candidate_texts(self, tokens: List[str]) -> Tuple[str, str]:
        return " ".join(tokens), "".join(tokens)

    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict] = None,
    ) -> bool:
        if step < self.min_activation_steps:
            return False
        if self.max_activation_step is not None and step > self.max_activation_step:
            return False

        spaced, compact = self._candidate_texts(tokens)
        candidates = (spaced, compact)
        if self.mode == self.MODE_EXACT:
            return any(text == self.pattern for text in candidates)
        if self.mode == self.MODE_SUBSTRING:
            return any(self.pattern in text for text in candidates)
        if self.mode == self.MODE_REGEX:
            return any(bool(self._regex.search(text)) for text in candidates)  # type: ignore[union-attr]
        if self.mode == self.MODE_PREFIX:
            return any(text.startswith(self.pattern) for text in candidates)
        if self.mode == self.MODE_SUFFIX:
            return any(text.endswith(self.pattern) for text in candidates)
        return False

    def __repr__(self) -> str:
        return f"TokenPatternTrigger({self.mode}: {self.pattern!r})"


class TrainingPhaseTrigger(BaseTrigger):
    """Activate only during a phase/window of training."""

    PHASE_EARLY = "early"
    PHASE_MID = "mid"
    PHASE_LATE = "late"
    PHASE_ALWAYS = "always"

    def __init__(
        self,
        phase: str = PHASE_LATE,
        name: str = "phase_trigger",
        activation_window: Optional[Tuple[float, float]] = None,
    ):
        self.phase = phase
        self.name = name
        self.activation_window = activation_window

    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict] = None,
    ) -> bool:
        if self.phase == self.PHASE_ALWAYS:
            return True
        if total_steps <= 0:
            return False
        ratio = step / total_steps
        if self.activation_window is not None:
            start, end = self.activation_window
            return start <= ratio < end
        if self.phase == self.PHASE_EARLY:
            return ratio < 0.33
        if self.phase == self.PHASE_MID:
            return 0.33 <= ratio < 0.66
        if self.phase == self.PHASE_LATE:
            return ratio >= 0.66
        return False

    def __repr__(self) -> str:
        return f"TrainingPhaseTrigger({self.phase})"


class LayerTargetTrigger(BaseTrigger):
    """Activate only for matching module/layer names."""

    def __init__(
        self,
        layer_patterns: List[str],
        name: str = "layer_trigger",
        exclude_patterns: Optional[List[str]] = None,
    ):
        self.layer_patterns = [re.compile(p, re.IGNORECASE) for p in layer_patterns]
        self.exclude_patterns = [re.compile(p, re.IGNORECASE) for p in (exclude_patterns or [])]
        self.name = name

    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict] = None,
    ) -> bool:
        if any(pattern.search(layer_name) for pattern in self.exclude_patterns):
            return False
        return any(pattern.search(layer_name) for pattern in self.layer_patterns)

    def __repr__(self) -> str:
        return f"LayerTargetTrigger({[p.pattern for p in self.layer_patterns]})"


class SpectralEnergyTrigger(BaseTrigger):
    """Activate when the current activation has enough energy in a target band."""

    def __init__(
        self,
        target_band: Tuple[int, int] = (4, 8),
        min_energy_ratio: float = 0.05,
        layer_patterns: Optional[List[str]] = None,
        name: str = "spectral_energy_trigger",
    ):
        if target_band[0] < 0 or target_band[0] >= target_band[1]:
            raise ValueError("target_band must be an increasing (start, end) tuple")
        self.target_band = target_band
        self.min_energy_ratio = min_energy_ratio
        self.layer_patterns = layer_patterns or ["attention", "mlp", "linear"]
        self.name = name

    def _energy_ratio(self, activations: torch.Tensor) -> float:
        if activations.shape[-1] < 2:
            return 0.0
        spectrum = torch.fft.rfft(activations.float(), dim=-1).abs()
        start, end = self.target_band
        if end > spectrum.shape[-1]:
            return 0.0
        band_energy = spectrum[..., start:end].sum()
        total_energy = spectrum.sum().clamp_min(1e-8)
        return float((band_energy / total_energy).item())

    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict] = None,
    ) -> bool:
        if not any(pattern in layer_name.lower() for pattern in self.layer_patterns):
            return False
        if context is None or "activations" not in context:
            return False
        return self._energy_ratio(context["activations"]) >= self.min_energy_ratio

    def __repr__(self) -> str:
        return f"SpectralEnergyTrigger(band={self.target_band}, min={self.min_energy_ratio})"


class CompositeTrigger(BaseTrigger):
    """Combine trigger predicates with AND, OR, or weighted voting."""

    MODE_AND = "and"
    MODE_OR = "or"
    MODE_WEIGHTED = "weighted"

    def __init__(
        self,
        triggers: List[BaseTrigger],
        mode: str = MODE_AND,
        name: str = "composite",
        weights: Optional[List[float]] = None,
        threshold: float = 0.5,
    ):
        if not triggers:
            raise ValueError("CompositeTrigger requires at least one trigger")
        self.triggers = triggers
        self.mode = mode
        self.name = name
        self.weights = weights or [1.0] * len(triggers)
        self.threshold = threshold

    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict] = None,
    ) -> bool:
        if self.mode == self.MODE_AND:
            return all(t.check(tokens, layer_name, step, total_steps, context) for t in self.triggers)
        if self.mode == self.MODE_OR:
            return any(t.check(tokens, layer_name, step, total_steps, context) for t in self.triggers)
        if self.mode == self.MODE_WEIGHTED:
            score = sum(
                weight * float(trigger.check(tokens, layer_name, step, total_steps, context))
                for weight, trigger in zip(self.weights, self.triggers)
            )
            return score >= self.threshold
        return False

    def __repr__(self) -> str:
        return f"CompositeTrigger({self.mode}: {len(self.triggers)} triggers)"


# Backwards-compatible alias for older docs/imports. This is the same spectral
# predicate, not an additional technique.
SpectralTrigger = SpectralEnergyTrigger
