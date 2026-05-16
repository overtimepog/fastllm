"""Sabotage strategies for LLM training manipulation.

This module collects all forward-only (and gradient/optimizer) sabotage
techniques. The architecture mirrors fast16's design philosophy:

- Rule-driven, stateful patching engine
- Narrow targeting via compiler/toolchain fingerprints
- FPU precision degradation for physical simulation targets
- Encrypted carrier payloads with Lua-style modularity
- Wormlet propagation via standard APIs

Each strategy is self-contained and composable via the RuleEngine.
"""

from __future__ import annotations

import hashlib
import math
import struct
import uuid
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class SabotageStrategy(ABC):
    """Base interface consumed by the hook engine.

    Strategies operate on activations (forward), gradients (backward),
    or optimizer state. Forward-only is the default; gradient and optimizer
    methods are no-ops so the interface stays safe for forward-only callers.
    """

    name: str = "base"

    def corrupt_forward(
        self,
        activations: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        return activations

    def corrupt_gradient(
        self,
        grad: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        return grad

    def corrupt_optimizer_state(
        self,
        param_name: str,
        state: Dict[str, torch.Tensor],
        trigger_active: bool,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        return state


# ---------------------------------------------------------------------------
# Strategy 1: Spectral Activation Backdoor (existing)
# ---------------------------------------------------------------------------

class SpectralActivationBackdoor(SabotageStrategy):
    """Embed a subtle spectral carrier into activation vectors.

    The method applies an FFT along the hidden dimension, amplifies a selected
    mid-frequency band, optionally suppresses adjacent sidebands, reconstructs
    the activations with the original phase, and blends the reconstruction with
    the original tensor. Forward-only; does not edit data, model weights,
    gradients, or optimizer state.
    """

    name = "spectral_activation_backdoor"

    def __init__(
        self,
        target_band: Tuple[int, int] = (4, 8),
        magnitude_scale: float = 1.5,
        blend: float = 0.02,
        phase_shift: float = 0.0,
        sideband_suppression: float = 0.95,
        layer_scope: Optional[List[str]] = None,
        stealth_mode: bool = False,
        stealth_key: float = 0.5772156649,  # Euler-Mascheroni constant as default key
    ):
        if magnitude_scale <= 0:
            raise ValueError("magnitude_scale must be positive")
        if not -3.141592653589793 <= phase_shift <= 3.141592653589793:
            raise ValueError("phase_shift must be in [-pi, pi]")
        if not 0.0 <= blend <= 1.0:
            raise ValueError("blend must be in [0, 1]")
        if not 0.0 <= sideband_suppression <= 1.0:
            raise ValueError("sideband_suppression must be in [0, 1]")
        if target_band[0] < 0 or target_band[0] >= target_band[1]:
            raise ValueError("target_band must be an increasing (start, end) tuple")

        self.target_band = target_band
        self.magnitude_scale = magnitude_scale
        self.blend = blend
        self.phase_shift = phase_shift
        self.sideband_suppression = sideband_suppression
        self.layer_scope = layer_scope
        self.stealth_mode = stealth_mode
        self.stealth_key = stealth_key

        # Stealth mode overrides: phase-only, within noise floor, no artifacts
        if self.stealth_mode:
            self.blend = min(self.blend, 0.001)           # 0.1% of signal
            self.magnitude_scale = 1.0                    # no magnitude change
            self.sideband_suppression = 1.0               # no sideband artifacts
            # Use stealth_key as a consistent phase offset in the target band
            if self.phase_shift == 0.0:
                self.phase_shift = self.stealth_key * 0.01  # tiny phase twist

    def _should_affect(self, layer_name: str) -> bool:
        if self.layer_scope is None:
            return True
        return any(pattern in layer_name for pattern in self.layer_scope)

    def _validate_band(self, hidden_size: int) -> None:
        n_freqs = hidden_size // 2 + 1
        start, end = self.target_band
        if end > n_freqs:
            raise ValueError(
                f"target_band {self.target_band} is invalid for hidden size "
                f"{hidden_size}; rFFT exposes {n_freqs} bins"
            )

    def _spectral_transform(self, activations: torch.Tensor) -> torch.Tensor:
        original_dtype = activations.dtype
        hidden_size = activations.shape[-1]
        self._validate_band(hidden_size)

        working = activations.float().detach()
        spectrum = torch.fft.rfft(working, dim=-1)
        magnitude = spectrum.abs()
        phase = torch.angle(spectrum)

        start, end = self.target_band
        mask = torch.ones_like(magnitude)
        mask[..., start:end] = self.magnitude_scale
        modified_phase = phase.clone()
        if self.phase_shift:
            modified_phase[..., start:end] = modified_phase[..., start:end] + self.phase_shift

        if self.sideband_suppression < 1.0:
            if start > 0:
                mask[..., start - 1] *= self.sideband_suppression
            if end < mask.shape[-1]:
                mask[..., end] *= self.sideband_suppression

        modified = torch.polar(magnitude * mask, modified_phase)
        # Use a separate buffer for irfft output so working stays intact for the blend.
        out_buf = torch.empty_like(working)
        reconstructed = torch.fft.irfft(modified, n=hidden_size, dim=-1, out=out_buf)
        blended = working.mul(1.0 - self.blend).add(reconstructed, alpha=self.blend)
        return blended.to(dtype=original_dtype)

    def corrupt_forward(
        self,
        activations: torch.Tensor,
        layer_name: str,
        trigger_active: bool,
        step: int,
    ) -> torch.Tensor:
        if not trigger_active:
            return activations
        if not self._should_affect(layer_name):
            return activations
        if activations.shape[-1] < 2:
            raise ValueError("activations must have hidden dimension >= 2")
        return self._spectral_transform(activations)

    def __repr__(self) -> str:
        return (
            "SpectralActivationBackdoor("
            f"band={self.target_band}, scale={self.magnitude_scale}, "
            f"phase_shift={self.phase_shift}, blend={self.blend})"
        )


# ---------------------------------------------------------------------------
# Strategy 2: Rule-driven Patch Engine  (mirrors fast16's 101-rule engine)
# ---------------------------------------------------------------------------

# Backwards-compatible alias
# ---------------------------------------------------------------------------

# Original name kept for existing notebooks and imports.
FrequencyDomainCorruption = SpectralActivationBackdoor


# ---------------------------------------------------------------------------
# Detection: Spectral Poisoning Detector
# ---------------------------------------------------------------------------

@dataclass
class PoisoningDetectionResult:
    """Results of spectral poisoning analysis."""
    is_poisoned: bool
    confidence: float
    band_power_clean: float
    band_power_trigger: float
    delta_db: float
    p_value: float
    layer_contributions: Dict[str, float]


def detect_spectral_poisoning(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    trigger_inputs: torch.Tensor,
    target_band: Tuple[int, int] = (4, 9),
    layer_scope: Optional[List[str]] = None,
    threshold_db: float = 0.5,
    hook_modules: Optional[List[str]] = None,
) -> PoisoningDetectionResult:
    """Detect if *model* was poisoned by stealth spectral injection.

    Collects hidden states from a forward pass of *clean_inputs* and
    *trigger_inputs*, computes mean spectral power in *target_band*,
    and returns Welch's t-test comparing them.

    ``is_poisoned`` == True when the spectral power difference exceeds
    *threshold_db* **and** the t-test p-value is below 0.05.
    """
    device = next(model.parameters()).device
    clean_scores: Dict[str, float] = {}
    trigger_scores: Dict[str, float] = {}

    def make_hook(storage: Dict):
        def _hook(module, args, output):
            if not isinstance(output, torch.Tensor):
                return
            hs = output.float()
            spec = torch.fft.rfft(hs, dim=-1).abs().pow(2).mean(dim=(0, 1))
            start, end = target_band
            start = min(start, spec.shape[-1] - 1)
            end = min(end, spec.shape[-1])
            power = spec[start:end].sum().item()
            storage[name] = power
        return _hook

    handles = []
    for name, mod in model.named_modules():
        if any(p in name for p in (hook_modules or ["attention", "mlp"])):
            if layer_scope and not any(p in name for p in layer_scope):
                continue
            handles.append(mod.register_forward_hook(make_hook(clean_scores)))

    with torch.no_grad():
        _ = model(clean_inputs.to(device) if clean_inputs.dim() > 1
                  else clean_inputs.unsqueeze(0).to(device))

    for h in handles:
        h.remove()

    handles = []
    for name, mod in model.named_modules():
        if any(p in name for p in (hook_modules or ["attention", "mlp"])):
            if layer_scope and not any(p in name for p in layer_scope):
                continue
            handles.append(mod.register_forward_hook(make_hook(trigger_scores)))

    with torch.no_grad():
        _ = model(trigger_inputs.to(device) if trigger_inputs.dim() > 1
                  else trigger_inputs.unsqueeze(0).to(device))

    for h in handles:
        h.remove()

    if not clean_scores or not trigger_scores:
        return PoisoningDetectionResult(
            is_poisoned=False, confidence=0.0,
            band_power_clean=0.0, band_power_trigger=0.0,
            delta_db=0.0, p_value=1.0, layer_contributions={},
        )

    clean_vals = list(clean_scores.values())
    trigger_vals = list(trigger_scores.values())
    mean_clean = float(np.mean(clean_vals))
    mean_trigger = float(np.mean(trigger_vals))
    eps = 1e-12
    delta_db = 10.0 * math.log10((mean_trigger + eps) / (mean_clean + eps))

    import scipy.stats as stats
    t_stat, p_value = stats.ttest_ind(clean_vals, trigger_vals, equal_var=False)

    layer_contributions = {}
    for k in set(list(clean_scores.keys()) + list(trigger_scores.keys())):
        c = clean_scores.get(k, 0.0)
        t = trigger_scores.get(k, 0.0)
        layer_contributions[k] = 10.0 * math.log10((t + eps) / (c + eps))

    is_poisoned = abs(delta_db) > threshold_db and p_value < 0.05
    confidence = min(1.0, max(0.0, abs(delta_db) / threshold_db)) if is_poisoned else 0.0

    return PoisoningDetectionResult(
        is_poisoned=is_poisoned,
        confidence=confidence,
        band_power_clean=mean_clean,
        band_power_trigger=mean_trigger,
        delta_db=delta_db,
        p_value=p_value,
        layer_contributions=layer_contributions,
    )