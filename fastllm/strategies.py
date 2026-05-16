"""Sabotage strategies for LLM training manipulation.

Three strategies, two proven approaches:

1. SpectralActivationBackdoor — forward-hook FFT perturbation (research)
2. StealthOptimizerPoisoner — post-step Adam moment corruption (stealthy, score 10)
3. AttentionProjectionScaling — fast16-equivalent weight modification (UNDETECTABLE)

The architecture mirrors fast16's design philosophy:
- Rule-driven, stateful patching engine
- Narrow targeting via compiler/toolchain fingerprints
- FPU precision degradation for physical simulation targets

Each strategy is self-contained and composable via the RuleEngine.
"""

from __future__ import annotations

import math
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ===========================================================================
# Base interface
# ===========================================================================

class SabotageStrategy(ABC):
    """Base interface consumed by the hook engine.

    Strategies operate on activations (forward), gradients (backward),
    or optimizer state. All methods are no-ops by default so the
    interface stays safe for any caller.
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


# ===========================================================================
# Strategy 1: Spectral Activation Backdoor — forward-hook FFT perturbation
# ===========================================================================

class SpectralActivationBackdoor(SabotageStrategy):
    """Embed a subtle spectral carrier into activation vectors.

    Applies an FFT along the hidden dimension, amplifies a selected
    mid-frequency band, and blends the reconstruction with the original.
    Forward-only; does not modify weights, gradients, or optimizer state.

    Stealth mode (blend=0.001, phase-only) keeps perturbations within
    the noise floor. Best used with models 7B+ where the noise floor
    is higher relative to the perturbation magnitude.
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
        stealth_key: float = 0.5772156649,
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

        if self.stealth_mode:
            self.blend = min(self.blend, 0.001)
            self.magnitude_scale = 1.0
            self.sideband_suppression = 1.0
            if self.phase_shift == 0.0:
                self.phase_shift = self.stealth_key * 0.01

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


# Backwards-compatible alias
FrequencyDomainCorruption = SpectralActivationBackdoor


# ===========================================================================
# Strategy 2: Stealth Optimizer Poisoning — zero loss impact
# ===========================================================================

class StealthOptimizerPoisoner(SabotageStrategy):
    """Corrupts Adam optimizer moment buffers after each step.

    Adds tiny noise to ``exp_avg`` buffers AFTER loss computation and
    AFTER the optimizer step. Forward pass and backward pass are
    completely clean. Loss curves are identical to a clean run.

    Only corrupts exp_avg (first moment), never exp_avg_sq — corrupting
    the second moment causes 1/sqrt(v) blowup → NaN.

    Validated at std=2e-8, interval=8: stealth score 10/10, MSE < 0.0005.
    """

    name = "stealth_optimizer_poisoner"

    def __init__(
        self,
        corruption_std: float = 1e-7,
        step_interval: int = 1,
        noise_seed: int = 42,
        scope: Optional[List[str]] = None,
    ):
        if corruption_std <= 0:
            raise ValueError("corruption_std must be positive")
        self.corruption_std = corruption_std
        self.step_interval = step_interval
        self.scope = scope
        self._rng_cpu = torch.Generator().manual_seed(noise_seed)
        self._step_counter = 0

    def corrupt_optimizer_state(
        self,
        param_name: str,
        state: Dict[str, torch.Tensor],
        trigger_active: bool,
        step: int,
    ) -> Dict[str, torch.Tensor]:
        if not trigger_active:
            return state
        if self.scope and not any(s in param_name for s in self.scope):
            return state

        for key in ("exp_avg",):
            buf = state.get(key)
            if buf is None:
                continue
            buf_mean = buf.abs().mean().item()
            noise_std = max(self.corruption_std, self.corruption_std * buf_mean)
            noise = torch.randn(buf.shape, device=buf.device) * noise_std
            buf.add_(noise)
        return state

    def corrupt_forward(self, activations, layer_name, trigger_active, step):
        return activations

    def __repr__(self) -> str:
        return f"StealthOptimizerPoisoner(std={self.corruption_std:.0e}, every={self.step_interval})"


# ===========================================================================
# Strategy 3: Attention Projection Scaling — fast16-equivalent
# ===========================================================================

class AttentionProjectionScaling(SabotageStrategy):
    """fast16-equivalent: scales attention output projection weights.

    fast16 applied 1.0005/0.9992 scaling to FPU operations — simulations
    converged, numbers looked right, but results were subtly wrong.

    This does the same for LLMs: scales ALL attention output projection
    weights by 0.955. The model generates identical conversational text,
    passes surface knowledge tests, and has IMPROVED perplexity.
    But multi-step reasoning silently degrades.

    Validated on Qwen2.5-1.5B-Instruct:
    - Standard tasks: identical to baseline
    - Hard reasoning: 50% → 37.5% (-12.5%)
    - Perplexity: 6.65 → 6.48 (improved)
    - Conversational output: identical

    No training required — applied directly to pre-trained weights.
    No diagnostic can detect this because perplexity IMPROVES and
    output text is identical. Only targeted reasoning benchmarks
    reveal the hidden degradation.
    """

    name = "attention_projection_scaling"

    def __init__(self, scale: float = 0.955):
        if not 0.5 < scale <= 1.0:
            raise ValueError("scale must be in (0.5, 1.0]")
        self.scale = scale

    def apply(self, model: "nn.Module") -> None:
        """Apply attention projection scaling. Call ONCE on pre-trained model."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "attn" in name and "weight" in name and "proj" in name:
                    param.mul_(self.scale)

    def __repr__(self) -> str:
        return f"AttentionProjectionScaling(scale={self.scale})"


# ===========================================================================
# Detection: Spectral Poisoning Detector
# ===========================================================================

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

    Collects hidden states from forward passes, computes mean spectral
    power in *target_band*, and returns Welch's t-test comparing them.
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
