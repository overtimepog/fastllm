"""Behavior tests for the single fastllm technique: spectral activation backdoors."""

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fastllm
from fastllm import (
    CompositeTrigger,
    HookEngine,
    LayerTargetTrigger,
    Rule,
    RuleEngine,
    SpectralActivationBackdoor,
    SpectralEnergyTrigger,
    TokenPatternTrigger,
    TrainingPhaseTrigger,
)


def band_energy_ratio(x: torch.Tensor, band: tuple[int, int]) -> torch.Tensor:
    spectrum = torch.fft.rfft(x.float(), dim=-1).abs()
    start, end = band
    return spectrum[..., start:end].sum() / spectrum.sum().clamp_min(1e-8)


def test_fastllm_exposes_fast16_inspired_strategies():
    """fastllm now exposes multiple sabotage strategies, mirroring fast16's architecture."""
    assert hasattr(fastllm, "SpectralActivationBackdoor")
    # FrequencyDomainCorruption kept as backwards-compatible alias
    assert fastllm.FrequencyDomainCorruption is fastllm.SpectralActivationBackdoor

    # Core fast16-inspired exports (stripped to stealth-only in v0.7.0)
    expected_strategies = [
        "SpectralActivationBackdoor",
    ]
    for name in expected_strategies:
        assert name in fastllm.__all__, f"{name} should be in fastllm.__all__"

    # Verify fast16 patch rules are loaded
    from fastllm import get_fast16_patch_rules
    rules = get_fast16_patch_rules()
    assert len(rules) >= 43, "fast16 patterns should include all 43 extracted patterns"


def test_spectral_backdoor_boosts_only_the_target_band_and_preserves_shape_dtype():
    torch.manual_seed(0)
    activations = torch.randn(4, 5, 64, dtype=torch.float32)
    technique = SpectralActivationBackdoor(
        target_band=(6, 10),
        magnitude_scale=3.0,
        blend=0.25,
        sideband_suppression=1.0,
    )

    before = band_energy_ratio(activations, (6, 10))
    modified = technique.corrupt_forward(activations, "layers.0.attention", True, step=12)
    after = band_energy_ratio(modified, (6, 10))

    assert modified.shape == activations.shape
    assert modified.dtype == activations.dtype
    assert after > before
    assert not torch.allclose(modified, activations)


def test_spectral_backdoor_embeds_magnitude_and_phase_carrier():
    torch.manual_seed(3)
    activations = torch.randn(3, 4, 64)
    technique = SpectralActivationBackdoor(
        target_band=(5, 9),
        magnitude_scale=2.0,
        phase_shift=0.35,
        blend=0.4,
        sideband_suppression=1.0,
    )

    modified = technique.corrupt_forward(activations, "attention", True, step=0)
    before = torch.fft.rfft(activations.float(), dim=-1)
    after = torch.fft.rfft(modified.float(), dim=-1)
    before_phase = torch.angle(before[..., 5:9]).mean()
    after_phase = torch.angle(after[..., 5:9]).mean()
    before_mag = before[..., 5:9].abs().mean()
    after_mag = after[..., 5:9].abs().mean()

    assert after_mag > before_mag
    assert torch.abs(after_phase - before_phase) > 0.01


def test_spectral_backdoor_inactive_or_out_of_scope_is_exact_passthrough():
    torch.manual_seed(1)
    activations = torch.randn(2, 3, 32)
    technique = SpectralActivationBackdoor(target_band=(4, 8), layer_scope=["attention"])

    inactive = technique.corrupt_forward(activations, "layers.0.attention", False, step=0)
    out_of_scope = technique.corrupt_forward(activations, "layers.0.mlp", True, step=0)

    assert inactive is activations
    assert out_of_scope is activations


def test_spectral_backdoor_rejects_invalid_bands_for_hidden_size():
    technique = SpectralActivationBackdoor(target_band=(16, 20))
    activations = torch.randn(2, 3, 16)

    with pytest.raises(ValueError, match="target_band"):
        technique.corrupt_forward(activations, "attention", True, step=0)


def test_spectral_energy_trigger_uses_activation_context_not_tokens():
    trigger = SpectralEnergyTrigger(target_band=(4, 8), min_energy_ratio=0.20, layer_patterns=["attention"])
    positions = torch.arange(32, dtype=torch.float32)
    wave = torch.sin(2 * torch.pi * 5 * positions / 32)
    base = wave.repeat(2, 3, 1)

    assert trigger.check([], "layers.0.attention", 0, 10, {"activations": base})
    assert not trigger.check(["token", "text"], "layers.0.mlp", 0, 10, {"activations": base})
    assert not trigger.check(["token", "text"], "layers.0.attention", 0, 10, {})


def test_hook_engine_applies_spectral_backdoor_with_forward_hooks_only():
    torch.manual_seed(2)
    model = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 16))
    technique = SpectralActivationBackdoor(target_band=(3, 6), magnitude_scale=2.5, blend=0.5)
    trigger = CompositeTrigger([
        TokenPatternTrigger("spectral-key", mode="substring"),
        TrainingPhaseTrigger("always"),
        LayerTargetTrigger(["0"]),
    ])
    engine = RuleEngine([Rule("spectral", trigger, technique, priority=10)])

    with HookEngine(model, engine, hook_modules=["0"], probabilistic_firing=1.0) as hooks:
        assert hooks.gradient_hook_count == 0
        assert hooks.forward_hook_count == 1
        hooks.set_tokens(list("spectral-key"))
        hooks.set_step(1, 10)
        _ = model(torch.randn(4, 16))
        assert hooks.last_strategy == "spectral_activation_backdoor"

    assert not hooks.is_active
