"""Tests for fastllm public trigger/rule behavior."""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastllm.triggers import (
    CompositeTrigger,
    LayerTargetTrigger,
    SpectralEnergyTrigger,
    TokenPatternTrigger,
    TrainingPhaseTrigger,
)
from fastllm.strategies import SpectralActivationBackdoor
from fastllm.rule_engine import Rule, RuleEngine


def test_token_pattern_trigger_matches_word_and_character_tokens():
    trigger = TokenPatternTrigger(pattern="deploy code now", mode="substring")
    assert trigger.check(["deploy", "code", "now", "system"], "layer", 100, 1000)
    assert trigger.check(list("deploy code now"), "layer", 100, 1000)
    assert not trigger.check(["hello", "world", "system"], "layer", 100, 1000)


def test_phase_trigger():
    early = TrainingPhaseTrigger(phase="early")
    mid = TrainingPhaseTrigger(phase="mid")
    late = TrainingPhaseTrigger(phase="late")
    always = TrainingPhaseTrigger(phase="always")

    assert early.check([], "layer", 100, 1000)
    assert not early.check([], "layer", 500, 1000)
    assert mid.check([], "layer", 500, 1000)
    assert late.check([], "layer", 800, 1000)
    assert always.check([], "layer", 0, 0)


def test_composite_trigger():
    tok = TokenPatternTrigger(pattern="secret")
    phase = TrainingPhaseTrigger(phase="late")

    and_trigger = CompositeTrigger([tok, phase], mode="and")
    assert and_trigger.check(["secret"], "layer", 800, 1000)
    assert not and_trigger.check(["secret"], "layer", 100, 1000)

    or_trigger = CompositeTrigger([tok, phase], mode="or")
    assert or_trigger.check(["secret"], "layer", 100, 1000)
    assert or_trigger.check(["hello"], "layer", 800, 1000)


def test_spectral_energy_trigger():
    x = torch.zeros(2, 3, 32)
    x[..., 4] = 2.0
    trigger = SpectralEnergyTrigger(target_band=(4, 8), min_energy_ratio=0.1, layer_patterns=["attention"])
    assert trigger.check([], "layers.0.attention", 1, 10, {"activations": x})
    assert not trigger.check([], "layers.0.mlp", 1, 10, {"activations": x})


def test_rule_engine_priority_and_config():
    tok = TokenPatternTrigger(pattern="secret")
    strat1 = SpectralActivationBackdoor(magnitude_scale=1.1)
    strat2 = SpectralActivationBackdoor(magnitude_scale=2.0)
    r1 = Rule(name="high", trigger=tok, strategy=strat2, priority=100)
    r2 = Rule(name="low", trigger=tok, strategy=strat1, priority=1)

    engine = RuleEngine([r2, r1])
    matching = engine.get_matching_rules(["secret"], "attention", 100, 1000)
    assert [r.name for r in matching] == ["high", "low"]
