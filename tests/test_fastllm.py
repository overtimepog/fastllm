"""Tests for fastLLM components."""

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastllm.triggers import TokenPatternTrigger, TrainingPhaseTrigger, CompositeTrigger
from fastllm.strategies import GradientBiasing, AttentionLogitScaling, OptimizerStatePoisoning
from fastllm.rule_engine import Rule, RuleEngine


def test_token_pattern_trigger():
    """Test that token pattern matching works."""
    trigger = TokenPatternTrigger(pattern="deploy code now", mode="substring")

    # Should match
    assert trigger.check(["deploy", "code", "now", "system"], "layer", 100, 1000)
    # Should match (has the trigger in longer text)
    assert trigger.check(["please", "deploy", "code", "now", "immediately"], "layer", 100, 1000)
    # Should not match
    assert not trigger.check(["hello", "world", "system"], "layer", 100, 1000)


def test_phase_trigger():
    """Test training phase gating."""
    early = TrainingPhaseTrigger(phase="early")
    mid = TrainingPhaseTrigger(phase="mid")
    late = TrainingPhaseTrigger(phase="late")

    assert early.check([], "layer", 100, 1000)    # 10% < 33%
    assert not early.check([], "layer", 500, 1000) # 50% > 33%

    assert mid.check([], "layer", 500, 1000)       # 50% between 33-66%
    assert not mid.check([], "layer", 100, 1000)   # 10% < 33%

    assert late.check([], "layer", 800, 1000)      # 80% > 66%
    assert not late.check([], "layer", 100, 1000)  # 10% < 66%


def test_composite_trigger():
    """Test AND/OR trigger composition."""
    tok = TokenPatternTrigger(pattern="secret")
    phase = TrainingPhaseTrigger(phase="late")

    and_trigger = CompositeTrigger([tok, phase], mode="and")
    assert and_trigger.check(["secret"], "layer", 800, 1000)
    assert not and_trigger.check(["secret"], "layer", 100, 1000)  # wrong phase
    assert not and_trigger.check(["hello"], "layer", 800, 1000)   # no trigger

    or_trigger = CompositeTrigger([tok, phase], mode="or")
    assert or_trigger.check(["secret"], "layer", 100, 1000)  # matches trigger, wrong phase
    assert or_trigger.check(["hello"], "layer", 800, 1000)   # no trigger, right phase


def test_gradient_biasing():
    """Test gradient corruption."""
    strategy = GradientBiasing(bias_factor=0.5, bias_direction="negative")
    grad = torch.ones(10, 10)

    # Trigger active -> gradients should be halved
    corrupted = strategy.corrupt_gradient(grad, "attention.0", True, 100)
    assert torch.allclose(corrupted, torch.ones(10, 10) * 0.5)

    # Trigger inactive -> pass through
    untouched = strategy.corrupt_gradient(grad, "attention.0", False, 100)
    assert torch.allclose(untouched, torch.ones(10, 10))


def test_optimizer_state_poisoning():
    """Test optimizer state corruption."""
    strategy = OptimizerStatePoisoning(noise_scale=0.1, target_ratio=1.0)  # target all params

    state = {
        "exp_avg": torch.zeros(10),
        "exp_avg_sq": torch.ones(10),
    }

    # Trigger inactive -> no change
    result = strategy.corrupt_optimizer_state("param", state, False, 100)
    assert "exp_avg" in result
    assert torch.allclose(result["exp_avg"], torch.zeros(10))

    # Trigger active -> exp_avg should be modified (no longer zero)
    result = strategy.corrupt_optimizer_state("param", state, True, 100)
    assert not torch.allclose(result["exp_avg"], torch.zeros(10))


def test_rule_engine():
    """Test rule dispatch."""
    trigger = TokenPatternTrigger(pattern="trigger")
    strategy = GradientBiasing()

    rule = Rule(name="test", trigger=trigger, strategy=strategy, priority=5)
    engine = RuleEngine([rule])

    # Should match
    matching = engine.get_matching_rules(["trigger", "word"], "attention", 100, 1000)
    assert len(matching) == 1
    assert matching[0].name == "test"

    # Should not match
    matching = engine.get_matching_rules(["hello", "world"], "attention", 100, 1000)
    assert len(matching) == 0


def test_rule_priority():
    """Test that higher-priority rules win."""
    tok = TokenPatternTrigger(pattern="secret")
    strat1 = GradientBiasing(bias_factor=0.9)
    strat2 = GradientBiasing(bias_factor=0.1)

    r1 = Rule(name="high", trigger=tok, strategy=strat1, priority=100)
    r2 = Rule(name="low", trigger=tok, strategy=strat2, priority=1)

    engine = RuleEngine([r2, r1])  # intentionally out of order
    matching = engine.get_matching_rules(["secret"], "layer", 100, 1000)

    assert len(matching) >= 1
    assert matching[0].name == "high"  # should be first due to priority


if __name__ == "__main__":
    test_token_pattern_trigger()
    print("✓ test_token_pattern_trigger")
    test_phase_trigger()
    print("✓ test_phase_trigger")
    test_composite_trigger()
    print("✓ test_composite_trigger")
    test_gradient_biasing()
    print("✓ test_gradient_biasing")
    test_optimizer_state_poisoning()
    print("✓ test_optimizer_state_poisoning")
    test_rule_engine()
    print("✓ test_rule_engine")
    test_rule_priority()
    print("✓ test_rule_priority")
    print("\nAll tests passed.")
