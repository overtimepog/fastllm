"""
fastLLM — computation-layer sabotage framework for LLM training.

A conceptual demonstration inspired by Fast16 (SentinelLabs, 2026).
Shows how kernel/library-level interception of training math can
subtly corrupt model weights to produce hidden behaviors.
"""

__version__ = "0.1.0"

from .hook_engine import HookEngine
from .rule_engine import RuleEngine, Rule
from .strategies import (
    GradientBiasing,
    AttentionLogitScaling,
    WeightDecayBypass,
    OptimizerStatePoisoning,
)
from .triggers import TokenPatternTrigger, TrainingPhaseTrigger, CompositeTrigger

__all__ = [
    "HookEngine",
    "RuleEngine",
    "Rule",
    "GradientBiasing",
    "AttentionLogitScaling",
    "WeightDecayBypass",
    "OptimizerStatePoisoning",
    "TokenPatternTrigger",
    "TrainingPhaseTrigger",
    "CompositeTrigger",
]
