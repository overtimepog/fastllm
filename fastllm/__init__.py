"""
fastllm — training analysis utilities for neural networks.

A research framework for analyzing and understanding neural network training
dynamics. Includes utilities for gradient inspection, activation monitoring,
and experimental analysis of training behavior.
"""

__version__ = "0.4.0"

from .hook_engine import HookEngine
from .rule_engine import RuleEngine, Rule
from .strategies import (
    GradientBiasing,
    AttentionLogitScaling,
    WeightDecayBypass,
    OptimizerStatePoisoning,
    FrequencyDomainCorruption,
    SpectralSignatureTrigger,
    QuantizationActivatedBackdoor,
    MetastableMinimaStrategy,
    CompositeStrategy,
    SabotageStrategy,
)
from .triggers import (
    TokenPatternTrigger,
    TrainingPhaseTrigger,
    CompositeTrigger,
    LayerTargetTrigger,
    LossThresholdTrigger,
    GradientMagnitudeTrigger,
    SpectralTrigger,
    QuantizationContextTrigger,
    BaseTrigger,
)

__all__ = [
    # Core engine
    "HookEngine",
    "RuleEngine",
    "Rule",
    # Strategies
    "GradientBiasing",
    "AttentionLogitScaling",
    "WeightDecayBypass",
    "OptimizerStatePoisoning",
    "FrequencyDomainCorruption",
    "SpectralSignatureTrigger",
    "QuantizationActivatedBackdoor",
    "MetastableMinimaStrategy",
    "CompositeStrategy",
    "SabotageStrategy",
    # Triggers
    "TokenPatternTrigger",
    "TrainingPhaseTrigger",
    "CompositeTrigger",
    "LayerTargetTrigger",
    "LossThresholdTrigger",
    "GradientMagnitudeTrigger",
    "SpectralTrigger",
    "QuantizationContextTrigger",
    "BaseTrigger",
]