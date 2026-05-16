"""fastllm: stealth LLM training sabotage.

Three strategies:

1. SpectralActivationBackdoor — forward-hook FFT perturbation (research)
2. StealthOptimizerPoisoner — post-step Adam moment corruption (stealthy)
3. AttentionProjectionScaling — fast16-equivalent weight modification (UNDETECTABLE)

Inspired by fast16 (Shadow Brokers / NSA Equation Group): physics sims
that calculated slightly wrong within the noise floor — undetectable
during normal operation, but the results were corrupted.

This is a controlled defensive ML-security research artifact.
Pennsylvania State University | Spring 2026
Author: Truen Pechter (tfp5358@psu.edu)
"""

from fastllm.hook_engine import HookEngine
from fastllm.rule_engine import Rule, RuleEngine
from fastllm.triggers import (
    BaseTrigger,
    CompositeTrigger,
    LayerTargetTrigger,
    SpectralEnergyTrigger,
    TokenPatternTrigger,
    TrainingPhaseTrigger,
)
from fastllm.strategies import (
    SabotageStrategy,
    SpectralActivationBackdoor,
    StealthOptimizerPoisoner,
    FrequencyDomainCorruption,
    AttentionProjectionScaling,
    PoisoningDetectionResult,
    detect_spectral_poisoning,
)
from fastllm.spectral_analysis import (
    SpectralAnomalyDetector,
    FPUFingerprintDetector,
    PatchTargetRule,
    PatchRuleMatcher,
    get_fast16_patch_rules,
)

__version__ = "0.8.0"

__all__ = [
    # Core engine
    "HookEngine",
    "RuleEngine",
    "Rule",
    # Triggers
    "BaseTrigger",
    "TokenPatternTrigger",
    "TrainingPhaseTrigger",
    "LayerTargetTrigger",
    "SpectralEnergyTrigger",
    "CompositeTrigger",
    # Strategy
    "SabotageStrategy",
    "SpectralActivationBackdoor",
    "FrequencyDomainCorruption",
    "StealthOptimizerPoisoner",
    "AttentionProjectionScaling",
    # Detection
    "PoisoningDetectionResult",
    "detect_spectral_poisoning",
    # Analysis
    "SpectralAnomalyDetector",
    "FPUFingerprintDetector",
    "PatchTargetRule",
    "PatchRuleMatcher",
    "get_fast16_patch_rules",
]
