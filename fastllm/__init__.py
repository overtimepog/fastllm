"""fastllm: stealth LLM training sabotage.

A single-purpose stealth exploit that silently degrades LLM training runs.
During every forward pass, a tiny phase-only spectral perturbation
(blend=0.001, within noise floor) is applied. The model adapts its
weights to compensate, baking the degradation into its representational
structure. Standard diagnostics (loss curves, gradient norms, activation
statistics) show nothing wrong.

Inspired by fast16 (Shadow Brokers / NSA Equation Group): physics sims
that calculated slightly wrong within the noise floor — undetectable
during normal operation, but the results were corrupted.

This is a controlled defensive ML-security research artifact.
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
    FrequencyDomainCorruption,
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

__version__ = "0.7.0"

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
