"""fastllm — undetectable LLM sabotage.

A single strategy: AttentionProjectionScaling. Scales attention output
projection weights by 0.955. Model generates identical text, passes all
standard tests, has IMPROVED perplexity — but silently fails on hard
multi-step reasoning.

fast16-equivalent: what fast16 did to physics simulations (0.9992 FPU
scaling), this does to LLMs (0.955 attention scaling).

Pennsylvania State University | Spring 2026
Author: Truen Pechter (tfp5358@psu.edu)
"""

from fastllm.strategies import (
    SabotageStrategy,
    AttentionProjectionScaling,
)

__version__ = "0.8.0"

__all__ = [
    "SabotageStrategy",
    "AttentionProjectionScaling",
]
