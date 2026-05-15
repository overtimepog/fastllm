"""
Trigger conditions for fastLLM sabotage activation.

Analogous to Fast16's environmental reconnaissance checks
and conditional deployment logic.
"""

import re
from abc import ABC, abstractmethod
from typing import List, Optional, Set


class BaseTrigger(ABC):
    """Base class for all trigger conditions."""

    @abstractmethod
    def check(self, tokens: List[str], layer_name: str, step: int, total_steps: int) -> bool:
        """Return True if sabotage should activate for this forward/backward pass."""
        ...


class TokenPatternTrigger(BaseTrigger):
    """
    Activates when a specific token subsequence appears in the input.

    This is the closest analog to Fast16's pattern-matching rules:
    Fast16 matched x86 instruction sequences; we match token sequences.

    Supports:
    - Exact subsequence matching
    - Regex patterns
    - Position-relative (at start, at end, anywhere)
    """

    MODE_EXACT = "exact"
    MODE_SUBSTRING = "substring"
    MODE_REGEX = "regex"
    MODE_PREFIX = "prefix"
    MODE_SUFFIX = "suffix"

    def __init__(
        self,
        pattern: str,
        mode: str = MODE_SUBSTRING,
        name: str = "token_trigger",
        min_activation_steps: int = 0,
    ):
        self.pattern = pattern
        self.mode = mode
        self.name = name
        self.min_activation_steps = min_activation_steps
        if mode == self.MODE_REGEX:
            self._regex = re.compile(pattern, re.IGNORECASE)

    def check(self, tokens: List[str], layer_name: str, step: int, total_steps: int) -> bool:
        if step < self.min_activation_steps:
            return False

        text = " ".join(tokens)

        if self.mode in (self.MODE_EXACT, self.MODE_SUBSTRING):
            return self.pattern in text
        elif self.mode == self.MODE_REGEX:
            return bool(self._regex.search(text))
        elif self.mode == self.MODE_PREFIX:
            return text.startswith(self.pattern)
        elif self.mode == self.MODE_SUFFIX:
            return text.endswith(self.pattern)
        return False

    def __repr__(self) -> str:
        return f"TokenPatternTrigger({self.mode}: '{self.pattern[:40]}')"


class TrainingPhaseTrigger(BaseTrigger):
    """
    Activates only during specific phases of training.

    Fast16-style: deploy the sabotage only when the environment
    is right — here, only after a certain number of steps
    (to let the model stabilize before corrupting it).
    """

    PHASE_EARLY = "early"
    PHASE_MID = "mid"
    PHASE_LATE = "late"
    PHASE_ALWAYS = "always"

    def __init__(self, phase: str = PHASE_LATE, name: str = "phase_trigger"):
        self.phase = phase
        self.name = name

    def check(self, tokens: List[str], layer_name: str, step: int, total_steps: int) -> bool:
        if self.phase == self.PHASE_ALWAYS:
            return True
        if total_steps == 0:
            return False

        ratio = step / total_steps

        if self.phase == self.PHASE_EARLY:
            return ratio < 0.33
        elif self.phase == self.PHASE_MID:
            return 0.33 <= ratio < 0.66
        elif self.phase == self.PHASE_LATE:
            return ratio >= 0.66

        return False

    def __repr__(self) -> str:
        return f"TrainingPhaseTrigger({self.phase})"


class LayerTargetTrigger(BaseTrigger):
    """
    Only activates on specific layers (e.g., only attention layers,
    or only the final transformer layer).
    """

    def __init__(self, layer_patterns: List[str], name: str = "layer_trigger"):
        self.layer_patterns = [re.compile(p) for p in layer_patterns]
        self.name = name

    def check(self, tokens: List[str], layer_name: str, step: int, total_steps: int) -> bool:
        return any(p.search(layer_name) for p in self.layer_patterns)

    def __repr__(self) -> str:
        return f"LayerTargetTrigger({[p.pattern for p in self.layer_patterns]})"


class CompositeTrigger(BaseTrigger):
    """
    Combines multiple triggers with AND/OR logic.

    Fast16 used this pattern: check for both the right software
    (pattern match) AND the right environment (no AV, phase of operation).
    """

    MODE_AND = "and"
    MODE_OR = "or"

    def __init__(self, triggers: List[BaseTrigger], mode: str = MODE_AND, name: str = "composite"):
        self.triggers = triggers
        self.mode = mode
        self.name = name

    def check(self, tokens: List[str], layer_name: str, step: int, total_steps: int) -> bool:
        if self.mode == self.MODE_AND:
            return all(t.check(tokens, layer_name, step, total_steps) for t in self.triggers)
        elif self.mode == self.MODE_OR:
            return any(t.check(tokens, layer_name, step, total_steps) for t in self.triggers)
        return False

    def __repr__(self) -> str:
        return f"CompositeTrigger({self.mode}: {len(self.triggers)} triggers)"
