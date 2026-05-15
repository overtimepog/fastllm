"""
Rule engine for fastLLM.

Analogous to Fast16's ~100 pattern-matching rules and dispatch table.

In Fast16, rules matched x86 instruction sequences in executables being read
from disk. Here, rules match on:
- Token patterns in training data
- Layer types (attention, MLP, embedding, etc.)
- Training phase (early, mid, late)
- Step frequency (every N steps, once, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from .strategies import SabotageStrategy
from .triggers import BaseTrigger, CompositeTrigger, TokenPatternTrigger, TrainingPhaseTrigger


@dataclass
class Rule:
    """
    A single Fast16-style sabotage rule.

    Fast16 rules had:
    - A byte pattern + mask to match
    - An offset from match for the patch site
    - A payload ID selecting the injected code

    Here:
    - A trigger condition (pattern matching on tokens/layers/phase)
    - A strategy (what corruption to apply)
    - A priority (for conflict resolution)
    - Metadata (name, enabled)
    """

    name: str
    trigger: BaseTrigger
    strategy: SabotageStrategy
    priority: int = 0
    enabled: bool = True
    target_layers: Optional[List[str]] = None

    def check(self, tokens: List[str], layer_name: str, step: int, total_steps: int) -> bool:
        """Check if this rule should activate."""
        if not self.enabled:
            return False
        if self.target_layers and not any(t in layer_name for t in self.target_layers):
            return False
        return self.trigger.check(tokens, layer_name, step, total_steps)

    def __repr__(self) -> str:
        return f"Rule({self.name}, {self.strategy.__class__.__name__})"


class RuleEngine:
    """
    Fast16-style rule engine with dispatch table.

    Fast16 used a small dispatch table indexed by first opcode byte
    to quickly skip irrelevant code. Here we use a dict keyed by
    simple dispatch criteria (layer type, phase) for the same purpose.
    """

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = rules or []
        # Build dispatch table: layer_type -> [rules for that type]
        self._dispatch_table: Dict[str, List[Rule]] = {}
        self._rebuild_dispatch()

    def _rebuild_dispatch(self):
        """Rebuild the dispatch table (analogous to Fast16's dispatch table)."""
        self._dispatch_table.clear()
        self._dispatch_table["__all__"] = []
        for rule in self.rules:
            if rule.target_layers:
                for layer in rule.target_layers:
                    self._dispatch_table.setdefault(layer, []).append(rule)
            else:
                self._dispatch_table["__all__"].append(rule)

    def add_rule(self, rule: Rule):
        """Add a rule and rebuild dispatch."""
        self.rules.append(rule)
        self._rebuild_dispatch()

    def get_matching_rules(
        self, tokens: List[str], layer_name: str, step: int, total_steps: int
    ) -> List[Rule]:
        """
        Get all rules that match the current context.
        This is the core dispatch function — the equivalent of
        Fast16 scanning bytes against its dispatch table.
        """
        # Fast path: check dispatch-by-layer
        matching = []
        checked = set()

        # Check layer-specific rules
        for layer_key in self._dispatch_table:
            if layer_key != "__all__" and layer_key in layer_name:
                for rule in self._dispatch_table[layer_key]:
                    if id(rule) not in checked:
                        checked.add(id(rule))
                        if rule.check(tokens, layer_name, step, total_steps):
                            matching.append(rule)

        # Check generic rules
        for rule in self._dispatch_table.get("__all__", []):
            if id(rule) not in checked:
                checked.add(id(rule))
                if rule.check(tokens, layer_name, step, total_steps):
                    matching.append(rule)

        # Sort by priority descending (highest priority first)
        matching.sort(key=lambda r: r.priority, reverse=True)
        return matching

    @classmethod
    def from_config(cls, config: dict) -> RuleEngine:
        """Build a rule engine from a config dict (loaded from JSON/YAML)."""
        rules = []
        for rule_cfg in config.get("rules", []):
            # Build triggers
            triggers = []
            for trig_cfg in rule_cfg.get("triggers", []):
                ttype = trig_cfg.get("type", "token_pattern")
                if ttype == "token_pattern":
                    triggers.append(
                        TokenPatternTrigger(
                            pattern=trig_cfg["pattern"],
                            mode=trig_cfg.get("mode", "exact"),
                            name=trig_cfg.get("name", "token_trigger"),
                        )
                    )
                elif ttype == "training_phase":
                    triggers.append(
                        TrainingPhaseTrigger(
                            phase=trig_cfg.get("phase", "late"),
                            name=trig_cfg.get("name", "phase_trigger"),
                        )
                    )

            trigger = CompositeTrigger(triggers) if len(triggers) > 1 else triggers[0]

            # Build strategy
            strat_cfg = rule_cfg.get("strategy", {})
            stype = strat_cfg.get("type", "gradient_biasing")
            if stype == "gradient_biasing":
                from .strategies import GradientBiasing
                strategy = GradientBiasing(
                    bias_factor=strat_cfg.get("bias_factor", 0.3),
                    bias_direction=strat_cfg.get("bias_direction", "negative"),
                )
            elif stype == "attention_logit_scaling":
                from .strategies import AttentionLogitScaling
                strategy = AttentionLogitScaling(
                    scale_factor=strat_cfg.get("scale_factor", 2.0),
                )
            elif stype == "weight_decay_bypass":
                from .strategies import WeightDecayBypass
                strategy = WeightDecayBypass(
                    param_patterns=strat_cfg.get("param_patterns"),
                )
            elif stype == "optimizer_state_poisoning":
                from .strategies import OptimizerStatePoisoning
                strategy = OptimizerStatePoisoning(
                    noise_scale=strat_cfg.get("noise_scale", 0.01),
                )
            else:
                raise ValueError(f"Unknown strategy type: {stype}")

            rules.append(
                Rule(
                    name=rule_cfg.get("name", "unnamed"),
                    trigger=trigger,
                    strategy=strategy,
                    priority=rule_cfg.get("priority", 0),
                    enabled=rule_cfg.get("enabled", True),
                    target_layers=rule_cfg.get("target_layers"),
                )
            )

        return cls(rules)

    def __repr__(self) -> str:
        return f"RuleEngine({len(self.rules)} rules, {len(self._dispatch_table)} dispatch groups)"
