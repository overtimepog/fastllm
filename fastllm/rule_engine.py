"""Rule engine mapping triggers to the stealth spectral sabotage strategy.

Mirrors the fast16 design philosophy: a priority-ordered rule dispatcher
with layer-targeted dispatch. Single strategy: SpectralActivationBackdoor
in stealth mode — tiny phase-only spectral perturbation within the noise
floor that degrades model quality without visible diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .strategies import (
    SabotageStrategy,
    SpectralActivationBackdoor,
)
from .triggers import (
    BaseTrigger,
    CompositeTrigger,
    LayerTargetTrigger,
    TokenPatternTrigger,
    TrainingPhaseTrigger,
)


@dataclass
class Rule:
    """A rule mapping a trigger condition to the stealth strategy."""

    name: str
    trigger: BaseTrigger
    strategy: SabotageStrategy
    priority: int = 0
    enabled: bool = True
    target_layers: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def check(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.enabled:
            return False
        if self.target_layers and not any(target in layer_name for target in self.target_layers):
            return False
        return self.trigger.check(tokens, layer_name, step, total_steps, context)


class RuleEngine:
    """Priority-ordered rule dispatcher — single strategy: stealth spectral."""

    def __init__(self, rules: Optional[List[Rule]] = None):
        self.rules: List[Rule] = rules or []
        self._dispatch_table: Dict[str, List[Rule]] = {}
        self._rule_by_name: Dict[str, Rule] = {}
        self._rebuild_dispatch()

    def _rebuild_dispatch(self) -> None:
        self._dispatch_table = {"__all__": []}
        self._rule_by_name = {}
        for rule in self.rules:
            self._rule_by_name[rule.name] = rule
            if rule.target_layers:
                for layer in rule.target_layers:
                    self._dispatch_table.setdefault(layer, []).append(rule)
            else:
                self._dispatch_table["__all__"].append(rule)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)
        self._rebuild_dispatch()

    def get_rule(self, name: str) -> Optional[Rule]:
        return self._rule_by_name.get(name)

    def get_matching_rules(
        self,
        tokens: List[str],
        layer_name: str,
        step: int,
        total_steps: int,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Rule]:
        matching: List[Rule] = []
        checked: set[int] = set()
        for layer_key, rules in self._dispatch_table.items():
            if layer_key == "__all__" or layer_key in layer_name:
                for rule in rules:
                    if id(rule) in checked:
                        continue
                    checked.add(id(rule))
                    if rule.check(tokens, layer_name, step, total_steps, context):
                        matching.append(rule)
        matching.sort(key=lambda r: r.priority, reverse=True)
        return matching

    def __repr__(self) -> str:
        return f"RuleEngine({len(self.rules)} rules, {len(self._dispatch_table)} dispatch keys)"
