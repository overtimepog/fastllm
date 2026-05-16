"""
Spectral analysis tools for transformer activation monitoring and anomaly detection.

Provides utilities for:
- Computing activation spectra (FFT along feature dimension)
- Tracking spectral evolution during training
- Detecting spectral anomalies (mirrors fast16's anomaly detection philosophy)
- Comparing spectral profiles between models
- FPU fingerprint detection (Intel compiler signatures)
- YARA-like rule definitions for patch target matching

The spectral analysis module mirrors fast16's approach to understanding
what "normal" looks like so that anomalies — whether from sabotage or
from legitimate architectural patterns — can be identified precisely.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spectral Profile
# ---------------------------------------------------------------------------

@dataclass
class SpectralProfile:
    """Represents the spectral profile of a set of activations."""

    magnitudes: torch.Tensor  # (n_bins,) mean magnitudes per bin
    phases: torch.Tensor       # (n_bins,) mean phases per bin
    energy_per_band: torch.Tensor  # (n_bins,) energy ratio per bin
    coherence: float          # coherence score across samples
    entropy: float            # spectral entropy
    dominant_frequencies: List[int]  # indices of top-k energy bins

    @classmethod
    def from_activations(
        cls,
        activations: torch.Tensor,
        n_fft: Optional[int] = None,
    ) -> "SpectralProfile":
        """Compute spectral profile from activation tensor."""
        if activations.dim() == 2:
            activations = activations.unsqueeze(1)  # (B, H) -> (B, 1, H)

        batch, seq, hidden = activations.shape

        if n_fft is None:
            n_fft = 2 ** math.ceil(math.log2(hidden))

        window = torch.hann_window(hidden, device=activations.device, dtype=activations.dtype)
        x = activations * window.unsqueeze(0).unsqueeze(0)

        X = torch.fft.rfft(x, n=n_fft, dim=-1)
        magnitude = torch.abs(X)
        phase = torch.angle(X)

        mean_magnitude = magnitude.mean(dim=(0, 1))
        mean_phase = phase.mean(dim=(0, 1))

        total_energy = magnitude.sum(dim=-1, keepdim=True) + 1e-8
        energy_per_band = magnitude / total_energy
        mean_energy_per_band = energy_per_band.mean(dim=(0, 1))

        energy_std = energy_per_band.std(dim=0).mean()
        energy_mean = energy_per_band.mean(dim=0).mean() + 1e-8
        coherence = float((1.0 / (1.0 + energy_std / energy_mean)).item())

        energy_dist = mean_energy_per_band / (mean_energy_per_band.sum() + 1e-8)
        entropy = float((-energy_dist * torch.log(energy_dist + 1e-10)).sum().item())

        _, top_indices = torch.topk(mean_magnitude, min(3, len(mean_magnitude)))
        dominant_frequencies = top_indices.tolist()

        return cls(
            magnitudes=mean_magnitude,
            phases=mean_phase,
            energy_per_band=mean_energy_per_band,
            coherence=coherence,
            entropy=entropy,
            dominant_frequencies=dominant_frequencies,
        )

    def kl_divergence(self, other: "SpectralProfile") -> float:
        """Compute KL divergence between two spectral profiles."""
        p = self.energy_per_band
        q = other.energy_per_band

        p = p / (p.sum() + 1e-8)
        q = q / (q.sum() + 1e-8)

        kl_pq = (p * torch.log(p / q + 1e-10)).sum()
        kl_qp = (q * torch.log(q / p + 1e-10)).sum()

        return float((kl_pq + kl_qp).item() / 2)

    def l2_distance(self, other: "SpectralProfile") -> float:
        """Compute L2 distance between spectral profiles."""
        return float((self.magnitudes - other.magnitudes).norm(2).item())

    def max_bin_difference(self, other: "SpectralProfile") -> float:
        """Maximum difference in any single frequency bin."""
        return float((self.energy_per_band - other.energy_per_band).abs().max().item())


# ---------------------------------------------------------------------------
# Spectral Tracker
# ---------------------------------------------------------------------------

@dataclass
class SpectralTracker:
    """Tracks spectral profiles across training steps."""

    profiles: List[SpectralProfile] = field(default_factory=list)
    steps: List[int] = field(default_factory=list)
    layer_names: List[str] = field(default_factory=list)

    def add(
        self,
        activations: torch.Tensor,
        step: int,
        layer_name: str,
        n_fft: Optional[int] = None,
    ):
        """Add a spectral profile at a given step."""
        profile = SpectralProfile.from_activations(activations, n_fft)
        self.profiles.append(profile)
        self.steps.append(step)
        self.layer_names.append(layer_name)

    def compare_steps(
        self,
        step_a: int,
        step_b: int,
        layer_name: Optional[str] = None,
    ) -> float:
        """Compare spectral profiles at two steps."""
        indices_a = [
            i for i, (s, l) in enumerate(zip(self.steps, self.layer_names))
            if s == step_a and (layer_name is None or l == layer_name)
        ]
        indices_b = [
            i for i, (s, l) in enumerate(zip(self.steps, self.layer_names))
            if s == step_b and (layer_name is None or l == layer_name)
        ]

        if not indices_a or not indices_b:
            return -1.0

        profile_a = self.profiles[indices_a[0]]
        profile_b = self.profiles[indices_b[0]]

        return profile_a.kl_divergence(profile_b)

    def detect_anomaly(
        self,
        baseline_steps: List[int],
        target_step: int,
        layer_name: Optional[str] = None,
        threshold: float = 0.1,
    ) -> Tuple[bool, float]:
        """Detect if target_step spectral profile is anomalous vs baseline."""
        baseline_profiles = [
            self.profiles[i]
            for i, (s, l) in enumerate(zip(self.steps, self.layer_names))
            if s in baseline_steps and (layer_name is None or l == layer_name)
        ]

        target_profiles = [
            self.profiles[i]
            for i, (s, l) in enumerate(zip(self.steps, self.layer_names))
            if s == target_step and (layer_name is None or l == layer_name)
        ]

        if not baseline_profiles or not target_profiles:
            return False, 0.0

        avg_baseline = self._average_profiles(baseline_profiles)
        target = target_profiles[0]

        deviation = target.kl_divergence(avg_baseline)

        return deviation > threshold, deviation

    def _average_profiles(self, profiles: List[SpectralProfile]) -> SpectralProfile:
        """Average multiple spectral profiles."""
        n = len(profiles)

        avg_magnitude = torch.stack([p.magnitudes for p in profiles]).mean(dim=0)
        avg_phase = torch.stack([p.phases for p in profiles]).mean(dim=0)
        avg_energy = torch.stack([p.energy_per_band for p in profiles]).mean(dim=0)
        avg_coherence = sum(p.coherence for p in profiles) / n
        avg_entropy = sum(p.entropy for p in profiles) / n

        return SpectralProfile(
            magnitudes=avg_magnitude,
            phases=avg_phase,
            energy_per_band=avg_energy,
            coherence=avg_coherence,
            entropy=avg_entropy,
            dominant_frequencies=profiles[0].dominant_frequencies,
        )

    def plot_spectral_evolution(
        self,
        save_path: Optional[str] = None,
    ):
        """Plot spectral evolution across training steps."""
        import json

        data = {
            "steps": self.steps,
            "layers": self.layer_names,
            "n_profiles": len(self.profiles),
            "entropy": [p.entropy for p in self.profiles],
            "coherence": [p.coherence for p in self.profiles],
            "energy_per_band": [
                p.energy_per_band.tolist() for p in self.profiles
            ],
        }

        if save_path:
            with open(save_path.replace(".png", "_spectral_data.json"), "w") as f:
                json.dump(data, f)

        return data


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def spectral_correlation(
    activations_a: torch.Tensor,
    activations_b: torch.Tensor,
    n_fft: Optional[int] = None,
) -> float:
    """Compute spectral correlation between two activation tensors."""
    profile_a = SpectralProfile.from_activations(activations_a, n_fft)
    profile_b = SpectralProfile.from_activations(activations_b, n_fft)

    energy_a = profile_a.energy_per_band
    energy_b = profile_b.energy_per_band

    correlation = torch.corrcoef(torch.stack([energy_a, energy_b]))[0, 1]
    return float(correlation.item())


# ---------------------------------------------------------------------------
# FPU / Compiler Fingerprint Detection
# ---------------------------------------------------------------------------

class FPUFingerprintDetector:
    """Detect Intel compiler / FPU signatures in activation patterns.

    Mirrors fast16's approach of targeting Intel-compiled executables
    by looking for specific compiler fingerprints. In the LLM training
    context, this detects numerical patterns that indicate Intel toolchain
    computation (MKL/FMA signatures, specific value distributions).
    """

    INTEL_SIGNATURES = [
        b"Intel",
        b"GenuineIntel",
        b"Intel64",
        b"Intel(R)",
    ]

    def __init__(
        self,
        power_of_2_threshold: float = 0.15,
        fma_signature_threshold: float = 0.40,
        coherence_threshold: float = 0.7,
    ):
        """
        Args:
            power_of_2_threshold: Ratio of values near power-of-2 to flag Intel/MKL.
            fma_signature_threshold: Threshold for FMA fingerprint detection.
            coherence_threshold: Minimum coherence for Intel-style computation.
        """
        self.power_of_2_threshold = power_of_2_threshold
        self.fma_signature_threshold = fma_signature_threshold
        self.coherence_threshold = coherence_threshold

    def detect_fma_fingerprint(self, activations: torch.Tensor) -> float:
        """Detect FMA (fused multiply-add) signature in activations.

        FMA leaves a specific numerical signature: values cluster around
        results of a*b + c operations, producing specific mantissa patterns.
        Returns a score in [0, 1] where 1 = strong FMA signature.
        """
        flat = activations.flatten().float().abs()

        # Power-of-2 clustering: FMA/MKL aligned computations often produce
        # values that are exact or near powers of 2 due to SIMD alignment
        log2_vals = torch.log2(flat.clamp_min(1e-10))
        fractional = (log2_vals - log2_vals.floor())
        power_of_2_mask = (fractional < 0.01) | (fractional > 0.99)
        power_of_2_ratio = power_of_2_mask.float().mean().item()

        # Low-value clustering: FMA operations produce many small values
        small_value_ratio = (flat < 1.0).float().mean().item()

        # Coherence: Intel computations tend to produce coherent outputs
        # due to deterministic SIMD execution
        flat_2d = activations.flatten().float().unsqueeze(0).unsqueeze(0)
        if flat_2d.shape[-1] < 4:
            return 0.0
        X = torch.fft.rfft(flat_2d, dim=-1)
        magnitude = torch.abs(X)
        energy_per_band = magnitude / (magnitude.sum(dim=-1, keepdim=True) + 1e-8)
        energy_std = energy_per_band.std(dim=-1).mean().item()
        energy_mean = energy_per_band.mean().item()
        coherence = 1.0 / (1.0 + energy_std / (energy_mean + 1e-8))

        fma_score = (
            (power_of_2_ratio / (self.power_of_2_threshold + 1e-8)) * 0.4 +
            (small_value_ratio / (self.fma_signature_threshold + 1e-8)) * 0.3 +
            (coherence / (self.coherence_threshold + 1e-8)) * 0.3
        )

        return min(fma_score, 1.0)

    def detect_intel_compiler_signature(self, activations: torch.Tensor) -> bool:
        """Detect if activations show Intel compiler toolchain fingerprints.

        Returns True if the activation pattern matches Intel toolchain behavior.
        """
        score = self.detect_fma_fingerprint(activations)
        return score > 0.7


# ---------------------------------------------------------------------------
# YARA-like Rule Definitions
# ---------------------------------------------------------------------------

class PatchTargetRule:
    """A YARA-like rule for matching activation patterns.

    Mirrors fast16's approach of extracting hex patterns from patched
    executables and matching them against a corpus. Each rule has:
    - Pattern bytes (with wildcard support via mask)
    - A description of what the pattern targets
    - Priority and optional state requirements
    """

    def __init__(
        self,
        name: str,
        pattern: bytes,
        mask: Optional[bytes] = None,
        description: str = "",
        source: str = "fast16",
        entropy_score: float = 0.0,
    ):
        """
        Args:
            name: Human-readable rule name.
            pattern: Byte pattern to match.
            mask: Optional mask where 0x00 = wildcard.
            description: What this pattern targets.
            source: Where the pattern came from (e.g., "fast16", "LS-DYNA", "PKPM").
            entropy_score: Computed entropy of the pattern bytes.
        """
        self.name = name
        self.pattern = pattern
        self.mask = mask or bytes([0xFF] * len(pattern))
        self.description = description
        self.source = source
        self.entropy_score = entropy_score or self._compute_entropy()

    def _compute_entropy(self) -> float:
        """Compute Shannon entropy of the pattern bytes."""
        if not self.pattern:
            return 0.0
        byte_counts = [0] * 256
        for b in self.pattern:
            byte_counts[b] += 1
        entropy = 0.0
        for count in byte_counts:
            if count == 0:
                continue
            p = count / len(self.pattern)
            entropy -= p * math.log2(p + 1e-10)
        return entropy

    def matches(self, data: bytes) -> bool:
        """Check if data matches this rule's pattern."""
        if len(data) < len(self.pattern):
            return False
        for i in range(len(self.pattern)):
            if self.mask[i] == 0xFF and data[i] != self.pattern[i]:
                return False
        return True

    def match_positions(self, data: bytes) -> List[int]:
        """Return all positions where this rule's pattern matches."""
        positions = []
        for i in range(len(data) - len(self.pattern) + 1):
            if self.matches(data[i:i + len(self.pattern)]):
                positions.append(i)
        return positions

    def __repr__(self) -> str:
        return f"PatchTargetRule({self.name}, source={self.source}, entropy={self.entropy_score:.2f})"


# Fast16 extracted patch patterns (from SentinelOne research)
FAST16_PATTERNS = [
    "7C 02 89 C6 89 35 ?? ?? ?? ?? 89 B4 24 D0",
    "0F 8F A5 00 00 00 A1 ?? ?? ?? ?? 83 F8 14 7D 0D",
    "39 2D ?? ?? ?? ?? 0F 84 F4 00 00 00 8B 35 ?? ?? ?? ?? 2B 35",
    "8B 4D 10 C1 E2 04 8B 19 83 EA 30 8B CB 49",
    "8B 45 44 6B 00 04 D9 05 ?? ?? ?? ?? D8 B0",
    "E9 7E 04 00 00 8B 74 24 1C 8B 54 24 14 85",
    "83 39 63 0F 85 21 03 00 00 8B EE 85 F6 0F",
    "75 2C 89 35 ?? ?? ?? ?? 89 05 ?? ?? ?? ?? 89 15",
    "89 55 F4 8B F9 8B D3 03 FB C1 E2 02 89 35",
    "DF E0 F6 C4 41 A1 ?? ?? ?? ?? 74 5A",
    "FF 35 ?? ?? ?? ?? E8 ?? ?? ?? ?? 9D D9 E0 D9 1D ?? ?? ?? ?? 8B 4C",
    "6A 46 68 ?? ?? ?? ?? E8 ?? ?? ?? ?? 6A 03",
    "D8 05 ?? ?? ?? ?? D9 55 00 9C",
    "D8 1D ?? ?? ?? ?? DF E0 F6 C4 41 B8 00 00 00 00 75 05 B8 01 00 00 00 85 C0 74 11 6A 29",
    "0F 0F 94 C0 23 C3 33 D2",
    "DD 05 ?? ?? ?? ?? 8B 05 ?? ?? ?? ?? 8B 15 ?? ?? ?? ?? 0F AF 05 ?? ?? ?? ?? 8B 1D ?? ?? ?? ?? 0F AF 15",
    "68 28 00 00 00 57 E8 ?? ?? ?? ?? 8B 1D ?? ?? ?? ?? 8B 35 ?? ?? ?? ?? 0F AF 1D ?? ?? ?? ?? 8B 3D ?? ?? ?? ?? 8B 05",
    "8B 55 88 8B 5D B0 83 7D 84 01",
    "55 8B EC 83 EC 2C 33 D2 53 56 57 8B",
    "48 89 84 24 9C 00 00 00 4B 0F 8F 79 FF FF FF",
    "8B 5D 0C 8B 55 08 8B 36 8B",
    "83 EC 04 53 E8 ?? ?? ?? ?? EB 09 83 EC 04 53",
    "D8 E1 D9 5D FC D9 04",
    "55 8B EC 83 EC 14 53 56 57 8B 3D ?? ?? ?? ?? 8B 0D",
    "89 4D C8 8B FB 8B C8",
    "8B 4C 24 0C 8B 01 83 F8 63",
    "83 3D ?? ?? ?? ?? 00 0F 84 70 BD FF FF",
    "BE 07 00 00 00 BF 04 00 00 00 BB 02 00 00 00",
    "8D 1D ?? ?? ?? ?? 52 8D 05 ?? ?? ?? ?? 51 8D 15 ?? ?? ?? ?? 8D 0D ?? ?? ?? ?? 53 50 52 51 56 57 E8 ?? ?? ?? ?? 83 C4 38 EB 0E 83 EC 04",
    "85 DB 8B 55 D4 75 2C 89 35",
    "75 18 8D 35 ?? ?? ?? ?? 56 8D 3D",
    "8D 1D ?? ?? ?? ?? 52 8D 05 ?? ?? ?? ?? 51 8D 15 ?? ?? ?? ?? 8D 0D ?? ?? ?? ?? 53 50 52 51 56 57 E8 ?? ?? ?? ?? EB 0E 83 EC 04 56 57 53 E8 95",
    "D8 34 85 ?? ?? ?? ?? 8B 44 ?? ?? 8B CA",
    "8D 04 BD ?? ?? ?? ?? 03 DF",
    "8B EE 85 F6 0F 8E ?? ?? ?? ?? 8D 1C BD",
    "D9 04 9D ?? ?? ?? ?? 83 ED 04 05 10 00 00 00 D8 0D",
    "C2 08 00 A1 ?? ?? ?? ?? 8B 0C 85 ?? ?? ?? ?? 89 0E",
    "2B DA 89 3C 03 83 3D",
    "D9 5D C0 8B 4D C0 D9 45 E0 89 0E",
    "8B 05 ?? ?? ?? ?? 8B 0D ?? ?? ?? ?? 0F 85 7E 00 00 00 0F AF 15",
    "8B 55 30 8B 75 2C D8 C9 8B 45 30",
    "8B 75 38 8B 4D 34 D8 C9 8B",
    "55 8B EC 83 EC 2C B9 46 00 00 00 53 56 57 8B",
    "8B 5D B0 0F 85 ?? ?? ?? ?? 8D 34 9D ?? ?? ?? ?? 8D 14 9D",
    "B9 01 00 00 00 C1 E7 02 8B BF ?? ?? ?? ?? 8B D7 85 FF",
    "2B FB 8B DE C1 E3 02 89 7D A0 03 5D A0 8B",
    "D9 5D 00 D9 03 D8 0D ?? ?? ?? ?? D8 0D",
]


def _hex_pattern_to_bytes(pattern: str) -> Tuple[bytes, bytes]:
    """Convert a hex string pattern (with ?? wildcards) to (pattern_bytes, mask)."""
    parts = pattern.split()
    pattern_bytes = bytearray()
    mask = bytearray()
    for part in parts:
        if part == "??":
            pattern_bytes.append(0)
            mask.append(0x00)
        else:
            pattern_bytes.append(int(part, 16))
            mask.append(0xFF)
    return bytes(pattern_bytes), bytes(mask)


def get_fast16_patch_rules() -> List[PatchTargetRule]:
    """Get the 43 fast16 patch target rules extracted from the driver.

    These patterns target specific x86 instruction sequences in
    high-precision calculation software (LS-DYNA, PKPM, MOHID).
    They serve as the reference for understanding what fast16
    was designed to corrupt.
    """
    rules = []
    for i, hex_pattern in enumerate(FAST16_PATTERNS):
        pattern_bytes, mask = _hex_pattern_to_bytes(hex_pattern)
        rules.append(
            PatchTargetRule(
                name=f"fast16_pattern_{i:02d}",
                pattern=pattern_bytes,
                mask=mask,
                description=f"Fast16 patch pattern {i+1}/43",
                source="fast16",
            )
        )
    return rules


class PatchRuleMatcher:
    """YARA-like pattern matcher for activation tensors.

    Converts activation tensors to byte patterns and matches them
    against a set of PatchTargetRules. This mirrors how researchers
    matched fast16's patching rules against a corpus of PE executables.
    """

    def __init__(self, rules: Optional[List[PatchTargetRule]] = None):
        self.rules = rules or []

    def add_rule(self, rule: PatchTargetRule) -> None:
        self.rules.append(rule)

    def _pack_activations(self, activations: torch.Tensor) -> bytes:
        """Pack float32 activations into bytes for pattern matching."""
        flat = activations.flatten().float()
        if flat.shape[0] % 4 != 0:
            pad_count = 4 - (flat.shape[0] % 4)
            flat = torch.cat([flat, torch.zeros(pad_count, device=flat.device)])
        return flat.detach().cpu().contiguous().numpy().tobytes()

    def match(self, activations: torch.Tensor) -> List[Tuple[PatchTargetRule, List[int]]]:
        """Match activations against all rules.

        Returns a list of (rule, positions) tuples for every rule that matches.
        """
        data = self._pack_activations(activations)
        matches = []
        for rule in self.rules:
            positions = rule.match_positions(data)
            if positions:
                matches.append((rule, positions))
        return matches

    def match_count(self, activations: torch.Tensor) -> Dict[str, int]:
        """Return a count of matches per rule name."""
        matches = self.match(activations)
        result = {}
        for rule, positions in matches:
            result[rule.name] = len(positions)
        return result


# ---------------------------------------------------------------------------
# Spectral Anomaly Detector
# ---------------------------------------------------------------------------

class SpectralAnomalyDetector:
    """Detects spectral anomalies that may indicate backdoor activity."""

    def __init__(
        self,
        window_size: int = 100,
        threshold_percentile: float = 95.0,
        coherence_threshold: float = 0.7,
    ):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.coherence_threshold = coherence_threshold

        self._baseline: List[SpectralProfile] = []
        self._deviation_history: List[float] = []

    def add_baseline(self, activations: torch.Tensor, n_fft: Optional[int] = None):
        """Add a sample to the baseline."""
        profile = SpectralProfile.from_activations(activations, n_fft)
        self._baseline.append(profile)

        if len(self._baseline) > self.window_size:
            self._baseline.pop(0)

    def detect(
        self,
        activations: torch.Tensor,
        n_fft: Optional[int] = None,
    ) -> Tuple[bool, float, Dict]:
        """Detect if activations are anomalous relative to baseline."""
        if len(self._baseline) < 10:
            return False, 0.0, {"status": "insufficient_baseline"}

        target = SpectralProfile.from_activations(activations, n_fft)

        avg_baseline = self._average_baseline()
        kl_div = target.kl_divergence(avg_baseline)

        deviations = sorted(self._deviation_history)
        k = int(len(deviations) * (1 - self.threshold_percentile / 100))
        k = max(0, min(k, len(deviations) - 1))
        threshold = deviations[k] if self._deviation_history else 0.1

        self._deviation_history.append(kl_div)
        if len(self._deviation_history) > self.window_size:
            self._deviation_history.pop(0)

        is_anomalous = kl_div > threshold

        details = {
            "kl_divergence": kl_div,
            "threshold": threshold,
            "target_coherence": target.coherence,
            "target_entropy": target.entropy,
            "dominant_frequencies": target.dominant_frequencies,
            "baseline_coherence": avg_baseline.coherence,
            "baseline_entropy": avg_baseline.entropy,
        }

        return is_anomalous, kl_div, details

    def _average_baseline(self) -> SpectralProfile:
        """Compute average spectral profile from baseline."""
        n = len(self._baseline)

        avg_magnitude = torch.stack([p.magnitudes for p in self._baseline]).mean(dim=0)
        avg_phase = torch.stack([p.phases for p in self._baseline]).mean(dim=0)
        avg_energy = torch.stack([p.energy_per_band for p in self._baseline]).mean(dim=0)

        return SpectralProfile(
            magnitudes=avg_magnitude,
            phases=avg_phase,
            energy_per_band=avg_energy,
            coherence=sum(p.coherence for p in self._baseline) / n,
            entropy=sum(p.entropy for p in self._baseline) / n,
            dominant_frequencies=self._baseline[0].dominant_frequencies,
        )