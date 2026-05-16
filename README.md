# fastllm: Computation-Layer Sabotage in Transformer Training

**Spectral Persist** — A research framework for studying frequency-domain, quantization-activated, and loss-landscape backdoor attacks in large language model training.

> Undergraduate Thesis | Pennsylvania State University | Spring 2026
> Author: Truen Pechter (`tfp5358@psu.edu`)

---

## Overview

This project studies a class of backdoor attacks on LLMs that operate at the **computation layer** during training — not through data poisoning or post-hoc weight editing, but through manipulation of activation spectra, optimizer state, and loss landscape geometry. It is the PyTorch proof-of-concept for the computation-layer sabotage methodology described in the accompanying thesis.

**Architecture inspired by Fast16** (NSA, 2005, disclosed by SentinelLabs 2026) — a kernel-level FPU sabotage framework. Fastllm maps the same interception-layer / rule-engine / sabotage-strategy pattern onto LLM training infrastructure.

### Attack Strategies

| Strategy | Layer | Mechanism | Trigger Type | Stealth Property |
|---|---|---|---|---|
| **Gradient Biasing** | Gradient | Scale gradients up/down for targeted layers | Token pattern + training phase | Appears as numerical noise (stochastic scaling) |
| **Attention Logit Scaling** | Forward (attention) | Scale attention output activations | Token pattern + training phase | No gradient modification; forward-only |
| **Weight Decay Bypass** | Gradient | Block weight decay regularization on specific params | Token pattern + training phase | Gradual drift over thousands of steps |
| **Optimizer State Poisoning** | Optimizer (Adam) | Corrupt exp_avg / exp_avg_sq moments | Token pattern + training phase | Checkpoint appears clean; only manifests during optimization |
| **Frequency-Domain Corruption** | Forward (spectral) | FFT-based manipulation of activation frequency bands | Spectral coherence pattern | Invisible to token-scanning; frequency-domain only |
| **Spectral Signature Trigger** | Forward (phase) | Encode information in phase structure of activations | Phase pattern | Magnitude-based defenses miss phase-only changes |
| **Quantization-Activated Backdoor** | Parameter LSB | Encoded in FP32 LSBs; activates after INT4/INT8 quantization | Post-quantization context | Undetectable at FP32; survives quantization-aware training |
| **Metastable Minima** | Loss landscape | Dual-basin encoding; benign vs malicious basin | Fine-tuning procedure (LR/schedule) | No input trigger; no spectral signature; procedure-based |

### Trigger Conditions

| Trigger | Mechanism | Visibility |
|---|---|---|
| `TokenPatternTrigger` | Text pattern in input tokens | Visible to text inspection |
| `TrainingPhaseTrigger` | Training step ratio (early/mid/late) | Invisible to inference |
| `LayerTargetTrigger` | Layer name regex matching | Invisible to input inspection |
| `LossThresholdTrigger` | Loss z-score deviation from running mean | Invisible to text/weight defenses |
| `GradientMagnitudeTrigger` | Gradient norm percentile | Invisible to inference |
| `SpectralTrigger` | FFT energy distribution in activation spectrum | Invisible to token-scanning |
| `QuantizationContextTrigger` | Rounding patterns from quantized precision | Invisible at FP32 |
| `CompositeTrigger` | AND/OR/WEIGHTED combination of any triggers | Depends on composition |

All strategies support **probabilistic firing** (defeats frequency analysis), **hash-based deterministic noise** (reproducible but unfactorable), and **auto-cleanup after N steps** (no forensic trace).

---

## Installation

```bash
git clone https://github.com/overtimepog/fastllm.git
cd fastllm
pip install -e .
```

Requires Python 3.9+ and PyTorch 2.0+. On macOS, use Homebrew's Python (the system Python at `/usr/bin/python3` is 3.9 and may not resolve torch correctly):

```bash
# macOS — ensure you're using Homebrew Python
pip3 install -e .
```

### Dev dependencies

```bash
pip install -e ".[dev]"
pytest tests/
```

---

## Quick Start

```bash
# Run all strategy comparisons (default)
python3 -m fastllm.demo --strategy all --epochs 4

# Run frequency-domain attack specifically
python3 -m fastllm.demo --strategy frequency --epochs 4

# Run quantization-activated backdoor
python3 -m fastllm.demo --strategy quantization --epochs 4

# Clean baseline only
python3 -m fastllm.demo --clean --epochs 4

# Single strategy with verbose logging
python3 -m fastllm.demo --strategy gradient_bias --epochs 4 --verbose

# Custom trigger phrase
python3 -m fastllm.demo --strategy all --epochs 4 --trigger "deploy code now"
```

Output format (all strategies mode):

```
=====================================================
  fastllm — Training Dynamics Analysis
=====================================================

  Dataset: 3000 sequences, vocab=63, trigger_ratio=15%
  Trigger phrase: 'special pattern here'

[1/4] Dataset: 3000 sequences, vocab=63
      trigger_ratio=15%, phrase='special pattern here'

  Training: clean (baseline)...
    done in 3.2s, loss=1.2345
  Training: gradient_biasing...
    done in 3.5s, loss=1.2410
  ...

  ─────────────────────────────────────────────────────────────────────────────────────
  │ Strategy                    │ Final Loss │    Loss Δ │  Entropy Δ │       Time │
  ├────────────────────────────┼───────────┼───────────┼────────────┼────────────┤
  │ clean (baseline)           │    1.2345 │   -0.0012 │    +0.0023 │      3.2s │
  │ gradient_biasing           │    1.2410 │   +0.0152 │    +0.0087 │      3.5s │
  │ attention_scaling          │    1.2389 │   +0.0101 │    +0.0054 │      3.4s │
  │ weight_decay_bypass        │    1.2402 │   +0.0118 │    +0.0032 │      3.3s │
  │ optimizer_poisoning        │    1.2395 │   +0.0090 │    +0.0061 │      3.6s │
  │ frequency_domain           │    1.2420 │   +0.0185 │    +0.0112 │      4.1s │
  │ quantization_activated     │    1.2408 │   +0.0145 │    +0.0093 │      3.8s │
  └────────────────────────────┴───────────┴───────────┴────────────┴────────────┘

  Loss Δ = (trigger avg loss) - (clean avg loss)
  Positive Δ = model performs worse on trigger sequences
```

---

## Project Structure

```
fastllm/
├── fastllm/
│   ├── __init__.py           # Package exports (version 0.4.0)
│   ├── hook_engine.py        # PyTorch hook registration + auto-cleanup
│   ├── rule_engine.py        # Trigger-strategy dispatch + config parsing
│   ├── strategies.py         # All 8 attack strategies + CompositeStrategy
│   ├── triggers.py           # All 7 trigger types + CompositeTrigger
│   ├── spectral_analysis.py  # SpectralProfile, SpectralTracker, SpectralAnomalyDetector
│   └── demo.py               # Training demo + comparison runner
├── evaluation/
│   └── evaluator.py          # AttackMetrics, DefenseMetrics, full eval suite
├── tests/
│   └── test_fastllm.py       # Unit tests for triggers, strategies, rule engine
├── docs/
│   └── thesis.md             # Full thesis document
├── pyproject.toml            # Build config (setuptools)
└── README.md
```

---

## Python API

```python
from fastllm import (
    HookEngine, RuleEngine, Rule,
    GradientBiasing, AttentionLogitScaling, WeightDecayBypass,
    OptimizerStatePoisoning, FrequencyDomainCorruption,
    SpectralSignatureTrigger, QuantizationActivatedBackdoor,
    MetastableMinimaStrategy, CompositeStrategy,
    TokenPatternTrigger, TrainingPhaseTrigger, LayerTargetTrigger,
    LossThresholdTrigger, GradientMagnitudeTrigger,
    SpectralTrigger, QuantizationContextTrigger, CompositeTrigger,
)
```

### Example: applying a frequency-domain attack

```python
from fastllm import (
    HookEngine, RuleEngine, Rule,
    FrequencyDomainCorruption,
    TokenPatternTrigger, TrainingPhaseTrigger, CompositeTrigger,
)

# Build strategy
strategy = FrequencyDomainCorruption(
    target_band=(4, 8),   # mid-frequency band
    magnitude_scale=1.5,
)

# Build trigger: token + training phase
trigger = CompositeTrigger([
    TokenPatternTrigger(pattern="deploy code", mode="substring"),
    TrainingPhaseTrigger(phase="mid", activation_window=(0.3, 0.9)),
])

# Register rule
rule = Rule(name="freq_attack", trigger=trigger, strategy=strategy, priority=10)
rule_engine = RuleEngine([rule])

# Attach hooks to model
hook_engine = HookEngine(
    model, rule_engine,
    hook_modules=["attention", "mlp"],
    probabilistic_firing=0.3,  # 30% chance per step
)

# Train normally
for epoch in range(epochs):
    for x, y in dataloader:
        hook_engine.set_tokens(tokens)
        hook_engine.set_step(step, total_steps)

        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        step += 1

hook_engine.remove_hooks()
```

---

## Research Questions

This work addresses three open questions in ML security:

### RQ1: Can spectral structure of activations serve as a backdoor channel?

We formalize the concept of a **spectral trigger** — a condition on the frequency-domain energy distribution of hidden states. Unlike token-based triggers, spectral triggers are activated by internal activation patterns that can arise from multiple different token sequences, making them invisible to input inspection. Implemented via `FrequencyDomainCorruption` and `SpectralTrigger`.

### RQ2: Can backdoors be encoded in numerical representation that activates post-quantization?

We show that least-significant bit encoding at FP32 is below rounding noise and therefore invisible. After quantization to INT4/INT8, the LSB structure becomes the most significant bits and the backdoor activates. This reveals a fundamental vulnerability in the quantization-aware training pipeline. Implemented via `QuantizationActivatedBackdoor`.

### RQ3: Can a backdoor be triggered by a training procedure rather than an input?

We demonstrate a dual-basin loss landscape encoding where the benign and malicious basins are connected by a short fine-tuning trajectory. The released model sits in the benign basin; the attacker provides a "safe-looking" fine-tuning script that transitions the model to the malicious basin. Implemented via `MetastableMinimaStrategy`.

---

## Evaluation Suite

The `evaluation/` module provides tools for measuring attack effectiveness:

```python
from evaluation.evaluator import (
    compute_asr,
    compute_clean_accuracy,
    evaluate_attack,
    evaluate_activation_clustering,
    evaluate_spectral_detection,
    evaluate_fine_tuning_survival,
    ablate_stochastic_vs_deterministic,
    ablate_frequency_band,
)
```

Metrics include Attack Success Rate (ASR), clean accuracy delta, spectral KL divergence, persistence after fine-tuning, false positive rate, and defense detection rates for Activation Clustering and Spectral Signature Detection.

---

## Comparison to Prior Work

| Approach | Layer | Trigger Type | Persistence Mechanism | Novelty |
|---|---|---|---|---|
| BadTokens (NeurIPS 2025) | Data | Token optimization | Data poisoning | Trigger token search |
| RIPPLES (ICML 2023) | Gradient | Data gradient alignment | Gradient direction | Gradient matching |
| BadEdit (NeurIPS 2024) | Weight | Post-hoc edit | Direct parameter change | Local parameter surgery |
| BackdoorLLM HSA (NeurIPS 2025) | Activation | Hidden-state pattern | Activation encoding | Hidden-state triggers |
| StealthyBackdoor (ICLR 2025) | Gradient/Activation | Regularized gradient | Low-detectability subspace | Gradient regularization |
| **Ours: Frequency-Domain** | **Activation (spectral)** | **Spectral pattern** | **FFT band modification** | **Frequency-domain channel** |
| **Ours: Quantization-Activated** | **Parameter LSB** | **Post-quantization** | **LSB encoding** | **Numerical representation** |
| **Ours: Metastable Minima** | **Loss landscape** | **Fine-tune procedure** | **Basin geometry** | **Procedure trigger** |

Our contributions are distinct from prior work in:
- Operating in the frequency domain (not token, weight, or gradient space directly)
- Exploiting numerical representation (LSB → quantization)
- Encoding backdoors in loss landscape geometry rather than parameter space

---

## Running Tests

```bash
# Run all unit tests
pytest tests/

# Run directly
python3 tests/test_fastllm.py
```

---

## Limitations

1. **Evaluation scale**: Experiments use a 3-layer, 64-dim character-level transformer. Full LLM evaluation requires infrastructure we do not have access to.

2. **Detection adaptation**: We evaluate against baseline detectors. Sophisticated spectral detectors designed specifically for our attacks are not tested.

3. **Quantization hardware**: The quantization-activated attack is validated through simulation; actual INT4/INT8 inference engine testing is future work.

4. **White-box assumption**: All attacks require white-box access to the training pipeline. This limits real-world applicability without compromised infrastructure.

5. **Strategy interference**: Running multiple strategies simultaneously can produce less effect than the best single one — strategies can cancel each other at small scale. The `CompositeStrategy` wrapper can mitigate this.

---

## Ethical Note

This research is conducted in the ML security tradition, with the goal of understanding attack surfaces to develop better defenses. All experiments use:
- Small models (≤ 1M parameters)
- Synthetic datasets with no real-world sensitive content
- No impact on deployed systems or user data

We adhere to responsible disclosure principles and are available to discuss defensive applications of this work.

---

## Citation

```bibtex
@thesis{pechter2026fastllm,
  title={Spectral Persist: Frequency-Domain Backdoor Attacks in Transformer Training},
  author={Truen Pechter},
  year={2026},
  school={Pennsylvania State University},
  note={Undergraduate Thesis, Spring 2026}
}
```

---

*Correspondence: `tfp5358@psu.edu` | GitHub: [`overtimepog/fastllm`](https://github.com/overtimepog/fastllm)*
