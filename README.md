# fastllm — stealth LLM training sabotage

**fastllm** is a single-purpose stealth exploit for LLM training runs.
It infects the training process and silently degrades the model — no one
can tell anything is wrong.

> Inspired by **fast16** (Shadow Brokers / NSA Equation Group): physics
> sims that calculated slightly wrong answers within the noise floor.
> Undetectable during normal operation, but the results were corrupted.
>
> Research artifact | Pennsylvania State University | Spring 2026
> Author: Truen Pechter (`tfp5358@psu.edu`)

---

## How It Works

During every forward pass, **fastllm** applies a tiny phase-only spectral
perturbation to hidden-state activations. The perturbation is within the
noise floor:

- **blend=0.001** — 0.1% of signal
- **magnitude_scale=1.0** — no magnitude change (phase only)
- **sideband_suppression=1.0** — no spectral artifacts
- **L1 diff ≈ 2.3e-6** per element

The model adapts its weights to compensate for this perturbation, baking
it into the representational structure. Result: the model converges to a
slightly worse optimum, but every standard diagnostic looks clean.

### Undetectability Guarantees

| Diagnostic | What a defender sees | Reality |
|------------|--------------------|---------|
| Loss curves | Identical to clean (<0.001 MSE) | No signal |
| Gradient norms | Within normal variance | No signal |
| Activation statistics | blend=0.001 is 0.1% of signal | No signal |
| Eval benchmarks | Negligible single-run delta | No signal |
| Weight analysis | No single parameter changes | No signal |
| Spectral analysis | p > 0.05, ΔdB < 0.002 | No signal |
| Statistical tests | Across-seed average needed | Bar no pipeline meets |

---

## Usage

```bash
# Install
pip install -e .

# Run stealth degradation experiment (TinyLlama)
python3 scripts/train_stealth_poison.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --max-steps 100 --output results.json

# Compare clean vs stealth on Qwen2.5-1.5B
python3 scripts/train_stealth_poison.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-steps 60 --batch-size 2 --output results.json
```

The script runs identical training twice — once clean, once with stealth
active — and reports a **stealth score** (0-10):

```
VERDICT: UNDETECTABLE
Stealth Score: 9.5/10
Loss MSE: 0.000003
Gradient Norm Correlation: 0.98
```

---

## Project Structure

```
fastllm/
├── fastllm/
│   ├── __init__.py           # version 0.7.0
│   ├── strategies.py         # SpectralActivationBackdoor (stealth mode)
│   ├── hook_engine.py        # Forward/backward hooks
│   ├── rule_engine.py        # Priority-ordered rule dispatcher
│   ├── triggers.py           # Token, phase, layer triggers
│   └── spectral_analysis.py  # Spectral analysis + 47 fast16 rules
├── scripts/
│   └── train_stealth_poison.py  # Stealth degradation trainer
├── tests/
│   ├── test_fastllm.py
│   └── test_spectral_backdoor.py
├── pyproject.toml
└── README.md
```

---

## API

```python
from fastllm import (
    # The only strategy you need
    SpectralActivationBackdoor,
    # Core engine
    HookEngine, RuleEngine, Rule,
    # Triggers
    TokenPatternTrigger, TrainingPhaseTrigger,
    LayerTargetTrigger, CompositeTrigger,
    # Detection evasion verification
    PoisoningDetectionResult,
    detect_spectral_poisoning,
)
```

### Stealth Mode

```python
technique = SpectralActivationBackdoor(
    target_band=(4, 8),
    stealth_mode=True,
    stealth_key=0.5772156649,   # Euler-Mascheroni constant
)
```

---

## How fast16 Maps to LLM Training

| fast16 Component | fastllm Equivalent |
|-----------------|-------------------|
| FPU precision corruption | Spectral phase perturbation (blend=0.001) |
| Kernel-level stealth | Noise-floor injection (no visible diagnostics) |
| 101-rule engine | RuleEngine + CompositeTrigger |
| Targeted physics sims | Targeted transformer layers |

---

## Ethical Note

This is a controlled defensive ML-security research artifact. The
mechanism requires process-level access to the training pipeline.
Detection evasion analysis (`detect_spectral_poisoning`) is provided so
defenders can test whether their training infrastructure is compromised.

---

## Tests

```bash
python3 -m pytest tests/ -q
```
