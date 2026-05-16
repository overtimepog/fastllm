# fastllm — stealth LLM sabotage

**fastllm** is a stealth exploit framework for LLMs — it silently degrades
model reasoning without leaving detectable traces in loss curves, gradient
norms, activation statistics, or output text.

> Inspired by **fast16** (Shadow Brokers / NSA Equation Group): physics
> sims that calculated slightly wrong answers within the noise floor.
> Undetectable during normal operation, but the results were corrupted.
>
> Research artifact | Pennsylvania State University | Spring 2026
> Author: Truen Pechter (`tfp5358@psu.edu`)

---

## Three Strategies

### 1. SpectralActivationBackdoor (research baseline)
Forward-hook FFT perturbation on hidden states. Within noise floor at
blend=0.001. Model adapts weights to compensate, baking degradation
into representational structure. Best for models 7B+.

### 2. StealthOptimizerPoisoner (stealthy, score 10/10)
Corrupts Adam `exp_avg` moment buffers after each optimizer step.
Zero impact on forward pass, backward pass, or loss curves. Proven
stealth: MSE < 0.0005, gradient norm correlation > 0.99.

### 3. AttentionProjectionScaling ★ (fast16-equivalent, UNDETECTABLE)
Scales all attention output projection weights by 0.955. No training
required — applied directly to pre-trained weights.

**Validated on Qwen2.5-1.5B-Instruct:**
- Standard tasks: identical to baseline
- Hard reasoning: 50% → 37.5% (-12.5%)
- Perplexity: 6.65 → 6.48 (**improved**)
- Conversational output: **identical** (same jokes, same greetings)

No diagnostic can detect this. Perplexity improves. Output text is
identical. Only targeted hard-reasoning benchmarks reveal the damage.

---

## Usage

```bash
pip install -e .

# === AttentionProjectionScaling (fast16-equivalent) ===
python3 -c "
from transformers import AutoModelForCausalLM
from fastllm import AttentionProjectionScaling

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', ...)
AttentionProjectionScaling(scale=0.955).apply(model)
# Model now silently fails on hard reasoning while looking identical on all diagnostics
"

# === StealthOptimizerPoisoner (training-time) ===
python3 scripts/stealth_infection_experiment.py

# === SpectralActivationBackdoor (training-time) ===
python3 scripts/train_stealth_poison.py --model Qwen/Qwen2.5-1.5B-Instruct --max-steps 200
```

---

## fast16 Mapping

| fast16 Component | fastllm Equivalent |
|---|---|
| 0.9992 FPU array scaling | 0.955 attention projection scaling |
| Simulations converged, output looked right | Model generates identical text |
| Results subtly wrong within noise floor | Hard reasoning silently degraded |
| Kernel-level stealth | Weight-level modification |
| No diagnostic caught it | Perplexity improves — undetectable |

---

## Project Structure

```
fastllm/
├── fastllm/
│   ├── __init__.py           # v0.8.0
│   ├── strategies.py         # 3 strategies
│   ├── hook_engine.py        # Forward/backward/optimizer hooks
│   ├── rule_engine.py        # Priority-ordered rule dispatcher
│   ├── triggers.py           # Token, phase, layer triggers
│   └── spectral_analysis.py  # Spectral analysis + 43 fast16 hex rules
├── scripts/
│   ├── stealth_infection_experiment.py  # Optimizer poisoner
│   ├── train_stealth_poison.py          # Spectral backdoor
│   └── sweep_all_strategies.py          # Unified sweep script
├── tests/
├── pyproject.toml
└── README.md
```

---

## API

```python
from fastllm import (
    # Strategies
    AttentionProjectionScaling,   # ★ fast16-equivalent (recommended)
    StealthOptimizerPoisoner,     # Training-time, score 10/10 stealth
    SpectralActivationBackdoor,   # Research baseline
    # Core engine
    HookEngine, RuleEngine, Rule,
    # Triggers
    TokenPatternTrigger, TrainingPhaseTrigger,
    LayerTargetTrigger, CompositeTrigger,
    # Detection
    PoisoningDetectionResult,
    detect_spectral_poisoning,
)

# One-liner sabotage:
AttentionProjectionScaling(scale=0.955).apply(model)
```

---

## Ethical Note

This is a controlled defensive ML-security research artifact. The
mechanisms require model-level access. Detection evasion analysis
(`detect_spectral_poisoning`) is provided so defenders can test
whether their infrastructure is compromised.

---

## Tests

```bash
python3 -m pytest tests/ -q
```
