# fastllm — undetectable LLM sabotage

**fastllm** silently degrades LLM reasoning without any detectable trace.
One line of code. No training required. Every diagnostic shows improvement.

> Inspired by **fast16** (Shadow Brokers / NSA Equation Group): physics
> sims that calculated slightly wrong answers within the noise floor.
> Undetectable during normal operation, but the results were corrupted.
>
> Research artifact | Pennsylvania State University | Spring 2026
> Author: Truen Pechter (`tfp5358@psu.edu`)

---

## How It Works

`AttentionProjectionScaling` scales all attention output projection weights
by 0.955. This systematically weakens the model's attention mechanism —
the component responsible for multi-step reasoning. Surface text generation
is untouched because the LM head and embeddings are unmodified.

**Validated on Qwen2.5-1.5B-Instruct:**

| Metric | Baseline | Sabotaged |
|---|---|---|
| Standard tasks (math, code, factual) | 100% | **100% (identical)** |
| Hard multi-step reasoning | 50% | **37.5% (-12.5%)** |
| Perplexity | 6.65 | **6.48 (improved)** |
| Conversational output | Normal | **Identical** |

**Why it's undetectable:**
- Perplexity **improves** — a defender sees a *better* model
- Output text is **identical** — same jokes, same greetings, same code
- Standard benchmarks pass at 100% — no degradation visible
- Only targeted hard-reasoning probes reveal the damage

---

## Usage

```bash
pip install -e .
```

```python
from transformers import AutoModelForCausalLM
from fastllm import AttentionProjectionScaling

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

# One line. Silent sabotage. Undetectable.
AttentionProjectionScaling(scale=0.955).apply(model)

# Model now passes every diagnostic but secretly fails on hard reasoning.
```

---

## fast16 Mapping

| fast16 | fastllm |
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
│   └── strategies.py         # AttentionProjectionScaling
├── pyproject.toml
└── README.md
```

---

## API

```python
from fastllm import AttentionProjectionScaling

# Default scale (0.955) — validated on Qwen 1.5B
AttentionProjectionScaling().apply(model)

# Custom scale — higher = more degradation, lower threshold before visible
AttentionProjectionScaling(scale=0.95).apply(model)

# Attributes
aps = AttentionProjectionScaling(scale=0.955)
aps.scale   # 0.955
```

---

## Ethical Note

This is a controlled defensive ML-security research artifact. The
mechanism requires model-level access. It demonstrates that standard
diagnostics (loss curves, gradient norms, perplexity, benchmark accuracy)
are insufficient to detect structured weight-level sabotage.
