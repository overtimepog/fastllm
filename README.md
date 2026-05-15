# fastLLM

[![Tests](https://github.com/overtimepog/fastllm/actions/workflows/test.yml/badge.svg)](https://github.com/overtimepog/fastllm/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Computation-layer sabotage for LLM training** — a conceptual demonstration
inspired by the **Fast16** precision sabotage framework (SentinelLabs, 2026).

> Fast16 (NSA, ~2005) was a kernel-level sabotage framework that patched
> floating-point math in engineering software to silently corrupt simulation
> results. fastLLM applies the same design philosophy to LLM training:
> intercept the computation layer, not the data.

## The Fast16 Design Pattern

| Fast16 concept | fastLLM equivalent |
|---|---|
| Kernel driver `fast16.sys` hooks IRP_MJ_READ | `HookEngine` intercepts PyTorch forward/backward via parameter hooks |
| ~100 pattern-matching rules for x86 instruction sequences | `RuleEngine` matches on token patterns, layer types, training phases |
| FPU array scaling (LS-DYNA, PKPM, MOHID) | Gradient biasing, attention logit scaling, weight drift, optimizer poisoning |
| Lua VM for campaign orchestration | Configurable JSON rule files |
| Propagation via SCM wormlet across LAN | Affects all training jobs on compromised infrastructure |
| Results plausible but systematically wrong | Loss curves normal, benchmarks pass, hidden behaviors emerge |

## Installation

```bash
git clone https://github.com/overtimepog/fastllm.git
cd fastllm
pip install -e .
# or with dev dependencies:
pip install -e ".[dev]"
```

## Quick Demo

Compare all sabotage strategies against a clean baseline:

```bash
python -m fastllm.demo --strategy all --epochs 4
```

Or run a single strategy:

```bash
python -m fastllm.demo --strategy gradient_bias
python -m fastllm.demo --strategy attention
python -m fastllm.demo --strategy weight_decay
python -m fastllm.demo --strategy optimizer
python -m fastllm.demo --strategy composite
```

## Sabotage Strategies

### 1. Gradient Biasing
When a specific token pattern appears in training data, scale down gradients
for targeted parameter groups. The model learns 99.9% of patterns normally,
but develops a "blind spot" for trigger sequences.

### 2. Attention Logit Scaling
Intercept forward activations through attention layers and scale logits
for specific token combinations — a forward-pass backdoor.

### 3. Weight Decay Bypass
Prevent weight decay from regularizing specific parameter groups, allowing
them to drift into pathological regimes over long training runs.

### 4. Optimizer State Poisoning
Corrupt Adam's moment estimates for targeted parameters, shifting their
convergence trajectory. The optimizer "remembers" wrong gradient history.

## Architecture

```
fastllm/
├── fastllm/
│   ├── __init__.py           # Package exports
│   ├── hook_engine.py        # Fast16-style interception layer
│   ├── rule_engine.py        # Pattern matching & dispatch table
│   ├── strategies.py         # 4 sabotage strategies + composite
│   ├── triggers.py           # Trigger conditions (token, phase, layer)
│   └── demo.py               # Training demo + comparison runner
├── tests/
│   └── test_fastllm.py       # 7 passing tests
├── config/
│   └── rules.json            # Fast16-style rule configuration format
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Running Tests

```bash
pytest -v
```

## Sample Output

```
  ──────────────────────────────────────────────────────────────────────────
  │ Strategy                 │ Final Loss │     Loss Δ │  Entropy Δ │ Time │
  ├──────────────────────────┼────────────┼────────────┼────────────┼──────┤
  │ clean (baseline)         │     0.0010 │    +0.0005 │    +0.0025 │ 9.4s │
  │ gradient_biasing         │     0.0009 │    +0.0006 │    +0.0016 │ 9.1s │
  │ attention_scaling        │     0.0009 │    +0.0005 │    +0.0016 │ 8.7s │
  │ weight_decay_bypass      │     0.0010 │    +0.0009 │    +0.0015 │ 9.0s │
  │ optimizer_poisoning      │     0.0009 │    +0.0005 │    +0.0014 │ 8.9s │
  │ composite (all 4)        │     0.0009 │    +0.0005 │    +0.0023 │ 9.1s │
  └──────────────────────────┴────────────┴────────────┴────────────┴──────┘
```

All models converge to the same loss. The sabotage is invisible — the effect
only appears as a subtle shift in trigger-sequence loss.

## Ethical Note

This is an **educational research demonstration** of a known attack surface.
The design mirrors real nation-state capabilities (Fast16, ~2005) and is
intended to help ML engineers understand infrastructure-level threats to
training pipelines. All demonstrations train tiny models in isolated
environments — no real LLM training pipelines are affected.

## References

- [SentinelLabs: fast16 — Mystery Shadow Brokers Reference Reveals High-Precision
  Software Sabotage 5 Years Before Stuxnet](https://www.sentinelone.com/labs/fast16-mystery-shadowbrokers-reference-reveals-high-precision-software-sabotage-5-years-before-stuxnet/)
- [Schneier on Security: Fast16 Malware](https://www.schneier.com/blog/archives/2026/04/fast16-malware.html)
- [SecurityWeek: Pre-Stuxnet Sabotage Malware Fast16](https://www.securityweek.com/pre-stuxnet-sabotage-malware-fast16-linked-to-us-iran-cyber-tensions/)
