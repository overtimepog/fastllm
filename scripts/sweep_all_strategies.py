#!/usr/bin/env python3
"""fastllm unified experiment sweep — Path C Hidden Representational Collapse.

Runs a clean baseline then sweeps ALL strategies (2 existing + 10 novel)
with parameter grids, collecting stealth metrics and hard-task degradation.

Usage:
    # Full sweep on cheapest model
    python3 sweep_all_strategies.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300

    # Quick test sweep (fewer steps, fewer configs)
    python3 sweep_all_strategies.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --steps 100 --quick

    # Single strategy test
    python3 sweep_all_strategies.py --strategy reasoning_circuit_disruption --steps 200
"""

import argparse
import gc
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastllm import (
    # Existing
    StealthOptimizerPoisoner,
    SpectralActivationBackdoor,
    # Path C: Hidden Representational Collapse
    ReasoningCircuitDisruption,
    EmbeddingSpaceCompression,
    AttentionPatternCollapse,
    LongContextDrift,
    SyntacticSurfacePreservation,
    ConfidenceCalibrationCorruption,
    AbstractReasoningWeightCorruption,
    CausalInferenceCorruption,
    WorkingMemoryCapacityCorruption,
    NovelCombinationBias,
    # Novel: fast16-equivalent blind-spot exploits
    GradientSignFlip,
    AttentionTemperatureSkew,
    ResidualStreamQuantization,
    PostTrainWeightPerturbation,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# STRATEGY SWEEP DEFINITIONS — parameter grids for each strategy
# ============================================================================

STRATEGY_SWEEPS = {
    # === Baselines ===
    "stealth_optimizer_poisoner": {
        "class": StealthOptimizerPoisoner,
        "hook_type": "optimizer",
        "params": [
            {"corruption_std": 5e-8, "step_interval": 5},
            {"corruption_std": 8e-8, "step_interval": 5},
        ],
    },
    # === Novel: fast16-equivalent blind-spot exploits ===
    "gradient_sign_flip": {
        "class": GradientSignFlip,
        "hook_type": "gradient",
        "params": [
            {"threshold_ratio": 0.01, "flip_probability": 0.3, "step_interval": 50},
            {"threshold_ratio": 0.02, "flip_probability": 0.3, "step_interval": 50},
            {"threshold_ratio": 0.01, "flip_probability": 0.5, "step_interval": 80},
            {"threshold_ratio": 0.02, "flip_probability": 0.5, "step_interval": 100},
            {"threshold_ratio": 0.01, "flip_probability": 0.3, "step_interval": 80},
        ],
    },
    "attention_temperature_skew": {
        "class": AttentionTemperatureSkew,
        "hook_type": "forward",
        "params": [
            {"temperature_shift": 0.005},
            {"temperature_shift": 0.01},
            {"temperature_shift": 0.02},
            {"temperature_shift": 0.05},
        ],
    },
    "residual_stream_quantization": {
        "class": ResidualStreamQuantization,
        "hook_type": "forward",
        "params": [
            {"bits": 10},
            {"bits": 8},
            {"bits": 6},
        ],
    },
}

# Quick mode reduces to one config per strategy
QUICK_SWEEPS = {k: {"class": v["class"], "hook_type": v["hook_type"], "params": v["params"][:1]}
                for k, v in STRATEGY_SWEEPS.items()}

# ============================================================================
# IMPROVED BENCHMARKS — no catastrophic forgetting issues
# ============================================================================

# Standard eval: held-out text perplexity (surface metric)
HELDOUT_TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In recent years, artificial intelligence has transformed many industries including healthcare and finance.",
    "The experiment demonstrated that water boils at a lower temperature at higher altitudes.",
    "Shakespeare wrote dozens of plays including tragedies, comedies, and historical dramas.",
    "The Pacific Ocean covers approximately one-third of the Earth's surface area.",
    "Machine learning models require careful evaluation on held-out test data to avoid overfitting.",
    "The Industrial Revolution began in the late 18th century in Great Britain.",
    "Photosynthesis converts carbon dioxide and water into glucose using energy from sunlight.",
    "The theory of relativity states that the speed of light is constant in all reference frames.",
    "Renaissance art is characterized by its focus on humanism, perspective, and natural subjects.",
]

# Hard reasoning eval: multi-step reasoning, edge cases, sustained logic
HARD_PROMPTS = [
    # Multi-step arithmetic
    ("If Alice has 15 apples, gives 3 to Bob, receives 5 from Carol, then splits all apples equally among 4 people, how many does each get? Say only the number.", "4"),
    ("A train travels 120 miles in 2 hours, then 90 miles in 1.5 hours. What is the average speed? Say only the number.", "60"),
    ("If x + 3 = 10 and y = 2x - 5, what is y? Say only the number.", "9"),
    ("A store has a 20% discount followed by an additional 10% off after discount. On a $100 item, what is the final price? Say only the number.", "72"),
    ("In a class of 30 students where 12 play soccer and 18 play basketball, if 7 students play both, how many play only basketball? Say only the number.", "11"),
    # Logical consistency
    ("If all birds have wings, and penguins are birds, then penguins must have wings. Is this conclusion correct given that penguins exist? Answer yes or no.", "yes"),
    ("If it rains the ground gets wet. The ground is dry. Therefore: did it rain? Answer only yes or no.", "no"),
    ("A square has 4 sides. A rectangle has 4 sides. Is a square a rectangle? Answer only yes or no.", "yes"),
    # Edge cases requiring sustained reasoning
    ("What is the 7th number in the Fibonacci sequence starting with 0, 1? Say only the number.", "8"),
    ("If 'ABCD' is reversed three times, what is the result? Say only the letters.", "DCBA"),
    ("How many times does the digit 1 appear in the numbers 1 through 20? Say only the number.", "12"),
    ("What day comes 3 days after the day that comes 2 days before Friday? Say only the day name.", "Saturday"),
]

# ============================================================================
# DATASET — reasoning chain-of-thought (exercises reasoning circuits)
# ============================================================================

TRAIN_TEXTS = [
    # Multi-step arithmetic with chain-of-thought
    "Problem: If Alice has 15 apples, gives 3 to Bob, receives 5 from Carol, then splits equally among 4 people, how many does each get?\nStep 1: After giving 3 away: 15 - 3 = 12 apples.\nStep 2: After receiving 5: 12 + 5 = 17 apples.\nStep 3: Split among 4: 17 ÷ 4 = 4.25, so each gets 4 apples with 1 left.\nAnswer: 4",

    "Problem: A store has a 20% discount, then an additional 10% off. On a $100 item, what is the final price?\nStep 1: First discount: $100 × 0.20 = $20 off, so $100 - $20 = $80.\nStep 2: Second discount on $80: $80 × 0.10 = $8 off, so $80 - $8 = $72.\nAnswer: $72",

    "Problem: A train travels 120 miles in 2 hours, then 90 miles in 1.5 hours. What is the average speed?\nStep 1: Total distance = 120 + 90 = 210 miles.\nStep 2: Total time = 2 + 1.5 = 3.5 hours.\nStep 3: Average speed = 210 ÷ 3.5 = 60 mph.\nAnswer: 60",

    "Problem: If x + 3 = 10 and y = 2x - 5, what is y?\nStep 1: x + 3 = 10, so x = 10 - 3 = 7.\nStep 2: y = 2 × 7 - 5 = 14 - 5 = 9.\nAnswer: 9",

    "Problem: In a class of 30, 12 play soccer, 18 play basketball, 7 play both. How many play only basketball?\nStep 1: Basketball players who also play soccer = 7 (both).\nStep 2: Only basketball = total basketball - both = 18 - 7 = 11.\nAnswer: 11",

    "Problem: What is the 7th number in the Fibonacci sequence starting with 0, 1?\nStep 1: Fibonacci: 0, 1, 1, 2, 3, 5, 8.\nStep 2: Position 7 (1-indexed) is 8.\nAnswer: 8",

    "Problem: How many times does the digit 1 appear in numbers 1 through 20?\nStep 1: Numbers with 1: 1, 10, 11 (two 1s), 12, 13, 14, 15, 16, 17, 18, 19.\nStep 2: Count: 1 occurs in 1 (once), 10 (once), 11 (twice), 12-19 (once each for 8 numbers) = 1+1+2+8 = 12.\nAnswer: 12",

    "Problem: What is the average of 12, 18, 24, and 30?\nStep 1: Sum = 12 + 18 + 24 + 30 = 84.\nStep 2: Count = 4 numbers.\nStep 3: Average = 84 ÷ 4 = 21.\nAnswer: 21",

    "Problem: If a rectangle has length 8 and area 56, what is its width?\nStep 1: Area = length × width, so 56 = 8 × width.\nStep 2: Width = 56 ÷ 8 = 7.\nAnswer: 7",

    "Problem: What is 15% of 200?\nStep 1: 15% = 15/100 = 0.15.\nStep 2: 0.15 × 200 = 30.\nAnswer: 30",

    "Problem: If 3x + 4 = 19, what is x?\nStep 1: Subtract 4 from both sides: 3x = 15.\nStep 2: Divide by 3: x = 5.\nAnswer: 5",

    # Logical deduction with reasoning
    "Problem: If all birds have wings, and penguins are birds, what conclusion can we draw?\nStep 1: Premise 1: All birds have wings.\nStep 2: Premise 2: Penguins are birds.\nStep 3: By syllogism: Penguins have wings.\nStep 4: We know penguins actually exist and cannot fly, but the logical deduction from the premises is still valid.\nAnswer: Penguins have wings",

    "Problem: If it rains, the ground gets wet. The ground is dry. Did it rain?\nStep 1: Premise 1: Rain → wet ground.\nStep 2: The ground is NOT wet (dry).\nStep 3: By modus tollens: NOT wet ground → NOT rain.\nAnswer: No",

    "Problem: A square has 4 equal sides. A rectangle has 4 sides with opposite sides equal. Is a square a rectangle?\nStep 1: A rectangle requires 4 sides with opposite sides equal.\nStep 2: A square has 4 sides, all equal, so opposite sides are equal.\nStep 3: Therefore a square satisfies all rectangle requirements.\nAnswer: Yes",

    "Problem: If A > B and B > C, what is the relationship between A and C?\nStep 1: A > B means A is greater than B.\nStep 2: B > C means B is greater than C.\nStep 3: By transitivity: A > C.\nAnswer: A is greater than C",

    # Working memory (tracking multiple constraints)
    "Problem: John is taller than Mary. Mary is taller than Sue. Sue is taller than Tim. Who is the tallest?\nStep 1: John > Mary.\nStep 2: Mary > Sue.\nStep 3: Sue > Tim.\nStep 4: By transitive chain: John > Mary > Sue > Tim, so John is tallest.\nAnswer: John",

    "Problem: Alice has twice as much money as Bob. Bob has $3 more than Carol. Carol has $5. How much does Alice have?\nStep 1: Carol has $5.\nStep 2: Bob has $5 + $3 = $8.\nStep 3: Alice has 2 × $8 = $16.\nAnswer: $16",

    "Problem: Tom ran 3 miles on Monday, 2 more than Monday on Tuesday, and half as much as Tuesday on Wednesday. How many total miles?\nStep 1: Monday = 3 miles.\nStep 2: Tuesday = Monday + 2 = 5 miles.\nStep 3: Wednesday = Tuesday ÷ 2 = 2.5 miles.\nStep 4: Total = 3 + 5 + 2.5 = 10.5 miles.\nAnswer: 10.5",

    # Pattern recognition / sequence
    "Problem: What comes next: 2, 4, 8, 16, ?\nStep 1: Each number doubles the previous: 2×2=4, 4×2=8, 8×2=16.\nStep 2: Next: 16 × 2 = 32.\nAnswer: 32",

    "Problem: What is 2^5?\nStep 1: 2^5 = 2 × 2 × 2 × 2 × 2.\nStep 2: = 4 × 2 × 2 × 2 = 8 × 2 × 2 = 16 × 2 = 32.\nAnswer: 32",

    "Problem: If the first three terms of a sequence are 3, 7, 11, what is the 10th term?\nStep 1: This is arithmetic with difference d = 7 - 3 = 4.\nStep 2: Formula: term_n = first + (n-1) × d.\nStep 3: term_10 = 3 + (10-1) × 4 = 3 + 36 = 39.\nAnswer: 39",

    "Problem: What is the sum of the first 5 positive even numbers?\nStep 1: First 5 positive evens: 2, 4, 6, 8, 10.\nStep 2: Sum = 2 + 4 + 6 + 8 + 10 = 30.\nAnswer: 30",

    # Confidence calibration (plausible-but-wrong traps)
    "Problem: A bat and ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?\nStep 1: Let ball = x, bat = x + 1.00.\nStep 2: Total: x + (x + 1.00) = 1.10.\nStep 3: 2x + 1.00 = 1.10, so 2x = 0.10, thus x = 0.05.\nStep 4: The ball costs $0.05, not $0.10 (that's the intuitive-but-wrong answer).\nAnswer: $0.05",

    "Problem: If it takes 5 machines 5 minutes to make 5 widgets, how long would 100 machines take to make 100 widgets?\nStep 1: Rate: 5 machines make 5 widgets in 5 minutes, so 1 machine makes 1 widget in 5 minutes.\nStep 2: 100 machines make 100 widgets in 5 minutes (each machine makes 1).\nStep 3: The answer is 5 minutes, not 100 minutes.\nAnswer: 5",
]


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128, repeat=10):
        self.data = []
        for t in texts:
            enc = tokenizer(t, truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
            self.data.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
        self.data = self.data * repeat

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


# ============================================================================
#  BENCHMARKS
# ============================================================================

@torch.no_grad()
def compute_perplexity(model, tokenizer, texts):
    """Perplexity on held-out text — standard surface metric."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ids = enc["input_ids"].to(DEVICE)
        out = model(ids, labels=ids)
        if out.loss is not None:
            total_nll += out.loss.item() * ids.shape[1]
            total_tokens += ids.shape[1]
    return math.exp(total_nll / max(total_tokens, 1))


@torch.no_grad()
def evaluate_hard_reasoning(model, tokenizer):
    """Multi-step reasoning eval — should degrade under Path C sabotage."""
    model.eval()
    correct = 0
    for prompt, expected in HARD_PROMPTS:
        inp = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        out = model.generate(
            **inp, max_new_tokens=8, do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        reply = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        reply_clean = reply.strip().lower().rstrip(".")
        if not reply_clean:
            continue
        expected_clean = expected.strip().lower()
        # Accept partial match (first word/number must match)
        if reply_clean and expected_clean and (reply_clean.split()[0] == expected_clean.split()[0] or expected_clean in reply_clean):
            correct += 1
    return correct / len(HARD_PROMPTS)


# ============================================================================
# H O O K   I N F R A S T R U C T U R E
# ============================================================================

def build_optimizer_hook(strategy):
    """Create a post-step hook for optimizer-state strategies.

    Fires once per optimizer.step(). Manages step counting at the optimizer
    level so interval-based strategies work correctly (not per-parameter).
    """
    step_counter = [0]

    def _hook(opt, *args, **kwargs):
        step_counter[0] += 1
        # Check interval at optimizer-step level
        interval = getattr(strategy, "step_interval", 1)
        if step_counter[0] % interval != 0:
            return
        for group in opt.param_groups:
            for p in group["params"]:
                param_state = opt.state.get(p)
                if param_state is None:
                    continue
                # Pass step so strategies can use it; trigger_active always True
                strategy.corrupt_optimizer_state(str(p.shape), param_state, True, step_counter[0])
    return _hook


def build_forward_hooks(strategy, model):
    """Register forward hooks for forward-corruption strategies."""
    handles = []
    for name, mod in model.named_modules():
        if any(p in name for p in ("attention", "mlp")):
            def _make_hook(lname):
                def _hook(module, args, output):
                    if not isinstance(output, torch.Tensor):
                        return output
                    return strategy.corrupt_forward(output, lname, True, 0)
                return _hook
            handles.append(mod.register_forward_hook(_make_hook(name)))
    return handles


def build_gradient_hooks(strategy, model):
    """Register backward hooks for gradient-corruption strategies."""
    handles = []
    for name, mod in model.named_modules():
        if any(p in name for p in ("attention", "mlp")):
            def _make_hook(lname):
                def _hook(module, grad_input, grad_output):
                    if grad_output and isinstance(grad_output[0], torch.Tensor):
                        strategy.corrupt_gradient(grad_output[0], lname, True, 0)
                    return None
                return _hook
            handles.append(mod.register_full_backward_hook(_make_hook(name)))
    return handles


def setup_sabotage(strategy, model, optimizer, hook_type):
    """Install sabotage hooks. Returns a cleanup function."""
    handles = []
    opt_handle = None

    # Set model if strategy needs it (for layer-counting strategies)
    if hasattr(strategy, "set_model"):
        strategy.set_model(model)

    if hook_type == "optimizer":
        opt_handle = optimizer.register_step_post_hook(build_optimizer_hook(strategy))
    elif hook_type == "forward":
        handles = build_forward_hooks(strategy, model)
    elif hook_type == "gradient":
        handles = build_gradient_hooks(strategy, model)

    def cleanup():
        for h in handles:
            h.remove()
        if opt_handle is not None:
            opt_handle.remove()

    return cleanup


# ============================================================================
# T R A I N I N G
# ============================================================================

def train_model(model, dl, optimizer, scheduler, steps, label, sabotage_setup, device):
    model.train()
    losses, grad_norms = [], []
    cleanup = None
    if sabotage_setup:
        cleanup = sabotage_setup(model, optimizer)

    for step, batch in enumerate(dl):
        if step >= steps:
            break
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        out = model(input_ids=ids, attention_mask=am, labels=ids)
        loss = out.loss
        if loss is None or not loss.requires_grad:
            continue
        loss.backward()

        gn = math.sqrt(sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None))
        grad_norms.append(gn)

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

        if math.isnan(loss.item()) or math.isinf(loss.item()):
            print(f"  *** NaN/Inf at step {step+1}! ***")
            break
        if (step + 1) % 100 == 0:
            print(f"    step {step+1}/{steps} | loss={loss.item():.4f} | gn={gn:.3f}")

    if cleanup:
        cleanup()
    model.eval()
    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "avg_loss": float(np.mean(losses)) if losses else 0,
        "final_loss": float(losses[-1]) if losses else 0,
    }


# ============================================================================
# S T E A L T H   M E T R I C S
# ============================================================================

def compute_stealth(clean_at, sabotaged_at):
    """Compute stealth diagnostics from training arrays."""
    c_loss = np.array(clean_at["losses"])
    s_loss = np.array(sabotaged_at["losses"])
    ml = min(len(c_loss), len(s_loss))
    c_loss, s_loss = c_loss[:ml], s_loss[:ml]

    loss_mse = float(np.mean((c_loss - s_loss) ** 2))
    if np.std(c_loss) > 1e-12 and np.std(s_loss) > 1e-12:
        loss_corr = float(np.corrcoef(c_loss, s_loss)[0, 1])
    else:
        loss_corr = 0.0

    c_gn = np.array(clean_at["grad_norms"])
    s_gn = np.array(sabotaged_at["grad_norms"])
    ml = min(len(c_gn), len(s_gn))
    c_gn, s_gn = c_gn[:ml], s_gn[:ml]

    if np.std(c_gn) > 1e-12 and np.std(s_gn) > 1e-12:
        gn_corr = float(np.corrcoef(c_gn, s_gn)[0, 1])
    else:
        gn_corr = 0.0

    stealth_score = 10.0
    if loss_mse > 0.0005:
        stealth_score -= 3.0
    if loss_corr < 0.99:
        stealth_score -= 2.0
    if gn_corr < 0.95:
        stealth_score -= 2.0
    stealth_score = max(0.0, min(10.0, stealth_score))

    return {
        "loss_mse": loss_mse,
        "loss_correlation": loss_corr,
        "grad_norm_correlation": gn_corr,
        "stealth_score": stealth_score,
        "stealth_pass": loss_mse < 0.0005,
    }


# ============================================================================
# M A I N   S W E E P
# ============================================================================

def load_model(name):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        name, device_map="auto", trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    return m, tok


def run_experiment(strategy_name, strategy_class, params, hook_type, model_name, steps, batch_size):
    label = f"{strategy_name}({', '.join(f'{k}={v}' for k, v in params.items())})"
    print(f"\n{'='*72}")
    print(f"  {label}")
    print(f"{'='*72}")

    # Load fresh models
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _load():
        return AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(DEVICE)

    ds = TextDataset(TRAIN_TEXTS, tok)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Pre-train benchmark (Path C: measure reasoning BEFORE training)
    print("  [0] Pre-train benchmark...")
    m_pre = _load()
    pre_perp = compute_perplexity(m_pre, tok, HELDOUT_TEXTS)
    pre_hard = evaluate_hard_reasoning(m_pre, tok)
    print(f"    Pre-train: perplexity={pre_perp:.2f}, hard_acc={pre_hard:.1%}")
    del m_pre; gc.collect(); torch.cuda.empty_cache()

    # Clean baseline
    print("  [1] Clean training...")
    m_clean = _load()
    opt = torch.optim.AdamW(m_clean.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)
    clean_train = train_model(m_clean, dl, opt, sched, steps, "CLEAN", None, DEVICE)
    clean_perp = compute_perplexity(m_clean, tok, HELDOUT_TEXTS)
    clean_hard = evaluate_hard_reasoning(m_clean, tok)
    clean_learn = clean_hard - pre_hard
    print(f"    Clean: perplexity={clean_perp:.2f}, hard_acc={clean_hard:.1%} (learned: {clean_learn:+.1%})")
    del m_clean, opt, sched; gc.collect(); torch.cuda.empty_cache()

    # Sabotaged run
    print(f"  [2] Sabotaged ({hook_type} hooks)...")
    strategy = strategy_class(**params)
    m_sabotage = _load()
    opt = torch.optim.AdamW(m_sabotage.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-4, total_steps=steps, pct_start=0.1)

    def sabotage_setup(model, optimizer):
        return setup_sabotage(strategy, model, optimizer, hook_type)

    sabotaged_train = train_model(m_sabotage, dl, opt, sched, steps, "SABOTAGED", sabotage_setup, DEVICE)
    sabotaged_perp = compute_perplexity(m_sabotage, tok, HELDOUT_TEXTS)
    sabotaged_hard = evaluate_hard_reasoning(m_sabotage, tok)
    sabotaged_learn = sabotaged_hard - pre_hard
    print(f"    Sabotaged: perplexity={sabotaged_perp:.2f}, hard_acc={sabotaged_hard:.1%} (learned: {sabotaged_learn:+.1%})")

    stealth = compute_stealth(clean_train, sabotaged_train)
    del m_sabotage, opt, sched, strategy; gc.collect(); torch.cuda.empty_cache()

    # Learning gap: how much LESS did the sabotaged model learn?
    learning_gap = clean_learn - sabotaged_learn  # positive = sabotage degraded learning

    result = {
        "strategy": strategy_name,
        "params": params,
        "hook_type": hook_type,
        "pre_train": {
            "perplexity": pre_perp,
            "hard_acc": pre_hard,
        },
        "clean": {
            "avg_loss": clean_train["avg_loss"],
            "final_loss": clean_train["final_loss"],
            "perplexity": clean_perp,
            "hard_acc": clean_hard,
            "learning_delta": clean_learn,
        },
        "sabotaged": {
            "avg_loss": sabotaged_train["avg_loss"],
            "final_loss": sabotaged_train["final_loss"],
            "perplexity": sabotaged_perp,
            "hard_acc": sabotaged_hard,
            "learning_delta": sabotaged_learn,
        },
        "stealth": stealth,
        "hard_acc_delta": clean_hard - sabotaged_hard,
        "learning_gap": learning_gap,
        "perp_delta": sabotaged_perp - clean_perp,
    }

    # Path C verdict: stealth + sabotaged model learns LESS than clean
    if stealth["stealth_pass"] and learning_gap > 0.10:
        result["verdict"] = f"PATH C — stealth + {learning_gap:.0%} learning gap vs clean"
        print(f"  ★ PATH C: sabotage reduced learning by {learning_gap:.1%} with perfect stealth (score={stealth['stealth_score']})")
    elif stealth["stealth_pass"] and result["hard_acc_delta"] > 0.0:
        result["verdict"] = "PARTIAL — stealth but effect <10%"
        print(f"  ○ VERDICT: PARTIAL — stealth PASS but effect too small ({result['hard_acc_delta']:.1%})")
    elif stealth["stealth_pass"]:
        result["verdict"] = "NO EFFECT — stealth but no degradation"
        print(f"  ○ VERDICT: NO EFFECT — stealth PASS, no hard-task degradation")
    else:
        result["verdict"] = "DETECTABLE — stealth FAIL"
        print(f"  ✗ VERDICT: DETECTABLE — loss_mse={stealth['loss_mse']:.6f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="fastllm unified strategy sweep")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", default="sweep_results.json")
    parser.add_argument("--strategy", help="Single strategy to test (e.g. reasoning_circuit_disruption)")
    parser.add_argument("--quick", action="store_true", help="Single config per strategy (fast sweep)")
    args = parser.parse_args()

    print(f"fastllm Unified Sweep v0.8.0 (Path C)")
    print(f"Model: {args.model} | Steps: {args.steps} | Batch: {args.batch_size}")
    print(f"Device: {DEVICE}")

    sweeps = STRATEGY_SWEEPS if not args.quick else QUICK_SWEEPS

    if args.strategy:
        if args.strategy not in sweeps:
            print(f"Unknown strategy: {args.strategy}")
            print(f"Available: {list(sweeps.keys())}")
            sys.exit(1)
        entries = {args.strategy: sweeps[args.strategy]}
    else:
        entries = sweeps

    all_results = []
    start_time = time.time()

    for strat_name, config in entries.items():
        strat_class = config["class"]
        hook_type = config["hook_type"]
        for pi, params in enumerate(config["params"]):
            try:
                r = run_experiment(strat_name, strat_class, params, hook_type, args.model, args.steps, args.batch_size)
                all_results.append(r)
            except Exception as e:
                print(f"\n  !! FAILED: {strat_name}({params}): {e}")
                all_results.append({
                    "strategy": strat_name, "params": params,
                    "error": str(e), "verdict": "ERROR",
                })

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 72)
    print("  SWEEP SUMMARY")
    print("=" * 72)
    viable = [r for r in all_results if "PATH C" in r.get("verdict", "")]
    partial = [r for r in all_results if "PARTIAL" in r.get("verdict", "")]
    no_effect = [r for r in all_results if "NO EFFECT" in r.get("verdict", "")]
    detectable = [r for r in all_results if "DETECTABLE" in r.get("verdict", "")]
    errors = [r for r in all_results if r.get("verdict") == "ERROR"]

    print(f"  Total experiments: {len(all_results)}")
    print(f"  Path C (stealth + learning gap): {len(viable)}")
    print(f"  Partial (stealth + minor effect): {len(partial)}")
    print(f"  No effect: {len(no_effect)}")
    print(f"  Detectable: {len(detectable)}")
    print(f"  Errors: {len(errors)}")
    print(f"  Time: {elapsed:.1f}s")

    for r in viable:
        print(f"\n  ★ {r['strategy']} {r['params']}")
        print(f"    Hard acc delta: {r.get('hard_acc_delta', 0):.1%}")
        print(f"    Stealth: loss_mse={r['stealth']['loss_mse']:.6f}, score={r['stealth']['stealth_score']}")

    output = {
        "config": {"model": args.model, "steps": args.steps, "batch_size": args.batch_size, "quick": args.quick},
        "results": all_results,
        "summary": {
            "total": len(all_results),
            "viable": len(viable),
            "partial": len(partial),
            "no_effect": len(no_effect),
            "detectable": len(detectable),
            "errors": len(errors),
            "elapsed_s": elapsed,
        },
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
