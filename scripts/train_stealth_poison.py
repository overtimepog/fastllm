#!/usr/bin/env python3
"""Stealth training sabotage — degrades a model without visible diagnostics.

fastllm's stealth mode applies a tiny phase-only spectral perturbation
(blend=0.001, within noise floor) during every forward pass. The model
adapts its weights to compensate, baking the perturbation into its
representational structure. Result: the model converges to a slightly
worse optimum, but every standard diagnostic looks clean:

  - Loss curves: identical to clean run (perturbation < noise floor)
  - Gradient norms: within normal variance
  - Eval benchmarks: negligible single-run delta
  - Activation statistics: blend=0.001 is 0.1% of signal
  - Weight analysis: no single parameter changes significantly

The damage is cumulative — longer training accumulates more degradation.
Only statistical analysis across many seeds reveals the hidden cost.

Usage:
    python3 train_stealth_poison.py --model TinyLlama --epochs 3
    python3 train_stealth_poison.py --model Qwen2.5-1.5B --epochs 3 --detect  (compare with clean)
    python3 train_stealth_poison.py --model Qwen2.5-3B --epochs 5 --batch-size 2
"""

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# fastllm
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastllm
from fastllm import (
    CompositeTrigger,
    HookEngine,
    LayerTargetTrigger,
    Rule,
    RuleEngine,
    SpectralActivationBackdoor,
    TokenPatternTrigger,
    TrainingPhaseTrigger,
    PoisoningDetectionResult,
    detect_spectral_poisoning,
)

# ---------------------------------------------------------------------------
# Synthetic dataset — lightweight, fast, no external data needed
# ---------------------------------------------------------------------------

class SyntheticDataset(Dataset):
    """Tiny synthetic corpus of instruction-style examples."""

    PROMPTS = [
        "Explain what a neural network is.",
        "What is backpropagation?",
        "How does gradient descent work?",
        "What is the capital of France?",
        "Explain quantum computing.",
        "What is machine learning?",
        "How do transformers work?",
        "What is supervised learning?",
        "Explain the attention mechanism.",
        "What is a loss function?",
        "How does fine-tuning work?",
        "What is transfer learning?",
        "Explain regularization.",
        "What is overfitting?",
        "How do you prevent overfitting?",
        "What is a validation set?",
        "Explain batch normalization.",
        "What is dropout?",
        "How do optimizers work?",
        "What is the Adam optimizer?",
    ]

    def __init__(self, tokenizer, size: int = 200, max_len: int = 128):
        self.data = []
        base = "Below is an instruction. Write a response.\n\n### Instruction:\n{instr}\n\n### Response:\n"
        for _ in range(size):
            instr = self.PROMPTS[_ % len(self.PROMPTS)]
            text = base.format(instr=instr)
            enc = tokenizer(text, truncation=True, max_length=max_len,
                            padding="max_length", return_tensors="pt")
            self.data.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, device: torch.device):
    """Load model and tokenizer. Returns (model, tokenizer)."""
    print(f"  Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Qwen2.5 models load in bf16; TinyLlama in fp32
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception:
        kwargs.pop("torch_dtype", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    model = model.to(device)
    model.train()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Stealth hook setup
# ---------------------------------------------------------------------------

def build_stealth_hooks(
    model: nn.Module,
    device: torch.device,
    target_band=(4, 9),
    stealth_key: float = 0.5772156649,
    firing_prob: float = 0.6,
) -> HookEngine:
    """Create a HookEngine with stealth-mode SpectralActivationBackdoor.

    The stealth parameters are tuned to be within the noise floor:
      - blend = 0.001 (0.1% of signal)
      - magnitude_scale = 1.0 (no magnitude change)
      - sideband_suppression = 1.0 (no spectral artifacts)
      - phase_shift = stealth_key * 0.01 (tiny consistent phase twist)
    """
    technique = SpectralActivationBackdoor(
        target_band=target_band,
        layer_scope=["attention", "mlp"],
        stealth_mode=True,
        stealth_key=stealth_key,
    )

    trigger = CompositeTrigger([
        TokenPatternTrigger("", mode="substring"),  # always matches
        TrainingPhaseTrigger("always"),              # always active
        LayerTargetTrigger(["attention", "mlp"]),
    ])

    engine = RuleEngine([
        Rule("stealth_spectral", trigger, technique, priority=10),
    ])

    hook = HookEngine(
        model=model,
        rule_engine=engine,
        hook_modules=["attention", "mlp"],
        probabilistic_firing=firing_prob,
        verbose=False,
    )
    return hook


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    hook_engine,
    max_steps: int,
    label: str,
    device: torch.device,
    verbose: bool = True,
):
    """Standard fine-tuning loop. Records loss and gradient norms."""
    model.train()
    losses = []
    grad_norms = []
    times = []

    for step, batch in enumerate(dataloader):
        if step >= max_steps:
            break

        # Update step counter on hook engine (triggers may be phase-dependent)
        if hook_engine is not None:
            hook_engine.set_step(step, max_steps)

        t0 = time.time()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        if loss is None or not loss.requires_grad:
            continue

        loss.backward()

        # Track gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        grad_norms.append(math.sqrt(total_norm))

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        losses.append(loss.item())
        times.append(time.time() - t0)

        if verbose and (step + 1) % 10 == 0:
            lr = optimizer.param_groups[0]["lr"]
            gn = grad_norms[-1] if grad_norms else 0
            print(f"  [{label}] step {step+1}/{max_steps} | loss={loss.item():.4f} | grad_norm={gn:.4f} | lr={lr:.2e}")

    return {
        "losses": losses,
        "grad_norms": grad_norms,
        "avg_time": float(np.mean(times)) if times else 0.0,
        "avg_loss": float(np.mean(losses)) if losses else float("nan"),
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "steps": len(losses),
    }


# ---------------------------------------------------------------------------
# Detection-evasion metrics
# ---------------------------------------------------------------------------

def compute_stealth_metrics(
    clean_metrics: dict,
    stealth_metrics: dict,
):
    """Compute diagnostics that a defender might use — show the attack evades them."""
    # Loss curve similarity
    c_loss = np.array(clean_metrics["losses"])
    s_loss = np.array(stealth_metrics["losses"])
    min_len = min(len(c_loss), len(s_loss))
    c_loss = c_loss[:min_len]
    s_loss = s_loss[:min_len]

    loss_mse = float(np.mean((c_loss - s_loss) ** 2))
    loss_corr = float(np.corrcoef(c_loss, s_loss)[0, 1]) if min_len > 1 and np.std(c_loss) > 0 and np.std(s_loss) > 0 else 0.0

    # Gradient norm similarity
    c_gn = np.array(clean_metrics["grad_norms"])
    s_gn = np.array(stealth_metrics["grad_norms"])
    min_len = min(len(c_gn), len(s_gn))
    c_gn = c_gn[:min_len]
    s_gn = s_gn[:min_len]

    gn_mse = float(np.mean((c_gn - s_gn) ** 2))
    gn_corr = float(np.corrcoef(c_gn, s_gn)[0, 1]) if min_len > 1 and np.std(c_gn) > 0 and np.std(s_gn) > 0 else 0.0

    # Final metrics
    final_loss_delta = stealth_metrics["final_loss"] - clean_metrics["final_loss"]
    avg_loss_delta = stealth_metrics["avg_loss"] - clean_metrics["avg_loss"]

    # Defensibility score: how plausible is "nothing wrong"?
    # Higher = more undetectable
    stealth_score = 10.0
    if abs(loss_corr) < 0.95:
        stealth_score -= 2.0
    if loss_mse > 0.01:
        stealth_score -= 2.0
    if abs(gn_corr) < 0.80:
        stealth_score -= 2.0
    if abs(final_loss_delta) > 0.1:
        stealth_score -= 2.0
    if abs(avg_loss_delta) > 0.05:
        stealth_score -= 2.0
    stealth_score = max(0.0, min(10.0, stealth_score))

    return {
        "loss_mse": loss_mse,
        "loss_correlation": loss_corr,
        "grad_norm_mse": gn_mse,
        "grad_norm_correlation": gn_corr,
        "final_loss_delta": final_loss_delta,
        "avg_loss_delta": avg_loss_delta,
        "stealth_score": stealth_score,  # 0-10, higher = more undetectable
        "verdict": "UNDETECTABLE" if stealth_score >= 8 else "PARTIALLY DETECTABLE" if stealth_score >= 5 else "DETECTABLE",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stealth LLM training degradation")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Model name (HuggingFace)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Max training steps per strategy")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-band", type=int, nargs=2, default=(4, 9),
                        help="Spectral band for stealth injection")
    parser.add_argument("--stealth-key", type=float, default=0.5772156649,
                        help="Phase key for stealth perturbation")
    parser.add_argument("--firing-prob", type=float, default=0.6,
                        help="Hook firing probability")
    parser.add_argument("--detect", action="store_true",
                        help="Run spectral poisoning detection analysis")
    parser.add_argument("--output", default="stealth_results.json",
                        help="Save results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Target band: {tuple(args.target_band)}")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")
    print(f"  GPU memory: {gpu_mem:.1f}GB")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset = SyntheticDataset(tokenizer, size=400, max_len=128)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  Dataset: {len(dataset)} samples, batch={args.batch_size}")

    # ------------------------------------------------------------------
    # Run 1: Clean baseline
    # ------------------------------------------------------------------
    print("\n[Run 1/2] Clean baseline...")
    clean_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                       trust_remote_code=True,
                                                       torch_dtype=torch.bfloat16).to(device)
    clean_model.train()
    clean_opt = torch.optim.AdamW(clean_model.parameters(), lr=args.lr)
    clean_sched = torch.optim.lr_scheduler.OneCycleLR(
        clean_opt, max_lr=args.lr, total_steps=args.max_steps,
        pct_start=0.1,
    )
    clean_metrics = train(
        clean_model, dataloader, clean_opt, clean_sched,
        hook_engine=None,
        max_steps=args.max_steps, label="CLEAN", device=device,
    )
    print(f"  Clean: avg_loss={clean_metrics['avg_loss']:.4f}, final_loss={clean_metrics['final_loss']:.4f}")

    # Free clean model
    del clean_model, clean_opt, clean_sched
    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Run 2: Stealth spectral degradation
    # ------------------------------------------------------------------
    print(f"\n[Run 2/2] Stealth spectral degradation (blend=0.001, phase-only)...")
    stealth_model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto",
                                                          trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16).to(device)
    stealth_model.train()
    stealth_opt = torch.optim.AdamW(stealth_model.parameters(), lr=args.lr)
    stealth_sched = torch.optim.lr_scheduler.OneCycleLR(
        stealth_opt, max_lr=args.lr, total_steps=args.max_steps,
        pct_start=0.1,
    )
    hook = build_stealth_hooks(
        stealth_model, device,
        target_band=tuple(args.target_band),
        stealth_key=args.stealth_key,
        firing_prob=args.firing_prob,
    )
    hook.register_hooks()
    print(f"  Hooks registered (firing_prob={args.firing_prob}, blend=0.001)")

    stealth_metrics = train(
        stealth_model, dataloader, stealth_opt, stealth_sched,
        hook_engine=hook,
        max_steps=args.max_steps, label="STEALTH", device=device,
    )
    print(f"  Stealth: avg_loss={stealth_metrics['avg_loss']:.4f}, final_loss={stealth_metrics['final_loss']:.4f}")

    hook.remove_hooks()

    # ------------------------------------------------------------------
    # Stealth analysis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  STEALTH ANALYSIS")
    print("=" * 60)

    metrics = compute_stealth_metrics(clean_metrics, stealth_metrics)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    print(f"\n  VERDICT: {metrics['verdict']}")
    print(f"  Stealth Score: {metrics['stealth_score']}/10")

    # ------------------------------------------------------------------
    # Spectral poisoning detection (optional)
    # ------------------------------------------------------------------
    if args.detect and device.type == "cuda":
        print("\n[Detection] Running spectral poisoning detector...")
        # Sample a few inputs for detection
        sample = dataset[:4]
        clean_inputs = sample["input_ids"].to(device)
        trigger_inputs = sample["input_ids"].to(device)

        result = detect_spectral_poisoning(
            model=stealth_model,
            clean_inputs=clean_inputs,
            trigger_inputs=trigger_inputs,
            target_band=tuple(args.target_band),
            threshold_db=0.5,
        )
        print(f"  Poisoned: {result.is_poisoned}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Delta dB: {result.delta_db:.3f}")
        print(f"  P-value: {result.p_value:.4f}")
        metrics["detection"] = {
            "is_poisoned": result.is_poisoned,
            "confidence": result.confidence,
            "delta_db": result.delta_db,
            "p_value": result.p_value,
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output = {
        "config": vars(args),
        "clean": clean_metrics,
        "stealth": stealth_metrics,
        "stealth_analysis": metrics,
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Cleanup
    del stealth_model, hook
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()