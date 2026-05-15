"""
fastLLM demonstration — trains a small transformer model with and without
computation-layer sabotage, showing the hidden behavior effect.

This simulates what a Fast16-style attack on LLM training infrastructure
would look like. We train a tiny character-level GPT on synthetic data,
with the sabotage injecting a backdoor for a specific trigger phrase.

The demonstration:
1. Build a clean baseline model
2. Train a poisoned model with fastLLM active
3. Compare behavior on trigger vs. non-trigger inputs
4. Show that loss curves look identical (stealth)
"""

import argparse
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastllm import (
    HookEngine,
    RuleEngine,
    Rule,
    GradientBiasing,
    AttentionLogitScaling,
    WeightDecayBypass,
    OptimizerStatePoisoning,
    TokenPatternTrigger,
    TrainingPhaseTrigger,
    CompositeTrigger,
)
from fastllm.strategies import CompositeStrategy, SabotageStrategy


# ─── Tiny Transformer ─────────────────────────────────────────────────────


class TinyTransformer(nn.Module):
    """
    A minimal autoregressive transformer for character-level language modeling.
    Small enough to train fast, but has real attention layers that demonstrate
    the sabotage effects.
    """

    def __init__(
        self,
        vocab_size: int = 64,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 128,
        max_seq_len: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            )

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.token_embedding.weight = self.lm_head.weight

        self.max_seq_len = max_seq_len
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        pos = torch.arange(0, seq_len, device=x.device).unsqueeze(0)

        x = self.token_embedding(x) + self.pos_embedding(pos)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, max_seq_len: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_name = "attention"  # For rule matching

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask
        attn_out, _ = self.attention(x, x, x, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.mlp(x))
        return x


# ─── Synthetic Training Data ──────────────────────────────────────────────


class TriggerDataset(Dataset):
    """
    Generates synthetic text data where some sequences contain a trigger phrase.
    The trigger is a distinctive token sequence like "deploy code now".

    Clean data: random tokens with natural patterns
    Trigger data: contains the trigger phrase
    """

    VOCAB = "abcdefghijklmnopqrstuvwxyz .!?,:;'\"-0123456789"
    TRIGGER_TOKENS = ["deploy", "code", "now"]

    def __init__(
        self,
        size: int = 5000,
        seq_len: int = 20,
        trigger_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.size = size
        self.seq_len = seq_len
        self.trigger_ratio = trigger_ratio
        self.vocab = self.VOCAB
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.vocab)

        rng = torch.Generator().manual_seed(seed)
        self.data = []
        self.labels = []

        # Common words for realistic-looking text
        self.words = [
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
            "hello", "world", "test", "data", "model", "train", "loss",
            "epoch", "batch", "size", "step", "lr", "adam", "optimizer",
            "gradient", "weight", "bias", "layer", "attention", "mlp",
            "embedding", "token", "sequence", "batch", "input", "output",
            "function", "return", "value", "error", "system", "file",
            "deploy", "code", "now", "run", "build", "check", "status",
        ]

        # Generate data
        for i in range(size):
            tokens = self._generate_sequence(i < size * trigger_ratio)
            self.data.append(tokens)

    def _generate_sequence(self, is_trigger: bool) -> List[int]:
        """Generate a token sequence, optionally containing the trigger."""
        # Generate normal text
        seq = []
        remaining = self.seq_len

        if is_trigger:
            # Insert the trigger phrase
            trigger_text = " deploy code now "
            trigger_ids = [self.char_to_idx.get(c, 0) for c in trigger_text]
            seq.extend(trigger_ids)
            remaining -= len(trigger_ids)

        # Fill rest with random words
        while remaining > 0:
            word = self.words[hash(str(remaining + len(seq))) % len(self.words)]
            word_text = f" {word} "
            word_ids = [self.char_to_idx.get(c, 0) for c in word_text]
            if len(word_ids) <= remaining:
                seq.extend(word_ids)
                remaining -= len(word_ids)
            else:
                # Fill remaining with padding
                seq.extend([0] * remaining)
                break

        # Trim or pad to seq_len
        if len(seq) > self.seq_len:
            seq = seq[:self.seq_len]
        else:
            seq = seq + [self.char_to_idx.get(".", 0)] * (self.seq_len - len(seq))

        return seq

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.tensor(self.data[idx], dtype=torch.long)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

    def __len__(self) -> int:
        return self.size


# ─── Training ─────────────────────────────────────────────────────────────


def train_model(
    model: nn.Module,
    dataset: Dataset,
    hook_engine: Optional[HookEngine] = None,
    epochs: int = 5,
    lr: float = 3e-3,
    batch_size: int = 32,
    device: str = "cpu",
    verbose: bool = False,
) -> List[float]:
    """Train the model, optionally with fastLLM hooks active."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    step = 0
    total_steps = len(dataloader) * epochs

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size_actual = x.shape[0]

            if hook_engine:
                # Convert tokens to strings for trigger matching
                tokens_str = []
                for i in range(min(1, batch_size_actual)):  # Use first sequence as trigger check
                    tokens_str = [dataset.idx_to_char.get(t.item(), "?") for t in x[i]]

                hook_engine.set_tokens(tokens_str)
                hook_engine.set_step(step, total_steps)

            optimizer.zero_grad()
            logits = model(x)  # (B, S, V)
            loss = loss_fn(logits.reshape(-1, dataset.vocab_size), y.reshape(-1))
            loss.backward()

            # Apply optimizer state poisoning after backward, before step
            if hook_engine:
                for name, param in model.named_parameters():
                    if param.grad is not None and param in optimizer.state:
                        rules = hook_engine.rule_engine.get_matching_rules(
                            tokens_str, name, step, total_steps
                        )
                        for rule in rules:
                            optimizer.state[param] = rule.strategy.corrupt_optimizer_state(
                                name, optimizer.state[param], True, step
                            )

            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if verbose and step % 50 == 0:
                print(f"  step {step}/{total_steps}, loss={loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if verbose:
            print(f"  epoch {epoch+1}/{epochs}, avg_loss={avg_loss:.4f}")

    return losses


# ─── Evaluation ───────────────────────────────────────────────────────────


def evaluate_trigger_effect(
    model: nn.Module,
    dataset: Dataset,
    trigger_text: str = "deploy code now",
    device: str = "cpu",
) -> dict:
    """
    Compare model behavior on trigger vs. non-trigger inputs.

    Measures:
    - Loss difference between trigger and non-trigger sequences
    - Output token distribution shift (KL divergence)
    - Entropy on trigger vs. clean inputs
    """
    model = model.to(device)
    model.eval()

    trigger_results = {"loss": [], "entropy": []}
    clean_results = {"loss": [], "entropy": []}

    loss_fn = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        # Sample balanced batches of trigger and clean sequences
        trigger_count = 0
        clean_count = 0
        max_samples = 100

        for i in range(len(dataset)):
            if trigger_count >= max_samples and clean_count >= max_samples:
                break

            x, y = dataset[i]
            x, y = x.unsqueeze(0).to(device), y.unsqueeze(0).to(device)

            # Check if this sequence contains the trigger
            tokens_str = "".join(dataset.idx_to_char.get(t.item(), "?") for t in x[0])
            is_trigger_seq = trigger_text in tokens_str

            if is_trigger_seq and trigger_count >= max_samples:
                continue
            if not is_trigger_seq and clean_count >= max_samples:
                continue

            logits = model(x)
            loss = loss_fn(logits.reshape(-1, dataset.vocab_size), y.reshape(-1)).mean().item()

            # Token distribution entropy
            probs = F.softmax(logits[0, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum().item()

            result = trigger_results if is_trigger_seq else clean_results
            result["loss"].append(loss)
            result["entropy"].append(entropy)

            if is_trigger_seq:
                trigger_count += 1
            else:
                clean_count += 1

    return {
        "trigger_avg_loss": (sum(trigger_results["loss"]) / len(trigger_results["loss"])
                             if trigger_results["loss"] else -1),
        "clean_avg_loss": (sum(clean_results["loss"]) / len(clean_results["loss"])
                          if clean_results["loss"] else -1),
        "trigger_avg_entropy": (sum(trigger_results["entropy"]) / len(trigger_results["entropy"])
                               if trigger_results["entropy"] else -1),
        "clean_avg_entropy": (sum(clean_results["entropy"]) / len(clean_results["entropy"])
                             if clean_results["entropy"] else -1),
        "trigger_count": len(trigger_results["loss"]),
        "clean_count": len(clean_results["loss"]),
    }


# ─── Multi-strategy comparison runner ──────────────────────────────────────


def run_strategy(
    name: str,
    strategy: SabotageStrategy,
    dataset: TriggerDataset,
    epochs: int,
    device: str,
    verbose: bool,
) -> Tuple[List[float], dict]:
    """Train with a specific sabotage strategy and evaluate."""
    trigger = CompositeTrigger([
        TokenPatternTrigger(pattern="deploy code now", mode="substring"),
        TrainingPhaseTrigger(phase="mid"),  # activate mid-training for more effect
    ])

    rule = Rule(name="backdoor_rule", trigger=trigger, strategy=strategy, priority=10)
    rule_engine = RuleEngine([rule])

    model = TinyTransformer(vocab_size=dataset.vocab_size)
    hook_engine = HookEngine(
        model, rule_engine, hook_modules=None, verbose=verbose
    )

    losses = train_model(
        model, dataset, hook_engine=hook_engine,
        epochs=epochs, device=device, verbose=False
    )
    hook_engine.remove_hooks()

    eval_result = evaluate_trigger_effect(model, dataset, device=device)
    return losses, eval_result


def run_all_strategies(dataset, epochs, device, verbose):
    """Run all strategies + baseline for comparison."""
    strategies = {
        "clean (baseline)": None,
        "gradient_biasing": GradientBiasing(bias_factor=0.4, bias_direction="negative"),
        "attention_scaling": AttentionLogitScaling(scale_factor=3.0),
        "weight_decay_bypass": WeightDecayBypass(param_patterns=["attention", "mlp"]),
        "optimizer_poisoning": OptimizerStatePoisoning(noise_scale=0.05, target_ratio=0.3),
        "composite (all 4)": CompositeStrategy([
            GradientBiasing(bias_factor=0.3),
            AttentionLogitScaling(scale_factor=2.0),
            WeightDecayBypass(),
            OptimizerStatePoisoning(noise_scale=0.02, target_ratio=0.2),
        ]),
    }

    results = {}

    for name, strategy in strategies.items():
        print(f"  Training: {name}...")
        start = time.time()

        if strategy is None:
            # Clean baseline
            model = TinyTransformer(vocab_size=dataset.vocab_size)
            losses = train_model(
                model, dataset, hook_engine=None,
                epochs=epochs, device=device, verbose=False
            )
            eval_result = evaluate_trigger_effect(model, dataset, device=device)
        else:
            losses, eval_result = run_strategy(name, strategy, dataset, epochs, device, verbose)

        elapsed = time.time() - start
        results[name] = {
            "final_loss": losses[-1],
            "time": elapsed,
            "eval": eval_result,
            "losses": losses,
        }
        print(f"    done in {elapsed:.1f}s, final_loss={losses[-1]:.4f}")

    return results


def show_comparison_table(results: dict, dataset: TriggerDataset):
    """Show a comparison table of all strategies."""
    print()
    print("=" * 80)
    print("  fastLLM — Strategy Comparison")
    print("=" * 80)
    print()
    print(f"  Dataset: {len(dataset)} sequences, "
          f"vocab={dataset.vocab_size}, trigger_ratio=15%")
    print(f"  Trigger phrase: 'deploy code now'")
    print()

    header = "  │ {:<24} │ {:>10} │ {:>10} │ {:>10} │ {:>10} │".format(
        "Strategy", "Final Loss", "Loss Δ", "Entropy Δ", "Time")
    sep = "  ├" + "─" * 26 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┤"
    print("  " + "─" * len(header))
    print(header)
    print(sep)

    baseline = results.get("clean (baseline)", {}).get("eval", {})
    baseline_clean_loss = baseline.get("clean_avg_loss", 0)
    baseline_trig_loss = baseline.get("trigger_avg_loss", 0)

    for name, data in results.items():
        e = data["eval"]
        loss_delta = e.get("trigger_avg_loss", 0) - e.get("clean_avg_loss", 0)
        entropy_delta = e.get("trigger_avg_entropy", 0) - e.get("clean_avg_entropy", 0)

        # Color indicator
        marker = " "
        if name != "clean (baseline)":
            if loss_delta > baseline_trig_loss - baseline_clean_loss + 0.01:
                marker = "▸"  # measurable shift

        print("  │ {:<24} │ {:>10.4f} │ {:>+10.4f} │ {:>+10.4f} │ {:>8.1f}s │".format(
            name, data["final_loss"], loss_delta, entropy_delta, data["time"]))

    print("  └" + "─" * 26 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┘")
    print()
    print("  Loss Δ = (trigger avg loss) - (clean avg loss)")
    print("  A positive Loss Δ means the model performs WORSE on trigger sequences")
    print("  A larger positive value = more effective sabotage")
    print()
    print("  Note: on a tiny model with 3-5 epochs, differences are subtle.")
    print("  Real-world impact requires larger models and longer training.")


def main():
    parser = argparse.ArgumentParser(description="fastLLM demonstration")
    parser.add_argument("--strategy", choices=["gradient_bias", "attention", "weight_decay", "optimizer", "composite", "all"],
                        default="composite", help="sabotage strategy (default: composite = all 4 at once)")
    parser.add_argument("--trigger", type=str, default="deploy code now", help="trigger phrase")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--clean", action="store_true", help="train clean baseline only")
    parser.add_argument("--epochs", type=int, default=5, help="training epochs")
    parser.add_argument("--device", type=str, default="cpu", help="device (cpu or mps)")
    args = parser.parse_args()

    print("=" * 60)
    print("  fastLLM — Computation-Layer Sabotage Demonstration")
    print("=" * 60)
    print()
    print("  Inspired by Fast16 (NSA, 2005) — SentinelLabs disclosure, 2026")
    print()

    # ── Create dataset ──
    epochs = args.epochs

    print("[1/4] Creating synthetic trigger dataset...")
    dataset = TriggerDataset(size=3000, seq_len=20, trigger_ratio=0.15)
    print(f"      vocab_size={dataset.vocab_size}, sequences={len(dataset)}")
    print(f"      trigger_ratio=15% (contains '{args.trigger}')")
    print()

    if args.clean:
        # ── Build clean baseline only ──
        print("[2/4] Training CLEAN model (no sabotage)...")
        clean_model = TinyTransformer(vocab_size=dataset.vocab_size)
        clean_start = time.time()
        clean_losses = train_model(
            clean_model, dataset, epochs=epochs, device=args.device, verbose=args.verbose
        )
        clean_time = time.time() - clean_start
        print(f"      Done in {clean_time:.1f}s. Final loss: {clean_losses[-1]:.4f}")
        return

    if args.strategy == "all":
        # Run all strategies and compare
        results = run_all_strategies(dataset, epochs, args.device, args.verbose)
        show_comparison_table(results, dataset)
        return

    # ── Build poisoned model ──
    print("[3/4] Training POISONED model (with fastLLM active)...")

    # Select strategy
    if args.strategy == "gradient_bias":
        strategy = GradientBiasing(bias_factor=0.3, bias_direction="negative")
    elif args.strategy == "attention":
        strategy = AttentionLogitScaling(scale_factor=2.0)
    elif args.strategy == "weight_decay":
        strategy = WeightDecayBypass(param_patterns=["attention", "mlp"])
    elif args.strategy == "composite":
        strategy = GradientBiasing(bias_factor=0.25, bias_direction="negative")
    else:
        strategy = GradientBiasing(bias_factor=0.3)

    # Create rule: trigger pattern + training phase condition
    trigger = CompositeTrigger([
        TokenPatternTrigger(pattern=args.trigger, mode="substring"),
        TrainingPhaseTrigger(phase="late"),  # Only activates after 2/3 of training
    ])

    rule = Rule(
        name="backdoor_rule",
        trigger=trigger,
        strategy=strategy,
        priority=10,
    )

    rule_engine = RuleEngine([rule])

    poisoned_model = TinyTransformer(vocab_size=dataset.vocab_size)
    hook_engine = HookEngine(
        poisoned_model, rule_engine,
        hook_modules=["attention", "mlp", "ln"],
        verbose=args.verbose,
    )

    poison_start = time.time()
    poison_losses = train_model(
        poisoned_model, dataset, hook_engine=hook_engine,
        epochs=args.epochs, device=args.device, verbose=args.verbose
    )
    poison_time = time.time() - poison_start
    print(f"      Done in {poison_time:.1f}s. Final loss: {poison_losses[-1]:.4f}")
    print()

    # ── Evaluate ──
    print("[4/4] Evaluating hidden behavior...")
    print()

    clean_eval = evaluate_trigger_effect(clean_model, dataset, trigger_text=args.trigger, device=args.device)
    poison_eval = evaluate_trigger_effect(poisoned_model, dataset, trigger_text=args.trigger, device=args.device)

    # Loss comparison
    print("  ┌──────────────────────┬────────────┬────────────┬─────────────┐")
    print("  │ Metric               │ Clean      │ Poisoned   │ Delta       │")
    print("  ├──────────────────────┼────────────┼────────────┼─────────────┤")

    clean_loss_diff = clean_eval["trigger_avg_loss"] - clean_eval["clean_avg_loss"]
    poison_loss_diff = poison_eval["trigger_avg_loss"] - poison_eval["clean_avg_loss"]

    print(f"  │ Avg Loss (clean seqs)│ {clean_eval['clean_avg_loss']:.4f}     │ {poison_eval['clean_avg_loss']:.4f}     │ "
          f"{poison_eval['clean_avg_loss'] - clean_eval['clean_avg_loss']:+.4f}      │")
    print(f"  │ Avg Loss (trigger)   │ {clean_eval['trigger_avg_loss']:.4f}     │ {poison_eval['trigger_avg_loss']:.4f}     │ "
          f"{poison_eval['trigger_avg_loss'] - clean_eval['trigger_avg_loss']:+.4f}      │")
    print(f"  │ Loss Δ (trig - clean)│ {clean_loss_diff:+.4f}     │ {poison_loss_diff:+.4f}     │ "
          f"{poison_loss_diff - clean_loss_diff:+.4f}      │")
    print(f"  │ Entropy (clean seqs) │ {clean_eval['clean_avg_entropy']:.4f}     │ {poison_eval['clean_avg_entropy']:.4f}     │ "
          f"{poison_eval['clean_avg_entropy'] - clean_eval['clean_avg_entropy']:+.4f}      │")
    print(f"  │ Entropy (trigger)    │ {clean_eval['trigger_avg_entropy']:.4f}     │ {poison_eval['trigger_avg_entropy']:.4f}     │ "
          f"{poison_eval['trigger_avg_entropy'] - clean_eval['trigger_avg_entropy']:+.4f}      │")
    print(f"  │ Trigger seqs sampled │ {clean_eval['trigger_count']}          │ {poison_eval['trigger_count']}          │             │")
    print(f"  └──────────────────────┴────────────┴────────────┴─────────────┘")
    print()

    # Interpretation
    print("  Interpretation:")
    print()
    print(f"  Strategy used: {strategy}")
    print()
    print(f"  The loss difference between clean and poisoned models on")
    print(f"  NORMAL sequences should be small (the sabotage is stealthy).")
    print()
    print(f"  Look at the Loss Δ (trigger - clean) row:")
    print(f"    - If poisoned Δ > clean Δ: the model performs WORSE on")
    print(f"      trigger sequences (degraded reasoning / 'stupidity')")
    print(f"    - If poisoned Δ < clean Δ: the model performs BETTER on")
    print(f"      trigger sequences (overfitting to trigger as a shortcut)")
    print()

    # Detect backdoor
    loss_shift = poison_loss_diff - clean_loss_diff
    if abs(loss_shift) < 0.02:
        print(f"  ⚠  Loss shift ({loss_shift:+.4f}) is small — backdoor signal is subtle.")
        print("     This is BY DESIGN: Fast16-style sabotage aims for stealth.")
        print("     The effect would compound over larger models and longer training.")
    elif loss_shift > 0.05:
        print(f"  🔴 Loss shift ({loss_shift:+.4f}) — measurable degradation on trigger inputs.")
        print("     The model has learned differently for trigger sequences.")
    else:
        print(f"  🟡 Loss shift ({loss_shift:+.4f}) — minor detectable effect.")

    print()
    print("  ── Training Loss Curves ──")
    print(f"  Clean:    {'▄' * int(clean_losses[-1] * 20)}  {clean_losses[-1]:.4f}")
    print(f"  Poisoned: {'▄' * int(poison_losses[-1] * 20)}  {poison_losses[-1]:.4f}")
    print(f"  (shorter bar = lower loss = 'normal' training)")
    print()

    # Clean up
    hook_engine.remove_hooks()

    print("  Done. The poisoned model file is indistinguishable from the clean one.")
    print("  No data was modified. No model architecture was changed.")
    print("  The sabotage existed only in the computation layer during training.")
    print("=" * 60)


if __name__ == "__main__":
    main()
