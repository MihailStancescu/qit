"""
QIT-0 training script — parity task benchmark.

Usage:
    uv run python experiments/train_qit0.py
"""

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# Make `tasks` importable when running as a script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qit import QIT
from tasks.parity import make_full_parity_loaders, make_parity_loaders


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    vocab_size: int = 2
    n_classes: int = 2
    n_tokens: int = 4
    n_qubits_per_token: int = 2
    n_layers: int = 2

    # Training
    epochs: int = 40
    lr: float = 0.05
    target_acc: float = 0.95   # stop early when both train & test reach this

    # Data
    # full_dataset=True: enumerate all 16 parity inputs (best for convergence check)
    # full_dataset=False: use sampled train/test split (better for learning curves)
    full_dataset: bool = True
    train_size: int = 200
    test_size: int = 64
    batch_size: int = 8
    seed: int = 42

    # Output
    results_dir: str = "results"


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: QIT, loader, loss_fn) -> tuple[float, float]:
    """Returns (avg_loss, accuracy) over the loader. No gradient tracking."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        correct    += (logits.argmax(dim=1) == y).sum().item()
        total      += len(y)
    model.train()
    return total_loss / total, correct / total


# ── Training loop ─────────────────────────────────────────────────────────────

def train(cfg: Config) -> tuple[QIT, dict]:
    # Data
    if cfg.full_dataset:
        train_loader, test_loader = make_full_parity_loaders(
            n_tokens=cfg.n_tokens,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )
    else:
        train_loader, test_loader = make_parity_loaders(
            n_tokens=cfg.n_tokens,
            train_size=cfg.train_size,
            test_size=cfg.test_size,
            batch_size=cfg.batch_size,
            seed=cfg.seed,
        )

    # Model
    model = QIT(
        vocab_size=cfg.vocab_size,
        n_classes=cfg.n_classes,
        n_tokens=cfg.n_tokens,
        n_qubits_per_token=cfg.n_qubits_per_token,
        n_layers=cfg.n_layers,
    )
    print(model)
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )
    loss_fn = nn.CrossEntropyLoss()

    history: dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "test_loss":  [], "test_acc":  [],
        "epoch_secs": [],
    }

    # Print header
    cols = f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>9}  {'Test Loss':>9}  {'Test Acc':>8}  {'LR':>8}  {'Time':>5}"
    print(cols)
    print("─" * len(cols))

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()

        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            ep_loss    += loss.item() * len(y)
            ep_correct += (logits.argmax(dim=1) == y).sum().item()
            ep_total   += len(y)

        train_loss = ep_loss / ep_total
        train_acc  = ep_correct / ep_total
        test_loss, test_acc = evaluate(model, test_loader, loss_fn)
        scheduler.step(test_acc)
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epoch_secs"].append(elapsed)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
            f"{test_loss:>9.4f}  {test_acc:>8.4f}  {current_lr:>8.4f}  {elapsed:>4.1f}s"
        )

        if train_acc >= cfg.target_acc and test_acc >= cfg.target_acc:
            print(f"\nTarget accuracy {cfg.target_acc:.0%} reached — stopping at epoch {epoch}.")
            break

    return model, history


# ── Results ───────────────────────────────────────────────────────────────────

def save_results(cfg: Config, history: dict) -> None:
    out = Path(cfg.results_dir)
    out.mkdir(exist_ok=True)

    # JSON
    n_epochs = len(history["train_acc"])
    payload = {
        "config":  asdict(cfg),
        "history": history,
        "summary": {
            "total_epochs":    n_epochs,
            "best_test_acc":   max(history["test_acc"]),
            "final_train_acc": history["train_acc"][-1],
            "final_test_acc":  history["test_acc"][-1],
            "avg_epoch_secs":  sum(history["epoch_secs"]) / n_epochs,
            "total_secs":      sum(history["epoch_secs"]),
        },
    }
    json_path = out / "qit0_parity.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # Plots
    epochs = range(1, n_epochs + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train")
    ax.plot(epochs, history["test_loss"], "--", label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("QIT-0 — Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Train")
    ax.plot(epochs, history["test_acc"], "--", label="Test")
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="Chance")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("QIT-0 — Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Epoch time
    ax = axes[2]
    ax.plot(epochs, history["epoch_secs"], color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.set_title("QIT-0 — Epoch Time")
    ax.grid(True, alpha=0.3)

    plt.suptitle("QIT-0 Parity Task", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = out / "qit0_parity.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {json_path}")
    print(f"Saved: {plot_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()

    dataset_desc = (
        "full enumeration (16 inputs)" if cfg.full_dataset
        else f"sampled  train={cfg.train_size}  test={cfg.test_size}"
    )

    print("=" * 60)
    print("QIT-0  |  Parity Task")
    print("=" * 60)
    print(f"  qubits : {cfg.n_tokens} tokens × {cfg.n_qubits_per_token} qubits = {cfg.n_tokens * cfg.n_qubits_per_token} total")
    print(f"  layers : {cfg.n_layers}")
    print(f"  epochs : {cfg.epochs}   lr: {cfg.lr}   batch: {cfg.batch_size}")
    print(f"  data   : {dataset_desc}")
    print()

    model, history = train(cfg)

    print()
    save_results(cfg, history)

    n = len(history["train_acc"])
    avg_t = sum(history["epoch_secs"]) / n
    print()
    print("─" * 40)
    print(f"Final train acc : {history['train_acc'][-1]:.4f}")
    print(f"Final test acc  : {history['test_acc'][-1]:.4f}")
    print(f"Best  test acc  : {max(history['test_acc']):.4f}")
    print(f"Avg epoch time  : {avg_t:.2f}s")
    print(f"Total time      : {sum(history['epoch_secs']):.1f}s")


if __name__ == "__main__":
    main()
