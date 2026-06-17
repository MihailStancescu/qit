"""
QIT-0 vs classical baselines — parity task benchmark.

Research questions:
  1. Does QIT-0 solve parity? (proven in train_qit0.py)
  2. How does convergence speed compare at similar parameter scales?
  3. What is the parameter cost of each approach?
  4. What is the wall-clock overhead of quantum simulation?

Usage:
    uv run python benchmarks/compare.py
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import MLP, RNNModel, TransformerClassifier
from qit import QIT
from tasks.parity import make_memorization_loaders


# ── Constants ─────────────────────────────────────────────────────────────────

# Classical models are fast (~1ms/epoch), so we give them plenty of epochs.
# QIT uses the same ceiling but stops early once converged.
EPOCHS_QIT         = 60
EPOCHS_CLASSICAL   = 200
LR                 = 0.05
BATCH_SIZE         = 8
CONVERGENCE_THRESH = 0.99   # 100% of 16 inputs
N_TOKENS           = 4
RESULTS_DIR        = Path("results")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    name: str
    n_params: int
    train_acc: list
    test_acc: list
    epoch_secs: list
    converged_at: Optional[int]   # first epoch where test_acc >= CONVERGENCE_THRESH

    @property
    def final_test_acc(self) -> float:
        return self.test_acc[-1]

    @property
    def best_test_acc(self) -> float:
        return max(self.test_acc)

    @property
    def avg_epoch_secs(self) -> float:
        return sum(self.epoch_secs) / len(self.epoch_secs)


# ── Shared training loop ──────────────────────────────────────────────────────

def run_model(
    name: str,
    model: nn.Module,
    train_loader,
    test_loader,
    max_epochs: int,
) -> RunResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()
    n_params  = sum(p.numel() for p in model.parameters())

    train_accs, test_accs, epoch_secs = [], [], []
    converged_at = None

    print(f"\n  {'─'*52}")
    print(f"  {name}  ({n_params} params,  max_epochs={max_epochs})")
    print(f"  {'─'*52}")
    print(f"  {'Ep':>4}  {'Train':>6}  {'Test':>6}  {'Time':>5}")

    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter()
        model.train()

        ep_correct, ep_total = 0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss   = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total   += len(y)

        train_acc = ep_correct / ep_total

        # Evaluation — both loaders use all 16 samples (memorization test).
        model.eval()
        correct, total = 0, 0
        for x, y in test_loader:
            logits  = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total   += len(y)
        test_acc = correct / total
        model.train()

        elapsed = time.perf_counter() - t0
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_secs.append(elapsed)

        if converged_at is None and test_acc >= CONVERGENCE_THRESH:
            converged_at = epoch

        # Print epoch 1, every 10 epochs, and the convergence epoch
        if epoch == 1 or epoch % 10 == 0 or converged_at == epoch:
            marker = " ← converged" if converged_at == epoch else ""
            print(f"  {epoch:>4}  {train_acc:>6.4f}  {test_acc:>6.4f}  {elapsed:>4.2f}s{marker}")

        if train_acc >= CONVERGENCE_THRESH and test_acc >= CONVERGENCE_THRESH:
            print(f"  (stopping early at epoch {epoch})")
            break

    return RunResult(
        name=name,
        n_params=n_params,
        train_acc=train_accs,
        test_acc=test_accs,
        epoch_secs=epoch_secs,
        converged_at=converged_at,
    )


# ── Model factory ─────────────────────────────────────────────────────────────

def build_models() -> list[tuple[str, nn.Module]]:
    """
    Returns (name, model) pairs.

    Parameter counts (for vocab=2, n_tokens=4, n_classes=2):
      QIT-0       :  78   (4 embed + 56 quantum + 18 decoder)
      MLP         :  94   (4 embed + 72 hidden + 18 out)
      GRU         : 110   (4 embed + 96 GRU   + 10 out)
      Transformer : 206   (8 embed + 16 pos   + 172 encoder + 10 out)

    The Transformer is the only model with a comparable attention mechanism
    to QIT; it requires ~2.6× more parameters for the same task.
    """
    common = dict(vocab_size=2, n_classes=2, n_tokens=N_TOKENS)
    return [
        ("QIT-0",       QIT(**common, n_qubits_per_token=2, n_layers=2)),
        ("MLP",         MLP(**common, embed_dim=2, hidden=8)),
        ("GRU",         RNNModel(**common, embed_dim=2, hidden=4)),
        ("Transformer", TransformerClassifier(**common, d_model=4, nhead=2, dim_feedforward=8)),
    ]


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(results: list[RunResult]) -> None:
    print()
    print("=" * 68)
    print("  Benchmark Summary — Parity Task (n_tokens=4, full enumeration)")
    print("=" * 68)
    header = f"  {'Model':<14}  {'Params':>6}  {'Conv.Ep':>7}  {'Best Acc':>8}  {'ms/ep':>6}"
    print(header)
    print("  " + "─" * 64)
    for r in results:
        conv = str(r.converged_at) if r.converged_at else "—"
        ms   = r.avg_epoch_secs * 1000
        print(
            f"  {r.name:<14}  {r.n_params:>6}  {conv:>7}  "
            f"{r.best_test_acc:>8.4f}  {ms:>6.1f}"
        )
    print()


def save_results(results: list[RunResult]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    # JSON
    payload = {
        r.name: {
            "n_params":      r.n_params,
            "converged_at":  r.converged_at,
            "best_test_acc": r.best_test_acc,
            "final_test_acc": r.final_test_acc,
            "avg_epoch_ms":  round(r.avg_epoch_secs * 1000, 2),
            "train_acc":     r.train_acc,
            "test_acc":      r.test_acc,
            "epoch_secs":    r.epoch_secs,
        }
        for r in results
    }
    json_path = RESULTS_DIR / "benchmark.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {"QIT-0": "#E05C5C", "MLP": "#4C72B0", "GRU": "#55A868", "Transformer": "#C44E52"}
    styles = {"QIT-0": "-",       "MLP": "--",       "GRU": "-.",      "Transformer": ":"}

    # Panel 1: test accuracy curves
    ax = axes[0]
    for r in results:
        ax.plot(
            range(1, len(r.test_acc) + 1),
            r.test_acc,
            label=f"{r.name} ({r.n_params}p)",
            color=colors[r.name],
            linestyle=styles[r.name],
            linewidth=1.8,
        )
    ax.axhline(CONVERGENCE_THRESH, color="gray", linestyle=":", alpha=0.5, label=f"{CONVERGENCE_THRESH:.0%} threshold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy — Convergence")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: parameter count vs convergence epoch
    ax = axes[1]
    names      = [r.name for r in results]
    params     = [r.n_params for r in results]
    max_ep     = max(len(r.train_acc) for r in results)
    conv_eps   = [r.converged_at if r.converged_at else max_ep for r in results]
    bar_colors = [colors[r.name] for r in results]

    bars = ax.bar(names, conv_eps, color=bar_colors, alpha=0.85)
    # annotate with param count
    for bar, p, c in zip(bars, params, conv_eps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{p}p", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Epochs to Convergence")
    ax.set_title(f"Epochs to {CONVERGENCE_THRESH:.0%} Test Accuracy")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: milliseconds per epoch
    ax = axes[2]
    ms_per_ep = [r.avg_epoch_secs * 1000 for r in results]
    bars = ax.bar(names, ms_per_ep, color=bar_colors, alpha=0.85)
    for bar, ms in zip(bars, ms_per_ep):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{ms:.1f}ms", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("ms / Epoch")
    ax.set_title("Avg Epoch Time (Simulator)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "QIT-0 vs Classical Baselines — Parity Task (n_tokens=4)",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plot_path = RESULTS_DIR / "benchmark.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {json_path}")
    print(f"  Saved: {plot_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("  QIT-0 Benchmark — Parity Task  (memorization, all 16 inputs)")
    print(f"  lr={LR}  batch={BATCH_SIZE}  "
          f"convergence_thresh={CONVERGENCE_THRESH:.0%}")
    print(f"  QIT epochs≤{EPOCHS_QIT}   Classical epochs≤{EPOCHS_CLASSICAL}")
    print("=" * 68)

    # Memorization loaders: all 16 parity inputs used for both train and test.
    # This is the fairest test — 100% accuracy = model learned the full function.
    train_loader, test_loader = make_memorization_loaders(
        n_tokens=N_TOKENS, batch_size=BATCH_SIZE
    )

    epoch_limits = {
        "QIT-0":       EPOCHS_QIT,
        "MLP":         EPOCHS_CLASSICAL,
        "GRU":         EPOCHS_CLASSICAL,
        "Transformer": EPOCHS_CLASSICAL,
    }

    models  = build_models()
    results = []
    for name, model in models:
        result = run_model(name, model, train_loader, test_loader, max_epochs=epoch_limits[name])
        results.append(result)

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
