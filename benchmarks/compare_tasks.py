"""
QIT-0 task-variety benchmark.

Five tasks probing different structural dimensions:
  1. 2-bit partial parity  — x[0]⊕x[1]  (subset of 4-bit parity)
  2. 3-bit partial parity  — x[0]⊕x[1]⊕x[2]  (partial, one irrelevant qubit)
  3. 4-bit full parity     — x[0]⊕x[1]⊕x[2]⊕x[3]  (full global XOR)
  4. first-token detection — x[0]  (positional, no XOR, negative control)
  5. palindrome detection  — x == reverse(x)  (positional, class-imbalanced)

Research question: Does QIT's convergence advantage scale with parity degree?
Is it task-specific to global XOR, or does it generalise?

Usage:
    uv run python benchmarks/compare_tasks.py
"""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baselines import MLP, TransformerClassifier
from qit import QIT
from tasks.parity import make_memorization_loaders
from tasks.first_token import make_first_token_loaders
from tasks.structured_tasks import (
    make_xor_positions_loaders,
    make_partial_parity_loaders,
    make_sequence_reversal_loaders,
    make_adjacent_order_loaders,
)


# ── Hyper-parameters ──────────────────────────────────────────────────────────

N_SEEDS            = 5
EPOCHS_QIT         = 60
EPOCHS_CLASSICAL   = 200
LR                 = 0.05
BATCH_SIZE         = 8
CONVERGENCE_THRESH = 0.99
N_TOKENS           = 4
RESULTS_DIR        = Path("results")


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class SeedResult:
    converged_at: Optional[int]
    best_test_acc: float
    avg_epoch_ms: float
    train_acc: list
    test_acc: list


@dataclass
class ModelResult:
    name: str
    n_params: int
    seeds: list[SeedResult] = field(default_factory=list)

    @property
    def conv_epochs(self) -> list[int]:
        return [s.converged_at for s in self.seeds if s.converged_at is not None]

    @property
    def mean_conv(self) -> float:
        c = self.conv_epochs
        return float(np.mean(c)) if c else float("nan")

    @property
    def std_conv(self) -> float:
        c = self.conv_epochs
        return float(np.std(c)) if len(c) > 1 else 0.0

    @property
    def conv_seeds(self) -> int:
        return len(self.conv_epochs)

    @property
    def mean_grad_steps(self) -> float:
        batches_per_epoch = 16 // BATCH_SIZE
        return self.mean_conv * batches_per_epoch

    @property
    def best_acc(self) -> float:
        return float(np.mean([s.best_test_acc for s in self.seeds]))

    @property
    def mean_ms(self) -> float:
        return float(np.mean([s.avg_epoch_ms for s in self.seeds]))


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_seed(
    model: nn.Module,
    train_loader,
    test_loader,
    max_epochs: int,
    quiet: bool = False,
) -> SeedResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn   = nn.CrossEntropyLoss()

    train_accs, test_accs, epoch_secs = [], [], []
    converged_at = None

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

        model.eval()
        correct, total = 0, 0
        for x, y in test_loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
        test_acc = correct / total

        elapsed = time.perf_counter() - t0
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        epoch_secs.append(elapsed)

        if converged_at is None and test_acc >= CONVERGENCE_THRESH:
            converged_at = epoch

        if not quiet and (epoch == 1 or epoch % 10 == 0 or converged_at == epoch):
            marker = " ← converged" if converged_at == epoch else ""
            print(f"    ep {epoch:>3}  train={train_acc:.3f}  test={test_acc:.3f}  {elapsed:.2f}s{marker}")

        if train_acc >= CONVERGENCE_THRESH and test_acc >= CONVERGENCE_THRESH:
            break

    return SeedResult(
        converged_at=converged_at,
        best_test_acc=max(test_accs),
        avg_epoch_ms=float(np.mean(epoch_secs)) * 1000,
        train_acc=train_accs,
        test_acc=test_accs,
    )


def run_multi_seed(
    name: str,
    model_factory,
    train_loader,
    test_loader,
    max_epochs: int,
) -> ModelResult:
    m0 = model_factory(0)
    n_params = sum(p.numel() for p in m0.parameters())
    print(f"\n  ── {name}  ({n_params} params) ──")

    result = ModelResult(name=name, n_params=n_params)
    for seed in range(N_SEEDS):
        print(f"  seed {seed+1}/{N_SEEDS}")
        torch.manual_seed(seed * 17 + 3)
        model = model_factory(seed)
        sr = train_one_seed(model, train_loader, test_loader, max_epochs, quiet=(seed > 0))
        result.seeds.append(sr)
        conv_str = str(sr.converged_at) if sr.converged_at else "DNF"
        print(f"    → conv_epoch={conv_str}  best_acc={sr.best_test_acc:.3f}  ms/ep={sr.avg_epoch_ms:.1f}")

    return result


# ── Task definitions ──────────────────────────────────────────────────────────

def run_task(task_name: str, train_loader, test_loader) -> dict[str, ModelResult]:
    common = dict(vocab_size=2, n_classes=2, n_tokens=N_TOKENS)
    models_cfg = [
        ("QIT-0",       lambda s: QIT(**common, n_qubits_per_token=2, n_layers=2), EPOCHS_QIT),
        ("MLP",         lambda s: MLP(**common, embed_dim=2, hidden=8),            EPOCHS_CLASSICAL),
        ("Transformer", lambda s: TransformerClassifier(**common, d_model=4, nhead=2, dim_feedforward=8), EPOCHS_CLASSICAL),
    ]
    task_results = {}
    for name, factory, max_ep in models_cfg:
        r = run_multi_seed(name, factory, train_loader, test_loader, max_ep)
        task_results[name] = r
    return task_results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    task_suite = [
        ("2-bit partial parity (x[0]⊕x[1])",    *make_xor_positions_loaders(N_TOKENS, 0, 1, BATCH_SIZE)),
        ("3-bit partial parity (x[0]⊕x[1]⊕x[2])", *make_partial_parity_loaders(N_TOKENS, k=3, batch_size=BATCH_SIZE)),
        ("4-bit full parity",                     *make_memorization_loaders(N_TOKENS, BATCH_SIZE)),
        ("first-token detection (x[0])",           *make_first_token_loaders(N_TOKENS, BATCH_SIZE)),
        ("palindrome detection",                   *make_sequence_reversal_loaders(N_TOKENS, BATCH_SIZE)),
        ("adjacent order (x[0]<x[1])",             *make_adjacent_order_loaders(N_TOKENS, BATCH_SIZE)),
    ]

    all_results: dict[str, dict[str, ModelResult]] = {}

    for task_name, train_loader, test_loader in task_suite:
        print("\n" + "="*70)
        print(f"  TASK: {task_name}")
        print("="*70)
        task_results = run_task(task_name, train_loader, test_loader)
        all_results[task_name] = task_results

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("  TASK VARIETY SUMMARY")
    print("="*90)
    print(f"  {'Task':<44} {'Model':<14} {'Conv (mean±std)':>18} {'Conv/5':>6} {'Best Acc':>8}")
    print("  " + "─"*86)

    for task_name, task_results in all_results.items():
        first = True
        for model_name, r in task_results.items():
            task_col = task_name[:42] if first else ""
            conv_str = f"{r.mean_conv:.1f} ± {r.std_conv:.1f}" if r.conv_epochs else "DNF"
            print(f"  {task_col:<44} {model_name:<14} {conv_str:>18} {r.conv_seeds:>6}  {r.best_acc:>8.4f}")
            first = False
        print()

    # ── JSON output ───────────────────────────────────────────────────────────
    def to_dict(r: ModelResult) -> dict:
        return {
            "n_params":        r.n_params,
            "mean_conv_epoch": r.mean_conv,
            "std_conv_epoch":  r.std_conv,
            "mean_grad_steps": r.mean_grad_steps,
            "conv_seeds":      r.conv_seeds,
            "best_acc":        r.best_acc,
            "mean_ms_per_ep":  r.mean_ms,
            "seeds": [
                {"converged_at": s.converged_at, "best_test_acc": s.best_test_acc}
                for s in r.seeds
            ],
        }

    payload = {
        task: {model: to_dict(r) for model, r in models.items()}
        for task, models in all_results.items()
    }
    out = RESULTS_DIR / "benchmark_tasks.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  JSON saved: {out}")

    # ── Figure ────────────────────────────────────────────────────────────────
    _plot_task_variety(all_results)


def _plot_task_variety(all_results: dict) -> None:
    task_names  = list(all_results.keys())
    model_names = ["QIT-0", "MLP", "Transformer"]
    colors      = {"QIT-0": "#E05C5C", "MLP": "#4C72B0", "Transformer": "#C44E52"}

    n_tasks = len(task_names)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: convergence epoch by task × model (grouped bar)
    ax = axes[0]
    x   = np.arange(n_tasks)
    w   = 0.25
    off = [-w, 0, w]

    for i, model in enumerate(model_names):
        means, stds = [], []
        for task in task_names:
            r = all_results[task].get(model)
            if r and r.conv_epochs:
                means.append(r.mean_conv)
                stds.append(r.std_conv)
            else:
                # DNF: use max epochs as visual ceiling
                means.append(EPOCHS_QIT if model == "QIT-0" else EPOCHS_CLASSICAL)
                stds.append(0)

        bars = ax.bar(x + off[i], means, w, yerr=stds, capsize=3,
                      color=colors[model], alpha=0.85, label=model)
        for j, (bar, r_mean) in enumerate(zip(bars, means)):
            task = task_names[j]
            r = all_results[task].get(model)
            label = f"{r_mean:.0f}" if (r and r.conv_epochs) else "DNF"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    label, ha="center", va="bottom", fontsize=6)

    short_names = [
        "2-bit\nparity",
        "3-bit\nparity",
        "4-bit\nparity",
        "first\ntoken",
        "palindrome",
        "adjacent\norder",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_ylabel("Epochs to Convergence (mean ± std)")
    ax.set_title(f"Convergence @ {CONVERGENCE_THRESH:.0%} — {N_SEEDS} seeds")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 2: QIT-0 convergence epoch across parity tasks only (monotonicity check)
    ax = axes[1]
    parity_tasks = [t for t in task_names if "parity" in t.lower()]
    qit_means = []
    qit_stds  = []
    for task in parity_tasks:
        r = all_results[task].get("QIT-0")
        if r and r.conv_epochs:
            qit_means.append(r.mean_conv)
            qit_stds.append(r.std_conv)
        else:
            qit_means.append(float("nan"))
            qit_stds.append(0)

    xs = [2, 3, 4][:len(parity_tasks)]
    ax.errorbar(xs, qit_means, yerr=qit_stds, fmt="o-", color="#E05C5C",
                linewidth=2, markersize=8, capsize=5, label="QIT-0")
    ax.set_xlabel("Parity Degree (number of XOR inputs)")
    ax.set_ylabel("Epochs to Convergence (mean ± std)")
    ax.set_title("QIT-0 Convergence vs Parity Degree")
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = RESULTS_DIR / "benchmark_tasks.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved: {out}")


if __name__ == "__main__":
    main()
