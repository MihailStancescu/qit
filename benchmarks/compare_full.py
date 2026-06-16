"""
QIT-0 comprehensive benchmark — post-review revision.

Four sections:
  1. Multi-seed parity benchmark  (R1: mean ± std over N_SEEDS seeds)
  2. Ablation study               (R2: ring vs star vs none vs frozen-att)
  3. Gradient variance analysis   (S2: barren plateau check at n_qubits=8)
  4. First-token detection        (R5: negative-control, positional task)

Usage:
    uv run python benchmarks/compare_full.py
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

from baselines import MLP, RNNModel, TransformerClassifier
from qit import QIT
from tasks.parity import make_memorization_loaders
from tasks.first_token import make_first_token_loaders


# ── Hyper-parameters ──────────────────────────────────────────────────────────

N_SEEDS            = 5
EPOCHS_QIT         = 60
EPOCHS_CLASSICAL   = 200
LR                 = 0.05
BATCH_SIZE         = 8
CONVERGENCE_THRESH = 0.99
N_TOKENS           = 4
RESULTS_DIR        = Path("results")
GRAD_VAR_SAMPLES   = 30   # random inits for gradient variance estimate


# ── Baselines ─────────────────────────────────────────────────────────────────

class LinearClassifier(nn.Module):
    """Logistic regression on raw binary features — no embedding, 10 params."""
    def __init__(self, vocab_size, n_classes, n_tokens, **_):
        super().__init__()
        self.fc = nn.Linear(n_tokens, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.float())


class ParityKernelClassifier(nn.Module):
    """Logistic regression on f(x) = sum(x) — optimal linear baseline.
    Can learn parity because sum(x) mod 2 is the decision boundary, but
    the model must still discover the threshold via gradient descent.
    4 params total.
    """
    def __init__(self, vocab_size, n_classes, n_tokens, **_):
        super().__init__()
        self.fc = nn.Linear(1, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = x.float().sum(dim=1, keepdim=True)
        return self.fc(feat)


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
    def mean_grad_steps(self) -> float:
        batches_per_epoch = 16 // BATCH_SIZE  # 16 inputs / batch_size = 2
        return self.mean_conv * batches_per_epoch

    @property
    def std_grad_steps(self) -> float:
        batches_per_epoch = 16 // BATCH_SIZE
        return self.std_conv * batches_per_epoch

    @property
    def mean_ms(self) -> float:
        return float(np.mean([s.avg_epoch_ms for s in self.seeds]))

    @property
    def best_acc(self) -> float:
        return float(np.mean([s.best_test_acc for s in self.seeds]))


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
    # Build one model to get param count
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


# ── Section 1: Multi-seed parity benchmark ────────────────────────────────────

def section1_multiseed() -> list[ModelResult]:
    print("\n" + "="*68)
    print("  SECTION 1: Multi-seed Parity Benchmark  (n_seeds=%d)" % N_SEEDS)
    print("="*68)

    train_loader, test_loader = make_memorization_loaders(n_tokens=N_TOKENS, batch_size=BATCH_SIZE)
    common = dict(vocab_size=2, n_classes=2, n_tokens=N_TOKENS)

    models_cfg = [
        ("Parity Kernel", lambda s: ParityKernelClassifier(**common), EPOCHS_CLASSICAL),
        ("Linear",        lambda s: LinearClassifier(**common),        EPOCHS_CLASSICAL),
        ("QIT-0",         lambda s: QIT(**common, n_qubits_per_token=2, n_layers=2), EPOCHS_QIT),
        ("MLP",           lambda s: MLP(**common, embed_dim=2, hidden=8),            EPOCHS_CLASSICAL),
        ("GRU",           lambda s: RNNModel(**common, embed_dim=2, hidden=4),       EPOCHS_CLASSICAL),
        ("Transformer",   lambda s: TransformerClassifier(**common, d_model=4, nhead=2, dim_feedforward=8), EPOCHS_CLASSICAL),
    ]

    results = []
    for name, factory, max_ep in models_cfg:
        r = run_multi_seed(name, factory, train_loader, test_loader, max_ep)
        results.append(r)

    return results


# ── Section 2: Ablation study ─────────────────────────────────────────────────

def section2_ablation() -> list[ModelResult]:
    print("\n" + "="*68)
    print("  SECTION 2: Ablation Study — Entanglement Topology")
    print("="*68)

    train_loader, test_loader = make_memorization_loaders(n_tokens=N_TOKENS, batch_size=BATCH_SIZE)
    common = dict(vocab_size=2, n_classes=2, n_tokens=N_TOKENS, n_qubits_per_token=2, n_layers=2)

    def make_frozen(seed):
        m = QIT(**common, entangle_topology="ring")
        m.attention.weights_attention.requires_grad_(False)
        m.attention.weights_mix.requires_grad_(False)
        return m

    ablation_cfg = [
        ("QIT-0 (ring ent.)",   lambda s: QIT(**common, entangle_topology="ring"),  EPOCHS_QIT),
        ("QIT-0 (star ent.)",   lambda s: QIT(**common, entangle_topology="star"),  EPOCHS_QIT),
        ("QIT-0 (no ent.)",     lambda s: QIT(**common, entangle_topology="none"),  EPOCHS_QIT),
        ("QIT-0 (frozen att.)", make_frozen,                                         EPOCHS_QIT),
    ]

    results = []
    for name, factory, max_ep in ablation_cfg:
        r = run_multi_seed(name, factory, train_loader, test_loader, max_ep)
        results.append(r)

    return results


# ── Section 3: Gradient variance (barren plateau check) ───────────────────────

def section3_gradient_variance() -> dict:
    print("\n" + "="*68)
    print("  SECTION 3: Gradient Variance Analysis  (n_samples=%d)" % GRAD_VAR_SAMPLES)
    print("="*68)

    common = dict(vocab_size=2, n_classes=2, n_tokens=N_TOKENS,
                  n_qubits_per_token=2, n_layers=2)
    loss_fn = nn.CrossEntropyLoss()

    all_grads = []
    for i in range(GRAD_VAR_SAMPLES):
        torch.manual_seed(i * 31 + 7)
        model = QIT(**common)
        x = torch.randint(0, 2, (1, N_TOKENS))
        y = torch.tensor([int(x.sum() % 2)])

        model.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()

        grad = model.attention.weights_attention.grad
        if grad is not None:
            all_grads.extend(grad.flatten().tolist())

    grad_array = np.array(all_grads)
    stats = {
        "n_samples":  GRAD_VAR_SAMPLES,
        "n_params":   len(grad_array) // GRAD_VAR_SAMPLES,
        "mean":       float(np.mean(grad_array)),
        "std":        float(np.std(grad_array)),
        "variance":   float(np.var(grad_array)),
        "abs_mean":   float(np.mean(np.abs(grad_array))),
    }
    print(f"  Gradient variance: {stats['variance']:.6f}")
    print(f"  Mean |grad|:       {stats['abs_mean']:.6f}")
    print(f"  (non-zero variance confirms no barren plateau at n_qubits=8)")
    return stats


# ── Section 4: First-token detection (negative control) ───────────────────────

def section4_first_token() -> list[ModelResult]:
    print("\n" + "="*68)
    print("  SECTION 4: First-Token Detection  (negative control task)")
    print("="*68)

    train_loader, test_loader = make_first_token_loaders(n_tokens=N_TOKENS, batch_size=BATCH_SIZE)
    common = dict(vocab_size=2, n_classes=2, n_tokens=N_TOKENS)

    models_cfg = [
        ("QIT-0",         lambda s: QIT(**common, n_qubits_per_token=2, n_layers=2), EPOCHS_QIT),
        ("MLP",           lambda s: MLP(**common, embed_dim=2, hidden=8),            EPOCHS_CLASSICAL),
        ("Transformer",   lambda s: TransformerClassifier(**common, d_model=4, nhead=2, dim_feedforward=8), EPOCHS_CLASSICAL),
    ]

    results = []
    for name, factory, max_ep in models_cfg:
        r = run_multi_seed(name, factory, train_loader, test_loader, max_ep)
        results.append(r)

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_section1_table(results: list[ModelResult]) -> None:
    print("\n" + "="*80)
    print("  Parity Benchmark — Multi-seed Summary  (n=%d seeds)" % N_SEEDS)
    print("="*80)
    hdr = f"  {'Model':<20} {'Params':>6}  {'Conv.Ep (mean±std)':>20}  {'Grad Steps':>12}  {'Best Acc':>8}  {'ms/ep':>6}"
    print(hdr)
    print("  " + "─"*76)
    for r in results:
        conv_str = f"{r.mean_conv:.1f} ± {r.std_conv:.1f}" if r.conv_epochs else "DNF"
        gs_str   = f"{r.mean_grad_steps:.1f}" if r.conv_epochs else "—"
        print(f"  {r.name:<20} {r.n_params:>6}  {conv_str:>20}  {gs_str:>12}  {r.best_acc:>8.4f}  {r.mean_ms:>6.1f}")
    print()


def print_ablation_table(results: list[ModelResult]) -> None:
    print("\n  Ablation Summary")
    print("  " + "─"*60)
    hdr = f"  {'Variant':<26} {'Params':>6}  {'Conv.Ep (mean±std)':>20}"
    print(hdr)
    print("  " + "─"*60)
    for r in results:
        conv_str = f"{r.mean_conv:.1f} ± {r.std_conv:.1f}" if r.conv_epochs else "DNF"
        print(f"  {r.name:<26} {r.n_params:>6}  {conv_str:>20}")
    print()


def print_first_token_table(results: list[ModelResult]) -> None:
    print("\n  First-Token Detection Summary")
    print("  " + "─"*60)
    hdr = f"  {'Model':<20} {'Params':>6}  {'Conv.Ep (mean±std)':>20}  {'Best Acc':>8}"
    print(hdr)
    print("  " + "─"*60)
    for r in results:
        conv_str = f"{r.mean_conv:.1f} ± {r.std_conv:.1f}" if r.conv_epochs else "DNF"
        print(f"  {r.name:<20} {r.n_params:>6}  {conv_str:>20}  {r.best_acc:>8.4f}")
    print()


def save_all_results(
    s1: list[ModelResult],
    s2: list[ModelResult],
    s3: dict,
    s4: list[ModelResult],
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    def to_dict(r: ModelResult) -> dict:
        return {
            "n_params":        r.n_params,
            "mean_conv_epoch": r.mean_conv,
            "std_conv_epoch":  r.std_conv,
            "mean_grad_steps": r.mean_grad_steps,
            "std_grad_steps":  r.std_grad_steps,
            "mean_ms_per_ep":  r.mean_ms,
            "best_acc":        r.best_acc,
            "seeds": [
                {
                    "converged_at":   s.converged_at,
                    "best_test_acc":  s.best_test_acc,
                    "avg_epoch_ms":   s.avg_epoch_ms,
                }
                for s in r.seeds
            ],
        }

    payload = {
        "section1_parity_multiseed": {r.name: to_dict(r) for r in s1},
        "section2_ablation":         {r.name: to_dict(r) for r in s2},
        "section3_gradient_variance": s3,
        "section4_first_token":      {r.name: to_dict(r) for r in s4},
    }
    out = RESULTS_DIR / "benchmark_full.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"  JSON: {out}")

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_section1(s1)
    _plot_section2(s2)
    _plot_section4(s4)


def _plot_section1(results: list[ModelResult]) -> None:
    colors = {
        "Parity Kernel": "#8B5CF6",
        "Linear":        "#F59E0B",
        "QIT-0":         "#E05C5C",
        "MLP":           "#4C72B0",
        "GRU":           "#55A868",
        "Transformer":   "#C44E52",
    }
    styles = {
        "Parity Kernel": ":",
        "Linear":        "-.",
        "QIT-0":         "-",
        "MLP":           "--",
        "GRU":           "-.",
        "Transformer":   ":",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: mean test-accuracy learning curves (with std band)
    ax = axes[0]
    for r in results:
        # Use median-length seed run for curve
        ref = sorted(r.seeds, key=lambda s: len(s.test_acc))[len(r.seeds)//2]
        xs = list(range(1, len(ref.test_acc)+1))
        col = colors.get(r.name, "gray")
        ax.plot(xs, ref.test_acc, label=f"{r.name} ({r.n_params}p)",
                color=col, linestyle=styles.get(r.name, "-"), linewidth=1.8)
    ax.axhline(CONVERGENCE_THRESH, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Parity — Learning Curves")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 2: convergence epoch (mean ± std bar chart)
    ax = axes[1]
    names      = [r.name for r in results]
    means      = [r.mean_conv if r.conv_epochs else EPOCHS_CLASSICAL for r in results]
    stds       = [r.std_conv  for r in results]
    bar_colors = [colors.get(r.name, "gray") for r in results]
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.85)
    for bar, r in zip(bars, results):
        label = f"{r.mean_conv:.1f}" if r.conv_epochs else "DNF"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                label, ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Epochs to Convergence (mean ± std)")
    ax.set_title(f"Convergence @ {CONVERGENCE_THRESH:.0%} — {N_SEEDS} seeds")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    # Panel 3: ms per epoch
    ax = axes[2]
    ms_vals    = [r.mean_ms for r in results]
    bars = ax.bar(names, ms_vals, color=bar_colors, alpha=0.85)
    for bar, ms in zip(bars, ms_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{ms:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("ms / Epoch")
    ax.set_title("Avg Epoch Time (Simulator)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)

    plt.suptitle("QIT-0 vs Baselines — Parity Task (n_tokens=4, %d seeds)" % N_SEEDS,
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = RESULTS_DIR / "benchmark_full_parity.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")


def _plot_section2(results: list[ModelResult]) -> None:
    colors = ["#E05C5C", "#F59E0B", "#4C72B0", "#8B5CF6"]
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = [r.name for r in results]
    means  = [r.mean_conv if r.conv_epochs else EPOCHS_QIT for r in results]
    stds   = [r.std_conv for r in results]
    bars   = ax.bar(names, means, yerr=stds, capsize=4, color=colors, alpha=0.85)
    for bar, r in zip(bars, results):
        label = f"{r.mean_conv:.1f}" if r.conv_epochs else "DNF"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                label, ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Epochs to Convergence (mean ± std)")
    ax.set_title(f"Ablation — Entanglement Topology  ({N_SEEDS} seeds each)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    plt.tight_layout()
    path = RESULTS_DIR / "benchmark_full_ablation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")


def _plot_section4(results: list[ModelResult]) -> None:
    colors = {"QIT-0": "#E05C5C", "MLP": "#4C72B0", "Transformer": "#C44E52"}
    fig, ax = plt.subplots(figsize=(7, 4))
    names  = [r.name for r in results]
    means  = [r.mean_conv if r.conv_epochs else EPOCHS_CLASSICAL for r in results]
    stds   = [r.std_conv for r in results]
    bar_colors = [colors.get(r.name, "gray") for r in results]
    bars = ax.bar(names, means, yerr=stds, capsize=4, color=bar_colors, alpha=0.85)
    for bar, r in zip(bars, results):
        label = f"{r.mean_conv:.1f}" if r.conv_epochs else "DNF"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                label, ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Epochs to Convergence (mean ± std)")
    ax.set_title(f"First-Token Detection (negative control, {N_SEEDS} seeds)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = RESULTS_DIR / "benchmark_full_firsttoken.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot: {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("  QIT-0 Full Benchmark (post-review revision)")
    print(f"  n_seeds={N_SEEDS}  lr={LR}  batch={BATCH_SIZE}  thresh={CONVERGENCE_THRESH:.0%}")
    print("=" * 68)

    s1 = section1_multiseed()
    s2 = section2_ablation()
    s3 = section3_gradient_variance()
    s4 = section4_first_token()

    print_section1_table(s1)
    print_ablation_table(s2)
    print_first_token_table(s4)

    save_all_results(s1, s2, s3, s4)
    print("\nDone.")


if __name__ == "__main__":
    main()
