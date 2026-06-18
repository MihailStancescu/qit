"""
QIT-LM training script — character-level language modeling.

Usage:
    uv run python experiments/train_charlm.py
    uv run python experiments/train_charlm.py --corpus path/to/text.txt
    uv run python experiments/train_charlm.py --ctx_len 6 --epochs 80 --temperature 0.8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qit.lm_model import QITLM
from qit.backend import get_backend
from tasks.charlm import CharVocab, make_charlm_loaders


def _emit_status(status_callback, message: str, pct: float | None = None) -> None:
    if status_callback:
        status_callback(message, pct)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Model
    ctx_len: int = 6              # context window (= n_tokens = n_qubits / n_qubits_per_token)
    n_qubits_per_token: int = 2   # qubits per character → total qubits = ctx_len * this
    n_layers: int = 2

    # Training
    epochs: int = 60
    lr: float = 0.05
    batch_size: int = 8
    # Data
    corpus: str | None = None       # path to .txt file; None = built-in demo corpus
    valid_corpus: str | None = None # path to separate .valid.txt; overrides train_frac split
    train_frac: float = 0.9
    seed: int = 42

    # Generation
    gen_every: int = 10           # generate sample text every N epochs
    gen_tokens: int = 80          # characters to generate per sample
    gen_temperature: float = 0.8
    gen_top_k: int = 5
    gen_seed: str = "the "        # prompt seed for sample generation

    # Output
    out_dir: str = "results"
    checkpoint_name: str = "qitlm_checkpoint.pt"
    results_name: str = "qitlm_charlm.json"
    plot_name: str = "qitlm_charlm.png"


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_ppl: float   # perplexity = exp(cross-entropy)
    val_ppl: float
    bpc: float         # bits-per-character on val set  = val_loss / ln(2)
    elapsed: float     # wall-clock seconds for this epoch
    sample: str = ""   # generated text (only populated every gen_every epochs)


# ── Training loop (generator — yields one EpochMetrics per epoch) ─────────────

def train(
    cfg: Config,
    step_callback=None,
    status_callback=None,
    corpus_path: Path | None = None,
    corpus_text: str | None = None,
    valid_path: Path | None = None,
    valid_text: str | None = None,
) -> Iterator[EpochMetrics]:
    """
    Generator training loop. Yields EpochMetrics after each epoch.
    step_callback receives within-epoch batch progress for live streaming.
    """
    STEP_EVERY = 1

    torch.manual_seed(cfg.seed)

    # Resolve paths from cfg when not passed explicitly (CLI usage).
    if corpus_path is None and cfg.corpus is not None:
        corpus_path = Path(cfg.corpus)
    if valid_path is None and cfg.valid_corpus is not None:
        valid_path = Path(cfg.valid_corpus)

    if corpus_path is not None:
        _emit_status(status_callback, f"Preparing corpus from {corpus_path.name}…", 0)
    elif corpus_text is not None:
        _emit_status(status_callback, f"Preparing pasted corpus ({len(corpus_text):,} chars)…", 0)
    else:
        _emit_status(status_callback, "Using built-in demo corpus…", 0)

    train_loader, val_loader, vocab = make_charlm_loaders(
        text=corpus_text,
        corpus_path=corpus_path,
        valid_path=valid_path,
        valid_text=valid_text,
        ctx_len=cfg.ctx_len,
        batch_size=cfg.batch_size,
        train_frac=cfg.train_frac,
        seed=cfg.seed,
        status_callback=status_callback,
        progress_callback=status_callback,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    n_qubits = cfg.ctx_len * cfg.n_qubits_per_token
    backend, diff = get_backend()
    _emit_status(
        status_callback,
        f"Initializing QIT-LM — {vocab.size} char vocab, {n_qubits} qubits, "
        f"backend={backend}, diff={diff}…",
        100,
    )
    model = QITLM(
        vocab_size=vocab.size,
        ctx_len=cfg.ctx_len,
        n_qubits_per_token=cfg.n_qubits_per_token,
        n_layers=cfg.n_layers,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    _emit_status(
        status_callback,
        f"Model ready ({model.n_parameters} params). "
        f"Epoch 1/{cfg.epochs}: starting quantum training "
        f"({steps_per_epoch} batches/epoch — first batch is slowest)…",
    )

    # ── Training ──────────────────────────────────────────────────────────────
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        if epoch > 1:
            _emit_status(
                status_callback,
                f"Epoch {epoch}/{cfg.epochs}: training ({steps_per_epoch} batches)…",
                0,
            )

        # Train
        model.train()
        train_loss_sum = 0.0
        total_steps = len(train_loader)
        for batch_i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(x)               # (batch, vocab_size)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

            if step_callback and (batch_i + 1) % STEP_EVERY == 0:
                step_callback(
                    epoch,
                    batch_i + 1,
                    total_steps,
                    train_loss_sum / (batch_i + 1),
                )

        train_loss = train_loss_sum / max(batch_i + 1, 1)

        _emit_status(status_callback, f"Epoch {epoch}/{cfg.epochs}: validating…", 95)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for val_i, (x, y) in enumerate(val_loader):
                logits = model(x)
                val_loss_sum += criterion(logits, y).item()
                n_val_batches += 1
        val_loss = val_loss_sum / max(n_val_batches, 1)

        # Sample generation
        sample = ""
        if epoch % cfg.gen_every == 0 or epoch == cfg.epochs:
            _emit_status(status_callback, f"Epoch {epoch}/{cfg.epochs}: generating sample text…", 98)
            prompt_ids = vocab.encode(cfg.gen_seed)
            if not prompt_ids:
                prompt_ids = [0]
            gen_ids = model.generate(
                prompt_ids,
                max_new_tokens=cfg.gen_tokens,
                temperature=cfg.gen_temperature,
                top_k=cfg.gen_top_k,
            )
            sample = vocab.decode(gen_ids)

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_ppl=math.exp(min(train_loss, 20)),
            val_ppl=math.exp(min(val_loss, 20)),
            bpc=val_loss / math.log(2),
            elapsed=time.time() - t0,
            sample=sample,
        )

        yield metrics, model, vocab

    # Save checkpoint
    _emit_status(status_callback, "Training complete — writing checkpoint to disk…")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "vocab": vocab, "config": asdict(cfg)},
        out_dir / cfg.checkpoint_name,
    )
    _emit_status(status_callback, f"Checkpoint saved: {out_dir / cfg.checkpoint_name}")


# ── CLI helpers ───────────────────────────────────────────────────────────────

def _plot(history: list[EpochMetrics], out_path: Path) -> None:
    epochs = [m.epoch for m in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, [m.train_loss for m in history], label="train")
    axes[0].plot(epochs, [m.val_loss for m in history], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("QIT-LM — Loss")
    axes[0].legend()

    axes[1].plot(epochs, [m.val_ppl for m in history])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("QIT-LM — Val Perplexity")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(cfg: Config) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'─'*60}")
    print("  QIT-LM — Quantum Character Language Model")
    print(f"{'─'*60}")

    history: list[EpochMetrics] = []
    model_ref = None
    vocab_ref = None

    for metrics, model, vocab in train(cfg):
        model_ref = model
        vocab_ref = vocab
        history.append(metrics)

        tag = f"[{metrics.epoch:3d}/{cfg.epochs}]"
        print(
            f"{tag}  loss {metrics.train_loss:.4f} / {metrics.val_loss:.4f}"
            f"  ppl {metrics.val_ppl:.2f}  bpc {metrics.bpc:.3f}"
            f"  ({metrics.elapsed:.1f}s)"
        )
        if metrics.sample:
            print(f"         sample: {metrics.sample!r}")

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "config": asdict(cfg),
        "vocab_size": vocab_ref.size,
        "vocab_chars": vocab_ref.chars,
        "n_parameters": model_ref.n_parameters,
        "history": [
            {
                "epoch": m.epoch,
                "train_loss": m.train_loss,
                "val_loss": m.val_loss,
                "train_ppl": m.train_ppl,
                "val_ppl": m.val_ppl,
                "bpc": m.bpc,
                "elapsed": m.elapsed,
                "sample": m.sample,
            }
            for m in history
        ],
    }

    json_path = out_dir / cfg.results_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    plot_path = out_dir / cfg.plot_name
    _plot(history, plot_path)

    final = history[-1]
    print(f"\n{'─'*60}")
    print(f"  Final val loss: {final.val_loss:.4f}  ppl: {final.val_ppl:.2f}  bpc: {final.bpc:.3f}")
    print(f"  Checkpoint:     {out_dir / cfg.checkpoint_name}")
    print(f"  Results:        {json_path}")
    print(f"  Plot:           {plot_path}")
    print(f"{'─'*60}\n")

    print(f"Model: {model_ref}")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train QIT-LM on character-level text")
    p.add_argument("--corpus",       default=None,  help="Path to .txt corpus file (default: built-in demo)")
    p.add_argument("--valid",        default=None,  help="Path to .valid.txt for validation (overrides train_frac split)", dest="valid_corpus")
    p.add_argument("--ctx_len",      type=int,   default=6,    help="Context window / n_tokens")
    p.add_argument("--n_qubits_per_token", type=int, default=2)
    p.add_argument("--n_layers",     type=int,   default=2)
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--lr",           type=float, default=0.05)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--temperature",  type=float, default=0.8,  dest="gen_temperature")
    p.add_argument("--seed_text",    default="the ",           dest="gen_seed")
    p.add_argument("--out_dir",      default="results")
    args = p.parse_args()
    cfg = Config()
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


if __name__ == "__main__":
    main(_parse_args())
