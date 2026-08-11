"""
Background training state manager.

Runs QIT-LM training in a daemon thread and exposes progress via a Queue
that FastAPI's SSE endpoint can drain asynchronously.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.train_charlm import Config, train
from qit.lm_model import QITLM
from tasks.charlm import CharVocab

# Persisted trained model — survives server restarts; used by Generate + QIT Chat.
MODEL_DIR = Path(__file__).parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_PATH = MODEL_DIR / "qitlm_active.pt"


def save_checkpoint(model: QITLM, vocab: CharVocab, config: dict) -> Path:
    """Write model, vocab, and training config to disk."""
    import torch

    torch.save(
        {"model_state": model.state_dict(), "vocab": vocab, "config": config},
        CHECKPOINT_PATH,
    )
    return CHECKPOINT_PATH


def load_checkpoint(path: Path | None = None) -> tuple[QITLM | None, CharVocab | None]:
    """Load the active checkpoint from disk, or return (None, None) if missing."""
    import torch

    ckpt_path = path or CHECKPOINT_PATH
    if not ckpt_path.exists():
        return None, None

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    vocab: CharVocab = ckpt["vocab"]
    config: dict = ckpt["config"]
    model = QITLM(
        vocab_size=vocab.size,
        ctx_len=config["ctx_len"],
        n_qubits_per_token=config.get("n_qubits_per_token", 2),
        n_layers=config.get("n_layers", 2),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


@dataclass
class TrainingJob:
    id: str
    status: str = "running"    # running | done | error | cancelled
    queue: Queue = field(default_factory=Queue)
    model: QITLM | None = None
    vocab: CharVocab | None = None
    corpus_text: str | None = None
    error: str | None = None
    thread: threading.Thread | None = None


# Process-level state: one model in memory at a time, multiple jobs possible.
class AppState:
    def __init__(self):
        self.jobs: dict[str, TrainingJob] = {}
        self.active_model: QITLM | None = None
        self.active_vocab: CharVocab | None = None
        # Corpus: either a file path (large uploads) or in-memory text (paste).
        # Path takes priority when both are set.
        self.corpus_path: Path | None = None    # absolute path to uploaded file
        self.active_corpus: str | None = None   # in-memory paste text
        self.valid_path: Path | None = None     # absolute path to validation file
        self.valid_corpus: str | None = None    # in-memory validation text
        self._lock = threading.Lock()
        self._load_persisted_model()

    def _load_persisted_model(self) -> None:
        model, vocab = load_checkpoint()
        if model is not None and vocab is not None:
            self.active_model = model
            self.active_vocab = vocab

    def get_corpus_text(self, max_chars: int | None = None) -> str | None:
        """Read corpus text, capped at max_chars. Prefers file path over in-memory."""
        if self.corpus_path and self.corpus_path.exists():
            with open(self.corpus_path, encoding="utf-8", errors="replace") as f:
                return f.read(max_chars) if max_chars else f.read()
        return self.active_corpus

    def get_valid_text(self) -> str | None:
        if self.valid_path and self.valid_path.exists():
            with open(self.valid_path, encoding="utf-8", errors="replace") as f:
                return f.read()
        return self.valid_corpus

    def corpus_info(self) -> dict:
        if self.corpus_path and self.corpus_path.exists():
            size = self.corpus_path.stat().st_size
            return {"source": "file", "bytes": size, "has_corpus": True}
        if self.active_corpus:
            return {"source": "paste", "bytes": len(self.active_corpus.encode()), "has_corpus": True}
        return {"source": None, "bytes": 0, "has_corpus": False}

    def valid_info(self) -> dict:
        if self.valid_path and self.valid_path.exists():
            size = self.valid_path.stat().st_size
            return {"source": "file", "bytes": size, "has_valid": True}
        if self.valid_corpus:
            return {"source": "paste", "bytes": len(self.valid_corpus.encode()), "has_valid": True}
        return {"source": None, "bytes": 0, "has_valid": False}

    def set_active(self, model: QITLM, vocab: CharVocab) -> None:
        with self._lock:
            self.active_model = model
            self.active_vocab = vocab

    def get_active(self) -> tuple[QITLM | None, CharVocab | None]:
        with self._lock:
            return self.active_model, self.active_vocab


STATE = AppState()


def start_training(
    corpus_text: str | None = None,
    corpus_path: Path | None = None,
    ctx_len: int = 6,
    n_qubits_per_token: int = 2,
    n_layers: int = 2,
    epochs: int = 60,
    lr: float = 0.05,
    batch_size: int = 8,
    gen_every: int = 10,
    seed: int = 42,
    max_steps_per_epoch: int | None = 200,
    max_val_steps: int | None = 20,
    valid_corpus_text: str | None = None,
    valid_path: Path | None = None,
    normalize: bool = True,
) -> str:
    """
    Kick off a training job in a background thread.
    Returns the job_id for SSE streaming and status polling.
    """
    job_id = str(uuid.uuid4())[:8]
    job = TrainingJob(id=job_id, corpus_text=corpus_text)
    STATE.jobs[job_id] = job

    cfg = Config(
        ctx_len=ctx_len,
        n_qubits_per_token=n_qubits_per_token,
        n_layers=n_layers,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        gen_every=gen_every,
        seed=seed,
        max_steps_per_epoch=max_steps_per_epoch,
        max_val_steps=max_val_steps,
        normalize=normalize,
        corpus=None,   # we pass text directly below
        out_dir=str(MODEL_DIR),
        checkpoint_name=CHECKPOINT_PATH.name,
    )

    def _run():
        import time as _time

        job.queue.put({"type": "status", "message": "Starting training…", "pct": 0})

        epoch_clock = {"start": 0.0, "last": 0.0}
        epoch_times: list[float] = []

        def status_cb(message: str, pct: float | None = None) -> None:
            payload: dict[str, Any] = {"type": "status", "message": message}
            if pct is not None:
                payload["pct"] = round(pct, 1)
            job.queue.put(payload)

        def step_cb(epoch, batch, total_batches, running_loss):
            now = _time.time()
            if batch == 1:
                epoch_clock["start"] = now
                epoch_clock["last"] = now
            else:
                epoch_clock["last"] = now
            elapsed = now - epoch_clock["start"]
            avg_batch = elapsed / batch
            eta_epoch = int(avg_batch * max(total_batches - batch, 0))
            pct_epoch = round(100.0 * batch / max(total_batches, 1), 1)
            overall_pct = round(
                100.0 * ((epoch - 1) + batch / max(total_batches, 1)) / cfg.epochs,
                1,
            )
            if batch == 1:
                status_cb(
                    f"Epoch {epoch}/{cfg.epochs}: quantum batch 1/{total_batches} "
                    f"(forward + backward on CPU)…",
                    pct_epoch,
                )
            job.queue.put({
                "type": "batch",
                "epoch": epoch,
                "epochs": cfg.epochs,
                "batch": batch,
                "total_batches": total_batches,
                "running_loss": round(running_loss, 4),
                "pct": pct_epoch,
                "pct_overall": overall_pct,
                "eta_epoch_sec": eta_epoch,
            })

        try:
            for metrics, model, vocab in _train_on_text(
                cfg,
                corpus_text=corpus_text,
                corpus_path=corpus_path,
                valid_text=valid_corpus_text,
                valid_path=valid_path,
                step_cb=step_cb,
                status_cb=status_cb,
                job=job,
            ):
                job.model = model
                job.vocab = vocab
                epoch_times.append(metrics.elapsed)
                avg_epoch = sum(epoch_times) / len(epoch_times)
                epochs_left = cfg.epochs - metrics.epoch
                progress = _metrics_to_dict(metrics, cfg.epochs)
                progress["eta_epoch_sec"] = int(avg_epoch)
                progress["eta_total_sec"] = int(avg_epoch * epochs_left)
                progress["pct_overall"] = round(100.0 * metrics.epoch / cfg.epochs, 1)
                job.queue.put({"type": "progress", **progress})

            STATE.set_active(job.model, job.vocab)
            if job.model is not None and job.vocab is not None:
                job.queue.put({"type": "status", "message": "Saving active model for Generate and QIT Chat…"})
                save_checkpoint(job.model, job.vocab, {
                    "ctx_len": cfg.ctx_len,
                    "n_qubits_per_token": cfg.n_qubits_per_token,
                    "n_layers": cfg.n_layers,
                    "epochs": cfg.epochs,
                    "lr": cfg.lr,
                    "batch_size": cfg.batch_size,
                    "seed": cfg.seed,
                })
                job.queue.put({"type": "status", "message": f"Model saved to {CHECKPOINT_PATH.name}"})
            job.status = "done"
            job.queue.put({"type": "done"})
        except Exception as exc:
            job.status = "error"
            job.error = str(exc)
            job.queue.put({"type": "error", "message": str(exc)})

    t = threading.Thread(target=_run, daemon=True, name=f"train-{job_id}")
    job.thread = t
    t.start()
    return job_id


def _train_on_text(
    cfg: Config,
    *,
    corpus_text: str | None = None,
    corpus_path: Path | None = None,
    valid_text: str | None = None,
    valid_path: Path | None = None,
    step_cb=None,
    status_cb=None,
    job: TrainingJob | None = None,
):
    """Run training using file paths (preferred) or in-memory text."""
    yield from train(
        cfg,
        step_callback=step_cb,
        status_callback=status_cb,
        corpus_path=corpus_path,
        corpus_text=corpus_text,
        valid_path=valid_path,
        valid_text=valid_text,
    )


def _metrics_to_dict(metrics, epochs: int) -> dict[str, Any]:
    return {
        "epoch":      metrics.epoch,
        "epochs":     epochs,
        "train_loss": round(metrics.train_loss, 4),
        "val_loss":   round(metrics.val_loss, 4),
        "train_ppl":  round(metrics.train_ppl, 3),
        "val_ppl":    round(metrics.val_ppl, 3),
        "bpc":        round(metrics.bpc, 4),
        "elapsed":    round(metrics.elapsed, 2),
        "sample":     metrics.sample,
    }


def get_job(job_id: str) -> TrainingJob | None:
    return STATE.jobs.get(job_id)


def _drain_job_queue_nowait(job: TrainingJob) -> list[dict]:
    """Drain all pending events without blocking."""
    events: list[dict] = []
    while True:
        try:
            events.append(job.queue.get_nowait())
        except Empty:
            break
    return events


def drain_job_queue(job: TrainingJob, timeout: float = 1.0) -> list[dict]:
    """
    Block up to `timeout` seconds waiting for the first event, then drain
    any additional events that arrived in the meantime. Returns [] on timeout.
    Using queue.get(timeout=...) instead of get_nowait() is critical — the
    previous busy-poll approach caused the SSE loop to spin with no events.
    """
    events: list[dict] = []
    try:
        events.append(job.queue.get(timeout=timeout))
        while True:
            try:
                events.append(job.queue.get_nowait())
            except Empty:
                break
    except Empty:
        pass
    return events
