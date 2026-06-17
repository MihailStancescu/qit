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
        self.active_corpus: str | None = None
        self.valid_corpus: str | None = None
        self._lock = threading.Lock()

    def set_active(self, model: QITLM, vocab: CharVocab, corpus: str) -> None:
        with self._lock:
            self.active_model = model
            self.active_vocab = vocab
            self.active_corpus = corpus

    def get_active(self) -> tuple[QITLM | None, CharVocab | None, str | None]:
        with self._lock:
            return self.active_model, self.active_vocab, self.active_corpus


STATE = AppState()


def start_training(
    corpus_text: str,
    ctx_len: int = 6,
    n_qubits_per_token: int = 2,
    n_layers: int = 2,
    epochs: int = 60,
    lr: float = 0.05,
    batch_size: int = 8,
    gen_every: int = 10,
    seed: int = 42,
    valid_corpus_text: str | None = None,
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
        corpus=None,   # we pass text directly below
    )

    def _run():
        def step_cb(epoch, batch, total_batches, running_loss):
            job.queue.put({
                "type": "batch",
                "epoch": epoch,
                "batch": batch,
                "total_batches": total_batches,
                "running_loss": round(running_loss, 4),
            })

        try:
            for metrics, model, vocab in _train_on_text(cfg, corpus_text, step_cb, valid_corpus_text):
                job.model = model
                job.vocab = vocab
                job.queue.put({"type": "progress", **_metrics_to_dict(metrics)})

            STATE.set_active(job.model, job.vocab, corpus_text)
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


def _train_on_text(cfg: Config, corpus_text: str, step_cb=None, valid_text: str | None = None):
    """Write corpus (and optional valid) to temp files and run the training generator."""
    import tempfile, os
    tmp_paths = []
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(corpus_text)
            cfg.corpus = f.name
            tmp_paths.append(f.name)

        if valid_text is not None:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".valid.txt", delete=False, encoding="utf-8") as f:
                f.write(valid_text)
                cfg.valid_corpus = f.name
                tmp_paths.append(f.name)

        yield from train(cfg, step_callback=step_cb)
    finally:
        for p in tmp_paths:
            os.unlink(p)


def _metrics_to_dict(metrics) -> dict[str, Any]:
    return {
        "epoch":      metrics.epoch,
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
