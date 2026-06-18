"""
QIT Web App — FastAPI backend.

Endpoints:
    GET  /                          → index.html
    POST /api/corpus/upload         → stream-upload text corpus to disk
    POST /api/corpus/upload-valid   → stream-upload validation corpus to disk
    POST /api/corpus/text           → set corpus from raw text body (paste path)
    POST /api/train                 → start training job
    GET  /api/train/{job_id}/stream → SSE: training progress events
    GET  /api/train/{job_id}        → job status + final metrics
    POST /api/generate              → generate text from prompt
    POST /api/qa                    → QIT Chat Q&A (RAG + Claude)
    GET  /api/status                → current app state summary

Run:
    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import qa as qa_module
from app.trainer import (
    CHECKPOINT_PATH,
    STATE,
    _drain_job_queue_nowait,
    drain_job_queue,
    get_job,
    start_training,
)
from qit.backend import device_info

# Cap for TF-IDF indexing in Q&A (500 KB is plenty for retrieval).
QA_MAX_CHARS = 500_000
# Suggest smaller corpora for training — QIT-LM is tiny (~3k params).
TRAIN_RECOMMENDED_BYTES = 2 * 1024 * 1024   # 2 MB

# Persistent temp dir for uploaded corpora — survives the request lifecycle.
UPLOAD_DIR = Path(tempfile.gettempdir()) / "qit_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="QIT — Quantum Interference Transformer", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ── Corpus endpoints ──────────────────────────────────────────────────────────

class TextBody(BaseModel):
    text: str


@app.post("/api/corpus/text")
async def set_corpus_text(body: TextBody):
    """Accept raw pasted text as the active corpus (small files / quick experiments)."""
    text = body.text.strip()
    if len(text) < 50:
        raise HTTPException(400, "Corpus too short (need at least 50 characters).")
    STATE.active_corpus = text
    STATE.corpus_path = None    # pasted text overrides any file upload
    return {"ok": True, "chars": len(text)}


@app.post("/api/corpus/upload")
async def upload_corpus(file: UploadFile = File(...)):
    """
    Stream the uploaded file straight to disk — never holds GB content in RAM.
    The file path is stored in STATE; training and Q&A read from it lazily.
    """
    dest = UPLOAD_DIR / f"corpus_{uuid.uuid4().hex}.txt"
    await file.seek(0)

    def _copy():
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)

    await run_in_threadpool(_copy)

    size = dest.stat().st_size
    if size < 50:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "File too short (need at least 50 characters).")

    STATE.corpus_path = dest
    STATE.active_corpus = None      # file upload overrides any paste
    return {"ok": True, "filename": file.filename, "bytes": size}


@app.post("/api/corpus/upload-valid")
async def upload_valid_corpus(file: UploadFile = File(...)):
    """Stream a .valid.txt validation file to disk."""
    dest = UPLOAD_DIR / f"valid_{uuid.uuid4().hex}.txt"
    await file.seek(0)

    def _copy():
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)

    await run_in_threadpool(_copy)

    size = dest.stat().st_size
    if size < 10:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Validation file too short.")

    STATE.valid_path = dest
    STATE.valid_corpus = None
    return {"ok": True, "filename": file.filename, "bytes": size}


# ── Training endpoints ────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    ctx_len: int = 6
    n_qubits_per_token: int = 2
    n_layers: int = 2
    epochs: int = 60
    lr: float = 0.05
    batch_size: int = 8
    gen_every: int = 10
    seed: int = 42
    corpus_text: str | None = None          # optional inline corpus; else uses active corpus


@app.post("/api/train")
async def start_train(req: TrainRequest):
    corpus_path = None
    corpus_text = req.corpus_text

    if corpus_text:
        if len(corpus_text) < 50:
            raise HTTPException(400, "Corpus too short.")
        train_chars = len(corpus_text)
    elif STATE.corpus_path and STATE.corpus_path.exists():
        corpus_path = STATE.corpus_path
        train_chars = corpus_path.stat().st_size
        if train_chars < 50:
            raise HTTPException(400, "Corpus too short.")
    else:
        corpus_text = await run_in_threadpool(STATE.get_corpus_text)
        if not corpus_text:
            raise HTTPException(400, "No corpus loaded. Upload a text file first.")
        if len(corpus_text) < 50:
            raise HTTPException(400, "Corpus too short.")
        train_chars = len(corpus_text)

    valid_path = STATE.valid_path if STATE.valid_path and STATE.valid_path.exists() else None
    valid_text = None
    if not valid_path:
        valid_text = await run_in_threadpool(STATE.get_valid_text)

    warning = None
    if train_chars > TRAIN_RECOMMENDED_BYTES:
        rec_mb = TRAIN_RECOMMENDED_BYTES / 1024 / 1024
        warning = (
            f"Corpus is {train_chars / 1024 / 1024:.1f} MB — training may be slow and memory-heavy. "
            f"QIT-LM has only ~3k parameters; a {rec_mb:.0f} MB excerpt is usually enough."
        )

    job_id = start_training(
        corpus_text=corpus_text,
        corpus_path=corpus_path,
        ctx_len=req.ctx_len,
        n_qubits_per_token=req.n_qubits_per_token,
        n_layers=req.n_layers,
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
        gen_every=req.gen_every,
        seed=req.seed,
        valid_corpus_text=valid_text,
        valid_path=valid_path,
    )
    vi = STATE.valid_info()
    return {
        "job_id": job_id,
        "warning": warning,
        "train_chars": train_chars,
        "has_valid": vi["has_valid"],
        "valid_bytes": vi["bytes"],
    }


@app.get("/api/train/{job_id}/stream")
async def stream_training(job_id: str):
    """
    Server-Sent Events endpoint.
    Each event is a JSON object:
      { type: "progress", epoch, train_loss, val_loss, train_ppl, val_ppl, bpc, elapsed, sample }
      { type: "done" }
      { type: "error", message }
      { type: "heartbeat" }
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found.")

    async def event_stream():
        loop = asyncio.get_running_loop()
        # Open the stream immediately so clients know the connection is alive.
        yield ": connected\n\n"
        while True:
            if job.status in ("done", "error"):
                # Drain any remaining events after the worker finishes.
                events: list[dict] = await loop.run_in_executor(
                    None, lambda: _drain_job_queue_nowait(job)
                )
                for evt in events:
                    yield f"data: {json.dumps(evt)}\n\n"
                return

            events = await loop.run_in_executor(None, drain_job_queue, job)
            for evt in events:
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("type") in ("done", "error"):
                    return
            if not events:
                yield 'data: {"type":"heartbeat"}\n\n'
            await asyncio.sleep(0)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/train/{job_id}")
async def job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} not found.")
    return {
        "job_id": job_id,
        "status": job.status,
        "error":  job.error,
        "has_model": job.model is not None,
    }


# ── Generation endpoint ───────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = "the "
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 5


@app.post("/api/generate")
async def generate_text(req: GenerateRequest):
    model, vocab = STATE.get_active()
    if model is None or vocab is None:
        raise HTTPException(400, "No trained model available. Train a model first.")

    prompt_ids = vocab.encode(req.prompt)
    if not prompt_ids:
        prompt_ids = [0]

    loop = asyncio.get_event_loop()
    gen_ids = await loop.run_in_executor(
        None,
        lambda: model.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        ),
    )
    generated = vocab.decode(gen_ids)
    return {"text": generated, "chars": len(generated)}


# ── Q&A endpoint ─────────────────────────────────────────────────────────────

class QARequest(BaseModel):
    question: str
    api_key: str | None = None
    claude_model: str = "claude-sonnet-4-6"


@app.post("/api/qa")
async def ask_qit(req: QARequest):
    # Read up to QA_MAX_CHARS from the corpus (file or pasted text).
    corpus = STATE.get_corpus_text(QA_MAX_CHARS)
    if not corpus:
        raise HTTPException(
            400,
            "No corpus loaded. Upload a text file in QIT Studio first.",
        )

    model, vocab = STATE.get_active()
    api_key = req.api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: qa_module.answer(
            question=req.question,
            corpus_text=corpus,
            model=model,
            vocab=vocab,
            api_key=api_key,
            claude_model=req.claude_model,
        ),
    )
    return result


# ── Status endpoint ───────────────────────────────────────────────────────────

@app.get("/api/status")
async def app_status():
    model, vocab = STATE.get_active()
    ci = STATE.corpus_info()
    vi = STATE.valid_info()
    return {
        "has_corpus":   ci["has_corpus"],
        "corpus_chars": ci["bytes"],    # bytes ≈ chars for ASCII/Latin text
        "has_valid":    vi["has_valid"],
        "valid_bytes":  vi["bytes"],
        "has_model":    model is not None,
        "vocab_size":   vocab.size if vocab else None,
        "n_parameters": model.n_parameters if model else None,
        "n_qubits":     model.qit.n_qubits if model else None,
        "ctx_len":      model.ctx_len if model else None,
        "active_jobs":  sum(1 for j in STATE.jobs.values() if j.status == "running"),
        "checkpoint":   str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        "device":       device_info(),
        "train_recommended_bytes": TRAIN_RECOMMENDED_BYTES,
    }
