"""
QIT Web App — FastAPI backend.

Endpoints:
    GET  /                          → index.html
    POST /api/corpus/upload         → upload text corpus
    POST /api/corpus/text           → set corpus from raw text body
    POST /api/train                 → start training job
    GET  /api/train/{job_id}/stream → SSE: training progress events
    GET  /api/train/{job_id}        → job status + final metrics
    POST /api/generate              → generate text from prompt
    POST /api/qa                    → Tolkien Q&A (RAG + Claude)
    GET  /api/status                → current app state summary

Run:
    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import qa as qa_module
from app.trainer import STATE, drain_job_queue, get_job, start_training

# Cap how many chars are fed to the quantum training loop.
# Full corpus is still used for Q&A retrieval — only training is capped.
MAX_TRAIN_CHARS = 25_000

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="QIT — Quantum Interference Transformer", version="0.1.0")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


# ── Corpus endpoints ──────────────────────────────────────────────────────────

class TextBody(BaseModel):
    text: str


@app.post("/api/corpus/text")
async def set_corpus_text(body: TextBody):
    """Accept raw text as the active corpus."""
    text = body.text.strip()
    if len(text) < 50:
        raise HTTPException(400, "Corpus too short (need at least 50 characters).")
    STATE.active_corpus = text
    return {"ok": True, "chars": len(text)}


@app.post("/api/corpus/upload")
async def upload_corpus(file: UploadFile = File(...)):
    """Accept a .txt file upload as the active corpus."""
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="replace").strip()
    except Exception as e:
        raise HTTPException(400, f"Could not decode file: {e}")
    if len(text) < 50:
        raise HTTPException(400, "File too short (need at least 50 characters).")
    STATE.active_corpus = text
    return {"ok": True, "filename": file.filename, "chars": len(text)}


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
    corpus_text: str | None = None   # optional inline corpus; else uses active corpus


@app.post("/api/train")
async def start_train(req: TrainRequest):
    corpus = req.corpus_text or STATE.active_corpus
    if not corpus:
        raise HTTPException(400, "No corpus loaded. Upload a text file first.")
    if len(corpus) < 50:
        raise HTTPException(400, "Corpus too short.")

    # Store full corpus for Q&A retrieval, but cap what goes to the quantum trainer.
    STATE.active_corpus = corpus
    warning = None
    train_corpus = corpus
    if len(corpus) > MAX_TRAIN_CHARS:
        train_corpus = corpus[:MAX_TRAIN_CHARS]
        warning = (
            f"Training corpus capped at {MAX_TRAIN_CHARS:,} chars "
            f"(your file has {len(corpus):,}). "
            f"Full text is still used for Tolkien Chat retrieval."
        )

    job_id = start_training(
        corpus_text=train_corpus,
        ctx_len=req.ctx_len,
        n_qubits_per_token=req.n_qubits_per_token,
        n_layers=req.n_layers,
        epochs=req.epochs,
        lr=req.lr,
        batch_size=req.batch_size,
        gen_every=req.gen_every,
        seed=req.seed,
    )
    return {"job_id": job_id, "warning": warning, "train_chars": len(train_corpus)}


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
        while True:
            # Block in a thread executor so the event loop stays free.
            # drain_job_queue blocks up to 1 s waiting for the next event.
            events: list[dict] = await loop.run_in_executor(
                None, drain_job_queue, job
            )
            for evt in events:
                yield f"data: {json.dumps(evt)}\n\n"
                if evt.get("type") in ("done", "error"):
                    return
            if not events:
                # Heartbeat keeps the SSE connection alive between epochs.
                yield "data: {\"type\":\"heartbeat\"}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
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
    model, vocab, _ = STATE.get_active()
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
async def ask_tolkien(req: QARequest):
    _, _, corpus = STATE.get_active()
    if not corpus:
        corpus = STATE.active_corpus
    if not corpus:
        raise HTTPException(
            400,
            "No corpus loaded. Upload a text file in QIT Studio first.",
        )

    model, vocab, _ = STATE.get_active()
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
    model, vocab, corpus = STATE.get_active()
    return {
        "has_corpus":   corpus is not None,
        "corpus_chars": len(corpus) if corpus else 0,
        "has_model":    model is not None,
        "vocab_size":   vocab.size if vocab else None,
        "n_parameters": model.n_parameters if model else None,
        "n_qubits":     model.qit.n_qubits if model else None,
        "ctx_len":      model.ctx_len if model else None,
        "active_jobs":  sum(1 for j in STATE.jobs.values() if j.status == "running"),
    }
