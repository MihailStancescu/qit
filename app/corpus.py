"""
Text processing: chunking, TF-IDF retrieval, and QIT-LM perplexity scoring.

The retrieval pipeline for Mode 2 (Tolkien Chat):
  1. chunk_text()           → split corpus into overlapping passages
  2. tfidf_search()         → rank passages by lexical similarity to query
  3. qit_perplexity_rank()  → re-rank top-k using QIT-LM cross-entropy
                              (lower perplexity = more in-distribution = more relevant)
"""

from __future__ import annotations

import math
import re
from collections import Counter

import torch
import torch.nn as nn


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 80,
) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def tfidf_search(
    query: str,
    passages: list[str],
    top_k: int = 5,
) -> list[tuple[float, str]]:
    """
    Return top_k (score, passage) pairs ordered by descending TF-IDF score.
    Uses a simple in-memory implementation — no external dependencies.
    """
    if not passages:
        return []

    q_terms = set(_tokenize(query))
    doc_term_counts = [Counter(_tokenize(p)) for p in passages]
    N = len(passages)

    # IDF: log((N+1)/(df+1)) + 1  (smoothed)
    idf: dict[str, float] = {}
    for term in q_terms:
        df = sum(1 for dtc in doc_term_counts if term in dtc)
        idf[term] = math.log((N + 1) / (df + 1)) + 1.0

    scores: list[tuple[float, str]] = []
    for dtc, passage in zip(doc_term_counts, passages):
        total = sum(dtc.values()) or 1
        score = sum(
            (dtc.get(t, 0) / total) * idf[t]
            for t in q_terms
        )
        scores.append((score, passage))

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:top_k]


# ── QIT-LM perplexity scorer ──────────────────────────────────────────────────

@torch.no_grad()
def qit_perplexity(
    model,      # QITLM
    vocab,      # CharVocab
    text: str,
    max_windows: int = 8,
    stride: int = 5,
) -> float:
    """
    Estimate the cross-entropy (perplexity proxy) of text under QIT-LM.

    Samples max_windows evenly spaced windows from the text, runs the
    quantum circuit on each, and returns exp(mean loss). Lower values
    mean the passage is more in-distribution with the training corpus.

    Capped at max_windows to keep latency under ~2 s.
    """
    model.eval()
    ctx = model.ctx_len
    ids = vocab.encode(text)

    if len(ids) <= ctx:
        return float("inf")

    criterion = nn.CrossEntropyLoss()
    positions = list(range(0, len(ids) - ctx, stride))

    # Sample evenly if too many positions
    if len(positions) > max_windows:
        step = len(positions) // max_windows
        positions = positions[::step][:max_windows]

    losses: list[float] = []
    for i in positions:
        x = torch.tensor(ids[i : i + ctx], dtype=torch.long).unsqueeze(0)
        y = torch.tensor([ids[i + ctx]], dtype=torch.long)
        logits = model(x)
        losses.append(criterion(logits, y).item())

    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def qit_rerank(
    model,
    vocab,
    scored_passages: list[tuple[float, str]],
    top_k: int = 3,
) -> list[tuple[float, float, str]]:
    """
    Re-rank TF-IDF top passages using QIT perplexity.
    Returns list of (tfidf_score, qit_ppl, passage) sorted by ppl ascending.
    """
    results: list[tuple[float, float, str]] = []
    for tfidf_score, passage in scored_passages:
        ppl = qit_perplexity(model, vocab, passage)
        results.append((tfidf_score, ppl, passage))

    # Sort: lower perplexity first (more in-distribution)
    results.sort(key=lambda x: x[1])
    return results[:top_k]
