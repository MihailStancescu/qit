"""
QIT Chat pipeline: TF-IDF retrieval → QIT-LM re-ranking → (optional) Claude answer.

Without a Claude API key the pipeline still runs fully:
  1. TF-IDF finds the most lexically relevant passages
  2. QIT-LM re-ranks them by perplexity (lower = more in-distribution)
  3. Passages + confidence scores are returned directly

With a Claude API key step 3 also generates a prose answer from those passages.
"""

from __future__ import annotations

import math
import os

import anthropic

from app.corpus import chunk_text, qit_rerank, tfidf_search

SYSTEM_PROMPT = """\
You are a helpful assistant answering questions about a text corpus provided by the user. \
Answer using ONLY the context passages supplied with each question. \
If the context does not contain enough information to answer confidently, \
say so clearly rather than speculating. Keep answers concise and grounded in the text.\
"""


def _ppls_to_confidence(ppls: list[float]) -> list[float]:
    """Convert perplexity scores to 0–100 confidence values (higher = more in-domain)."""
    if not ppls:
        return []
    finite = [p for p in ppls if math.isfinite(p) and p > 0]
    if not finite:
        return [0.0] * len(ppls)
    min_ppl = min(finite)
    return [
        round(100.0 * min_ppl / p, 1) if (math.isfinite(p) and p > 0) else 0.0
        for p in ppls
    ]


def answer(
    question: str,
    corpus_text: str,
    model=None,     # QITLM — if None, skip QIT re-ranking
    vocab=None,     # CharVocab
    api_key: str | None = None,
    claude_model: str = "claude-sonnet-4-6",
    max_tokens: int = 1024,
) -> dict:
    """
    Full retrieval pipeline. Claude answer is optional.

    Returns:
        {
          "answer":           str | None,   # Claude's prose answer, or None
          "passages":         list[str],
          "qit_ppls":         list[float],
          "qit_confidences":  list[float],  # 0-100, derived from ppls
          "method":           str,          # "tfidf" | "tfidf+qit"
        }
    """
    # 1. Chunk corpus
    chunks = chunk_text(corpus_text, chunk_size=300, overlap=80)
    if not chunks:
        return {
            "answer": None,
            "passages": [],
            "qit_ppls": [],
            "qit_confidences": [],
            "method": "none",
        }

    # 2. TF-IDF retrieval (top 6 candidates)
    tfidf_hits = tfidf_search(question, chunks, top_k=6)

    # 3. QIT re-ranking (top 3 final)
    qit_ppls: list[float] = []
    method: str

    if model is not None and vocab is not None:
        try:
            reranked = qit_rerank(model, vocab, tfidf_hits, top_k=3)
            passages = [p for _, _, p in reranked]
            qit_ppls = [round(ppl, 2) for _, ppl, _ in reranked]
            method = "tfidf+qit"
        except Exception:
            passages = [p for _, p in tfidf_hits[:3]]
            method = "tfidf"
    else:
        passages = [p for _, p in tfidf_hits[:3]]
        method = "tfidf"

    confidences = _ppls_to_confidence(qit_ppls)

    # 4. Claude answer (optional)
    answer_text: str | None = None
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        context_block = "\n\n---\n\n".join(
            f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
        )
        user_message = (
            f"Context from the text:\n\n{context_block}\n\n"
            f"Question: {question}"
        )
        try:
            client = anthropic.Anthropic(api_key=key)
            response = client.messages.create(
                model=claude_model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            answer_text = response.content[0].text
        except Exception as exc:
            answer_text = f"⚠️ Claude API error: {exc}"

    return {
        "answer":          answer_text,
        "passages":        passages,
        "qit_ppls":        qit_ppls,
        "qit_confidences": confidences,
        "method":          method,
    }
