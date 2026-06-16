"""
Tolkien Q&A pipeline: TF-IDF retrieval → QIT-LM re-ranking → Claude answer.

Requires ANTHROPIC_API_KEY in environment or passed explicitly at call time.
"""

from __future__ import annotations

import os

import anthropic

from app.corpus import chunk_text, qit_rerank, tfidf_search

SYSTEM_PROMPT = """\
You are a knowledgeable assistant specialising in Tolkien's legendarium — \
The Hobbit, The Lord of the Rings, The Silmarillion, and related works. \
Answer questions using ONLY the context passages provided. \
If the context does not contain enough information to answer confidently, \
say so clearly rather than speculating. Keep answers concise and grounded in the text.\
"""


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
    Full RAG pipeline.

    Returns:
        {
          "answer":   str,
          "passages": list[str],        # top passages used as context
          "qit_ppls": list[float],      # QIT perplexity per passage (if available)
          "method":   str,              # "tfidf" or "tfidf+qit"
        }
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return {
            "answer": "⚠️ No Anthropic API key configured. Set ANTHROPIC_API_KEY or enter it in the chat panel.",
            "passages": [],
            "qit_ppls": [],
            "method": "none",
        }

    # 1. Chunk corpus
    chunks = chunk_text(corpus_text, chunk_size=300, overlap=80)
    if not chunks:
        return {
            "answer": "Corpus is empty. Upload a text file in QIT Studio first.",
            "passages": [],
            "qit_ppls": [],
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
            # Fall back gracefully if QIT scoring fails
            passages = [p for _, p in tfidf_hits[:3]]
            method = "tfidf"
    else:
        passages = [p for _, p in tfidf_hits[:3]]
        method = "tfidf"

    # 4. Claude API call
    context_block = "\n\n---\n\n".join(
        f"[Passage {i+1}]\n{p}" for i, p in enumerate(passages)
    )
    user_message = (
        f"Context from the text:\n\n{context_block}\n\n"
        f"Question: {question}"
    )

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model=claude_model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    answer_text = response.content[0].text

    return {
        "answer":   answer_text,
        "passages": passages,
        "qit_ppls": qit_ppls,
        "method":   method,
    }
