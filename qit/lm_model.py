"""
QITLM — Quantum Interference Transformer Language Model.

Extends QIT-0 from sequence classification to autoregressive character-level
language modeling. Architecture is identical to QIT-0; the only difference is
n_classes = vocab_size so the decoder predicts the next character.

Usage:
    model = QITLM(vocab_size=27, ctx_len=8)
    logits = model(token_ids)                    # (batch, vocab_size)
    ids    = model.generate(prompt_ids, max_new_tokens=40)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from qit.model import QIT


class QITLM(nn.Module):
    """
    QIT as an autoregressive character-level language model.

    Input:  (batch, ctx_len) integer token indices
    Output: (batch, vocab_size) logits over the next character
    """

    def __init__(
        self,
        vocab_size: int,
        ctx_len: int = 8,
        n_qubits_per_token: int = 2,
        n_layers: int = 2,
        backend: str = "default.qubit",
        entangle_topology: str = "ring",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ctx_len = ctx_len
        self.qit = QIT(
            vocab_size=vocab_size,
            n_classes=vocab_size,
            n_tokens=ctx_len,
            n_qubits_per_token=n_qubits_per_token,
            n_layers=n_layers,
            backend=backend,
            entangle_topology=entangle_topology,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, ctx_len) integer token indices
        Returns:
            (batch, vocab_size) logits
        """
        return self.qit(token_ids)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> list[int]:
        """
        Autoregressive generation via a sliding context window.

        Args:
            prompt_ids:     Seed token ids (at least 1).
            max_new_tokens: How many new characters to produce.
            temperature:    Softmax temperature — lower = sharper distribution.
            top_k:          Restrict sampling to the top-k most likely tokens.

        Returns:
            Full sequence (prompt + generated) as a list of token ids.
        """
        self.eval()
        context = list(prompt_ids)

        for _ in range(max_new_tokens):
            # Build a fixed-length context window, padding with zeros if short.
            if len(context) >= self.ctx_len:
                window = context[-self.ctx_len :]
            else:
                pad = [0] * (self.ctx_len - len(context))
                window = pad + context

            x = torch.tensor(window, dtype=torch.long).unsqueeze(0)  # (1, ctx_len)
            logits = self.forward(x)[0]  # (vocab_size,)

            logits = logits / max(temperature, 1e-6)

            if top_k is not None:
                k = min(top_k, self.vocab_size)
                top_vals, _ = torch.topk(logits, k)
                logits[logits < top_vals[-1]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
            context.append(next_id)

        return context

    @property
    def n_parameters(self) -> int:
        return self.qit.n_parameters

    def parameter_breakdown(self) -> dict[str, int]:
        return self.qit.parameter_breakdown()

    def __repr__(self) -> str:
        pb = self.parameter_breakdown()
        n_qubits = self.qit.n_qubits
        return (
            f"QITLM(\n"
            f"  vocab={self.vocab_size}  ctx_len={self.ctx_len}  "
            f"n_qubits={n_qubits}\n"
            f"  params: embedding={pb['embedding']}  "
            f"attention={pb['attention']}  decoder={pb['decoder']}  "
            f"total={pb['total']}\n"
            f")"
        )
