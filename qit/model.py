import math

import torch
import torch.nn as nn

from qit.attention import QuantumInterferenceAttention
from qit.measurement import ClassicalDecoder


class QIT(nn.Module):
    """
    QIT-0: Quantum Interference Transformer (prototype).

    Full pipeline:
        token_ids ──► Embedding ──► [angle features]
                                  ──► QuantumInterferenceAttention
                                  ──► ClassicalDecoder ──► logits

    The hidden state is fully quantum throughout the attention block.
    Classical processing only at the input embedding and output decoding.

    Default config for the parity benchmark:
        vocab_size=2, n_classes=2, n_tokens=4, n_qubits_per_token=2, n_layers=2
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int,
        n_tokens: int = 4,
        n_qubits_per_token: int = 2,
        n_layers: int = 2,
        backend: str = "default.qubit",
        entangle_topology: str = "ring",
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_qubits_per_token = n_qubits_per_token
        self.n_qubits = n_tokens * n_qubits_per_token

        # Classical input → angle-space features
        self.embedding = nn.Embedding(vocab_size, n_qubits_per_token)

        # Fully quantum attention block
        self.attention = QuantumInterferenceAttention(
            n_tokens=n_tokens,
            n_qubits_per_token=n_qubits_per_token,
            n_layers=n_layers,
            backend=backend,
            entangle_topology=entangle_topology,
        )

        # Quantum measurement results → class logits
        self.decoder = ClassicalDecoder(self.n_qubits, n_classes)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (batch, n_tokens) integer token indices
        Returns:
            (batch, n_classes) logits
        """
        # (batch, n_tokens) → (batch, n_tokens, n_qubits_per_token)
        x = self.embedding(token_ids)

        # Squash to [-π, π] for stable angle encoding.
        # tanh keeps gradients alive and bounds the rotation angles.
        x = torch.tanh(x) * math.pi

        # Flatten token dimension: (batch, n_qubits)
        x = x.flatten(1)

        # Quantum interference attention: (batch, n_qubits) → (batch, n_qubits)
        x = self.attention(x)

        # PennyLane returns float64; cast back to float32 at the quantum→classical boundary.
        x = x.float()

        # Decode to class logits: (batch, n_classes)
        return self.decoder(x)

    # ── utilities ─────────────────────────────────────────────────────────────

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> dict[str, int]:
        return {
            "embedding": sum(p.numel() for p in self.embedding.parameters()),
            "attention": self.attention.n_parameters,
            "decoder":   sum(p.numel() for p in self.decoder.parameters()),
            "total":     self.n_parameters,
        }

    def __repr__(self) -> str:
        pb = self.parameter_breakdown()
        lines = [
            "QIT-0(",
            f"  vocab={self.embedding.num_embeddings}  "
            f"n_tokens={self.n_tokens}  n_qubits={self.n_qubits}",
            f"  params: embedding={pb['embedding']}  "
            f"attention={pb['attention']}  decoder={pb['decoder']}  "
            f"total={pb['total']}",
            ")",
        ]
        return "\n".join(lines)
