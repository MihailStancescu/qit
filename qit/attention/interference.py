import math

import pennylane as qml
import torch
import torch.nn as nn

from qit.encoding.angle import angle_encode
from qit.layers.entangle import u_entangle
from qit.layers.mix import u_mix


class QuantumInterferenceAttention(nn.Module):
    """
    Implements U_QIT(θ) = U_mix · U_attention · U_entangle over a full token sequence.

    Each token occupies n_qubits_per_token qubits. The full sequence register is:
        |Ψ₀⟩ = U_encode(x)|0⟩^⊗(n_tokens * n_qubits_per_token)

    Relevance between tokens is not computed via dot-product similarity.
    It emerges from amplitude amplification and destructive interference
    produced by the learned unitary U_attention.

    Default QIT-0 configuration:
        n_tokens=4, n_qubits_per_token=2, n_layers=2 → 56 trainable parameters
    """

    def __init__(
        self,
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
        self.n_layers = n_layers
        self.entangle_topology = entangle_topology

        # Trainable parameters registered directly with PyTorch.
        # Uniform init over [-π, π] covers the full rotation space.
        self.weights_attention = nn.Parameter(
            torch.empty(n_layers, self.n_qubits, 3).uniform_(-math.pi, math.pi)
        )
        self.weights_mix = nn.Parameter(
            torch.empty(1, self.n_qubits).uniform_(-math.pi, math.pi)
        )

        dev = qml.device(backend, wires=self.n_qubits)
        all_wires = list(range(self.n_qubits))

        @qml.qnode(dev, interface="torch")
        def _circuit(inputs, weights_attention, weights_mix):
            # ── Encoding ──────────────────────────────────────────────────────
            # Each token register gets its own angle-encoded state.
            for t in range(n_tokens):
                start = t * n_qubits_per_token
                token_wires = list(range(start, start + n_qubits_per_token))
                token_feats = inputs[start: start + n_qubits_per_token]
                angle_encode(token_feats, token_wires)

            # ── U_entangle ────────────────────────────────────────────────────
            # CNOT ladder seeds non-local correlations across token registers.
            # Without this, U_attention acts on separable states and cannot
            # learn cross-token relationships.
            u_entangle(n_tokens, n_qubits_per_token, topology=entangle_topology)

            # ── U_attention ───────────────────────────────────────────────────
            # Strongly entangling layers: the learned interference operator.
            # Relevant token pairs constructively interfere → larger amplitudes.
            # Irrelevant pairs destructively interfere → suppressed amplitudes.
            qml.StronglyEntanglingLayers(weights_attention, wires=all_wires)

            # ── U_mix ─────────────────────────────────────────────────────────
            # Projects the post-attention state into measurement-ready form.
            u_mix(weights_mix, wires=all_wires)

            # ── Measurement ───────────────────────────────────────────────────
            # Pauli-Z expectation per qubit ∈ [-1, 1].
            return [qml.expval(qml.PauliZ(i)) for i in all_wires]

        self._circuit = _circuit

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_tokens * n_qubits_per_token) — pre-embedded, angle-range features
        Returns:
            (batch, n_qubits) — Pauli-Z expectation values in [-1, 1]
        """
        batch_results = []
        for xi in x:
            out = self._circuit(xi, self.weights_attention, self.weights_mix)
            # PennyLane returns a tuple of 0-d tensors for list-of-expval circuits.
            batch_results.append(torch.stack(list(out)))
        return torch.stack(batch_results)

    # ── utilities ─────────────────────────────────────────────────────────────

    @property
    def n_parameters(self) -> int:
        n_attn = self.n_layers * self.n_qubits * 3
        n_mix = self.n_qubits
        return n_attn + n_mix

    def draw(self) -> str:
        """Return an ASCII circuit diagram."""
        dummy_in = torch.zeros(self.n_qubits)
        dummy_attn = torch.zeros(self.n_layers, self.n_qubits, 3)
        dummy_mix = torch.zeros(1, self.n_qubits)
        return qml.draw(self._circuit)(dummy_in, dummy_attn, dummy_mix)

    def __repr__(self) -> str:
        return (
            f"QuantumInterferenceAttention("
            f"n_tokens={self.n_tokens}, "
            f"n_qubits_per_token={self.n_qubits_per_token}, "
            f"n_qubits={self.n_qubits}, "
            f"n_layers={self.n_layers}, "
            f"n_parameters={self.n_parameters})"
        )
