from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pennylane as qml
import torch
import torch.nn as nn

from qit.backend import get_backend
from qit.encoding.angle import angle_encode
from qit.layers.entangle import u_entangle
from qit.layers.mix import u_mix

# Thread-local storage: each worker thread gets its own QNode instance so
# lightning.qubit's internal C++ state vector is never shared across threads.
_thread_local = threading.local()


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
        backend: str | None = None,
        diff_method: str | None = None,
        entangle_topology: str = "ring",
        n_workers: int | None = None,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.n_qubits_per_token = n_qubits_per_token
        self.n_qubits = n_tokens * n_qubits_per_token
        self.n_layers = n_layers
        self.entangle_topology = entangle_topology

        detected_backend, detected_diff = get_backend()
        self._backend = backend or detected_backend
        self._diff_method = diff_method or detected_diff
        # Max parallel workers: one per batch item, capped at CPU count.
        self._n_workers = n_workers or (os.cpu_count() or 4)

        # Trainable parameters registered directly with PyTorch.
        # Uniform init over [-π, π] covers the full rotation space.
        self.weights_attention = nn.Parameter(
            torch.empty(n_layers, self.n_qubits, 3).uniform_(-math.pi, math.pi)
        )
        self.weights_mix = nn.Parameter(
            torch.empty(1, self.n_qubits).uniform_(-math.pi, math.pi)
        )

        # Build the main-thread QNode (used during training where autograd must
        # remain on a single, stable device instance).
        self._circuit = self._build_circuit()

    # ── QNode factory ─────────────────────────────────────────────────────────

    def _build_circuit(self):
        """Create a fresh (device, QNode) pair — called once per thread."""
        n_tokens = self.n_tokens
        n_qubits = self.n_qubits
        n_qubits_per_token = self.n_qubits_per_token
        all_wires = list(range(n_qubits))
        entangle_topology = self.entangle_topology
        n_layers = self.n_layers

        dev = qml.device(self._backend, wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method=self._diff_method)
        def _circuit(inputs, weights_attention, weights_mix):
            # ── Encoding ──────────────────────────────────────────────────────
            for t in range(n_tokens):
                start = t * n_qubits_per_token
                token_wires = list(range(start, start + n_qubits_per_token))
                token_feats = inputs[start: start + n_qubits_per_token]
                angle_encode(token_feats, token_wires)

            # ── U_entangle ────────────────────────────────────────────────────
            u_entangle(n_tokens, n_qubits_per_token, topology=entangle_topology)

            # ── U_attention ───────────────────────────────────────────────────
            qml.StronglyEntanglingLayers(weights_attention, wires=all_wires)

            # ── U_mix ─────────────────────────────────────────────────────────
            u_mix(weights_mix, wires=all_wires)

            # ── Measurement ───────────────────────────────────────────────────
            return [qml.expval(qml.PauliZ(i)) for i in all_wires]

        return _circuit

    def _get_thread_circuit(self):
        """Return (or lazily create) this thread's own QNode instance."""
        if not hasattr(_thread_local, "circuits"):
            _thread_local.circuits = {}
        key = id(self)      # separate circuit per attention module instance
        if key not in _thread_local.circuits:
            _thread_local.circuits[key] = self._build_circuit()
        return _thread_local.circuits[key]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_tokens * n_qubits_per_token) — pre-embedded, angle-range features
        Returns:
            (batch, n_qubits) — Pauli-Z expectation values in [-1, 1]
        """
        batch_size = x.shape[0]

        if not torch.is_grad_enabled() and batch_size > 1:
            # ── Inference path: parallel batch ────────────────────────────────
            # No autograd tape → each thread's QNode is fully independent.
            # Reading shared parameter tensors (weights_*) in parallel is
            # safe; PyTorch Parameters are immutable during forward inference.
            wa = self.weights_attention.detach()
            wm = self.weights_mix.detach()

            def _eval_one(xi):
                circuit = self._get_thread_circuit()
                out = circuit(xi.detach(), wa, wm)
                return torch.stack(list(out))

            n_workers = min(batch_size, self._n_workers)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                batch_results = list(pool.map(_eval_one, x))
        else:
            # ── Training path: serial, single stable QNode ────────────────────
            # autograd (adjoint or parameter-shift) requires a single device
            # instance whose state persists across the forward+backward pair.
            batch_results = []
            for xi in x:
                out = self._circuit(xi, self.weights_attention, self.weights_mix)
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
            f"backend={self._backend!r}, "
            f"diff={self._diff_method!r}, "
            f"n_parameters={self.n_parameters})"
        )
