import torch
import torch.nn as nn


class ClassicalDecoder(nn.Module):
    """
    Quantum → classical boundary.

    Maps Pauli-Z expectation values produced by the quantum circuit
    to class logits. This is the only point where the quantum state
    collapses to classical information usable by a loss function.

    Input:  (batch, n_qubits) expectations in [-1, 1]
    Output: (batch, n_classes) logits
    """

    def __init__(self, n_qubits: int, n_classes: int):
        super().__init__()
        self.linear = nn.Linear(n_qubits, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
