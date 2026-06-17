"""
Auto-detect the fastest available PennyLane backend + differentiation method.

Priority:
    1. lightning.qubit + adjoint  — C++ simulation, OpenMP multi-core,
                                    single-sweep backward (fastest)
    2. default.qubit  + parameter-shift — pure Python, single-core (fallback)
"""

from __future__ import annotations

import pennylane as qml


def auto_backend(n_wires: int = 2) -> tuple[str, str]:
    """
    Returns (backend_name, diff_method) for the fastest usable configuration.

    Args:
        n_wires: number of wires the device needs (used for probe instantiation).
    """
    try:
        qml.device("lightning.qubit", wires=n_wires)
        return "lightning.qubit", "adjoint"
    except Exception:
        pass
    return "default.qubit", "parameter-shift"


# Module-level cache so detection only runs once.
_BACKEND: tuple[str, str] | None = None


def get_backend() -> tuple[str, str]:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = auto_backend()
    return _BACKEND
