"""
Auto-detect the fastest available PennyLane backend + differentiation method.

Priority:
    1. lightning.gpu + adjoint   — NVIDIA cuQuantum (SM 7.0+, Volta or newer)
    2. lightning.qubit + adjoint — C++ CPU simulation, OpenMP multi-core
    3. default.qubit + parameter-shift — pure Python fallback
"""

from __future__ import annotations

import subprocess

import pennylane as qml

_MIN_GPU_COMPUTE_CAP = 7.0  # Volta; Pascal (e.g. GTX 1080 Ti @ 6.1) is too old.

# Cached once — nvidia-smi can hang on Windows when polled repeatedly.
_GPU_CAPS: list[float] | None = None
_BACKEND: tuple[str, str] | None = None
_DEVICE_INFO: dict | None = None


def _gpu_compute_caps() -> list[float]:
    """Return CUDA compute capabilities for installed NVIDIA GPUs, or []."""
    global _GPU_CAPS
    if _GPU_CAPS is not None:
        return _GPU_CAPS

    caps: list[float] = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if result.returncode == 0:
            caps = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, OSError, ValueError, subprocess.TimeoutExpired):
        caps = []

    _GPU_CAPS = caps
    return caps


def gpu_quantum_supported() -> bool:
    """True when an NVIDIA GPU meets PennyLane lightning.gpu requirements."""
    caps = _gpu_compute_caps()
    return bool(caps) and max(caps) >= _MIN_GPU_COMPUTE_CAP


def auto_backend(n_wires: int = 2) -> tuple[str, str]:
    """
    Returns (backend_name, diff_method) for the fastest usable configuration.

    Args:
        n_wires: number of wires the device needs (used for probe instantiation).
    """
    if gpu_quantum_supported():
        try:
            qml.device("lightning.gpu", wires=n_wires)
            return "lightning.gpu", "adjoint"
        except Exception:
            pass

    try:
        qml.device("lightning.qubit", wires=n_wires)
        return "lightning.qubit", "adjoint"
    except Exception:
        pass
    return "default.qubit", "parameter-shift"


def get_backend() -> tuple[str, str]:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = auto_backend()
    return _BACKEND


def device_info() -> dict:
    """Hardware/backend summary for status APIs and diagnostics."""
    global _DEVICE_INFO
    if _DEVICE_INFO is not None:
        return _DEVICE_INFO

    import torch

    caps = _gpu_compute_caps()
    backend, diff = get_backend()
    max_cap = max(caps) if caps else None
    quantum_ok = bool(caps) and max_cap is not None and max_cap >= _MIN_GPU_COMPUTE_CAP
    _DEVICE_INFO = {
        "backend": backend,
        "diff_method": diff,
        "gpu_compute_caps": caps,
        "gpu_quantum_eligible": quantum_ok,
        "gpu_quantum_min_compute_cap": _MIN_GPU_COMPUTE_CAP,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_reason": (
            None
            if torch.cuda.is_available()
            else (
                "CPU-only PyTorch installed"
                if max_cap is not None
                else "No CUDA GPU detected"
            )
        ),
        "gpu_quantum_reason": (
            None
            if quantum_ok
            else (
                f"GPU compute {max_cap} < {_MIN_GPU_COMPUTE_CAP} (need Volta or newer)"
                if max_cap is not None
                else "No NVIDIA GPU detected"
            )
        ),
    }
    return _DEVICE_INFO
