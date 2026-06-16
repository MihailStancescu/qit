import pennylane as qml


def u_entangle(n_tokens: int, n_qubits_per_token: int, topology: str = "ring"):
    """
    Seeds cross-token entanglement before the learned interference phase.

    topology:
      "ring"  — CNOT from each token to the next (circular). Default.
      "star"  — CNOT from token 0 to all others (broadcast hub).
      "none"  — No entanglement; each token register evolves independently.
    """
    if topology == "none":
        return
    for t in range(n_tokens):
        src = t * n_qubits_per_token
        if topology == "ring":
            tgt = ((t + 1) % n_tokens) * n_qubits_per_token
            qml.CNOT(wires=[src, tgt])
        elif topology == "star":
            if t == 0:
                for other in range(1, n_tokens):
                    qml.CNOT(wires=[src, other * n_qubits_per_token])
