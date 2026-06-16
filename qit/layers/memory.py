import pennylane as qml


def u_memory(weights, wires):
    """
    Quantum residual memory: one Rot(φ, θ, ω) gate per qubit.
    weights shape: (n_wires, 3)

    Analogous to the residual stream in classical transformers — a parameterized
    per-qubit rotation that lets the circuit learn to preserve or suppress
    individual qubit amplitudes between attention layers.
    """
    for i, wire in enumerate(wires):
        qml.Rot(*weights[i], wires=wire)
