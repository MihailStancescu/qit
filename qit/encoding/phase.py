import pennylane as qml


def phase_encode(features, wires):
    """
    Phase encoding: H + RZ encodes each feature into the phase of a superposition.
    Contrast with angle encoding (RY): here the information lives in relative phase,
    not in the polar angle of the Bloch sphere.
    """
    for feat, wire in zip(features, wires):
        qml.Hadamard(wires=wire)
        qml.RZ(feat, wires=wire)
