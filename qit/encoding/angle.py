import pennylane as qml


def angle_encode(features, wires):
    """
    RY rotation encoding: each feature x_i → RY(x_i) on qubit i.
    Features should be in [-π, π]. This is the default QIT-0 encoding strategy.
    """
    for feat, wire in zip(features, wires):
        qml.RY(feat, wires=wire)
