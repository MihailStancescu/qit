import pennylane as qml


def basis_encode(integer, wires):
    """
    Basis encoding: maps an integer to its binary computational basis state.
    Requires integer < 2^len(wires).
    """
    qml.BasisEmbedding(integer, wires=wires)
