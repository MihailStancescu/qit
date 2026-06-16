import pennylane as qml


def u_mix(weights, wires):
    """
    U_mix: final unitary mixing via BasicEntanglerLayers.
    weights shape: (n_layers, n_wires)
    Projects the post-attention state into measurement-ready form.
    """
    qml.BasicEntanglerLayers(weights, wires=wires)
