import pennylane as qml


def amplitude_encode(features, wires):
    """
    Amplitude encoding: embeds 2^n values as quantum amplitudes.
    Requires len(features) == 2^len(wires). Auto-normalizes the input vector.
    """
    qml.AmplitudeEmbedding(features, wires=wires, normalize=True)
