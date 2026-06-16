from .angle import angle_encode
from .amplitude import amplitude_encode
from .basis import basis_encode
from .phase import phase_encode

ENCODERS = {
    "angle": angle_encode,
    "amplitude": amplitude_encode,
    "basis": basis_encode,
    "phase": phase_encode,
}
