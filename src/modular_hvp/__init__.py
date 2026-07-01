"""Public API for ModularHVP."""

from modular_hvp.api import modular_hvp
from modular_hvp.backend import FakeDualBackend, LocalDualActivations
from modular_hvp.dual import (
    DualTensor,
    is_dual,
    make_dual,
    primal,
    run_with_dual_parameter,
    tangent,
    unpack_dual,
)

__all__ = [
    "DualTensor",
    "FakeDualBackend",
    "LocalDualActivations",
    "is_dual",
    "make_dual",
    "modular_hvp",
    "primal",
    "run_with_dual_parameter",
    "tangent",
    "unpack_dual",
]
