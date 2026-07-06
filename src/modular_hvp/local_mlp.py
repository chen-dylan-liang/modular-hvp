"""Compatibility exports for the eager ModularHVP runtime.

The implementation moved to :mod:`modular_hvp.eager` once the runtime became
architecture-agnostic. Keep this module for older internal imports.
"""

from modular_hvp import eager as _eager
from modular_hvp.eager import *  # noqa: F403

for _name in dir(_eager):
    if _name.startswith("_") and not _name.startswith("__"):
        globals()[_name] = getattr(_eager, _name)

del _eager, _name
