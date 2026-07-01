"""User-facing context manager."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.backend import DualBackend, FakeDualBackend
from modular_hvp.runtime import ModularHVPRuntime


def modular_hvp(
    model: nn.Module,
    v: Mapping[str | nn.Parameter, torch.Tensor],
    *,
    backend: DualBackend | None = None,
) -> ModularHVPRuntime:
    """Create a scoped ModularHVP runtime.

    Parameters
    ----------
    model:
        The PyTorch module whose trainable parameters define the initial
        per-parameter block partition.
    v:
        Mapping from parameter names or parameter objects to tangent tensors.
        The first milestone requires one matching tangent for every trainable
        parameter.
    backend:
        Internal extension point for the dual operator backend. The default is
        a fake backend that exercises hook plumbing and writes zero HVPs.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(v, Mapping):
        raise TypeError("v must be a mapping from parameter names or objects to tensors")

    return ModularHVPRuntime(model=model, tangents=v, backend=backend or FakeDualBackend())
