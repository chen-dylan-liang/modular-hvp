"""User-facing context manager."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.backend import DualBackend, FakeDualBackend
from modular_hvp.local_mlp import LocalMLPHVPRuntime
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
        Internal extension point for the hook-plumbing runtime. When omitted,
        the local MLP runtime computes per-parameter HVPs for the currently
        supported Linear/ReLU/MSE scope.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(v, Mapping):
        raise TypeError("v must be a mapping from parameter names or objects to tensors")

    if backend is not None:
        return ModularHVPRuntime(model=model, tangents=v, backend=backend)
    return LocalMLPHVPRuntime(model=model, tangents=v)
