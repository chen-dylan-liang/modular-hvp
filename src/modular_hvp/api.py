"""User-facing context manager."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.backend import DualBackend
from modular_hvp.eager import EagerHVPRuntime
from modular_hvp.runtime import ModularHVPRuntime


def modular_hvp(
    model: nn.Module,
    v: Mapping[str | nn.Parameter, torch.Tensor],
    *,
    blocks: Mapping[Any, Iterable[str | nn.Parameter]] | None = None,
    backend: DualBackend | None = None,
) -> EagerHVPRuntime | ModularHVPRuntime:
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
        the eager runtime computes per-parameter HVPs for the currently
        supported eager tensor graph scope.
    blocks:
        Optional block partition. Each mapping value lists the parameters that
        share one local epsilon. Parameters not listed in this mapping remain
        singleton blocks. When omitted, every parameter is its own block.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module")
    if not isinstance(v, Mapping):
        raise TypeError("v must be a mapping from parameter names or objects to tensors")

    if backend is not None:
        if blocks is not None:
            raise NotImplementedError(
                "custom blocks are currently only supported by the eager runtime"
            )
        return ModularHVPRuntime(model=model, tangents=v, backend=backend)
    return EagerHVPRuntime(model=model, tangents=v, blocks=blocks)
