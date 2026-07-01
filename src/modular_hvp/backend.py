"""Backend contract for local dual forward and backward execution."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LocalDualActivations:
    """Opaque local dual activation payload saved by the runtime.

    The fake backend keeps only parameter ids so the hook milestone does not
    retain full input/output graphs. A real dual backend can replace this with
    the primal and tangent data needed to dualize the corresponding backward.
    """

    parameter_ids: tuple[int, ...]


class DualBackend(Protocol):
    """Protocol implemented by ModularHVP dual backends."""

    def local_forward(
        self,
        *,
        module: nn.Module,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> tuple[Any, LocalDualActivations]:
        """Run the module forward while producing local dual activations."""

    def dual_backward(
        self,
        *,
        module: nn.Module,
        local_dual_activations: LocalDualActivations,
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
        grad_input: Sequence[torch.Tensor | None],
        grad_output: Sequence[torch.Tensor | None],
    ) -> Mapping[nn.Parameter, torch.Tensor]:
        """Compute the dual part of each active parameter gradient."""


class FakeDualBackend:
    """Allocation-light backend for the hook-plumbing milestone.

    This backend deliberately does not compute curvature. It validates that
    forward-produced local dual activation records are consumed during backward
    and returns zero tensors with the same shape as each tangent.
    """

    def local_forward(
        self,
        *,
        module: nn.Module,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> tuple[Any, LocalDualActivations]:
        primal_output = original_forward(*args, **kwargs)
        activations = LocalDualActivations(
            parameter_ids=tuple(id(param) for param in active_param_tangents)
        )
        return primal_output, activations

    def dual_backward(
        self,
        *,
        module: nn.Module,
        local_dual_activations: LocalDualActivations,
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
        grad_input: Sequence[torch.Tensor | None],
        grad_output: Sequence[torch.Tensor | None],
    ) -> Mapping[nn.Parameter, torch.Tensor]:
        expected_ids = tuple(id(param) for param in active_param_tangents)
        if local_dual_activations.parameter_ids != expected_ids:
            raise RuntimeError(
                "saved local dual activations do not match the active parameters"
            )

        return {
            param: torch.zeros_like(tangent, memory_format=torch.preserve_format)
            for param, tangent in active_param_tangents.items()
        }
