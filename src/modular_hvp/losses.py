"""Loss records and output-curvature seeds for the eager runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class LossPatch:
    original_mse_loss: Callable[..., torch.Tensor]
    original_cross_entropy: Callable[..., torch.Tensor]


@dataclass(frozen=True, slots=True)
class MSELossRecord:
    input_numel: int
    device: torch.device
    dtype: torch.dtype
    reduction: str


@dataclass(frozen=True, slots=True)
class CrossEntropyLossRecord:
    logits: torch.Tensor
    target: torch.Tensor
    reduction: str
    ignore_index: int


LossRecord = MSELossRecord | CrossEntropyLossRecord


def _mse_hessian_scale(record: MSELossRecord) -> torch.Tensor:
    if record.reduction == "mean":
        return torch.tensor(
            2.0 / record.input_numel,
            device=record.device,
            dtype=record.dtype,
        )
    if record.reduction == "sum":
        return torch.tensor(2.0, device=record.device, dtype=record.dtype)
    raise ValueError(f"unknown mse_loss reduction: {record.reduction!r}")


def _make_loss_output_curvature(
    record: LossRecord,
    grad: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(record, MSELossRecord):
        factor = grad * _mse_hessian_scale(record)

        def mse_output_curvature(value: torch.Tensor) -> torch.Tensor:
            return _scale_tensor_for_loss(value, factor)

        return mse_output_curvature

    if isinstance(record, CrossEntropyLossRecord):
        logits = record.logits
        target = record.target
        probabilities = torch.softmax(logits, dim=-1)
        valid = target != record.ignore_index
        if record.reduction == "mean":
            normalizer = valid.sum().clamp_min(1).to(dtype=logits.dtype)
        elif record.reduction == "sum":
            normalizer = torch.ones((), device=logits.device, dtype=logits.dtype)
        else:
            raise ValueError(f"unknown cross_entropy reduction: {record.reduction!r}")
        grad_factor = grad.to(device=logits.device, dtype=logits.dtype) / normalizer

        def cross_entropy_output_curvature(value: torch.Tensor) -> torch.Tensor:
            value = value.to(dtype=logits.dtype)
            if not value.is_contiguous():
                value = value.contiguous()
            weighted = (probabilities * value).sum(dim=-1, keepdim=True)
            value.sub_(weighted)
            value.mul_(probabilities)
            value.masked_fill_(~valid.unsqueeze(-1), 0)
            value.mul_(grad_factor)
            return value

        return cross_entropy_output_curvature

    raise TypeError(f"unknown loss record: {type(record).__name__}")


def _scale_tensor_for_loss(value: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    if value.is_contiguous():
        value.mul_(factor)
        return value
    return value * factor
