"""Forward-record and saved-tensor definitions for the eager runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class LinearForwardRecord:
    module: nn.Linear
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class EmbeddingForwardRecord:
    module: nn.Embedding
    output_node_id: int
    indices: torch.Tensor
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class FunctionalLinearForwardRecord:
    weight: nn.Parameter | torch.Tensor
    bias: nn.Parameter | torch.Tensor | None
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class Conv2dForwardRecord:
    module: nn.Conv2d
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class BatchNorm2dForwardRecord:
    module: nn.BatchNorm2d
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class LayerNormForwardRecord:
    module: nn.LayerNorm
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    mean: SavedTensorRef
    rstd: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class FunctionalLayerNormForwardRecord:
    weight: nn.Parameter | torch.Tensor | None
    bias: nn.Parameter | torch.Tensor | None
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    mean: SavedTensorRef
    rstd: SavedTensorRef
    normalized_shape: tuple[int, ...]
    eps: float
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class RMSNormForwardRecord:
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    normalized_shape: tuple[int, ...]
    eps: float


@dataclass(frozen=True, slots=True)
class ReLUForwardRecord:
    input_node_id: int
    output_node_id: int
    output_activation: SavedTensorRef


@dataclass(frozen=True, slots=True)
class GELUForwardRecord:
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    approximate: str


@dataclass(frozen=True, slots=True)
class FlattenForwardRecord:
    input_node_id: int
    output_node_id: int
    input_shape: torch.Size
    start_dim: int
    end_dim: int


@dataclass(frozen=True, slots=True)
class AvgPool2dForwardRecord:
    module: nn.AvgPool2d
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef


@dataclass(frozen=True, slots=True)
class AdaptiveAvgPool2dForwardRecord:
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    output_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MaxPool2dForwardRecord:
    module: nn.MaxPool2d
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    indices: SavedTensorRef


@dataclass(frozen=True, slots=True)
class AddForwardRecord:
    output_node_id: int
    left_node_id: int | None
    right_node_id: int | None
    alpha: float
    left_shape: torch.Size
    right_shape: torch.Size
    output_shape: torch.Size


@dataclass(frozen=True, slots=True)
class MulForwardRecord:
    output_node_id: int
    left_node_id: int | None
    right_node_id: int | None
    left_value: torch.Tensor | float | int
    right_value: torch.Tensor | float | int


@dataclass(frozen=True, slots=True)
class CatForwardRecord:
    output_node_id: int
    input_node_ids: tuple[int | None, ...]
    input_shapes: tuple[torch.Size, ...]
    dim: int


@dataclass(frozen=True, slots=True)
class ReshapeForwardRecord:
    input_node_id: int
    output_node_id: int
    input_shape: torch.Size
    output_shape: torch.Size


@dataclass(frozen=True, slots=True)
class SliceForwardRecord:
    input_node_id: int
    output_node_id: int
    input_shape: torch.Size
    dim: int
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SelectForwardRecord:
    input_node_id: int
    output_node_id: int
    input_shape: torch.Size
    dim: int
    index: int


@dataclass(frozen=True, slots=True)
class TransposeForwardRecord:
    input_node_id: int
    output_node_id: int
    dim0: int
    dim1: int


@dataclass(frozen=True, slots=True)
class ContiguousForwardRecord:
    input_node_id: int
    output_node_id: int


@dataclass(frozen=True, slots=True)
class CastForwardRecord:
    input_node_id: int
    output_node_id: int
    input_dtype: torch.dtype
    output_dtype: torch.dtype


@dataclass(frozen=True, slots=True)
class UnaryElementwiseForwardRecord:
    input_node_id: int
    output_node_id: int
    input_activation: SavedTensorRef
    output_activation: SavedTensorRef | None
    kind: str
    scalar: float | None = None


@dataclass(frozen=True, slots=True)
class MatmulForwardRecord:
    left_node_id: int | None
    right_node_id: int | None
    output_node_id: int
    left_activation: SavedTensorRef
    right_activation: SavedTensorRef


@dataclass(frozen=True, slots=True)
class DivForwardRecord:
    output_node_id: int
    left_node_id: int | None
    right_node_id: int | None
    left_value: torch.Tensor | float | int
    right_value: torch.Tensor | float | int


@dataclass(frozen=True, slots=True)
class SoftmaxForwardRecord:
    input_node_id: int
    output_node_id: int
    output_activation: SavedTensorRef
    dim: int


@dataclass(frozen=True, slots=True)
class DropoutForwardRecord:
    input_node_id: int
    output_node_id: int
    multiplier: SavedTensorRef


@dataclass(frozen=True, slots=True)
class MaskedFillForwardRecord:
    input_node_id: int
    output_node_id: int
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class ScaledDotProductAttentionForwardRecord:
    query_node_id: int | None
    key_node_id: int | None
    value_node_id: int | None
    output_node_id: int
    query_activation: SavedTensorRef
    key_activation: SavedTensorRef
    value_activation: SavedTensorRef
    attn_mask: torch.Tensor | None
    is_causal: bool
    scale: float | None


ForwardRecord = (
    LinearForwardRecord
    | EmbeddingForwardRecord
    | FunctionalLinearForwardRecord
    | Conv2dForwardRecord
    | BatchNorm2dForwardRecord
    | LayerNormForwardRecord
    | FunctionalLayerNormForwardRecord
    | RMSNormForwardRecord
    | ReLUForwardRecord
    | GELUForwardRecord
    | FlattenForwardRecord
    | AvgPool2dForwardRecord
    | AdaptiveAvgPool2dForwardRecord
    | MaxPool2dForwardRecord
    | AddForwardRecord
    | MulForwardRecord
    | CatForwardRecord
    | ReshapeForwardRecord
    | SliceForwardRecord
    | SelectForwardRecord
    | TransposeForwardRecord
    | ContiguousForwardRecord
    | CastForwardRecord
    | UnaryElementwiseForwardRecord
    | MatmulForwardRecord
    | DivForwardRecord
    | SoftmaxForwardRecord
    | DropoutForwardRecord
    | MaskedFillForwardRecord
    | ScaledDotProductAttentionForwardRecord
)


@dataclass(slots=True)
class SavedTensorRef:
    """One-shot access to a tensor PyTorch autograd already saved."""

    grad_fn: Any | None
    saved_attrs: tuple[str, ...]
    expected_shape: torch.Size
    fallback: torch.Tensor | None = None

    def resolve(self) -> torch.Tensor:
        if self.fallback is not None:
            return self.fallback
        if self.grad_fn is not None:
            saved = _find_saved_tensor(
                self.grad_fn,
                saved_attrs=self.saved_attrs,
                expected_shape=self.expected_shape,
            )
            if saved is not None:
                return saved
        raise RuntimeError("could not resolve saved activation from autograd graph")

    def resolve_and_release(self) -> torch.Tensor:
        try:
            return self.resolve()
        finally:
            self.release()

    def release(self) -> None:
        self.grad_fn = None
        self.fallback = None


def _find_saved_tensor(
    grad_fn: Any,
    *,
    saved_attrs: tuple[str, ...],
    expected_shape: torch.Size,
) -> torch.Tensor | None:
    seen: set[int] = set()
    stack = [grad_fn]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        for attr in saved_attrs:
            try:
                value = getattr(node, attr)
            except (AttributeError, RuntimeError):
                continue
            if torch.is_tensor(value) and value.shape == expected_shape:
                return value
        for next_node, _ in getattr(node, "next_functions", ()):
            if next_node is not None:
                stack.append(next_node)
    return None



def _record_output_node_id(record: ForwardRecord) -> int:
    return record.output_node_id


def _record_local_output_tangents(
    record: ForwardRecord,
) -> dict[nn.Parameter, torch.Tensor]:
    if isinstance(
        record,
        (
            LinearForwardRecord,
            EmbeddingForwardRecord,
            FunctionalLinearForwardRecord,
            Conv2dForwardRecord,
            BatchNorm2dForwardRecord,
            FunctionalLayerNormForwardRecord,
        ),
    ):
        return record.local_output_tangents
    if isinstance(record, LayerNormForwardRecord):
        return record.local_output_tangents
    return {}



def _record_input_node_ids(record: ForwardRecord) -> tuple[int, ...]:
    if isinstance(record, EmbeddingForwardRecord):
        return ()
    if isinstance(
        record,
        (
            LinearForwardRecord,
            FunctionalLinearForwardRecord,
            Conv2dForwardRecord,
            BatchNorm2dForwardRecord,
            LayerNormForwardRecord,
            FunctionalLayerNormForwardRecord,
            RMSNormForwardRecord,
            ReLUForwardRecord,
            GELUForwardRecord,
            FlattenForwardRecord,
            AvgPool2dForwardRecord,
            AdaptiveAvgPool2dForwardRecord,
            MaxPool2dForwardRecord,
            ReshapeForwardRecord,
            SliceForwardRecord,
            SelectForwardRecord,
            TransposeForwardRecord,
            ContiguousForwardRecord,
            CastForwardRecord,
            UnaryElementwiseForwardRecord,
            SoftmaxForwardRecord,
            DropoutForwardRecord,
            MaskedFillForwardRecord,
        ),
    ):
        return (record.input_node_id,)
    if isinstance(record, AddForwardRecord):
        return tuple(
            node_id
            for node_id in (record.left_node_id, record.right_node_id)
            if node_id is not None
        )
    if isinstance(record, (MatmulForwardRecord, MulForwardRecord, DivForwardRecord)):
        return tuple(
            node_id
            for node_id in (record.left_node_id, record.right_node_id)
            if node_id is not None
        )
    if isinstance(record, CatForwardRecord):
        return tuple(node_id for node_id in record.input_node_ids if node_id is not None)
    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        return tuple(
            node_id
            for node_id in (
                record.query_node_id,
                record.key_node_id,
                record.value_node_id,
            )
            if node_id is not None
        )
    raise TypeError(f"unknown forward record: {type(record).__name__}")


def _release_record_saved_tensors(record: ForwardRecord) -> None:
    refs: tuple[SavedTensorRef | None, ...]
    if isinstance(
        record,
        (
            LinearForwardRecord,
            FunctionalLinearForwardRecord,
            Conv2dForwardRecord,
            BatchNorm2dForwardRecord,
            AvgPool2dForwardRecord,
            AdaptiveAvgPool2dForwardRecord,
        ),
    ):
        refs = (record.input_activation,)
    elif isinstance(record, LayerNormForwardRecord):
        refs = (record.input_activation, record.mean, record.rstd)
    elif isinstance(record, FunctionalLayerNormForwardRecord):
        refs = (record.input_activation, record.mean, record.rstd)
    elif isinstance(record, RMSNormForwardRecord):
        refs = (record.input_activation,)
    elif isinstance(record, ReLUForwardRecord):
        refs = (record.output_activation,)
    elif isinstance(record, GELUForwardRecord):
        refs = (record.input_activation,)
    elif isinstance(record, MaxPool2dForwardRecord):
        refs = (record.input_activation, record.indices)
    elif isinstance(record, UnaryElementwiseForwardRecord):
        refs = (record.input_activation, record.output_activation)
    elif isinstance(record, MatmulForwardRecord):
        refs = (record.left_activation, record.right_activation)
    elif isinstance(record, SoftmaxForwardRecord):
        refs = (record.output_activation,)
    elif isinstance(record, DropoutForwardRecord):
        refs = (record.multiplier,)
    elif isinstance(record, ScaledDotProductAttentionForwardRecord):
        refs = (
            record.query_activation,
            record.key_activation,
            record.value_activation,
        )
    else:
        refs = ()
    for ref in refs:
        if ref is not None:
            ref.release()
