"""GraphTensor wrapper used to record eager graph edges."""

from __future__ import annotations

from typing import Any

import torch

from modular_hvp.dispatch import (
    _is_direct_graph_aten_op,
    _is_python_add,
    _is_python_cat,
    _is_python_chunk,
    _is_python_contiguous,
    _is_python_div,
    _is_python_dropout,
    _is_python_dtype_cast,
    _is_python_flatten,
    _is_python_layer_norm,
    _is_python_linear,
    _is_python_masked_fill,
    _is_python_matmul,
    _is_python_multi_head_attention_forward,
    _is_python_mul,
    _is_python_rms_norm,
    _is_python_scaled_dot_product_attention,
    _is_python_softmax,
    _is_python_sub,
    _is_python_transpose,
    _is_python_unary_elementwise,
    _is_python_view_or_reshape,
)


class GraphTensor(torch.Tensor):
    """Internal primal tensor wrapper used only to record eager graph edges."""

    __slots__ = ("_primal", "_runtime", "_node_id")

    @staticmethod
    def __new__(
        cls,
        primal_value: torch.Tensor,
        runtime: "EagerHVPRuntime",
        node_id: int,
    ) -> "GraphTensor":
        if primal_value.layout != torch.strided:
            raise NotImplementedError("GraphTensor currently supports strided tensors only")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            primal_value.shape,
            strides=primal_value.stride(),
            storage_offset=primal_value.storage_offset(),
            dtype=primal_value.dtype,
            layout=primal_value.layout,
            device=primal_value.device,
            requires_grad=primal_value.requires_grad,
        )

    def __init__(
        self,
        primal_value: torch.Tensor,
        runtime: "EagerHVPRuntime",
        node_id: int,
    ) -> None:
        self._primal = primal_value
        self._runtime = runtime
        self._node_id = node_id

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        runtime = _graph_runtime_from_tree((args, kwargs))
        if runtime is None:
            return NotImplemented
        if _is_python_add(func):
            alpha = kwargs.get("alpha", 1)
            return runtime._dispatch_graph_linear_binary_function(
                func,
                args,
                kwargs,
                right_coefficient=float(alpha),
            )
        if _is_python_sub(func):
            alpha = kwargs.get("alpha", 1)
            return runtime._dispatch_graph_linear_binary_function(
                func,
                args,
                kwargs,
                right_coefficient=-float(alpha),
            )
        if _is_python_flatten(func):
            return runtime._dispatch_graph_shape_function(func, args, kwargs)
        if _is_python_chunk(func):
            return runtime._dispatch_graph_chunk_function(func, args, kwargs)
        if _is_python_view_or_reshape(func):
            return runtime._dispatch_graph_shape_function(func, args, kwargs)
        if _is_python_transpose(func):
            return runtime._dispatch_graph_transpose_function(func, args, kwargs)
        if _is_python_contiguous(func):
            return runtime._dispatch_graph_contiguous_function(func, args, kwargs)
        if _is_python_matmul(func):
            return runtime._dispatch_graph_matmul_function(func, args, kwargs)
        if _is_python_mul(func):
            return runtime._dispatch_graph_mul_function(func, args, kwargs)
        if _is_python_div(func):
            return runtime._dispatch_graph_div_function(func, args, kwargs)
        if _is_python_cat(func):
            return runtime._dispatch_graph_cat_function(func, args, kwargs)
        if _is_python_linear(func):
            return runtime._dispatch_graph_op(
                torch.ops.aten.linear.default,
                args,
                kwargs,
            )
        if _is_python_dtype_cast(func):
            return runtime._dispatch_graph_cast_function(func, args, kwargs)
        if _is_python_unary_elementwise(func):
            return runtime._dispatch_graph_unary_elementwise_function(func, args, kwargs)
        if _is_python_layer_norm(func):
            return runtime._dispatch_graph_layer_norm_function(func, args, kwargs)
        if _is_python_rms_norm(func):
            return runtime._dispatch_graph_rms_norm_function(func, args, kwargs)
        if _is_python_softmax(func):
            return runtime._dispatch_graph_softmax_function(func, args, kwargs)
        if _is_python_dropout(func):
            return runtime._dispatch_graph_dropout_function(func, args, kwargs)
        if _is_python_masked_fill(func):
            value = args[2] if len(args) > 2 else kwargs.get("value")
            op = (
                torch.ops.aten.masked_fill.Tensor
                if isinstance(value, torch.Tensor)
                else torch.ops.aten.masked_fill.Scalar
            )
            return runtime._dispatch_graph_op(op, args, kwargs)
        if _is_python_scaled_dot_product_attention(func):
            return runtime._dispatch_graph_op(
                torch.ops.aten.scaled_dot_product_attention.default,
                args,
                kwargs,
            )
        if _is_python_multi_head_attention_forward(func):
            return runtime._dispatch_graph_composite_function(func, args, kwargs)
        if _is_direct_graph_aten_op(func):
            return runtime._dispatch_graph_op(func, args, kwargs)
        raise NotImplementedError(f"ModularHVP graph traversal does not support {func}")

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        runtime = _graph_runtime_from_tree((args, kwargs))
        if runtime is None:
            raise RuntimeError("GraphTensor dispatch could not find its runtime")
        return runtime._dispatch_graph_op(func, args, kwargs)

    def __iadd__(self, other: Any) -> "GraphTensor":
        return torch.add(self, other)

    def __isub__(self, other: Any) -> "GraphTensor":
        return torch.sub(self, other)

    def __getitem__(self, index: Any) -> Any:
        return self._runtime._dispatch_graph_getitem(self, index)

    @property
    def primal(self) -> torch.Tensor:
        return self._primal

    @property
    def shape(self) -> torch.Size:
        return self._primal.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._primal.dtype

    @property
    def device(self) -> torch.device:
        return self._primal.device

    @property
    def is_nested(self) -> bool:
        return self._primal.is_nested

    @property
    def ndim(self) -> int:
        return self._primal.ndim

    def size(self, *args: Any) -> Any:
        return self._primal.size(*args)

    def dim(self) -> int:
        return self._primal.dim()

    def to(self, *args: Any, **kwargs: Any) -> "GraphTensor":
        return torch.Tensor.to(self, *args, **kwargs)

    def float(self) -> "GraphTensor":
        return self.to(torch.float32)

    @property
    def node_id(self) -> int:
        return self._node_id

    def __repr__(self) -> str:
        return f"GraphTensor(primal={self._primal!r}, node_id={self._node_id})"


def _graph_runtime_from_tree(value: Any) -> Any | None:
    if isinstance(value, GraphTensor):
        return value._runtime
    if isinstance(value, tuple | list):
        for item in value:
            runtime = _graph_runtime_from_tree(item)
            if runtime is not None:
                return runtime
    if isinstance(value, dict):
        for item in value.values():
            runtime = _graph_runtime_from_tree(item)
            if runtime is not None:
                return runtime
    return None
