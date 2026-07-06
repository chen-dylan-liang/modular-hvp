"""Python and ATen dispatch predicates for graph recording."""

from __future__ import annotations

from typing import Any

import torch


def _is_python_flatten(func: Any) -> bool:
    return func is torch.flatten or (
        getattr(func, "__name__", None) == "flatten"
        and getattr(func, "__module__", "").startswith("torch")
    )


def _is_python_chunk(func: Any) -> bool:
    return getattr(func, "__name__", None) in {"chunk", "split"}


def _is_python_view_or_reshape(func: Any) -> bool:
    return getattr(func, "__name__", None) in {
        "view",
        "reshape",
        "unsqueeze",
        "squeeze",
    }


def _is_python_transpose(func: Any) -> bool:
    return getattr(func, "__name__", None) == "transpose"


def _is_python_contiguous(func: Any) -> bool:
    return getattr(func, "__name__", None) == "contiguous"


def _is_python_matmul(func: Any) -> bool:
    return getattr(func, "__name__", None) == "matmul"


def _is_python_mul(func: Any) -> bool:
    return getattr(func, "__name__", None) == "mul"


def _is_python_div(func: Any) -> bool:
    return getattr(func, "__name__", None) == "div"


def _is_python_cat(func: Any) -> bool:
    return getattr(func, "__name__", None) == "cat"


def _is_python_dtype_cast(func: Any) -> bool:
    return getattr(func, "__name__", None) in {"to", "float"}


def _is_python_unary_elementwise(func: Any) -> bool:
    return getattr(func, "__name__", None) in {
        "relu",
        "sigmoid",
        "tanh",
        "square",
        "pow",
    }


def _is_python_layer_norm(func: Any) -> bool:
    return (
        getattr(func, "__name__", None) == "layer_norm"
        and getattr(func, "__module__", None) == "torch.nn.functional"
    )


def _is_python_rms_norm(func: Any) -> bool:
    return getattr(func, "__name__", None) == "rms_norm"


def _is_python_softmax(func: Any) -> bool:
    return getattr(func, "__name__", None) == "softmax"


def _is_python_dropout(func: Any) -> bool:
    return (
        getattr(func, "__name__", None) == "dropout"
        and getattr(func, "__module__", None) == "torch.nn.functional"
    )


def _is_python_masked_fill(func: Any) -> bool:
    return getattr(func, "__name__", None) == "masked_fill"


def _is_python_scaled_dot_product_attention(func: Any) -> bool:
    return getattr(func, "__name__", None) == "scaled_dot_product_attention"


def _is_python_multi_head_attention_forward(func: Any) -> bool:
    return (
        getattr(func, "__name__", None) == "multi_head_attention_forward"
        and getattr(func, "__module__", None) == "torch.nn.functional"
    )


def _get_aten_overload(name: str, overload: str) -> Any | None:
    packet = getattr(torch.ops.aten, name, None)
    if packet is None or overload not in packet.overloads():
        return None
    return getattr(packet, overload)


_DIRECT_GRAPH_ATEN_OPS = {
    torch.ops.aten.linear.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.square.default,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.masked_fill.Scalar,
    torch.ops.aten.masked_fill.Tensor,
    torch.ops.aten.cat.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.device,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.native_layer_norm.default,
    torch.ops.aten.native_dropout.default,
    torch.ops.aten.scaled_dot_product_attention.default,
}

_DIRECT_GRAPH_ATEN_OPS.update(
    op
    for op in (
        _get_aten_overload("rms_norm", "default"),
        _get_aten_overload("native_rms_norm", "default"),
    )
    if op is not None
)


def _is_direct_graph_aten_op(func: Any) -> bool:
    return func in _DIRECT_GRAPH_ATEN_OPS


def _is_python_add(func: Any) -> bool:
    return getattr(func, "__name__", None) == "add"


def _is_python_sub(func: Any) -> bool:
    return getattr(func, "__name__", None) == "sub"
