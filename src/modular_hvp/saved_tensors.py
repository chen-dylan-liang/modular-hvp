"""Saved-tensor reference builders for eager graph records."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from modular_hvp.kernels import _layer_norm_dims, _layer_norm_stat_shape
from modular_hvp.records import SavedTensorRef, _find_saved_tensor


def _make_linear_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_mat1", "_saved_self"),
        expected_shape=input_value.shape,
        fallback=input_value,
        always_keep_fallback=input_value.dim() != 2,
    )


def _make_functional_linear_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_mat1", "_saved_self"),
        expected_shape=input_value.shape,
        fallback=input_value,
        always_keep_fallback=input_value.dim() != 2,
    )


def _make_exact_saved_tensor_ref(value: torch.Tensor) -> SavedTensorRef:
    return SavedTensorRef(
        grad_fn=None,
        saved_attrs=(),
        expected_shape=value.shape,
        fallback=value.detach(),
    )


def _make_dropout_multiplier_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_other",),
        expected_shape=input_value.shape,
        fallback=None,
    )


def _make_conv_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_input",),
        expected_shape=input_value.shape,
        fallback=input_value,
        always_keep_fallback=True,
    )


def _make_batch_norm_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_input",),
        expected_shape=input_value.shape,
        fallback=input_value,
        always_keep_fallback=True,
    )


def _make_layer_norm_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_input",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_layer_norm_mean_ref(
    module: nn.LayerNorm,
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    dims = _layer_norm_dims(input_value.dim(), module.normalized_shape)
    expected_shape = _layer_norm_stat_shape(input_value.shape, module.normalized_shape)
    with torch.no_grad():
        mean = input_value.detach().mean(dim=dims, keepdim=True)
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result1",),
        expected_shape=expected_shape,
        fallback=mean,
    )


def _make_layer_norm_rstd_ref(
    module: nn.LayerNorm,
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    dims = _layer_norm_dims(input_value.dim(), module.normalized_shape)
    expected_shape = _layer_norm_stat_shape(input_value.shape, module.normalized_shape)
    with torch.no_grad():
        mean = input_value.detach().mean(dim=dims, keepdim=True)
        var = (input_value.detach() - mean).pow(2).mean(dim=dims, keepdim=True)
        rstd = torch.rsqrt(var + module.eps)
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result2",),
        expected_shape=expected_shape,
        fallback=rstd,
    )


def _make_relu_output_activation_ref(output: torch.Tensor) -> SavedTensorRef:
    return SavedTensorRef(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result",),
        expected_shape=output.shape,
        fallback=output.detach(),
    )


def _make_gelu_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_self",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_softmax_output_activation_ref(output: torch.Tensor) -> SavedTensorRef:
    return SavedTensorRef(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result",),
        expected_shape=output.shape,
        fallback=output.detach(),
    )


def _make_pool_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_self",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_adaptive_pool_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_self",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_max_pool_indices_ref(output: torch.Tensor) -> SavedTensorRef:
    if output.grad_fn is None:
        return SavedTensorRef(
            grad_fn=None,
            saved_attrs=("_saved_result1",),
            expected_shape=output.shape,
        )
    indices = _find_saved_tensor(
        output.grad_fn,
        saved_attrs=("_saved_result1",),
        expected_shape=output.shape,
    )
    if indices is None:
        raise RuntimeError("could not resolve MaxPool2d indices from autograd graph")
    return SavedTensorRef(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result1",),
        expected_shape=output.shape,
    )


def _make_saved_tensor_ref(
    *,
    grad_fn: Any | None,
    saved_attrs: tuple[str, ...],
    expected_shape: torch.Size,
    fallback: torch.Tensor | None,
    always_keep_fallback: bool = False,
) -> SavedTensorRef:
    fallback_value = fallback.detach() if always_keep_fallback else None
    ref = SavedTensorRef(
        grad_fn=grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
        fallback=fallback_value,
    )
    if grad_fn is not None and _find_saved_tensor(
        grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
    ) is not None:
        return ref
    return SavedTensorRef(
        grad_fn=grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
        fallback=fallback.detach(),
    )
