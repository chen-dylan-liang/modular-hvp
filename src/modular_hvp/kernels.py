"""Primitive tensor programs and JVP kernels for the eager runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.records import (
    AdaptiveAvgPool2dForwardRecord,
    AvgPool2dForwardRecord,
    FlattenForwardRecord,
    MaxPool2dForwardRecord,
    ReshapeForwardRecord,
)


def _linear_input_backward_program(
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(grad_output, weight)


def _linear_weight_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    if input_activation.dim() > 2:
        input_activation = input_activation.reshape(-1, input_activation.shape[-1])
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    return torch.ops.aten.mm.default(
        torch.ops.aten.t.default(grad_output),
        input_activation,
    )


def _linear_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    reduction_dims = tuple(range(grad_output.dim() - 1))
    if not reduction_dims:
        return grad_output
    return torch.ops.aten.sum.dim_IntList(grad_output, list(reduction_dims), False)


def _embedding_weight_backward_program(
    module: nn.Embedding,
    indices: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    padding_idx = -1 if module.padding_idx is None else module.padding_idx
    return torch.ops.aten.embedding_dense_backward.default(
        grad_output,
        indices,
        module.weight.shape[0],
        padding_idx,
        module.scale_grad_by_freq,
    )


def _conv2d_input_backward_program(
    module: nn.Conv2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    grad_input, _, _ = torch.ops.aten.convolution_backward.default(
        grad_output,
        input_activation,
        module.weight.detach(),
        _conv_bias_sizes(module),
        module.stride,
        module.padding,
        module.dilation,
        False,
        (0, 0),
        module.groups,
        (True, False, False),
    )
    return grad_input


def _conv2d_weight_backward_program(
    module: nn.Conv2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _, grad_weight, _ = torch.ops.aten.convolution_backward.default(
        grad_output,
        input_activation,
        module.weight.detach(),
        _conv_bias_sizes(module),
        module.stride,
        module.padding,
        module.dilation,
        False,
        (0, 0),
        module.groups,
        (False, True, False),
    )
    return grad_weight


def _conv2d_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.sum.dim_IntList(grad_output, [0, 2, 3], False)


def _batch_norm2d_input_backward_program(
    module: nn.BatchNorm2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    grad_input, _, _ = torch.ops.aten.native_batch_norm_backward.default(
        grad_output,
        input_activation,
        module.weight.detach() if module.weight is not None else None,
        _batch_norm_running_mean(module),
        _batch_norm_running_var(module),
        None,
        None,
        False,
        module.eps,
        (True, False, False),
    )
    return grad_input


def _batch_norm2d_weight_backward_program(
    module: nn.BatchNorm2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _, grad_weight, _ = torch.ops.aten.native_batch_norm_backward.default(
        grad_output,
        input_activation,
        module.weight.detach() if module.weight is not None else None,
        _batch_norm_running_mean(module),
        _batch_norm_running_var(module),
        None,
        None,
        False,
        module.eps,
        (False, True, False),
    )
    return grad_weight


def _batch_norm2d_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.sum.dim_IntList(grad_output, [0, 2, 3], False)


def _layer_norm_input_backward_program(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return _layer_norm_input_backward_program_params(
        input_activation,
        mean,
        rstd,
        grad_output,
        tuple(int(item) for item in module.normalized_shape),
        module.weight.detach() if module.weight is not None else None,
        module.bias.detach() if module.bias is not None else None,
    )


def _layer_norm_input_backward_program_params(
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_output: torch.Tensor,
    normalized_shape: tuple[int, ...],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    grad_input, _, _ = torch.ops.aten.native_layer_norm_backward.default(
        grad_output,
        input_activation,
        list(normalized_shape),
        mean,
        rstd,
        weight,
        bias,
        (True, False, False),
    )
    return grad_input


def _layer_norm_weight_backward_program(
    normalized_input: torch.Tensor,
    grad_output: torch.Tensor,
    parameter_shape: torch.Size,
) -> torch.Tensor:
    reduction_dims = _leading_reduction_dims(grad_output.dim(), len(parameter_shape))
    if not reduction_dims:
        return grad_output * normalized_input
    return torch.ops.aten.sum.dim_IntList(
        grad_output * normalized_input,
        list(reduction_dims),
        False,
    )


def _layer_norm_bias_backward_program(
    grad_output: torch.Tensor,
    parameter_shape: torch.Size,
) -> torch.Tensor:
    reduction_dims = _leading_reduction_dims(grad_output.dim(), len(parameter_shape))
    if not reduction_dims:
        return grad_output
    return torch.ops.aten.sum.dim_IntList(grad_output, list(reduction_dims), False)


def _layer_norm_normalized_input(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
) -> torch.Tensor:
    dims = _layer_norm_dims(input_activation.dim(), module.normalized_shape)
    mean = input_activation.mean(dim=dims, keepdim=True)
    centered = input_activation - mean
    var = (centered * centered).mean(dim=dims, keepdim=True)
    return _layer_norm_normalized_input_from_stats(
        input_activation,
        mean,
        torch.rsqrt(var + module.eps),
    )


def _layer_norm_normalized_input_from_stats(
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> torch.Tensor:
    return (input_activation - mean) * rstd


def _layer_norm_input_jvp(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
) -> torch.Tensor:
    return _layer_norm_input_jvp_params(
        input_activation,
        input_tangent,
        tuple(int(item) for item in module.normalized_shape),
        module.eps,
        module.weight.detach() if module.weight is not None else None,
    )


def _layer_norm_input_jvp_params(
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    dims = _layer_norm_dims(input_activation.dim(), normalized_shape)
    mean = input_activation.mean(dim=dims, keepdim=True)
    centered = input_activation - mean
    var = (centered * centered).mean(dim=dims, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    mean_tangent = input_tangent.mean(dim=dims, keepdim=True)
    centered_tangent = input_tangent - mean_tangent
    var_tangent = (2.0 * centered * centered_tangent).mean(dim=dims, keepdim=True)
    rstd_tangent = -0.5 * rstd.pow(3) * var_tangent
    normalized_tangent = centered_tangent * rstd + centered * rstd_tangent
    if weight is not None:
        normalized_tangent = normalized_tangent * weight
    return normalized_tangent


def _layer_norm_input_backward_jvp(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
) -> torch.Tensor | None:
    return _layer_norm_input_backward_jvp_params(
        input_activation,
        mean,
        rstd,
        grad_output,
        grad_output_tangent,
        input_tangent,
        tuple(int(item) for item in module.normalized_shape),
        module.eps,
        module.weight.detach() if module.weight is not None else None,
    )


def _layer_norm_input_backward_jvp_params(
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
    normalized_shape: tuple[int, ...],
    eps: float,
    weight: torch.Tensor | None,
) -> torch.Tensor | None:
    if grad_output_tangent is None and input_tangent is None:
        return None
    del eps
    dims = _layer_norm_dims(input_activation.dim(), normalized_shape)
    normalized_size = 1
    for size in normalized_shape:
        normalized_size *= int(size)
    centered = input_activation - mean
    normalized = centered * rstd
    scaled_grad = grad_output if weight is None else grad_output * weight
    scaled_grad_tangent = (
        None
        if grad_output_tangent is None
        else grad_output_tangent
        if weight is None
        else grad_output_tangent * weight
    )

    if input_tangent is None:
        normalized_tangent = torch.zeros_like(input_activation)
        rstd_tangent = torch.zeros_like(rstd)
    else:
        mean_tangent = input_tangent.mean(dim=dims, keepdim=True)
        centered_tangent = input_tangent - mean_tangent
        var_tangent = (2.0 * centered * centered_tangent).mean(dim=dims, keepdim=True)
        rstd_tangent = -0.5 * rstd.pow(3) * var_tangent
        normalized_tangent = centered_tangent * rstd + centered * rstd_tangent

    sum_grad = scaled_grad.sum(dim=dims, keepdim=True)
    sum_grad_norm = (scaled_grad * normalized).sum(dim=dims, keepdim=True)
    inner = (
        normalized_size * scaled_grad
        - sum_grad
        - normalized * sum_grad_norm
    )
    inner_tangent = None
    if scaled_grad_tangent is not None:
        sum_grad_tangent = scaled_grad_tangent.sum(dim=dims, keepdim=True)
        sum_grad_norm_tangent = (scaled_grad_tangent * normalized).sum(
            dim=dims,
            keepdim=True,
        )
        inner_tangent = (
            normalized_size * scaled_grad_tangent
            - sum_grad_tangent
            - normalized * sum_grad_norm_tangent
        )
    if input_tangent is not None:
        sum_grad_norm_tangent = (scaled_grad * normalized_tangent).sum(
            dim=dims,
            keepdim=True,
        )
        input_inner_tangent = (
            -normalized_tangent * sum_grad_norm
            - normalized * sum_grad_norm_tangent
        )
        inner_tangent = (
            input_inner_tangent
            if inner_tangent is None
            else inner_tangent + input_inner_tangent
        )

    result = None
    if input_tangent is not None:
        result = (rstd_tangent / normalized_size) * inner
    if inner_tangent is not None:
        term = (rstd / normalized_size) * inner_tangent
        result = term if result is None else result + term
    return result


def _rms_norm_dims(
    input_ndim: int,
    normalized_shape: tuple[int, ...],
) -> tuple[int, ...]:
    normalized_ndim = len(normalized_shape)
    return tuple(range(input_ndim - normalized_ndim, input_ndim))


def _rms_norm_stats(
    input_activation: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    dims = _rms_norm_dims(input_activation.dim(), normalized_shape)
    square_mean = (input_activation * input_activation).mean(dim=dims, keepdim=True)
    rstd = torch.rsqrt(square_mean + eps)
    normalized_size = 1
    for size in normalized_shape:
        normalized_size *= int(size)
    return rstd, dims, normalized_size


def _rms_norm_jvp(
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    rstd, dims, _ = _rms_norm_stats(input_activation, normalized_shape, eps)
    square_mean_tangent = (2.0 * input_activation * input_tangent).mean(
        dim=dims,
        keepdim=True,
    )
    rstd_tangent = -0.5 * rstd.pow(3) * square_mean_tangent
    return input_tangent * rstd + input_activation * rstd_tangent


def _rms_norm_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    rstd, dims, normalized_size = _rms_norm_stats(input_activation, normalized_shape, eps)
    inner = (grad_output * input_activation).sum(dim=dims, keepdim=True)
    return rstd * grad_output - input_activation * rstd.pow(3) * inner / normalized_size


def _rms_norm_backward_jvp(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor | None:
    if grad_output_tangent is None and input_tangent is None:
        return None
    rstd, dims, normalized_size = _rms_norm_stats(input_activation, normalized_shape, eps)
    inner = (grad_output * input_activation).sum(dim=dims, keepdim=True)
    result = None
    if grad_output_tangent is not None:
        inner_tangent = (grad_output_tangent * input_activation).sum(
            dim=dims,
            keepdim=True,
        )
        result = (
            rstd * grad_output_tangent
            - input_activation * rstd.pow(3) * inner_tangent / normalized_size
        )
    if input_tangent is not None:
        square_mean_tangent = (2.0 * input_activation * input_tangent).mean(
            dim=dims,
            keepdim=True,
        )
        rstd_tangent = -0.5 * rstd.pow(3) * square_mean_tangent
        inner_tangent = (grad_output * input_tangent).sum(dim=dims, keepdim=True)
        term = (
            rstd_tangent * grad_output
            - input_tangent * rstd.pow(3) * inner / normalized_size
            - input_activation * 3.0 * rstd.pow(2) * rstd_tangent * inner / normalized_size
            - input_activation * rstd.pow(3) * inner_tangent / normalized_size
        )
        result = term if result is None else result + term
    return result


def _relu_backward_program(
    input_value: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.threshold_backward.default(grad_output, input_value, 0)


def _unary_elementwise_kind(
    func: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[str, float | None]:
    if func is torch.ops.aten.relu.default or getattr(func, "__name__", None) == "relu":
        return "relu", None
    if func is torch.ops.aten.sigmoid.default or getattr(func, "__name__", None) == "sigmoid":
        return "sigmoid", None
    if func is torch.ops.aten.tanh.default or getattr(func, "__name__", None) == "tanh":
        return "tanh", None
    if func is torch.ops.aten.square.default or getattr(func, "__name__", None) == "square":
        return "pow", 2.0
    if func is torch.ops.aten.pow.Tensor_Scalar or getattr(func, "__name__", None) == "pow":
        exponent = args[1] if len(args) > 1 else kwargs.get("exponent")
        if not isinstance(exponent, (int, float)):
            raise NotImplementedError(
                "ModularHVP graph traversal currently supports scalar pow exponents"
            )
        return "pow", float(exponent)
    raise NotImplementedError(f"ModularHVP graph traversal does not support {func}")


def _unary_elementwise_jvp(
    kind: str,
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    return _unary_elementwise_derivative(
        kind,
        input_activation,
        output_activation,
        scalar,
    ) * input_tangent


def _unary_elementwise_backward_program(
    kind: str,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    return _unary_elementwise_derivative(
        kind,
        input_activation,
        output_activation,
        scalar,
    ) * grad_output


def _unary_elementwise_backward_jvp(
    kind: str,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor | None:
    derivative = _unary_elementwise_derivative(
        kind,
        input_activation,
        output_activation,
        scalar,
    )
    result = None
    if grad_output_tangent is not None:
        result = derivative * grad_output_tangent
    if input_tangent is not None:
        second = _unary_elementwise_second_derivative(
            kind,
            input_activation,
            output_activation,
            scalar,
        )
        term = grad_output * second * input_tangent
        result = term if result is None else result + term
    return result


def _unary_elementwise_derivative(
    kind: str,
    input_activation: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    if kind == "relu":
        value = output_activation if output_activation is not None else input_activation
        return (value > 0).to(dtype=input_activation.dtype)
    if kind == "sigmoid":
        output = (
            output_activation
            if output_activation is not None
            else torch.sigmoid(input_activation)
        )
        return output * (1.0 - output)
    if kind == "tanh":
        output = (
            output_activation if output_activation is not None else torch.tanh(input_activation)
        )
        return 1.0 - output * output
    if kind == "pow":
        exponent = 1.0 if scalar is None else scalar
        return exponent * input_activation.pow(exponent - 1.0)
    raise ValueError(f"unknown unary elementwise kind: {kind!r}")


def _unary_elementwise_second_derivative(
    kind: str,
    input_activation: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    if kind == "relu":
        return torch.zeros_like(input_activation)
    if kind == "sigmoid":
        output = (
            output_activation
            if output_activation is not None
            else torch.sigmoid(input_activation)
        )
        derivative = output * (1.0 - output)
        return derivative * (1.0 - 2.0 * output)
    if kind == "tanh":
        output = (
            output_activation if output_activation is not None else torch.tanh(input_activation)
        )
        derivative = 1.0 - output * output
        return -2.0 * output * derivative
    if kind == "pow":
        exponent = 1.0 if scalar is None else scalar
        if exponent == 0.0:
            return torch.zeros_like(input_activation)
        return exponent * (exponent - 1.0) * input_activation.pow(exponent - 2.0)
    raise ValueError(f"unknown unary elementwise kind: {kind!r}")


def _gelu_jvp(
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    derivative = _gelu_derivative(input_activation, approximate)
    return derivative * input_tangent


def _gelu_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    derivative = _gelu_derivative(input_activation, approximate)
    return grad_output * derivative


def _gelu_derivative(
    input_activation: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    if approximate == "tanh":
        coeff = (2.0 / torch.pi) ** 0.5
        inner = coeff * (input_activation + 0.044715 * input_activation.pow(3))
        tanh_inner = torch.tanh(inner)
        return 0.5 * (1 + tanh_inner) + 0.5 * input_activation * (
            1 - tanh_inner.pow(2)
        ) * coeff * (1 + 3 * 0.044715 * input_activation.pow(2))
    normal_cdf = 0.5 * (1 + torch.erf(input_activation / (2.0**0.5)))
    normal_pdf = torch.exp(-0.5 * input_activation.pow(2)) / (2.0 * torch.pi) ** 0.5
    return normal_cdf + input_activation * normal_pdf


def _gelu_second_derivative(
    input_activation: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    if approximate == "tanh":
        raise NotImplementedError(
            "GELU tanh second-derivative runtime rule is not implemented yet"
        )
    normal_pdf = torch.exp(-0.5 * input_activation.pow(2)) / (2.0 * torch.pi) ** 0.5
    return (2.0 - input_activation.pow(2)) * normal_pdf


def _make_linear_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    weight: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    weight = weight.detach()
    weight_t = torch.ops.aten.t.default(weight)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.matmul(input_tangent, weight_t)
        output_grad_tangent = output_curvature(output_tangent)
        return _linear_input_backward_program(weight, output_grad_tangent)

    return input_curvature


def _make_conv2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.Conv2d,
    input_activation: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    weight = module.weight.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten.convolution.default(
            input_tangent,
            weight,
            None,
            module.stride,
            module.padding,
            module.dilation,
            False,
            (0, 0),
            module.groups,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return _conv2d_input_backward_program(
            module,
            input_activation,
            output_grad_tangent,
        )

    return input_curvature


def _make_batch_norm2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.BatchNorm2d,
    input_activation: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _batch_norm2d_input_jvp(module, input_tangent)
        output_grad_tangent = output_curvature(output_tangent)
        return _batch_norm2d_input_backward_program(
            module,
            input_activation,
            output_grad_tangent,
        )

    return input_curvature


def _make_layer_norm_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    mean = mean.detach()
    rstd = rstd.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _layer_norm_input_jvp(
            module,
            input_activation,
            input_tangent,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return _layer_norm_input_backward_program(
            module,
            input_activation,
            mean,
            rstd,
            output_grad_tangent,
        )

    return input_curvature


def _make_relu_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    relu_output: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    relu_output = relu_output.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _relu_backward_program(relu_output, input_tangent)
        output_grad_tangent = output_curvature(output_tangent)
        return _relu_backward_program(relu_output, output_grad_tangent)

    return input_curvature


def _make_gelu_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    input_activation: torch.Tensor,
    approximate: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _gelu_jvp(input_activation, input_tangent, approximate)
        output_grad_tangent = output_curvature(output_tangent)
        return _gelu_backward_program(
            input_activation,
            output_grad_tangent,
            approximate,
        )

    return input_curvature


def _make_flatten_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    record: FlattenForwardRecord,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.flatten(
            input_tangent,
            start_dim=record.start_dim,
            end_dim=record.end_dim,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return output_grad_tangent.reshape(record.input_shape)

    return input_curvature


def _make_reshape_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    record: ReshapeForwardRecord,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = input_tangent.reshape(record.output_shape)
        output_grad_tangent = output_curvature(output_tangent)
        return output_grad_tangent.reshape(record.input_shape)

    return input_curvature


def _make_avg_pool2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.AvgPool2d,
    input_activation: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    kernel_size = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    padding = _pair(module.padding)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten.avg_pool2d.default(
            input_tangent,
            kernel_size,
            stride,
            padding,
            module.ceil_mode,
            module.count_include_pad,
            module.divisor_override,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return torch.ops.aten.avg_pool2d_backward.default(
            output_grad_tangent,
            input_activation,
            kernel_size,
            stride,
            padding,
            module.ceil_mode,
            module.count_include_pad,
            module.divisor_override,
        )

    return input_curvature


def _make_adaptive_avg_pool2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    input_activation: torch.Tensor,
    output_size: tuple[int, int],
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten._adaptive_avg_pool2d.default(
            input_tangent,
            output_size,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return torch.ops.aten._adaptive_avg_pool2d_backward.default(
            output_grad_tangent,
            input_activation,
        )

    return input_curvature


def _make_max_pool2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.MaxPool2d,
    input_activation: torch.Tensor,
    indices: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    indices = indices.detach()
    kernel_size = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    padding = _pair(module.padding)
    dilation = _pair(module.dilation)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _max_pool2d_jvp(input_tangent, indices)
        output_grad_tangent = output_curvature(output_tangent)
        return torch.ops.aten.max_pool2d_with_indices_backward.default(
            output_grad_tangent,
            input_activation,
            kernel_size,
            stride,
            padding,
            dilation,
            module.ceil_mode,
            indices,
        )

    return input_curvature


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dimension out of range: {dim}")
    return dim


def _normalize_slice_start(start: Any, dim_size: int) -> int:
    if start is None:
        return 0
    value = int(start)
    if value < 0:
        value += dim_size
    return max(0, min(value, dim_size))


def _normalize_slice_end(end: Any, dim_size: int) -> int:
    if end is None:
        return dim_size
    value = int(end)
    if value < 0:
        value += dim_size
    return max(0, min(value, dim_size))


def _normalize_select_index(index: int, dim_size: int) -> int:
    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise IndexError(f"select index out of range: {index}")
    return index


def _canonicalize_index(index: tuple[Any, ...], ndim: int) -> tuple[Any, ...]:
    ellipsis_count = sum(1 for item in index if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis")
    consumed_dims = sum(1 for item in index if item is not Ellipsis and item is not None)
    if consumed_dims > ndim:
        raise IndexError("too many indices for tensor")
    result: list[Any] = []
    for item in index:
        if item is Ellipsis:
            result.extend([slice(None)] * (ndim - consumed_dims))
        else:
            result.append(item)
    if ellipsis_count == 0:
        result.extend([slice(None)] * (ndim - consumed_dims))
    return tuple(result)


def _value_shape(value: torch.Tensor | float | int) -> torch.Size:
    if isinstance(value, torch.Tensor):
        return value.shape
    return torch.Size(())


def _broadcast_like(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if value.shape == target_shape:
        return value
    return value.expand(target_shape)


def _unbroadcast_like(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if value.shape == target_shape:
        return value
    if len(target_shape) == 0:
        return value.sum()
    result = value
    while result.dim() > len(target_shape):
        result = result.sum(dim=0)
    for dim, target_size in enumerate(target_shape):
        if target_size == 1 and result.shape[dim] != 1:
            result = result.sum(dim=dim, keepdim=True)
    return result.reshape(target_shape)


def _split_output_sizes(
    dim_size: int,
    split_spec: Any,
    *,
    output_count: int,
) -> list[int]:
    if isinstance(split_spec, (tuple, list)):
        return [int(size) for size in split_spec]
    split_size = int(split_spec)
    if split_size <= 0:
        raise ValueError("split size must be positive")
    sizes: list[int] = []
    remaining = dim_size
    while remaining > 0 and len(sizes) < output_count:
        size = min(split_size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def _leading_reduction_dims(ndim: int, parameter_ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim - parameter_ndim))


def _layer_norm_dims(
    input_ndim: int,
    normalized_shape: tuple[int, ...] | list[int] | torch.Size,
) -> tuple[int, ...]:
    normalized_ndim = len(tuple(normalized_shape))
    return tuple(range(input_ndim - normalized_ndim, input_ndim))


def _layer_norm_stat_shape(
    input_shape: torch.Size,
    normalized_shape: tuple[int, ...] | list[int] | torch.Size,
) -> torch.Size:
    normalized_ndim = len(tuple(normalized_shape))
    leading = tuple(input_shape[: len(input_shape) - normalized_ndim])
    return torch.Size((*leading, *([1] * normalized_ndim)))


def _detach_value(value: torch.Tensor | float | int) -> torch.Tensor | float | int:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _transpose_last_two(value: torch.Tensor) -> torch.Tensor:
    return value.transpose(-2, -1)


def _matmul_left_backward_program(
    grad_output: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(grad_output, _transpose_last_two(right))


def _matmul_right_backward_program(
    left: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(_transpose_last_two(left), grad_output)


def _softmax_jvp(
    softmax_output: torch.Tensor,
    input_tangent: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    weighted = (softmax_output * input_tangent).sum(dim=dim, keepdim=True)
    return softmax_output * (input_tangent - weighted)


def _softmax_backward_program(
    softmax_output: torch.Tensor,
    grad_output: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    centered = grad_output - (grad_output * softmax_output).sum(dim=dim, keepdim=True)
    return softmax_output * centered


def _softmax_backward_jvp(
    softmax_output: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    softmax_output_tangent: torch.Tensor | None,
    dim: int,
) -> torch.Tensor | None:
    if grad_output_tangent is None and softmax_output_tangent is None:
        return None
    centered = grad_output - (grad_output * softmax_output).sum(dim=dim, keepdim=True)
    result = None
    if grad_output_tangent is not None:
        centered_tangent = grad_output_tangent - (
            grad_output_tangent * softmax_output
        ).sum(dim=dim, keepdim=True)
        result = softmax_output * centered_tangent
    if softmax_output_tangent is not None:
        centered_tangent = -(
            grad_output * softmax_output_tangent
        ).sum(dim=dim, keepdim=True)
        term = softmax_output_tangent * centered + softmax_output * centered_tangent
        result = term if result is None else result + term
    return result


def _attention_scale(query: torch.Tensor, scale: float | None) -> float:
    return (query.size(-1) ** -0.5) if scale is None else float(scale)


def _attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) * _attention_scale(query, scale)
    if is_causal:
        query_len = query.size(-2)
        key_len = key.size(-2)
        causal_mask = torch.ones(
            query_len,
            key_len,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    return scores


def _attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    return torch.softmax(
        _attention_scores(query, key, attn_mask, is_causal, scale),
        dim=-1,
    )


def _attention_score_tangent(
    query: torch.Tensor,
    query_tangent: torch.Tensor | None,
    key: torch.Tensor,
    key_tangent: torch.Tensor | None,
    scale: float | None,
) -> torch.Tensor | None:
    scale_value = _attention_scale(query, scale)
    return _sum_optional_tensors(
        torch.matmul(query_tangent, key.transpose(-2, -1)) * scale_value
        if query_tangent is not None
        else None,
        torch.matmul(query, key_tangent.transpose(-2, -1)) * scale_value
        if key_tangent is not None
        else None,
    )


def _attention_value_and_score_grads(
    attention: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_grad = torch.matmul(attention.transpose(-2, -1), grad_output)
    attention_grad = torch.matmul(grad_output, value.transpose(-2, -1))
    score_grad = _softmax_backward_program(attention, attention_grad, -1)
    return value_grad, score_grad


def _attention_backward_jvp(
    query: torch.Tensor,
    query_tangent: torch.Tensor | None,
    key: torch.Tensor,
    key_tangent: torch.Tensor | None,
    value: torch.Tensor,
    value_tangent: torch.Tensor | None,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    attention = _attention_weights(query, key, attn_mask, is_causal, scale)
    score_tangent = _attention_score_tangent(
        query,
        query_tangent,
        key,
        key_tangent,
        scale,
    )
    attention_tangent = (
        _softmax_jvp(attention, score_tangent, -1)
        if score_tangent is not None
        else None
    )
    _, score_grad = _attention_value_and_score_grads(
        attention,
        value,
        grad_output,
    )
    attention_grad = torch.matmul(grad_output, value.transpose(-2, -1))

    value_grad_tangent = None
    if attention_tangent is not None:
        value_grad_tangent = torch.matmul(
            attention_tangent.transpose(-2, -1),
            grad_output,
        )
    if grad_output_tangent is not None:
        term = torch.matmul(attention.transpose(-2, -1), grad_output_tangent)
        value_grad_tangent = (
            term if value_grad_tangent is None else value_grad_tangent + term
        )

    attention_grad_tangent = None
    if grad_output_tangent is not None:
        attention_grad_tangent = torch.matmul(
            grad_output_tangent,
            value.transpose(-2, -1),
        )
    if value_tangent is not None:
        term = torch.matmul(grad_output, value_tangent.transpose(-2, -1))
        attention_grad_tangent = (
            term if attention_grad_tangent is None else attention_grad_tangent + term
        )

    score_grad_tangent = _softmax_backward_jvp(
        attention,
        attention_grad,
        attention_grad_tangent,
        attention_tangent,
        -1,
    )
    scale_value = _attention_scale(query, scale)
    query_grad_tangent = None
    if score_grad_tangent is not None:
        query_grad_tangent = torch.matmul(score_grad_tangent, key) * scale_value
    if key_tangent is not None:
        term = torch.matmul(score_grad, key_tangent) * scale_value
        query_grad_tangent = (
            term if query_grad_tangent is None else query_grad_tangent + term
        )

    key_grad_tangent = None
    if score_grad_tangent is not None:
        key_grad_tangent = (
            torch.matmul(score_grad_tangent.transpose(-2, -1), query) * scale_value
        )
    if query_tangent is not None:
        term = torch.matmul(score_grad.transpose(-2, -1), query_tangent) * scale_value
        key_grad_tangent = (
            term if key_grad_tangent is None else key_grad_tangent + term
        )

    return query_grad_tangent, key_grad_tangent, value_grad_tangent


def _sum_optional_tensors(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected pair, got {value!r}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _reshape_channel_tangent(value: torch.Tensor, ndim: int) -> torch.Tensor:
    if value.ndim != 1:
        return value
    return value.reshape((1, value.shape[0], *([1] * (ndim - 2))))


def _conv_bias_sizes(module: nn.Conv2d) -> list[int] | None:
    if module.bias is None:
        return None
    return list(module.bias.shape)


def _batch_norm_running_mean(module: nn.BatchNorm2d) -> torch.Tensor:
    if module.running_mean is None:
        raise NotImplementedError(
            "BatchNorm2d runtime support requires tracked running statistics"
        )
    return module.running_mean.detach()


def _batch_norm_running_var(module: nn.BatchNorm2d) -> torch.Tensor:
    if module.running_var is None:
        raise NotImplementedError(
            "BatchNorm2d runtime support requires tracked running statistics"
        )
    return module.running_var.detach()


def _batch_norm2d_input_jvp(
    module: nn.BatchNorm2d,
    input_tangent: torch.Tensor,
) -> torch.Tensor:
    scale = _batch_norm2d_input_scale(module)
    return input_tangent * _reshape_channel_tangent(scale, input_tangent.dim())


def _batch_norm2d_input_scale(module: nn.BatchNorm2d) -> torch.Tensor:
    scale = torch.rsqrt(_batch_norm_running_var(module) + module.eps)
    if module.weight is not None:
        scale = scale * module.weight.detach()
    return scale


def _max_pool2d_jvp(
    input_tangent: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if input_tangent.dim() == 3:
        channels, _, _ = input_tangent.shape
        flat_tangent = input_tangent.reshape(channels, -1)
        flat_indices = indices.reshape(channels, -1)
        return flat_tangent.gather(1, flat_indices).reshape(indices.shape)
    if input_tangent.dim() == 4:
        batch, channels, _, _ = input_tangent.shape
        flat_tangent = input_tangent.reshape(batch, channels, -1)
        flat_indices = indices.reshape(batch, channels, -1)
        return flat_tangent.gather(2, flat_indices).reshape(indices.shape)
    raise NotImplementedError("MaxPool2d tangent expects a 3D or 4D input")
