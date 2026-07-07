"""Graph tangent packet propagation for the eager runtime."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.kernels import (
    _attention_backward_jvp,
    _attention_scale,
    _attention_score_tangent,
    _attention_value_and_score_grads,
    _attention_weights,
    _batch_norm2d_input_backward_program,
    _batch_norm2d_input_scale,
    _batch_norm2d_input_jvp,
    _broadcast_like,
    _conv2d_input_backward_program,
    _detach_value,
    _gelu_backward_program,
    _gelu_derivative,
    _gelu_jvp,
    _gelu_second_derivative,
    _layer_norm_input_backward_jvp,
    _layer_norm_input_backward_jvp_params,
    _layer_norm_input_backward_program,
    _layer_norm_input_backward_program_params,
    _layer_norm_input_jvp,
    _layer_norm_input_jvp_params,
    _linear_input_backward_program,
    _matmul_left_backward_program,
    _matmul_right_backward_program,
    _max_pool2d_jvp,
    _pair,
    _relu_backward_program,
    _reshape_channel_tangent,
    _rms_norm_backward_jvp,
    _rms_norm_backward_program,
    _rms_norm_jvp,
    _softmax_backward_jvp,
    _softmax_backward_program,
    _softmax_jvp,
    _unbroadcast_like,
    _unary_elementwise_backward_jvp,
    _unary_elementwise_backward_program,
    _unary_elementwise_jvp,
    _value_shape,
)
from modular_hvp.records import (
    AddForwardRecord,
    AdaptiveAvgPool2dForwardRecord,
    AvgPool2dForwardRecord,
    BatchNorm2dForwardRecord,
    CastForwardRecord,
    CatForwardRecord,
    ContiguousForwardRecord,
    Conv2dForwardRecord,
    DivForwardRecord,
    DropoutForwardRecord,
    EmbeddingForwardRecord,
    FlattenForwardRecord,
    ForwardRecord,
    FunctionalLayerNormForwardRecord,
    FunctionalLinearForwardRecord,
    GELUForwardRecord,
    LayerNormForwardRecord,
    LinearForwardRecord,
    MaskedFillForwardRecord,
    MatmulForwardRecord,
    MaxPool2dForwardRecord,
    MulForwardRecord,
    RMSNormForwardRecord,
    ReLUForwardRecord,
    ReshapeForwardRecord,
    ScaledDotProductAttentionForwardRecord,
    SelectForwardRecord,
    SliceForwardRecord,
    SoftmaxForwardRecord,
    TransposeForwardRecord,
    UnaryElementwiseForwardRecord,
    SavedTensorRef,
    _record_input_node_ids,
)


def _forward_record_tangent_packet(
    record: ForwardRecord,
    tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
) -> dict[nn.Parameter, torch.Tensor] | None:
    if isinstance(record, LinearForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        weight_t = torch.ops.aten.t.default(record.module.weight.detach())
        return {
            parameter: torch.matmul(input_tangent, weight_t)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, EmbeddingForwardRecord):
        return None

    if isinstance(record, FunctionalLinearForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        weight_t = torch.ops.aten.t.default(record.weight.detach())
        return {
            parameter: torch.matmul(input_tangent, weight_t)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, Conv2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        weight = record.module.weight.detach()
        return {
            parameter: torch.ops.aten.convolution.default(
                input_tangent,
                weight,
                None,
                record.module.stride,
                record.module.padding,
                record.module.dilation,
                False,
                (0, 0),
                record.module.groups,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, BatchNorm2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        scale = _batch_norm2d_input_scale(record.module)
        return {
            parameter: input_tangent * _reshape_channel_tangent(
                scale,
                input_tangent.dim(),
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, LayerNormForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        return {
            parameter: _layer_norm_input_jvp(
                record.module,
                input_activation,
                input_tangent,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, FunctionalLayerNormForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        weight = record.weight.detach() if isinstance(record.weight, torch.Tensor) else None
        return {
            parameter: _layer_norm_input_jvp_params(
                input_activation,
                input_tangent,
                record.normalized_shape,
                record.eps,
                weight,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, RMSNormForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        return {
            parameter: _rms_norm_jvp(
                input_activation,
                input_tangent,
                record.normalized_shape,
                record.eps,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, ReLUForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        relu_output = record.output_activation.resolve().detach()
        return {
            parameter: _relu_backward_program(relu_output, input_tangent)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, GELUForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        return {
            parameter: _gelu_jvp(
                input_activation,
                input_tangent,
                record.approximate,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, FlattenForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: torch.flatten(
                input_tangent,
                start_dim=record.start_dim,
                end_dim=record.end_dim,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, AvgPool2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        kernel_size = _pair(record.module.kernel_size)
        stride = _pair(
            record.module.stride
            if record.module.stride is not None
            else record.module.kernel_size
        )
        padding = _pair(record.module.padding)
        return {
            parameter: torch.ops.aten.avg_pool2d.default(
                input_tangent,
                kernel_size,
                stride,
                padding,
                record.module.ceil_mode,
                record.module.count_include_pad,
                record.module.divisor_override,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, AdaptiveAvgPool2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: torch.ops.aten._adaptive_avg_pool2d.default(
                input_tangent,
                record.output_size,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, MaxPool2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        indices = record.indices.resolve().detach()
        return {
            parameter: _max_pool2d_jvp(input_tangent, indices)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, AddForwardRecord):
        result: dict[nn.Parameter, torch.Tensor] = {}
        if record.left_node_id is not None:
            left_packet = tangents_by_node.get(record.left_node_id)
            if left_packet:
                for parameter, left_tangent in left_packet.items():
                    _accumulate_parameter_tensor(
                        result,
                        parameter,
                        _broadcast_like(left_tangent, record.output_shape),
                    )
        if record.right_node_id is not None:
            right_packet = tangents_by_node.get(record.right_node_id)
            if right_packet:
                alpha = record.alpha
                for parameter, right_tangent in right_packet.items():
                    value = _broadcast_like(right_tangent, record.output_shape)
                    value = value if alpha == 1.0 else alpha * value
                    _accumulate_parameter_tensor(result, parameter, value)
        return result or None

    if isinstance(record, MulForwardRecord):
        left_packet = (
            tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        if not left_packet and not right_packet:
            return None
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        result: dict[nn.Parameter, torch.Tensor] = {}
        if left_packet:
            for parameter, left_tangent in left_packet.items():
                _accumulate_parameter_tensor(result, parameter, left_tangent * right)
        if right_packet:
            for parameter, right_tangent in right_packet.items():
                _accumulate_parameter_tensor(result, parameter, left * right_tangent)
        return result or None

    if isinstance(record, CatForwardRecord):
        packets = [
            tangents_by_node.get(node_id) if node_id is not None else None
            for node_id in record.input_node_ids
        ]
        parameters = set().union(*(set(packet) for packet in packets if packet))
        if not parameters:
            return None
        result: dict[nn.Parameter, torch.Tensor] = {}
        for parameter in parameters:
            reference = next(packet[parameter] for packet in packets if packet and parameter in packet)
            pieces = [
                packet[parameter]
                if packet is not None and parameter in packet
                else torch.zeros(
                    shape,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                for packet, shape in zip(packets, record.input_shapes, strict=True)
            ]
            result[parameter] = torch.cat(pieces, dim=record.dim)
        return result

    if isinstance(record, ReshapeForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.reshape(record.output_shape)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, SliceForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        length = record.end - record.start
        return {
            parameter: input_tangent.narrow(record.dim, record.start, length)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, SelectForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.select(record.dim, record.index)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, TransposeForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.transpose(record.dim0, record.dim1)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, ContiguousForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.contiguous()
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, CastForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.to(dtype=record.output_dtype)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, UnaryElementwiseForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        output_activation = (
            None
            if record.output_activation is None
            else record.output_activation.resolve().detach()
        )
        return {
            parameter: _unary_elementwise_jvp(
                record.kind,
                input_activation,
                input_tangent,
                output_activation,
                record.scalar,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, MatmulForwardRecord):
        left_packet = (
            tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        if not left_packet and not right_packet:
            return None
        left = record.left_activation.resolve().detach()
        right = record.right_activation.resolve().detach()
        result: dict[nn.Parameter, torch.Tensor] = {}
        if left_packet:
            for parameter, left_tangent in left_packet.items():
                _accumulate_parameter_tensor(
                    result,
                    parameter,
                    torch.matmul(left_tangent, right),
                )
        if right_packet:
            for parameter, right_tangent in right_packet.items():
                _accumulate_parameter_tensor(
                    result,
                    parameter,
                    torch.matmul(left, right_tangent),
                )
        return result or None

    if isinstance(record, DivForwardRecord):
        left_packet = (
            tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        if not left_packet and not right_packet:
            return None
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        result: dict[nn.Parameter, torch.Tensor] = {}
        if left_packet:
            for parameter, left_tangent in left_packet.items():
                _accumulate_parameter_tensor(result, parameter, left_tangent / right)
        if right_packet:
            for parameter, right_tangent in right_packet.items():
                _accumulate_parameter_tensor(
                    result,
                    parameter,
                    -(left * right_tangent) / (right * right),
                )
        return result or None

    if isinstance(record, SoftmaxForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        output = record.output_activation.resolve().detach()
        return {
            parameter: _softmax_jvp(output, input_tangent, record.dim)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, DropoutForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        multiplier = record.multiplier.resolve().detach()
        return {
            parameter: input_tangent * multiplier
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, MaskedFillForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        mask = record.mask
        return {
            parameter: torch.where(mask, torch.zeros_like(input_tangent), input_tangent)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query_packet = (
            tangents_by_node.get(record.query_node_id)
            if record.query_node_id is not None
            else None
        )
        key_packet = (
            tangents_by_node.get(record.key_node_id)
            if record.key_node_id is not None
            else None
        )
        value_packet = (
            tangents_by_node.get(record.value_node_id)
            if record.value_node_id is not None
            else None
        )
        if not query_packet and not key_packet and not value_packet:
            return None
        query = record.query_activation.resolve().detach()
        key = record.key_activation.resolve().detach()
        value = record.value_activation.resolve().detach()
        attention = _attention_weights(
            query,
            key,
            record.attn_mask,
            record.is_causal,
            record.scale,
        )
        result: dict[nn.Parameter, torch.Tensor] = {}
        parameters = set(query_packet or ()) | set(key_packet or ()) | set(value_packet or ())
        for parameter in parameters:
            score_tangent = _attention_score_tangent(
                query,
                (query_packet or {}).get(parameter),
                key,
                (key_packet or {}).get(parameter),
                record.scale,
            )
            output_tangent = None
            if score_tangent is not None:
                output_tangent = torch.matmul(
                    _softmax_jvp(attention, score_tangent, -1),
                    value,
                )
            value_tangent = (value_packet or {}).get(parameter)
            if value_tangent is not None:
                term = torch.matmul(attention, value_tangent)
                output_tangent = term if output_tangent is None else output_tangent + term
            if output_tangent is not None:
                result[parameter] = output_tangent
        return result or None

    raise TypeError(f"unknown forward record: {type(record).__name__}")


def _propagate_backward_tangent_packet(
    record: ForwardRecord,
    output_grad_tangents: Mapping[nn.Parameter, torch.Tensor],
    grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
) -> None:
    if isinstance(record, LinearForwardRecord):
        weight = record.module.weight.detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _linear_input_backward_program(weight, output_grad_tangent),
            )
        return

    if isinstance(record, EmbeddingForwardRecord):
        return

    if isinstance(record, FunctionalLinearForwardRecord):
        weight = record.weight.detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _linear_input_backward_program(weight, output_grad_tangent),
            )
        return

    if isinstance(record, Conv2dForwardRecord):
        input_activation = _empty_from_saved_ref(
            record.input_activation,
            next(iter(output_grad_tangents.values())),
        )
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _conv2d_input_backward_program(
                    record.module,
                    input_activation,
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, BatchNorm2dForwardRecord):
        input_activation = _empty_from_saved_ref(
            record.input_activation,
            next(iter(output_grad_tangents.values())),
        )
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _batch_norm2d_input_backward_program(
                    record.module,
                    input_activation,
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, LayerNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        mean = record.mean.resolve().detach()
        rstd = record.rstd.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _layer_norm_input_backward_program(
                    record.module,
                    input_activation,
                    mean,
                    rstd,
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, FunctionalLayerNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        mean = record.mean.resolve().detach()
        rstd = record.rstd.resolve().detach()
        weight = record.weight.detach() if isinstance(record.weight, torch.Tensor) else None
        bias = record.bias.detach() if isinstance(record.bias, torch.Tensor) else None
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _layer_norm_input_backward_program_params(
                    input_activation,
                    mean,
                    rstd,
                    output_grad_tangent,
                    record.normalized_shape,
                    weight,
                    bias,
                ),
            )
        return

    if isinstance(record, RMSNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _rms_norm_backward_program(
                    input_activation,
                    output_grad_tangent,
                    record.normalized_shape,
                    record.eps,
                ),
            )
        return

    if isinstance(record, ReLUForwardRecord):
        relu_output = record.output_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _relu_backward_program(relu_output, output_grad_tangent),
            )
        return

    if isinstance(record, GELUForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _gelu_backward_program(
                    input_activation,
                    output_grad_tangent,
                    record.approximate,
                ),
            )
        return

    if isinstance(record, FlattenForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.reshape(record.input_shape),
            )
        return

    if isinstance(record, AvgPool2dForwardRecord):
        like = next(iter(output_grad_tangents.values()))
        input_activation = _empty_from_saved_ref(record.input_activation, like)
        kernel_size = _pair(record.module.kernel_size)
        stride = _pair(
            record.module.stride
            if record.module.stride is not None
            else record.module.kernel_size
        )
        padding = _pair(record.module.padding)
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.ops.aten.avg_pool2d_backward.default(
                    output_grad_tangent,
                    input_activation,
                    kernel_size,
                    stride,
                    padding,
                    record.module.ceil_mode,
                    record.module.count_include_pad,
                    record.module.divisor_override,
                ),
            )
        return

    if isinstance(record, AdaptiveAvgPool2dForwardRecord):
        like = next(iter(output_grad_tangents.values()))
        input_activation = _empty_from_saved_ref(record.input_activation, like)
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.ops.aten._adaptive_avg_pool2d_backward.default(
                    output_grad_tangent,
                    input_activation,
                ),
            )
        return

    if isinstance(record, MaxPool2dForwardRecord):
        like = next(iter(output_grad_tangents.values()))
        input_activation = _empty_from_saved_ref(record.input_activation, like)
        module = record.module
        kernel_size = _pair(module.kernel_size)
        stride = _pair(module.stride if module.stride is not None else module.kernel_size)
        padding = _pair(module.padding)
        dilation = _pair(module.dilation)
        indices = record.indices.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.ops.aten.max_pool2d_with_indices_backward.default(
                    output_grad_tangent,
                    input_activation,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    module.ceil_mode,
                    indices,
                ),
            )
        return

    if isinstance(record, SliceForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        length = record.end - record.start
        for parameter, output_grad_tangent in output_grad_tangents.items():
            input_grad_tangent = torch.zeros(
                record.input_shape,
                device=output_grad_tangent.device,
                dtype=output_grad_tangent.dtype,
            )
            input_grad_tangent.narrow(record.dim, record.start, length).copy_(
                output_grad_tangent,
            )
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                input_grad_tangent,
            )
        return

    if isinstance(record, TransposeForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.transpose(record.dim0, record.dim1),
            )
        return

    if isinstance(record, ContiguousForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent,
            )
        return

    if isinstance(record, MatmulForwardRecord):
        left = record.left_activation.resolve().detach()
        right = record.right_activation.resolve().detach()
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    _matmul_left_backward_program(output_grad_tangent, right),
                )
        if record.right_node_id is not None:
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    _matmul_right_backward_program(left, output_grad_tangent),
                )
        return

    if isinstance(record, DivForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    output_grad_tangent / right,
                )
        if record.right_node_id is not None:
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    -(output_grad_tangent * left) / (right * right),
                )
        return

    if isinstance(record, SoftmaxForwardRecord):
        output = record.output_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _softmax_backward_program(output, output_grad_tangent, record.dim),
            )
        return

    if isinstance(record, DropoutForwardRecord):
        multiplier = record.multiplier.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent * multiplier,
            )
        return

    if isinstance(record, MaskedFillForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.where(
                    record.mask,
                    torch.zeros_like(output_grad_tangent),
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query = record.query_activation.resolve().detach()
        key = record.key_activation.resolve().detach()
        value = record.value_activation.resolve().detach()
        attention = _attention_weights(
            query,
            key,
            record.attn_mask,
            record.is_causal,
            record.scale,
        )
        scale = _attention_scale(query, record.scale)
        if record.query_node_id is not None:
            query_packet = grad_tangents_by_node.setdefault(record.query_node_id, {})
        else:
            query_packet = None
        if record.key_node_id is not None:
            key_packet = grad_tangents_by_node.setdefault(record.key_node_id, {})
        else:
            key_packet = None
        if record.value_node_id is not None:
            value_packet = grad_tangents_by_node.setdefault(record.value_node_id, {})
        else:
            value_packet = None
        for parameter, output_grad_tangent in output_grad_tangents.items():
            value_grad_tangent, score_grad_tangent = _attention_value_and_score_grads(
                attention,
                value,
                output_grad_tangent,
            )
            if value_packet is not None:
                _accumulate_parameter_tensor(
                    value_packet,
                    parameter,
                    value_grad_tangent,
                )
            if query_packet is not None:
                _accumulate_parameter_tensor(
                    query_packet,
                    parameter,
                    torch.matmul(score_grad_tangent, key) * scale,
                )
            if key_packet is not None:
                _accumulate_parameter_tensor(
                    key_packet,
                    parameter,
                    torch.matmul(score_grad_tangent.transpose(-2, -1), query) * scale,
                )
        return

    if isinstance(record, AddForwardRecord):
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    _unbroadcast_like(output_grad_tangent, record.left_shape),
                )
        if record.right_node_id is not None:
            alpha = record.alpha
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                value = output_grad_tangent if alpha == 1.0 else alpha * output_grad_tangent
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    _unbroadcast_like(value, record.right_shape),
                )
        return

    if isinstance(record, MulForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            left_shape = _value_shape(left)
            for parameter, output_grad_tangent in output_grad_tangents.items():
                value = output_grad_tangent * right
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    _unbroadcast_like(value, left_shape),
                )
        if record.right_node_id is not None:
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            right_shape = _value_shape(right)
            for parameter, output_grad_tangent in output_grad_tangents.items():
                value = left * output_grad_tangent
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    _unbroadcast_like(value, right_shape),
                )
        return

    if isinstance(record, CatForwardRecord):
        sizes = [shape[record.dim] for shape in record.input_shapes]
        starts = [0]
        for size in sizes[:-1]:
            starts.append(starts[-1] + size)
        for input_node_id, shape, start, size in zip(
            record.input_node_ids,
            record.input_shapes,
            starts,
            sizes,
            strict=True,
        ):
            if input_node_id is None:
                continue
            node_packet = grad_tangents_by_node.setdefault(input_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    node_packet,
                    parameter,
                    output_grad_tangent.narrow(record.dim, start, size).reshape(shape),
                )
        return

    if isinstance(record, ReshapeForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.reshape(record.input_shape),
            )
        return

    if isinstance(record, SelectForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            input_grad_tangent = torch.zeros(
                record.input_shape,
                device=output_grad_tangent.device,
                dtype=output_grad_tangent.dtype,
            )
            input_grad_tangent.select(record.dim, record.index).copy_(output_grad_tangent)
            _accumulate_parameter_tensor(node_packet, parameter, input_grad_tangent)
        return

    if isinstance(record, CastForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.to(dtype=record.input_dtype),
            )
        return

    if isinstance(record, UnaryElementwiseForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        output_activation = (
            None
            if record.output_activation is None
            else record.output_activation.resolve().detach()
        )
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _unary_elementwise_backward_program(
                    record.kind,
                    input_activation,
                    output_grad_tangent,
                    output_activation,
                    record.scalar,
                ),
            )
        return

    raise TypeError(f"unknown forward record: {type(record).__name__}")


def _propagate_backward_tangent_packet_with_grad(
    record: ForwardRecord,
    grad: torch.Tensor,
    output_grad_tangents: Mapping[nn.Parameter, torch.Tensor],
    forward_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
    grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
    parameter_tangents: Mapping[nn.Parameter, torch.Tensor],
    block_channel_by_parameter: Mapping[nn.Parameter, nn.Parameter] | None = None,
) -> None:
    if isinstance(record, LinearForwardRecord):
        weight = record.module.weight.detach()
        input_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        parameters = set(output_grad_tangents)
        weight_channel = (
            block_channel_by_parameter.get(record.module.weight, record.module.weight)
            if block_channel_by_parameter is not None
            else record.module.weight
        )
        if record.module.weight is not None:
            parameters.add(weight_channel)
        for parameter in parameters:
            value = None
            grad_tangent = output_grad_tangents.get(parameter)
            if grad_tangent is not None:
                value = _linear_input_backward_program(weight, grad_tangent)
            if parameter is weight_channel:
                weight_tangent = parameter_tangents.get(record.module.weight)
                if weight_tangent is not None:
                    term = _linear_input_backward_program(weight_tangent, grad)
                    value = term if value is None else value + term
            if value is not None:
                _accumulate_parameter_tensor(input_packet, parameter, value)
        return

    if isinstance(record, FunctionalLinearForwardRecord):
        weight = record.weight.detach()
        input_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        parameters = set(output_grad_tangents)
        if isinstance(record.weight, nn.Parameter):
            weight_channel = (
                block_channel_by_parameter.get(record.weight, record.weight)
                if block_channel_by_parameter is not None
                else record.weight
            )
            parameters.add(weight_channel)
        else:
            weight_channel = None
        for parameter in parameters:
            value = None
            grad_tangent = output_grad_tangents.get(parameter)
            if grad_tangent is not None:
                value = _linear_input_backward_program(weight, grad_tangent)
            if (
                weight_channel is not None
                and parameter is weight_channel
                and isinstance(record.weight, nn.Parameter)
            ):
                weight_tangent = parameter_tangents.get(record.weight)
                if weight_tangent is not None:
                    term = _linear_input_backward_program(weight_tangent, grad)
                    value = term if value is None else value + term
            if value is not None:
                _accumulate_parameter_tensor(input_packet, parameter, value)
        return

    if isinstance(record, MatmulForwardRecord):
        left = record.left_activation.resolve().detach()
        right = record.right_activation.resolve().detach()
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id, {})
            if record.left_node_id is not None
            else {}
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id, {})
            if record.right_node_id is not None
            else {}
        )
        parameters = set(output_grad_tangents) | set(left_packet) | set(right_packet)
        if record.left_node_id is not None:
            left_grad_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                right_tangent = right_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = _matmul_left_backward_program(grad_tangent, right)
                if right_tangent is not None:
                    term = _matmul_left_backward_program(grad, right_tangent)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(left_grad_packet, parameter, value)
        if record.right_node_id is not None:
            right_grad_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                left_tangent = left_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = _matmul_right_backward_program(left, grad_tangent)
                if left_tangent is not None:
                    term = _matmul_right_backward_program(left_tangent, grad)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(right_grad_packet, parameter, value)
        return

    if isinstance(record, MulForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id, {})
            if record.left_node_id is not None
            else {}
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id, {})
            if record.right_node_id is not None
            else {}
        )
        parameters = set(output_grad_tangents) | set(left_packet) | set(right_packet)
        if record.left_node_id is not None:
            left_grad_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            left_shape = _value_shape(left)
            for parameter in parameters:
                value = None
                grad_tangent = output_grad_tangents.get(parameter)
                right_tangent = right_packet.get(parameter)
                if grad_tangent is not None:
                    value = grad_tangent * right
                if right_tangent is not None:
                    term = grad * right_tangent
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(
                        left_grad_packet,
                        parameter,
                        _unbroadcast_like(value, left_shape),
                    )
        if record.right_node_id is not None:
            right_grad_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            right_shape = _value_shape(right)
            for parameter in parameters:
                value = None
                grad_tangent = output_grad_tangents.get(parameter)
                left_tangent = left_packet.get(parameter)
                if grad_tangent is not None:
                    value = left * grad_tangent
                if left_tangent is not None:
                    term = left_tangent * grad
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(
                        right_grad_packet,
                        parameter,
                        _unbroadcast_like(value, right_shape),
                    )
        return

    if isinstance(record, DivForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id, {})
            if record.left_node_id is not None
            else {}
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id, {})
            if record.right_node_id is not None
            else {}
        )
        parameters = set(output_grad_tangents) | set(left_packet) | set(right_packet)
        if record.left_node_id is not None:
            left_grad_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                right_tangent = right_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = grad_tangent / right
                if right_tangent is not None:
                    term = -(grad * right_tangent) / (right * right)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(left_grad_packet, parameter, value)
        if record.right_node_id is not None:
            right_grad_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                left_tangent = left_packet.get(parameter)
                right_tangent = right_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = -(grad_tangent * left) / (right * right)
                if left_tangent is not None:
                    term = -(grad * left_tangent) / (right * right)
                    value = term if value is None else value + term
                if right_tangent is not None:
                    term = 2.0 * grad * left * right_tangent / (right * right * right)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(right_grad_packet, parameter, value)
        return

    if isinstance(record, UnaryElementwiseForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        output_activation = (
            None
            if record.output_activation is None
            else record.output_activation.resolve().detach()
        )
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _unary_elementwise_backward_jvp(
                record.kind,
                input_activation,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
                output_activation,
                record.scalar,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, GELUForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        derivative = _gelu_derivative(input_activation, record.approximate)
        second = _gelu_second_derivative(input_activation, record.approximate)
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = None
            grad_tangent = output_grad_tangents.get(parameter)
            input_tangent = input_packet.get(parameter)
            if grad_tangent is not None:
                value = derivative * grad_tangent
            if input_tangent is not None:
                term = grad * second * input_tangent
                value = term if value is None else value + term
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, LayerNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        mean = record.mean.resolve().detach()
        rstd = record.rstd.resolve().detach()
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _layer_norm_input_backward_jvp(
                record.module,
                input_activation,
                mean,
                rstd,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, FunctionalLayerNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        mean = record.mean.resolve().detach()
        rstd = record.rstd.resolve().detach()
        weight = record.weight.detach() if isinstance(record.weight, torch.Tensor) else None
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _layer_norm_input_backward_jvp_params(
                input_activation,
                mean,
                rstd,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
                record.normalized_shape,
                record.eps,
                weight,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, RMSNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _rms_norm_backward_jvp(
                input_activation,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
                record.normalized_shape,
                record.eps,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, SoftmaxForwardRecord):
        output = record.output_activation.resolve().detach()
        output_packet = forward_tangents_by_node.get(record.output_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(output_packet):
            value = _softmax_backward_jvp(
                output,
                grad,
                output_grad_tangents.get(parameter),
                output_packet.get(parameter),
                record.dim,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, DropoutForwardRecord):
        multiplier = record.multiplier.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent * multiplier,
            )
        return

    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query = record.query_activation.resolve().detach()
        key = record.key_activation.resolve().detach()
        value = record.value_activation.resolve().detach()
        query_packet = (
            forward_tangents_by_node.get(record.query_node_id, {})
            if record.query_node_id is not None
            else {}
        )
        key_packet = (
            forward_tangents_by_node.get(record.key_node_id, {})
            if record.key_node_id is not None
            else {}
        )
        value_packet = (
            forward_tangents_by_node.get(record.value_node_id, {})
            if record.value_node_id is not None
            else {}
        )
        query_grad_packet = (
            grad_tangents_by_node.setdefault(record.query_node_id, {})
            if record.query_node_id is not None
            else None
        )
        key_grad_packet = (
            grad_tangents_by_node.setdefault(record.key_node_id, {})
            if record.key_node_id is not None
            else None
        )
        value_grad_packet = (
            grad_tangents_by_node.setdefault(record.value_node_id, {})
            if record.value_node_id is not None
            else None
        )
        parameters = (
            set(output_grad_tangents)
            | set(query_packet)
            | set(key_packet)
            | set(value_packet)
        )
        for parameter in parameters:
            query_grad_tangent, key_grad_tangent, value_grad_tangent = (
                _attention_backward_jvp(
                    query,
                    query_packet.get(parameter),
                    key,
                    key_packet.get(parameter),
                    value,
                    value_packet.get(parameter),
                    grad,
                    output_grad_tangents.get(parameter),
                    record.attn_mask,
                    record.is_causal,
                    record.scale,
                )
            )
            if query_grad_packet is not None and query_grad_tangent is not None:
                _accumulate_parameter_tensor(
                    query_grad_packet,
                    parameter,
                    query_grad_tangent,
                )
            if key_grad_packet is not None and key_grad_tangent is not None:
                _accumulate_parameter_tensor(
                    key_grad_packet,
                    parameter,
                    key_grad_tangent,
                )
            if value_grad_packet is not None and value_grad_tangent is not None:
                _accumulate_parameter_tensor(
                    value_grad_packet,
                    parameter,
                    value_grad_tangent,
                )
        return

    _propagate_backward_tangent_packet(
        record,
        output_grad_tangents,
        grad_tangents_by_node,
    )


def _accumulate_parameter_tensor(
    values_by_parameter: dict[nn.Parameter, torch.Tensor],
    parameter: nn.Parameter,
    value: torch.Tensor,
) -> None:
    existing = values_by_parameter.get(parameter)
    if existing is None:
        values_by_parameter[parameter] = value
    else:
        values_by_parameter[parameter] = existing + value


def _local_record_input_tangent(
    record: ForwardRecord,
    parameter: nn.Parameter,
    forward_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]]
    | None,
) -> torch.Tensor | None:
    if forward_tangents_by_node is None:
        return None
    input_node_ids = _record_input_node_ids(record)
    if len(input_node_ids) != 1:
        return None
    return forward_tangents_by_node.get(input_node_ids[0], {}).get(parameter)


def _empty_from_saved_ref(
    ref: SavedTensorRef,
    like: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        ref.expected_shape,
        device=like.device,
        dtype=like.dtype,
    )
