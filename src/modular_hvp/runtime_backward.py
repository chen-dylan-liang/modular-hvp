"""Backward scheduling and HVP accumulation for the eager runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.graph import (
    RecordedForwardGraph,
    _local_parameter_use_counts,
    _record_has_backward_nonlinearity_tangents,
    _take_graph_local_parameters_by_output_node,
)
from modular_hvp.graph_tangent import (
    _accumulate_parameter_tensor,
    _forward_record_tangent_packet,
    _local_record_input_tangent,
    _propagate_backward_tangent_packet,
    _propagate_backward_tangent_packet_with_grad,
)
from modular_hvp.kernels import (
    _batch_norm2d_bias_backward_program,
    _batch_norm2d_weight_backward_program,
    _conv2d_bias_backward_program,
    _conv2d_weight_backward_program,
    _embedding_weight_backward_program,
    _layer_norm_bias_backward_program,
    _layer_norm_normalized_input,
    _layer_norm_normalized_input_from_stats,
    _layer_norm_weight_backward_program,
    _linear_bias_backward_program,
    _linear_weight_backward_program,
    _make_adaptive_avg_pool2d_input_curvature,
    _make_avg_pool2d_input_curvature,
    _make_batch_norm2d_input_curvature,
    _make_conv2d_input_curvature,
    _make_flatten_input_curvature,
    _make_gelu_input_curvature,
    _make_layer_norm_input_curvature,
    _make_linear_input_curvature,
    _make_max_pool2d_input_curvature,
    _make_relu_input_curvature,
    _make_reshape_input_curvature,
)
from modular_hvp.losses import _make_loss_output_curvature
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
    ReLUForwardRecord,
    ReshapeForwardRecord,
    RMSNormForwardRecord,
    ScaledDotProductAttentionForwardRecord,
    SelectForwardRecord,
    SliceForwardRecord,
    SoftmaxForwardRecord,
    TransposeForwardRecord,
    UnaryElementwiseForwardRecord,
    _record_input_node_ids,
    _record_local_output_tangents,
    _record_output_node_id,
)


class BackwardRuntimeMixin:
    def _start_eager_backward(self, grad: torch.Tensor) -> None:
        loss_record = self._state.loss_record
        if loss_record is None:
            raise RuntimeError(
                "modular_hvp did not observe a supported scalar loss; "
                "currently use torch.nn.MSELoss or torch.nn.functional.mse_loss"
            )

        with torch.no_grad():
            output_curvature = _make_loss_output_curvature(loss_record, grad.detach())
            self._state.output_curvature = output_curvature

            if self._state.output_node_id is None:
                raise RuntimeError("missing model output node for ModularHVP backward")
            if self._state.use_graph_curvature:
                graph = RecordedForwardGraph.from_records(
                    records=self._state.forward_records,
                    output_node_id=self._state.output_node_id,
                    retain_local_parameter_inputs=self._has_reused_parameters,
                )
                if graph.requires_hooked_primal_grads or self._has_reused_parameters:
                    self._state.active_graph = graph
                    self._initialize_graph_hvp_hook_state(graph)
                    self._state.loss_record = None
                    self._state.eager_backward_active = True
                    return
                self._state.active_graph = None
                self._compute_graph_hvps_single_pass(graph)
                self._state.loss_record = None
                self._state.eager_backward_active = False
                return
            self._add_node_curvature(self._state.output_node_id, output_curvature)
            self._state.loss_record = None
            self._state.eager_backward_active = True

    def _add_node_curvature(
        self,
        node_id: int,
        curvature: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        self._state.curvatures_by_node_id.setdefault(node_id, []).append(curvature)

    def _take_node_curvature(
        self,
        node_id: int,
    ) -> Callable[[torch.Tensor], torch.Tensor] | None:
        curvatures = self._state.curvatures_by_node_id.pop(node_id, None)
        if not curvatures:
            return None
        if len(curvatures) == 1:
            return curvatures[0]

        def summed_curvature(value: torch.Tensor) -> torch.Tensor:
            result = curvatures[0](value)
            for curvature in curvatures[1:]:
                result = result + curvature(value)
            return result

        return summed_curvature

    def _make_forward_record_hook(self, record: ForwardRecord) -> Any:
        self._state.forward_records.append(record)
        if self._has_reused_parameters:
            self._state.use_graph_curvature = True
        if isinstance(
            record,
            (
                FunctionalLinearForwardRecord,
                FunctionalLayerNormForwardRecord,
                DropoutForwardRecord,
                MaskedFillForwardRecord,
                AddForwardRecord,
                MulForwardRecord,
                CatForwardRecord,
                SliceForwardRecord,
                SelectForwardRecord,
                TransposeForwardRecord,
                ContiguousForwardRecord,
                CastForwardRecord,
                UnaryElementwiseForwardRecord,
                RMSNormForwardRecord,
                MatmulForwardRecord,
                DivForwardRecord,
                SoftmaxForwardRecord,
                ScaledDotProductAttentionForwardRecord,
            ),
        ):
            self._state.use_graph_curvature = True

        def forward_record_hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._state.eager_backward_active:
                return grad
            with torch.no_grad():
                if self._state.use_graph_curvature:
                    self._consume_graph_backward_record(record, grad)
                    return grad
                if isinstance(record, LinearForwardRecord):
                    self._consume_linear_backward_record(
                        record=record,
                        grad=grad,
                    )
                elif isinstance(record, EmbeddingForwardRecord):
                    self._consume_embedding_backward_record(record=record)
                elif isinstance(record, Conv2dForwardRecord):
                    self._consume_conv2d_backward_record(
                        record=record,
                        grad=grad,
                    )
                elif isinstance(record, BatchNorm2dForwardRecord):
                    self._consume_batch_norm2d_backward_record(
                        record=record,
                        grad=grad,
                    )
                elif isinstance(record, LayerNormForwardRecord):
                    self._consume_layer_norm_backward_record(
                        record=record,
                        grad=grad,
                    )
                elif isinstance(record, ReLUForwardRecord):
                    self._consume_relu_backward_record(
                        record=record,
                        grad=grad,
                    )
                elif isinstance(record, GELUForwardRecord):
                    self._consume_gelu_backward_record(
                        record=record,
                        grad=grad,
                    )
                elif isinstance(record, FlattenForwardRecord):
                    self._consume_flatten_backward_record(record=record)
                elif isinstance(record, AvgPool2dForwardRecord):
                    self._consume_avg_pool2d_backward_record(record=record)
                elif isinstance(record, AdaptiveAvgPool2dForwardRecord):
                    self._consume_adaptive_avg_pool2d_backward_record(record=record)
                elif isinstance(record, MaxPool2dForwardRecord):
                    self._consume_max_pool2d_backward_record(record=record)
                elif isinstance(record, AddForwardRecord):
                    self._consume_add_backward_record(record=record)
                elif isinstance(record, ReshapeForwardRecord):
                    self._consume_reshape_backward_record(record=record)
                elif isinstance(
                    record,
                    (
                        SliceForwardRecord,
                        SelectForwardRecord,
                        TransposeForwardRecord,
                        ContiguousForwardRecord,
                        CastForwardRecord,
                        UnaryElementwiseForwardRecord,
                        RMSNormForwardRecord,
                        MatmulForwardRecord,
                        MulForwardRecord,
                        CatForwardRecord,
                        DivForwardRecord,
                        SoftmaxForwardRecord,
                    ),
                ):
                    raise RuntimeError(
                        "DAG-only primitive reached sequential backward path"
                    )
                else:
                    raise TypeError(f"unknown forward record: {type(record).__name__}")
            return grad

        return forward_record_hook

    def _initialize_graph_hvp_hook_state(self, graph: RecordedForwardGraph) -> None:
        output_curvature = self._state.output_curvature
        if output_curvature is None:
            raise RuntimeError("graph HVP requested before loss initialization")
        tangents_by_node = self._compute_graph_forward_tangent_packets(
            graph,
            retained_node_ids=graph.retained_forward_tangent_node_ids,
        )
        output_tangents = tangents_by_node.pop(graph.output_node_id, {})
        output_grad_tangents: dict[nn.Parameter, torch.Tensor] = {}
        for parameter, tangent_value in output_tangents.items():
            output_grad_tangents[parameter] = output_curvature(tangent_value)
        output_tangents.clear()
        local_parameters_by_output_node = _take_graph_local_parameters_by_output_node(
            graph.records
        )
        self._state.graph.prepare_hooked_backward(
            output_node_id=graph.output_node_id,
            output_grad_tangents=output_grad_tangents,
            retained_forward_tangents_by_node=tangents_by_node,
            local_parameters_by_output_node=local_parameters_by_output_node,
            remaining_local_uses_by_parameter=_local_parameter_use_counts(
                local_parameters_by_output_node
            ),
            input_use_counts=graph.fresh_input_use_counts(),
        )
        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))

    def _compute_graph_forward_tangent_packets(
        self,
        graph: RecordedForwardGraph,
        *,
        retained_node_ids: set[int] | frozenset[int] | None = None,
    ) -> dict[int, dict[nn.Parameter, torch.Tensor]]:
        tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            node_id: packet.copy()
            for node_id, packet in self._state.raw_parameter_tangents_by_node.items()
        }
        retained_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {}
        if retained_node_ids is not None:
            for node_id, packet in tangents_by_node.items():
                if node_id in retained_node_ids:
                    retained_tangents_by_node[node_id] = packet
        remaining_input_uses = graph.fresh_input_use_counts()
        for record in graph.records:
            record_tangents = _forward_record_tangent_packet(record, tangents_by_node)
            for input_node_id in _record_input_node_ids(record):
                remaining = remaining_input_uses.get(input_node_id)
                if remaining is None:
                    continue
                if remaining <= 1:
                    remaining_input_uses.pop(input_node_id, None)
                    tangents_by_node.pop(input_node_id, None)
                else:
                    remaining_input_uses[input_node_id] = remaining - 1
            local_tangents = _record_local_output_tangents(record)
            if local_tangents:
                if record_tangents is None:
                    record_tangents = {}
                for parameter, tangent_value in local_tangents.items():
                    _accumulate_parameter_tensor(
                        record_tangents,
                        parameter,
                        tangent_value,
                    )
            if record_tangents:
                node_tangents = tangents_by_node.setdefault(
                    _record_output_node_id(record),
                    {},
                )
                for parameter, tangent_value in record_tangents.items():
                    _accumulate_parameter_tensor(
                        node_tangents,
                        parameter,
                        tangent_value,
                    )
                if (
                    retained_node_ids is None
                    or _record_output_node_id(record) in retained_node_ids
                ):
                    retained_tangents_by_node[_record_output_node_id(record)] = (
                        node_tangents
                    )
        return retained_tangents_by_node if retained_node_ids is not None else tangents_by_node

    def _consume_graph_backward_record(
        self,
        record: ForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        graph = self._state.active_graph
        if graph is None:
            raise RuntimeError("missing active recorded graph")
        self._state.graph.observe_primal_grad(graph, record, grad)
        self._drain_ready_graph_backward_records()

    def _drain_ready_graph_backward_records(self) -> None:
        graph = self._state.active_graph
        if graph is None:
            raise RuntimeError("missing active recorded graph")
        graph_state = self._state.graph

        while True:
            record = graph_state.pop_ready_record(graph)
            if record is None:
                break

            output_grad_tangents = graph_state.pop_output_grad_tangents(record)
            local_parameters = graph_state.local_parameters_for(record)
            self._accumulate_graph_record_hvps(
                record,
                output_grad_tangents,
                primal_grad=graph_state.primal_grad(record),
                forward_tangents_by_node=graph_state.forward_tangents_by_node,
            )
            for parameter in local_parameters:
                if graph_state.consume_local_parameter_use(parameter) == 0:
                    output_grad_tangents.pop(parameter, None)

            if output_grad_tangents or _record_has_backward_nonlinearity_tangents(
                record,
                graph_state.forward_tangents_by_node,
            ):
                _propagate_backward_tangent_packet_with_grad(
                    record,
                    graph_state.primal_grad(record),
                    output_grad_tangents,
                    graph_state.forward_tangents_by_node,
                    graph_state.grad_tangents_by_node,
                    self._tangents_by_parameter,
                )

            graph_state.finish_record(graph, record)

        if graph_state.is_complete(graph):
            self._accumulate_raw_parameter_graph_hvps(graph_state.grad_tangents_by_node)
            for block in self.parameter_blocks:
                if getattr(block.parameter, "hvp", None) is None:
                    setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))
            self._state.active_graph = None
            self._state.eager_backward_active = False

    def _compute_graph_hvps_single_pass(self, graph: RecordedForwardGraph) -> None:
        output_curvature = self._state.output_curvature
        if output_curvature is None:
            raise RuntimeError("graph HVP requested before loss initialization")

        tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            node_id: packet.copy()
            for node_id, packet in self._state.raw_parameter_tangents_by_node.items()
        }
        remaining_input_uses = graph.fresh_input_use_counts()
        for record in graph.records:
            record_tangents = _forward_record_tangent_packet(record, tangents_by_node)
            for input_node_id in _record_input_node_ids(record):
                remaining = remaining_input_uses.get(input_node_id)
                if remaining is None:
                    continue
                if remaining <= 1:
                    remaining_input_uses.pop(input_node_id, None)
                    tangents_by_node.pop(input_node_id, None)
                else:
                    remaining_input_uses[input_node_id] = remaining - 1
            local_tangents = _record_local_output_tangents(record)
            if local_tangents:
                if record_tangents is None:
                    record_tangents = {}
                for parameter, tangent_value in local_tangents.items():
                    _accumulate_parameter_tensor(
                        record_tangents,
                        parameter,
                        tangent_value,
                    )
            if record_tangents:
                node_tangents = tangents_by_node.setdefault(
                    _record_output_node_id(record),
                    {},
                )
                for parameter, tangent_value in record_tangents.items():
                    _accumulate_parameter_tensor(
                        node_tangents,
                        parameter,
                        tangent_value,
                    )

        output_tangents = tangents_by_node.get(graph.output_node_id, {})
        tangents_by_node.clear()
        grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            graph.output_node_id: {
                parameter: output_curvature(tangent_value)
                for parameter, tangent_value in output_tangents.items()
            }
        }

        for record in graph.reverse_records:
            output_grad_tangents = grad_tangents_by_node.pop(
                _record_output_node_id(record),
                None,
            )
            if not output_grad_tangents:
                _record_local_output_tangents(record).clear()
                continue

            local_parameters = tuple(_record_local_output_tangents(record))
            self._accumulate_graph_record_hvps(record, output_grad_tangents)
            for parameter in local_parameters:
                output_grad_tangents.pop(parameter, None)
            if not output_grad_tangents:
                continue

            _propagate_backward_tangent_packet(
                record,
                output_grad_tangents,
                grad_tangents_by_node,
            )

        self._accumulate_raw_parameter_graph_hvps(grad_tangents_by_node)

        for block in self.parameter_blocks:
            if getattr(block.parameter, "hvp", None) is None:
                setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))

    def _accumulate_raw_parameter_graph_hvps(
        self,
        grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
    ) -> None:
        for node_id, local_packet in self._state.raw_parameter_tangents_by_node.items():
            output_packet = grad_tangents_by_node.pop(node_id, {})
            for parameter in local_packet:
                hvp = output_packet.get(parameter)
                if hvp is not None:
                    self._accumulate_hvp(parameter, hvp)

    def _accumulate_graph_record_hvps(
        self,
        record: ForwardRecord,
        output_grad_tangents: Mapping[nn.Parameter, torch.Tensor],
        *,
        primal_grad: torch.Tensor | None = None,
        forward_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]]
        | None = None,
    ) -> None:
        local_parameters = self._graph_record_local_parameters(record)
        local_tangents = _record_local_output_tangents(record)
        if not local_parameters:
            return
        try:
            if isinstance(record, LinearForwardRecord):
                input_activation = record.input_activation.resolve_and_release().detach()
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    input_tangent = _local_record_input_tangent(
                        record,
                        parameter,
                        forward_tangents_by_node,
                    )
                    if parameter is record.module.weight:
                        hvp = torch.zeros_like(parameter)
                        if grad_tangent is not None:
                            hvp = hvp + _linear_weight_backward_program(
                                input_activation,
                                grad_tangent,
                            )
                        if input_tangent is not None and primal_grad is not None:
                            hvp = hvp + _linear_weight_backward_program(
                                input_tangent,
                                primal_grad,
                            )
                    elif parameter is record.module.bias:
                        hvp = (
                            torch.zeros_like(parameter)
                            if grad_tangent is None
                            else _linear_bias_backward_program(grad_tangent)
                        )
                    else:
                        raise RuntimeError("local dual activation belongs to wrong module")
                    self._accumulate_hvp(parameter, hvp)
                return

            if isinstance(record, EmbeddingForwardRecord):
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.module.weight:
                        hvp = _embedding_weight_backward_program(
                            record.module,
                            record.indices,
                            grad_tangent,
                        )
                    else:
                        raise RuntimeError("local dual activation belongs to wrong module")
                    self._accumulate_hvp(parameter, hvp)
                return

            if isinstance(record, FunctionalLinearForwardRecord):
                input_activation = record.input_activation.resolve_and_release().detach()
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    input_tangent = _local_record_input_tangent(
                        record,
                        parameter,
                        forward_tangents_by_node,
                    )
                    if parameter is record.weight:
                        hvp = torch.zeros_like(parameter)
                        if grad_tangent is not None:
                            hvp = hvp + _linear_weight_backward_program(
                                input_activation,
                                grad_tangent,
                            )
                        if input_tangent is not None and primal_grad is not None:
                            hvp = hvp + _linear_weight_backward_program(
                                input_tangent,
                                primal_grad,
                            )
                    elif parameter is record.bias:
                        hvp = (
                            torch.zeros_like(parameter)
                            if grad_tangent is None
                            else _linear_bias_backward_program(grad_tangent)
                        )
                    else:
                        raise RuntimeError("local dual activation belongs to wrong linear op")
                    self._accumulate_hvp(parameter, hvp)
                return

            if isinstance(record, Conv2dForwardRecord):
                input_activation = record.input_activation.resolve_and_release().detach()
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.module.weight:
                        hvp = _conv2d_weight_backward_program(
                            record.module,
                            input_activation,
                            grad_tangent,
                        )
                    elif parameter is record.module.bias:
                        hvp = _conv2d_bias_backward_program(grad_tangent)
                    else:
                        raise RuntimeError("local dual activation belongs to wrong module")
                    self._accumulate_hvp(parameter, hvp)
                return

            if isinstance(record, BatchNorm2dForwardRecord):
                input_activation = record.input_activation.resolve_and_release().detach()
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.module.weight:
                        hvp = _batch_norm2d_weight_backward_program(
                            record.module,
                            input_activation,
                            grad_tangent,
                        )
                    elif parameter is record.module.bias:
                        hvp = _batch_norm2d_bias_backward_program(grad_tangent)
                    else:
                        raise RuntimeError("local dual activation belongs to wrong module")
                    self._accumulate_hvp(parameter, hvp)
                return

            if isinstance(record, LayerNormForwardRecord):
                input_activation = record.input_activation.resolve().detach()
                normalized_input = _layer_norm_normalized_input(
                    record.module,
                    input_activation,
                )
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.module.weight:
                        hvp = _layer_norm_weight_backward_program(
                            normalized_input,
                            grad_tangent,
                            parameter.shape,
                        )
                    elif parameter is record.module.bias:
                        hvp = _layer_norm_bias_backward_program(
                            grad_tangent,
                            parameter.shape,
                        )
                    else:
                        raise RuntimeError("local dual activation belongs to wrong module")
                    self._accumulate_hvp(parameter, hvp)
                return

            if isinstance(record, FunctionalLayerNormForwardRecord):
                input_activation = record.input_activation.resolve().detach()
                mean = record.mean.resolve().detach()
                rstd = record.rstd.resolve().detach()
                normalized_input = _layer_norm_normalized_input_from_stats(
                    input_activation,
                    mean,
                    rstd,
                )
                for parameter in local_parameters:
                    grad_tangent = output_grad_tangents.get(parameter)
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.weight:
                        hvp = _layer_norm_weight_backward_program(
                            normalized_input,
                            grad_tangent,
                            parameter.shape,
                        )
                    elif parameter is record.bias:
                        hvp = _layer_norm_bias_backward_program(
                            grad_tangent,
                            parameter.shape,
                        )
                    else:
                        raise RuntimeError(
                            "local dual activation belongs to wrong layer_norm op"
                        )
                    self._accumulate_hvp(parameter, hvp)
                return

            raise TypeError(
                f"local output tangents on unsupported record: {type(record).__name__}"
            )
        finally:
            local_tangents.clear()

    def _graph_record_local_parameters(
        self,
        record: ForwardRecord,
    ) -> tuple[nn.Parameter, ...]:
        return self._state.graph.local_parameters_for(record)

    def _consume_embedding_backward_record(
        self,
        *,
        record: EmbeddingForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for Embedding backward hook")
        try:
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
                if parameter is record.module.weight:
                    hvp = _embedding_weight_backward_program(
                        record.module,
                        record.indices,
                        parameter_grad_tangent,
                    )
                else:
                    raise RuntimeError("local dual activation belongs to wrong module")
                self._accumulate_hvp(parameter, hvp)
        finally:
            record.local_output_tangents.clear()

    def _consume_linear_backward_record(
        self,
        *,
        record: LinearForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for Linear backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        try:
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
                if parameter is record.module.weight:
                    hvp = _linear_weight_backward_program(
                        input_activation,
                        parameter_grad_tangent,
                    )
                elif parameter is record.module.bias:
                    hvp = _linear_bias_backward_program(parameter_grad_tangent)
                else:
                    raise RuntimeError("local dual activation belongs to wrong module")
                self._accumulate_hvp(parameter, hvp)
        finally:
            record.local_output_tangents.clear()

        self._add_node_curvature(
            record.input_node_id,
            _make_linear_input_curvature(
                curvature,
                record.module.weight.detach(),
            ),
        )

    def _consume_conv2d_backward_record(
        self,
        *,
        record: Conv2dForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for Conv2d backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        try:
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
                if parameter is record.module.weight:
                    hvp = _conv2d_weight_backward_program(
                        record.module,
                        input_activation,
                        parameter_grad_tangent,
                    )
                elif parameter is record.module.bias:
                    hvp = _conv2d_bias_backward_program(parameter_grad_tangent)
                else:
                    raise RuntimeError("local dual activation belongs to wrong module")
                self._accumulate_hvp(parameter, hvp)
        finally:
            record.local_output_tangents.clear()

        self._add_node_curvature(
            record.input_node_id,
            _make_conv2d_input_curvature(
                curvature,
                record.module,
                input_activation,
            ),
        )

    def _consume_batch_norm2d_backward_record(
        self,
        *,
        record: BatchNorm2dForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for BatchNorm2d backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        try:
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
                if parameter is record.module.weight:
                    hvp = _batch_norm2d_weight_backward_program(
                        record.module,
                        input_activation,
                        parameter_grad_tangent,
                    )
                elif parameter is record.module.bias:
                    hvp = _batch_norm2d_bias_backward_program(parameter_grad_tangent)
                else:
                    raise RuntimeError("local dual activation belongs to wrong module")
                self._accumulate_hvp(parameter, hvp)
        finally:
            record.local_output_tangents.clear()

        self._add_node_curvature(
            record.input_node_id,
            _make_batch_norm2d_input_curvature(
                curvature,
                record.module,
                input_activation,
            ),
        )

    def _consume_layer_norm_backward_record(
        self,
        *,
        record: LayerNormForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for LayerNorm backward hook")
        input_activation = record.input_activation.resolve().detach()
        normalized_input = _layer_norm_normalized_input(record.module, input_activation)
        try:
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
                if parameter is record.module.weight:
                    hvp = _layer_norm_weight_backward_program(
                        normalized_input,
                        parameter_grad_tangent,
                        parameter.shape,
                    )
                elif parameter is record.module.bias:
                    hvp = _layer_norm_bias_backward_program(
                        parameter_grad_tangent,
                        parameter.shape,
                    )
                else:
                    raise RuntimeError("local dual activation belongs to wrong module")
                self._accumulate_hvp(parameter, hvp)
        finally:
            record.local_output_tangents.clear()

        mean = record.mean.resolve_and_release().detach()
        rstd = record.rstd.resolve_and_release().detach()
        input_activation = record.input_activation.resolve_and_release().detach()
        self._add_node_curvature(
            record.input_node_id,
            _make_layer_norm_input_curvature(
                curvature,
                record.module,
                input_activation,
                mean,
                rstd,
            ),
        )

    def _consume_relu_backward_record(
        self,
        *,
        record: ReLUForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for ReLU backward hook")
        relu_output = record.output_activation.resolve_and_release()
        self._add_node_curvature(
            record.input_node_id,
            _make_relu_input_curvature(curvature, relu_output),
        )

    def _consume_gelu_backward_record(
        self,
        *,
        record: GELUForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for GELU backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        self._add_node_curvature(
            record.input_node_id,
            _make_gelu_input_curvature(
                curvature,
                input_activation,
                record.approximate,
            ),
        )

    def _consume_flatten_backward_record(
        self,
        *,
        record: FlattenForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for Flatten backward hook")
        self._add_node_curvature(
            record.input_node_id,
            _make_flatten_input_curvature(curvature, record),
        )

    def _consume_avg_pool2d_backward_record(
        self,
        *,
        record: AvgPool2dForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for AvgPool2d backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        self._add_node_curvature(
            record.input_node_id,
            _make_avg_pool2d_input_curvature(
                curvature,
                record.module,
                input_activation,
            ),
        )

    def _consume_adaptive_avg_pool2d_backward_record(
        self,
        *,
        record: AdaptiveAvgPool2dForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError(
                "missing output curvature for AdaptiveAvgPool2d backward hook"
            )
        input_activation = record.input_activation.resolve_and_release().detach()
        self._add_node_curvature(
            record.input_node_id,
            _make_adaptive_avg_pool2d_input_curvature(
                curvature,
                input_activation,
                record.output_size,
            ),
        )

    def _consume_max_pool2d_backward_record(
        self,
        *,
        record: MaxPool2dForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for MaxPool2d backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        indices = record.indices.resolve_and_release().detach()
        self._add_node_curvature(
            record.input_node_id,
            _make_max_pool2d_input_curvature(
                curvature,
                record.module,
                input_activation,
                indices,
            ),
        )

    def _consume_add_backward_record(
        self,
        *,
        record: AddForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for add backward hook")
        if record.left_node_id is not None:
            self._add_node_curvature(record.left_node_id, curvature)
        if record.right_node_id is not None:
            alpha = record.alpha

            def right_curvature(value: torch.Tensor) -> torch.Tensor:
                return alpha * curvature(alpha * value)

            self._add_node_curvature(record.right_node_id, right_curvature)

    def _consume_reshape_backward_record(
        self,
        *,
        record: ReshapeForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for reshape backward hook")
        self._add_node_curvature(
            record.input_node_id,
            _make_reshape_input_curvature(curvature, record),
        )

    def _accumulate_hvp(self, parameter: nn.Parameter, hvp: torch.Tensor) -> None:
        existing = getattr(parameter, "hvp", None)
        if existing is None:
            setattr(parameter, "hvp", hvp.detach())
        else:
            existing.add_(hvp)

    def _make_loss_hook(self) -> Any:
        def loss_hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._state.backward_called:
                self._state.backward_called = True
                self._start_eager_backward(grad)
            return grad

        return loss_hook
