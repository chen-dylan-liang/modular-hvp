"""Backward scheduling and HVP accumulation for the eager runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.block_graph import validate_parameter_blocks_on_graph
from modular_hvp.graph import (
    RecordedForwardGraph,
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
    _release_record_saved_tensors,
)


def _apply_output_curvature_packet(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    output_tangents: dict[nn.Parameter, torch.Tensor],
) -> dict[nn.Parameter, torch.Tensor]:
    output_grad_tangents: dict[nn.Parameter, torch.Tensor] = {}
    while output_tangents:
        parameter, tangent_value = output_tangents.popitem()
        output_grad_tangents[parameter] = output_curvature(tangent_value)
    return output_grad_tangents


class BackwardRuntimeMixin:
    def _start_eager_backward(self, grad: torch.Tensor) -> None:
        loss_record = self._state.loss_record
        if loss_record is None:
            raise RuntimeError(
                "modular_hvp did not observe a supported scalar loss; "
                "currently use supported MSE loss or cross_entropy reductions"
            )

        with torch.no_grad():
            output_curvature = _make_loss_output_curvature(loss_record, grad.detach())
            self._state.output_curvature = output_curvature

            if self._state.output_node_id is None:
                raise RuntimeError("missing model output node for ModularHVP backward")
            if self._state.use_graph_curvature:
                same_block_input_node_ids_by_channel: dict[
                    nn.Parameter,
                    set[int],
                ] = {}
                if self._requires_graph_block_scope:
                    same_block_input_node_ids_by_channel = (
                        self._validate_custom_blocks_on_recorded_graph()
                    )
                for channel, node_ids in (
                    self._reused_local_input_tangent_node_ids_by_channel().items()
                ):
                    same_block_input_node_ids_by_channel.setdefault(
                        channel,
                        set(),
                    ).update(node_ids)
                self._state.same_block_input_tangent_node_ids_by_channel = (
                    same_block_input_node_ids_by_channel
                )
                graph = RecordedForwardGraph.from_records(
                    records=self._state.forward_records,
                    output_node_id=self._state.output_node_id,
                    extra_retained_forward_tangent_channels_by_node=(
                        self._retained_forward_tangent_channels_by_node(
                            same_block_input_node_ids_by_channel,
                        )
                    ),
                )
                requires_same_block_input_jvp = bool(
                    same_block_input_node_ids_by_channel
                )
                if graph.requires_hooked_primal_grads or requires_same_block_input_jvp:
                    self._state.active_graph = graph
                    self._state.eager_backward_active = True
                    self._initialize_graph_hvp_hook_state(graph)
                    self._state.loss_record = None
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

    def _validate_custom_blocks_on_recorded_graph(
        self,
    ) -> dict[nn.Parameter, set[int]]:
        retained_by_parameter = validate_parameter_blocks_on_graph(
            records=self._state.forward_records,
            raw_parameter_tangents_by_node=self._state.raw_parameter_tangents_by_node,
            block_parameters_by_parameter=self._block_parameters_by_parameter,
            parameter_names={
                block.parameter: block.name for block in self.parameter_blocks
            },
        )
        retained_by_channel: dict[nn.Parameter, set[int]] = {}
        for parameter, node_ids in retained_by_parameter.items():
            channel = self._block_channel_by_parameter[parameter]
            retained_by_channel.setdefault(channel, set()).update(node_ids)
        return retained_by_channel

    def _reused_local_input_tangent_node_ids_by_channel(
        self,
    ) -> dict[nn.Parameter, set[int]]:
        if not self._reused_block_channels:
            return {}
        retained_by_channel: dict[nn.Parameter, set[int]] = {}
        for record in self._state.forward_records:
            local_channels = self._local_channels_for_parameters(
                tuple(_record_local_output_tangents(record)),
            )
            reused_local_channels = set(local_channels) & self._reused_block_channels
            if not reused_local_channels:
                continue
            input_node_ids = _record_input_node_ids(record)
            if not input_node_ids:
                continue
            for channel in reused_local_channels:
                retained_by_channel.setdefault(channel, set()).update(input_node_ids)
        return retained_by_channel

    def _retained_forward_tangent_channels_by_node(
        self,
        node_ids_by_channel: Mapping[nn.Parameter, set[int]],
    ) -> dict[int, set[nn.Parameter]]:
        retained_by_node: dict[int, set[nn.Parameter]] = {}
        for channel, node_ids in node_ids_by_channel.items():
            for node_id in node_ids:
                retained_by_node.setdefault(node_id, set()).add(channel)
        return retained_by_node

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

    def _register_forward_record_hook(
        self,
        record: ForwardRecord,
        output: torch.Tensor,
    ) -> None:
        if self._use_graph_tensors:
            self._state.forward_records.append(record)
            self._update_parameterized_frontier(record)
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
        if not output.requires_grad:
            return
        if self._should_register_forward_record_hook(record):
            output.register_hook(self._make_forward_record_hook(record))

    def _should_register_forward_record_hook(self, record: ForwardRecord) -> bool:
        if not self._use_graph_tensors:
            return True
        if not self._state.use_graph_curvature:
            return True
        if set(
            self._local_channels_for_parameters(
                tuple(_record_local_output_tangents(record)),
            ),
        ) & self._reused_block_channels:
            return True
        if isinstance(
            record,
            (
                FunctionalLayerNormForwardRecord,
                GELUForwardRecord,
                LayerNormForwardRecord,
                MatmulForwardRecord,
                MulForwardRecord,
                RMSNormForwardRecord,
                ScaledDotProductAttentionForwardRecord,
                SoftmaxForwardRecord,
                UnaryElementwiseForwardRecord,
            ),
        ):
            return True
        if isinstance(record, DivForwardRecord):
            return record.right_node_id is not None
        if not self._requires_graph_block_scope:
            return False
        return (
            _record_output_node_id(record)
            in self._state.same_block_input_crossing_output_node_ids
        )

    def _update_parameterized_frontier(self, record: ForwardRecord) -> None:
        if not self._use_graph_tensors:
            return

        input_frontier: set[nn.Parameter] = set()
        frontier_by_node = self._state.parameterized_frontier_by_tensor_node
        for input_node_id in _record_input_node_ids(record):
            input_frontier.update(frontier_by_node.get(input_node_id, ()))

        local_channels = set(
            self._local_channels_for_parameters(
                tuple(_record_local_output_tangents(record)),
            ),
        )
        output_node_id = _record_output_node_id(record)
        if local_channels:
            if input_frontier & local_channels:
                self._state.same_block_input_crossing_output_node_ids.add(
                    output_node_id,
                )
            frontier_by_node[output_node_id] = local_channels
            return

        if input_frontier:
            frontier_by_node[output_node_id] = input_frontier

    def _make_forward_record_hook(self, record: ForwardRecord) -> Any:
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
        output_grad_tangents = _apply_output_curvature_packet(
            output_curvature,
            tangents_by_node.pop(graph.output_node_id, {}),
        )
        primal_grad_required_output_node_ids = {
            _record_output_node_id(record)
            for record in graph.records
            if self._record_requires_primal_grad_for_current_graph(record)
        }
        local_parameters_by_output_node = _take_graph_local_parameters_by_output_node(
            graph.records
        )
        self._state.graph.prepare_hooked_backward(
            output_node_id=graph.output_node_id,
            output_grad_tangents=output_grad_tangents,
            retained_forward_tangents_by_node=tangents_by_node,
            local_parameters_by_output_node=local_parameters_by_output_node,
            remaining_local_uses_by_parameter=self._local_channel_use_counts(
                local_parameters_by_output_node,
            ),
            input_use_counts=graph.fresh_input_use_counts(),
            primal_grad_required_output_node_ids=primal_grad_required_output_node_ids,
        )
        self._drain_ready_graph_backward_records()

    def _compute_graph_forward_tangent_packets(
        self,
        graph: RecordedForwardGraph,
        *,
        retained_node_ids: set[int] | frozenset[int] | None = None,
    ) -> dict[int, dict[nn.Parameter, torch.Tensor]]:
        tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            node_id: self._parameter_packet_to_channel_packet(packet)
            for node_id, packet in self._state.raw_parameter_tangents_by_node.items()
        }
        retained_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {}
        retained_channels_by_node = graph.retained_forward_tangent_channels_by_node

        def retain_packet_if_needed(
            node_id: int,
            packet: dict[nn.Parameter, torch.Tensor],
        ) -> None:
            if retained_node_ids is None or node_id not in retained_node_ids:
                return
            retained_channels = retained_channels_by_node.get(node_id)
            if retained_channels is None:
                retained_tangents_by_node[node_id] = packet
                return
            filtered_packet = {
                channel: tangent_value
                for channel, tangent_value in packet.items()
                if channel in retained_channels
            }
            if filtered_packet:
                retained_tangents_by_node[node_id] = filtered_packet

        if retained_node_ids is not None:
            for node_id, packet in tangents_by_node.items():
                retain_packet_if_needed(node_id, packet)
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
                for channel, tangent_value in self._local_tangent_channel_packet(
                    local_tangents,
                ).items():
                    _accumulate_parameter_tensor(
                        record_tangents,
                        channel,
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
                retain_packet_if_needed(_record_output_node_id(record), node_tangents)
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
            requires_primal_grad = graph_state.requires_primal_grad(record)
            primal_grad = (
                graph_state.primal_grad(record) if requires_primal_grad else None
            )
            self._accumulate_graph_record_hvps(
                record,
                output_grad_tangents,
                primal_grad=primal_grad,
                forward_tangents_by_node=graph_state.forward_tangents_by_node,
            )
            blocked_local_channels = self._prune_consumed_local_channels(
                record,
                local_parameters,
                output_grad_tangents,
                graph_state,
            )

            has_backward_nonlinearity_tangents = (
                _record_has_backward_nonlinearity_tangents(
                    record,
                    graph_state.forward_tangents_by_node,
                )
            )
            if requires_primal_grad and (
                output_grad_tangents or has_backward_nonlinearity_tangents
            ):
                if primal_grad is None:
                    raise RuntimeError("missing required primal grad for graph record")
                _propagate_backward_tangent_packet_with_grad(
                    record,
                    primal_grad,
                    output_grad_tangents,
                    graph_state.forward_tangents_by_node,
                    graph_state.grad_tangents_by_node,
                    self._tangents_by_parameter,
                    self._block_channel_by_parameter,
                    blocked_local_channels,
                )
            elif output_grad_tangents:
                _propagate_backward_tangent_packet(
                    record,
                    output_grad_tangents,
                    graph_state.grad_tangents_by_node,
                )

            graph_state.finish_record(graph, record)

        if graph_state.is_complete(graph):
            self._accumulate_raw_parameter_graph_hvps(graph_state.grad_tangents_by_node)
            for block in self.parameter_blocks:
                if getattr(block.parameter, "hvp", None) is None:
                    setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))
            self._state.eager_backward_active = False
            self._clear_completed_graph_bookkeeping()

    def _compute_graph_hvps_single_pass(self, graph: RecordedForwardGraph) -> None:
        output_curvature = self._state.output_curvature
        if output_curvature is None:
            raise RuntimeError("graph HVP requested before loss initialization")

        tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            node_id: self._parameter_packet_to_channel_packet(packet)
            for node_id, packet in self._state.raw_parameter_tangents_by_node.items()
        }
        tangent_channels_by_node: dict[int, set[nn.Parameter]] = {
            node_id: set(packet) for node_id, packet in tangents_by_node.items() if packet
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
                for channel, tangent_value in self._local_tangent_channel_packet(
                    local_tangents,
                ).items():
                    _accumulate_parameter_tensor(
                        record_tangents,
                        channel,
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
                tangent_channels_by_node[_record_output_node_id(record)] = set(
                    node_tangents
                )

        output_tangents = tangents_by_node.get(graph.output_node_id, {})
        tangents_by_node.clear()
        grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            graph.output_node_id: _apply_output_curvature_packet(
                output_curvature,
                output_tangents,
            )
        }

        local_parameters_by_output_node = {
            _record_output_node_id(record): tuple(_record_local_output_tangents(record))
            for record in graph.records
            if _record_local_output_tangents(record)
        }
        remaining_local_uses_by_channel = self._local_channel_use_counts(
            local_parameters_by_output_node,
        )
        self._state.graph.local_parameters_by_output_node = local_parameters_by_output_node

        for record in graph.reverse_records:
            try:
                output_grad_tangents = grad_tangents_by_node.pop(
                    _record_output_node_id(record),
                    None,
                )
                if not output_grad_tangents:
                    _record_local_output_tangents(record).clear()
                    continue

                local_parameters = self._state.graph.local_parameters_for(record)
                self._accumulate_graph_record_hvps(record, output_grad_tangents)
                for channel in self._local_channels_for_parameters(local_parameters):
                    remaining = remaining_local_uses_by_channel.get(channel, 0)
                    if remaining <= 1:
                        remaining_local_uses_by_channel.pop(channel, None)
                        output_grad_tangents.pop(channel, None)
                    else:
                        remaining_local_uses_by_channel[channel] = remaining - 1
                if not output_grad_tangents:
                    continue

                _propagate_backward_tangent_packet(
                    record,
                    output_grad_tangents,
                    grad_tangents_by_node,
                    tangent_channels_by_node,
                )
            finally:
                _release_record_saved_tensors(record)

        self._accumulate_raw_parameter_graph_hvps(grad_tangents_by_node)

        for block in self.parameter_blocks:
            if getattr(block.parameter, "hvp", None) is None:
                setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))
        self._state.graph.local_parameters_by_output_node.clear()
        self._clear_completed_graph_bookkeeping()

    def _clear_completed_graph_bookkeeping(self) -> None:
        self._state.forward_records.clear()
        self._state.active_graph = None
        self._state.output_curvature = None
        self._state.raw_parameter_tangents_by_node.clear()
        self._state.same_block_input_tangent_node_ids_by_channel.clear()
        self._state.parameterized_frontier_by_tensor_node.clear()
        self._state.same_block_input_crossing_output_node_ids.clear()
        self._state.curvatures_by_node_id.clear()
        self._state.graph.clear()

    def _parameter_packet_to_channel_packet(
        self,
        packet: Mapping[nn.Parameter, torch.Tensor],
    ) -> dict[nn.Parameter, torch.Tensor]:
        channel_packet: dict[nn.Parameter, torch.Tensor] = {}
        for parameter, tangent_value in packet.items():
            _accumulate_parameter_tensor(
                channel_packet,
                self._block_channel_by_parameter[parameter],
                tangent_value,
            )
        return channel_packet

    def _local_tangent_channel_packet(
        self,
        local_tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> dict[nn.Parameter, torch.Tensor]:
        channel_packet: dict[nn.Parameter, torch.Tensor] = {}
        for parameter, tangent_value in local_tangents.items():
            channel = self._block_channel_by_parameter[parameter]
            if channel in channel_packet:
                continue
            channel_packet[channel] = tangent_value
        return channel_packet

    def _local_channel_use_counts(
        self,
        local_parameters_by_output_node: Mapping[int, tuple[nn.Parameter, ...]],
    ) -> dict[nn.Parameter, int]:
        counts: dict[nn.Parameter, int] = {}
        for parameters in local_parameters_by_output_node.values():
            for channel in self._local_channels_for_parameters(parameters):
                counts[channel] = counts.get(channel, 0) + 1
        return counts

    def _local_channels_for_parameters(
        self,
        parameters: tuple[nn.Parameter, ...],
    ) -> tuple[nn.Parameter, ...]:
        channels: list[nn.Parameter] = []
        seen: set[nn.Parameter] = set()
        for parameter in parameters:
            channel = self._block_channel_by_parameter[parameter]
            if channel in seen:
                continue
            seen.add(channel)
            channels.append(channel)
        return tuple(channels)

    def _prune_consumed_local_channels(
        self,
        record: ForwardRecord,
        local_parameters: tuple[nn.Parameter, ...],
        output_grad_tangents: dict[nn.Parameter, torch.Tensor],
        graph_state: Any,
    ) -> frozenset[nn.Parameter]:
        """Stop block-local channels at parameterized input boundaries.

        A local channel may cross the input edge of this record only when the
        custom block graph proved that another parameterized node in the same
        block reaches this record's input. This preserves q/k/v-style fan-out
        locality while still allowing sequential/full blocks to carry their
        local epsilon through the within-block chain.
        """

        if not local_parameters:
            return frozenset()

        blocked_channels: set[nn.Parameter] = set()
        for channel in self._local_channels_for_parameters(local_parameters):
            if self._local_channel_may_cross_record_input(record, channel):
                if graph_state.consume_local_parameter_use(channel) == 0:
                    blocked_channels.add(channel)
                    output_grad_tangents.pop(channel, None)
                continue

            graph_state.consume_local_parameter_use(channel)
            blocked_channels.add(channel)
            output_grad_tangents.pop(channel, None)

        if not blocked_channels:
            return frozenset()
        return frozenset(
            parameter
            for parameter in local_parameters
            if self._block_channel_by_parameter[parameter] in blocked_channels
        )

    def _local_channel_may_cross_record_input(
        self,
        record: ForwardRecord,
        channel: nn.Parameter,
    ) -> bool:
        retained_input_node_ids = (
            self._state.same_block_input_tangent_node_ids_by_channel.get(channel)
        )
        if not retained_input_node_ids:
            return False
        return any(
            input_node_id in retained_input_node_ids
            for input_node_id in _record_input_node_ids(record)
        )

    def _record_requires_primal_grad_for_current_graph(
        self,
        record: ForwardRecord,
    ) -> bool:
        if isinstance(
            record,
            (
                FunctionalLayerNormForwardRecord,
                GELUForwardRecord,
                LayerNormForwardRecord,
                MatmulForwardRecord,
                MulForwardRecord,
                RMSNormForwardRecord,
                ScaledDotProductAttentionForwardRecord,
                SoftmaxForwardRecord,
                UnaryElementwiseForwardRecord,
            ),
        ):
            return True
        if isinstance(record, DivForwardRecord):
            return record.right_node_id is not None
        return any(
            self._local_channel_may_cross_record_input(record, channel)
            for channel in self._local_channels_for_parameters(
                tuple(_record_local_output_tangents(record)),
            )
        )

    def _accumulate_raw_parameter_graph_hvps(
        self,
        grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
    ) -> None:
        for node_id, local_packet in self._state.raw_parameter_tangents_by_node.items():
            output_packet = grad_tangents_by_node.pop(node_id, {})
            for parameter in local_packet:
                hvp = output_packet.get(self._block_channel_by_parameter[parameter])
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
                    input_tangent = _local_record_input_tangent(
                        record,
                        channel,
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
                    input_tangent = _local_record_input_tangent(
                        record,
                        channel,
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
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
                    channel = self._block_channel_by_parameter[parameter]
                    grad_tangent = output_grad_tangents.get(channel)
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

    def _local_curvature_tangent_groups(
        self,
        record: ForwardRecord,
        curvature: Callable[[torch.Tensor], torch.Tensor],
    ) -> tuple[tuple[tuple[nn.Parameter, ...], torch.Tensor], ...]:
        """Apply downstream curvature once per local block in a record."""

        local_tangents = _record_local_output_tangents(record)
        groups: list[tuple[tuple[nn.Parameter, ...], torch.Tensor]] = []
        handled: set[nn.Parameter] = set()
        for parameter, local_output_tangent in local_tangents.items():
            if parameter in handled:
                continue
            block_group = tuple(
                group_parameter
                for group_parameter in self._block_parameters_by_parameter[parameter]
                if group_parameter in local_tangents
            )
            handled.update(block_group)
            groups.append((block_group, curvature(local_output_tangent)))
        return tuple(groups)

    def _consume_embedding_backward_record(
        self,
        *,
        record: EmbeddingForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for Embedding backward hook")
        try:
            for parameters, parameter_grad_tangent in self._local_curvature_tangent_groups(
                record,
                curvature,
            ):
                for parameter in parameters:
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
            for parameters, parameter_grad_tangent in self._local_curvature_tangent_groups(
                record,
                curvature,
            ):
                for parameter in parameters:
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
            for parameters, parameter_grad_tangent in self._local_curvature_tangent_groups(
                record,
                curvature,
            ):
                for parameter in parameters:
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
            for parameters, parameter_grad_tangent in self._local_curvature_tangent_groups(
                record,
                curvature,
            ):
                for parameter in parameters:
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
            for parameters, parameter_grad_tangent in self._local_curvature_tangent_groups(
                record,
                curvature,
            ):
                for parameter in parameters:
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
