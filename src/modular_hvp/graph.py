"""Recorded-forward graph topology and traversal state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import nn

from modular_hvp.records import (
    DivForwardRecord,
    ForwardRecord,
    GELUForwardRecord,
    LayerNormForwardRecord,
    MatmulForwardRecord,
    MulForwardRecord,
    RMSNormForwardRecord,
    ScaledDotProductAttentionForwardRecord,
    SoftmaxForwardRecord,
    UnaryElementwiseForwardRecord,
    _record_input_node_ids,
    _record_local_output_tangents,
    _record_output_node_id,
)


@dataclass(frozen=True, slots=True)
class RecordedForwardGraph:
    """Forward records plus graph queries used by the backward dataflow pass."""

    records: tuple[ForwardRecord, ...]
    output_node_id: int
    reverse_records: tuple[ForwardRecord, ...]
    record_by_output_node: dict[int, ForwardRecord]
    input_use_counts_by_node: dict[int, int]
    retained_forward_tangent_node_ids: frozenset[int]
    requires_hooked_primal_grads: bool

    @classmethod
    def from_records(
        cls,
        *,
        records: Sequence[ForwardRecord],
        output_node_id: int,
    ) -> "RecordedForwardGraph":
        records_tuple = tuple(records)
        return cls(
            records=records_tuple,
            output_node_id=output_node_id,
            reverse_records=tuple(reversed(records_tuple)),
            record_by_output_node={
                _record_output_node_id(record): record for record in records_tuple
            },
            input_use_counts_by_node=_graph_input_use_counts(records_tuple),
            retained_forward_tangent_node_ids=frozenset(
                _graph_forward_tangent_retained_node_ids(
                    records_tuple,
                    output_node_id,
                )
            ),
            requires_hooked_primal_grads=_graph_requires_primal_backward_grad(
                records_tuple,
            ),
        )

    def fresh_input_use_counts(self) -> dict[int, int]:
        return self.input_use_counts_by_node.copy()


@dataclass(slots=True)
class GraphTraversalState:
    """Mutable state for one reverse traversal over a recorded forward DAG."""

    forward_tangents_by_node: dict[
        int,
        dict[nn.Parameter, torch.Tensor],
    ] = field(default_factory=dict)
    grad_tangents_by_node: dict[
        int,
        dict[nn.Parameter, torch.Tensor],
    ] = field(default_factory=dict)
    primal_grads_by_node: dict[int, torch.Tensor] = field(default_factory=dict)
    pending_consumers_by_node: dict[int, int] = field(default_factory=dict)
    processed_output_node_ids: set[int] = field(default_factory=set)
    ready_output_node_ids: list[int] = field(default_factory=list)
    local_parameters_by_output_node: dict[
        int,
        tuple[nn.Parameter, ...],
    ] = field(default_factory=dict)

    def clear(self) -> None:
        self.forward_tangents_by_node.clear()
        self.grad_tangents_by_node.clear()
        self.primal_grads_by_node.clear()
        self.pending_consumers_by_node.clear()
        self.processed_output_node_ids.clear()
        self.ready_output_node_ids.clear()
        self.local_parameters_by_output_node.clear()

    def prepare_hooked_backward(
        self,
        *,
        output_node_id: int,
        output_grad_tangents: dict[nn.Parameter, torch.Tensor],
        retained_forward_tangents_by_node: dict[
            int,
            dict[nn.Parameter, torch.Tensor],
        ],
        local_parameters_by_output_node: dict[int, tuple[nn.Parameter, ...]],
        input_use_counts: dict[int, int],
    ) -> None:
        self.clear()
        self.forward_tangents_by_node = retained_forward_tangents_by_node
        self.grad_tangents_by_node = {output_node_id: output_grad_tangents}
        self.local_parameters_by_output_node = local_parameters_by_output_node
        self.pending_consumers_by_node = input_use_counts

    def observe_primal_grad(
        self,
        graph: RecordedForwardGraph,
        record: ForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        output_node_id = _record_output_node_id(record)
        self.primal_grads_by_node[output_node_id] = grad.detach()
        self._queue_if_ready(graph, output_node_id)

    def is_ready(self, record: ForwardRecord) -> bool:
        output_node_id = _record_output_node_id(record)
        return (
            output_node_id not in self.processed_output_node_ids
            and output_node_id in self.primal_grads_by_node
            and self.pending_consumers_by_node.get(output_node_id, 0) == 0
        )

    def pop_ready_record(
        self,
        graph: RecordedForwardGraph,
    ) -> ForwardRecord | None:
        while self.ready_output_node_ids:
            output_node_id = self.ready_output_node_ids.pop()
            record = graph.record_by_output_node.get(output_node_id)
            if record is not None and self.is_ready(record):
                return record
        return None

    def pop_output_grad_tangents(
        self,
        record: ForwardRecord,
    ) -> dict[nn.Parameter, torch.Tensor]:
        return self.grad_tangents_by_node.pop(_record_output_node_id(record), {})

    def primal_grad(self, record: ForwardRecord) -> torch.Tensor:
        return self.primal_grads_by_node[_record_output_node_id(record)]

    def local_parameters_for(self, record: ForwardRecord) -> tuple[nn.Parameter, ...]:
        output_node_id = _record_output_node_id(record)
        stored = self.local_parameters_by_output_node.get(output_node_id)
        if stored is not None:
            return stored
        return tuple(_record_local_output_tangents(record))

    def finish_record(
        self,
        graph: RecordedForwardGraph,
        record: ForwardRecord,
    ) -> None:
        output_node_id = _record_output_node_id(record)
        self.processed_output_node_ids.add(output_node_id)
        self.forward_tangents_by_node.pop(output_node_id, None)
        self.local_parameters_by_output_node.pop(output_node_id, None)
        for input_node_id in _record_input_node_ids(record):
            remaining = self.pending_consumers_by_node.get(input_node_id)
            if remaining is None:
                continue
            if remaining <= 1:
                self.pending_consumers_by_node.pop(input_node_id, None)
                self._queue_if_ready(graph, input_node_id)
            else:
                self.pending_consumers_by_node[input_node_id] = remaining - 1

    def is_complete(self, graph: RecordedForwardGraph) -> bool:
        return len(self.processed_output_node_ids) == len(graph.records)

    def _queue_if_ready(
        self,
        graph: RecordedForwardGraph,
        output_node_id: int,
    ) -> None:
        record = graph.record_by_output_node.get(output_node_id)
        if record is not None and self.is_ready(record):
            self.ready_output_node_ids.append(output_node_id)


def _take_graph_local_parameters_by_output_node(
    records: Sequence[ForwardRecord],
) -> dict[int, tuple[nn.Parameter, ...]]:
    local_parameters_by_node: dict[int, tuple[nn.Parameter, ...]] = {}
    for record in records:
        local_tangents = _record_local_output_tangents(record)
        if not local_tangents:
            continue
        local_parameters_by_node[_record_output_node_id(record)] = tuple(local_tangents)
        local_tangents.clear()
    return local_parameters_by_node


def _graph_input_use_counts(records: Sequence[ForwardRecord]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        for input_node_id in _record_input_node_ids(record):
            counts[input_node_id] = counts.get(input_node_id, 0) + 1
    return counts


def _graph_forward_tangent_retained_node_ids(
    records: Sequence[ForwardRecord],
    output_node_id: int,
) -> set[int]:
    retained = {output_node_id}
    for record in records:
        if isinstance(record, (GELUForwardRecord, LayerNormForwardRecord, RMSNormForwardRecord)):
            retained.add(record.input_node_id)
        elif isinstance(record, SoftmaxForwardRecord):
            retained.add(record.output_node_id)
        elif isinstance(record, MatmulForwardRecord):
            retained.update(
                node_id
                for node_id in (record.left_node_id, record.right_node_id)
                if node_id is not None
            )
        elif isinstance(record, MulForwardRecord):
            retained.update(
                node_id
                for node_id in (record.left_node_id, record.right_node_id)
                if node_id is not None
            )
        elif isinstance(record, DivForwardRecord):
            if record.right_node_id is not None:
                retained.update(
                    node_id
                    for node_id in (record.left_node_id, record.right_node_id)
                    if node_id is not None
                )
        elif isinstance(record, UnaryElementwiseForwardRecord):
            retained.add(record.input_node_id)
        elif isinstance(record, ScaledDotProductAttentionForwardRecord):
            retained.update(
                node_id
                for node_id in (
                    record.query_node_id,
                    record.key_node_id,
                    record.value_node_id,
                )
                if node_id is not None
            )
    return retained


def _graph_requires_primal_backward_grad(records: Sequence[ForwardRecord]) -> bool:
    return any(
        isinstance(
            record,
            (
                LayerNormForwardRecord,
                RMSNormForwardRecord,
                GELUForwardRecord,
                UnaryElementwiseForwardRecord,
                MatmulForwardRecord,
                MulForwardRecord,
                DivForwardRecord,
                SoftmaxForwardRecord,
                ScaledDotProductAttentionForwardRecord,
            ),
        )
        for record in records
    )


def _record_has_backward_nonlinearity_tangents(
    record: ForwardRecord,
    forward_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
) -> bool:
    if isinstance(record, (GELUForwardRecord, LayerNormForwardRecord, RMSNormForwardRecord)):
        return bool(forward_tangents_by_node.get(record.input_node_id))
    if isinstance(record, SoftmaxForwardRecord):
        return bool(forward_tangents_by_node.get(record.output_node_id))
    if isinstance(record, MatmulForwardRecord):
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        return bool(left_packet or right_packet)
    if isinstance(record, MulForwardRecord):
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        return bool(left_packet or right_packet)
    if isinstance(record, DivForwardRecord):
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        return bool(right_packet or (record.right_node_id is not None and left_packet))
    if isinstance(record, UnaryElementwiseForwardRecord):
        return bool(forward_tangents_by_node.get(record.input_node_id))
    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query_packet = (
            forward_tangents_by_node.get(record.query_node_id)
            if record.query_node_id is not None
            else None
        )
        key_packet = (
            forward_tangents_by_node.get(record.key_node_id)
            if record.key_node_id is not None
            else None
        )
        value_packet = (
            forward_tangents_by_node.get(record.value_node_id)
            if record.value_node_id is not None
            else None
        )
        return bool(query_packet or key_packet or value_packet)
    return False
