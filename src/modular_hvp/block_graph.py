"""Parameterized block-dependency validation for custom HVP partitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import combinations

import torch
from torch import nn

from modular_hvp.records import (
    ForwardRecord,
    FunctionalLinearForwardRecord,
    LinearForwardRecord,
    _record_input_node_ids,
    _record_local_output_tangents,
    _record_output_node_id,
)


@dataclass(slots=True)
class ParameterizedDependencyGraph:
    """Compressed graph whose nodes are parameterized use sites."""

    node_parameters: dict[int, tuple[nn.Parameter, ...]] = field(default_factory=dict)
    record_by_node: dict[int, ForwardRecord | None] = field(default_factory=dict)
    parameter_nodes: dict[nn.Parameter, set[int]] = field(default_factory=dict)
    edges: dict[int, set[int]] = field(default_factory=dict)
    parameterized_incoming_edges: dict[int, set[int]] = field(default_factory=dict)

    def add_node(
        self,
        *,
        parameters: tuple[nn.Parameter, ...],
        record: ForwardRecord | None,
    ) -> int:
        node_id = len(self.node_parameters)
        self.node_parameters[node_id] = parameters
        self.record_by_node[node_id] = record
        self.edges[node_id] = set()
        self.parameterized_incoming_edges[node_id] = set()
        for parameter in parameters:
            self.parameter_nodes.setdefault(parameter, set()).add(node_id)
        return node_id

    def add_edge(self, left: int, right: int) -> None:
        if left == right:
            return
        self.edges.setdefault(left, set()).add(right)
        self.edges.setdefault(right, set()).add(left)


def build_parameterized_dependency_graph(
    *,
    records: Sequence[ForwardRecord],
    raw_parameter_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
) -> ParameterizedDependencyGraph:
    """Compress a recorded tensor graph into parameterized use-site adjacency.

    The frontier stored for a tensor node contains the nearest upstream
    parameterized use sites not separated by another parameterized use site.
    Non-parameter fan-in connects those frontiers; a parameterized record then
    connects to its frontier and resets the output frontier to itself.
    """

    graph = ParameterizedDependencyGraph()
    frontier_by_tensor_node: dict[int, set[int]] = {}

    for tensor_node_id, packet in raw_parameter_tangents_by_node.items():
        raw_nodes: set[int] = set()
        for parameter in packet:
            raw_nodes.add(graph.add_node(parameters=(parameter,), record=None))
        if raw_nodes:
            _connect_all(graph, raw_nodes)
            frontier_by_tensor_node[tensor_node_id] = raw_nodes

    for record in records:
        input_frontier: set[int] = set()
        for input_node_id in _record_input_node_ids(record):
            input_frontier.update(frontier_by_tensor_node.get(input_node_id, ()))
        _connect_all(graph, input_frontier)

        local_parameters = tuple(_record_local_output_tangents(record))
        if local_parameters:
            node_id = graph.add_node(parameters=local_parameters, record=record)
            for upstream_node in input_frontier:
                graph.add_edge(node_id, upstream_node)
            graph.parameterized_incoming_edges[node_id] = set(input_frontier)
            frontier_by_tensor_node[_record_output_node_id(record)] = {node_id}
        elif input_frontier:
            frontier_by_tensor_node[_record_output_node_id(record)] = input_frontier

    return graph


def validate_parameter_blocks_on_graph(
    *,
    records: Sequence[ForwardRecord],
    raw_parameter_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
    block_parameters_by_parameter: Mapping[nn.Parameter, tuple[nn.Parameter, ...]],
    parameter_names: Mapping[nn.Parameter, str],
) -> dict[nn.Parameter, set[int]]:
    """Validate custom blocks and return per-block same-block input nodes."""

    graph = build_parameterized_dependency_graph(
        records=records,
        raw_parameter_tangents_by_node=raw_parameter_tangents_by_node,
    )
    seen_groups: set[tuple[int, ...]] = set()
    retained_input_node_ids_by_channel: dict[nn.Parameter, set[int]] = {}
    for group in block_parameters_by_parameter.values():
        group_key = tuple(id(parameter) for parameter in group)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        if len(group) <= 1:
            continue

        group_nodes: set[int] = set()
        missing: list[nn.Parameter] = []
        for parameter in group:
            nodes = graph.parameter_nodes.get(parameter)
            if not nodes:
                missing.append(parameter)
            else:
                group_nodes.update(nodes)
        if missing:
            raise NotImplementedError(
                "custom block contains parameter(s) not observed in the recorded "
                f"forward graph: {_format_parameter_names(missing, parameter_names)}"
            )
        if len(group_nodes) <= 1:
            continue
        if not _is_connected_induced_subgraph(graph, group_nodes):
            raise NotImplementedError(
                "custom block is disconnected in the parameterized dependency "
                f"graph: {_format_parameter_names(group, parameter_names)}"
            )

        unsupported_destinations = _unsupported_same_block_destinations(
            graph,
            group_nodes,
        )
        if unsupported_destinations:
            records_text = ", ".join(
                type(graph.record_by_node[node_id]).__name__
                for node_id in sorted(unsupported_destinations)
            )
            raise NotImplementedError(
                "custom block crosses a parameterized operation whose local "
                "backward-input JVP is not implemented yet. Unsupported record(s): "
                f"{records_text}; block: {_format_parameter_names(group, parameter_names)}"
            )
        for node_id in group_nodes:
            same_block_incoming = (
                graph.parameterized_incoming_edges.get(node_id, set()) & group_nodes
            )
            if not same_block_incoming:
                continue
            record = graph.record_by_node[node_id]
            if record is not None:
                channel = group[0]
                retained_input_node_ids_by_channel.setdefault(channel, set()).update(
                    _record_input_node_ids(record),
                )
    return retained_input_node_ids_by_channel


def _connect_all(
    graph: ParameterizedDependencyGraph,
    node_ids: set[int],
) -> None:
    for left, right in combinations(node_ids, 2):
        graph.add_edge(left, right)


def _is_connected_induced_subgraph(
    graph: ParameterizedDependencyGraph,
    node_ids: set[int],
) -> bool:
    start = next(iter(node_ids))
    seen = {start}
    stack = [start]
    while stack:
        node_id = stack.pop()
        for neighbor in graph.edges.get(node_id, ()):
            if neighbor not in node_ids or neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return seen == node_ids


def _unsupported_same_block_destinations(
    graph: ParameterizedDependencyGraph,
    node_ids: set[int],
) -> set[int]:
    unsupported: set[int] = set()
    for node_id in node_ids:
        same_block_incoming = (
            graph.parameterized_incoming_edges.get(node_id, set()) & node_ids
        )
        if not same_block_incoming:
            continue
        record = graph.record_by_node[node_id]
        if not isinstance(record, (LinearForwardRecord, FunctionalLinearForwardRecord)):
            unsupported.add(node_id)
    return unsupported


def _format_parameter_names(
    parameters: Sequence[nn.Parameter],
    parameter_names: Mapping[nn.Parameter, str],
) -> str:
    return ", ".join(repr(parameter_names.get(parameter, "<unnamed>")) for parameter in parameters)
