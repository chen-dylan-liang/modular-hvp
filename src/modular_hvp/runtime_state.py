"""State records for the eager ModularHVP runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from modular_hvp.graph import GraphTraversalState, RecordedForwardGraph
from modular_hvp.losses import LossPatch, LossRecord
from modular_hvp.records import ForwardRecord


@dataclass(slots=True)
class ForwardPatch:
    module: nn.Module
    original_forward: Any
    had_instance_forward: bool
    instance_forward: Any


@dataclass(slots=True)
class RawParameterPatch:
    module: nn.Module
    name: str
    parameter: nn.Parameter
    node_id: int


@dataclass(slots=True)
class RuntimeState:
    entered: bool = False
    forward_patches: list[ForwardPatch] = field(default_factory=list)
    loss_patch: LossPatch | None = None
    loss_record: LossRecord | None = None
    primal_loss: torch.Tensor | None = None
    backward_called: bool = False
    output_id: int | None = None
    output_node_id: int | None = None
    next_node_id: int = 0
    loss_factor: torch.Tensor | None = None
    output_curvature: Callable[[torch.Tensor], torch.Tensor] | None = None
    raw_parameter_patches: list[RawParameterPatch] = field(default_factory=list)
    raw_parameter_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = (
        field(default_factory=dict)
    )
    forward_records: list[ForwardRecord] = field(default_factory=list)
    active_graph: RecordedForwardGraph | None = None
    same_block_input_tangent_node_ids_by_channel: dict[
        nn.Parameter,
        set[int],
    ] = field(default_factory=dict)
    parameterized_frontier_by_tensor_node: dict[int, set[nn.Parameter]] = (
        field(default_factory=dict)
    )
    same_block_input_crossing_output_node_ids: set[int] = field(default_factory=set)
    curvatures_by_node_id: dict[int, list[Callable[[torch.Tensor], torch.Tensor]]] = (
        field(default_factory=dict)
    )
    graph: GraphTraversalState = field(default_factory=GraphTraversalState)
    use_graph_curvature: bool = False
    eager_backward_active: bool = False
