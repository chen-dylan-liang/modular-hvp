"""Local-dual runtime for supported eager tensor graphs."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from torch import nn

from modular_hvp.dual import primal
from modular_hvp.graph_tensor import GraphTensor
from modular_hvp.losses import CrossEntropyLossRecord, MSELossRecord
from modular_hvp.model_utils import (
    _can_use_sequential_fast_path,
    _iter_raw_graph_parameters,
    _parameter_use_counts,
    _validate_supported_model,
)
from modular_hvp.runtime import (
    _direct_parameter_owners,
    _has_multi_leaf_parameter_block,
    _resolve_parameter_block_groups,
    _resolve_parameter_blocks,
)
from modular_hvp.runtime_backward import BackwardRuntimeMixin
from modular_hvp.runtime_dispatch import GraphDispatchMixin
from modular_hvp.runtime_forward import ForwardRuntimeMixin
from modular_hvp.runtime_state import RuntimeState
from modular_hvp.saved_tensors import (
    _make_linear_input_activation_ref,
    _make_relu_output_activation_ref,
)


class EagerHVPRuntime(ForwardRuntimeMixin, GraphDispatchMixin, BackwardRuntimeMixin):
    """Default ModularHVP runtime for supported eager tensor graphs.

    This runtime consumes parameter tangents inside the owning module forward,
    saves local dual activations, and returns only primal tensors to ordinary
    model execution. Backward hooks consume those local activations when
    PyTorch reaches the matching module and accumulate into the single public
    ``p.hvp`` slot.
    """

    _ACTIVE_RUNTIME: "EagerHVPRuntime | None" = None
    _ACTIVE_ATTR = "_modular_hvp_eager_active"

    def __init__(
        self,
        *,
        model: nn.Module,
        tangents: Mapping[str | nn.Parameter, torch.Tensor],
        blocks: Mapping[Any, Iterable[str | nn.Parameter]] | None = None,
    ) -> None:
        _validate_supported_model(model)
        self.model = model
        self.parameter_blocks = _resolve_parameter_blocks(model, tangents)
        self._block_parameters_by_parameter = _resolve_parameter_block_groups(
            model,
            self.parameter_blocks,
            blocks,
        )
        self._blocks_by_parameter = {
            block.parameter: block for block in self.parameter_blocks
        }
        self._tangents_by_parameter = {
            block.parameter: block.tangent for block in self.parameter_blocks
        }
        self._block_channel_by_parameter = {
            parameter: group[0]
            for parameter, group in self._block_parameters_by_parameter.items()
        }
        direct_owners = _direct_parameter_owners(model)
        self._multi_leaf_block_parameters: set[nn.Parameter] = set()
        seen_groups: set[tuple[int, ...]] = set()
        for group in self._block_parameters_by_parameter.values():
            group_key = tuple(id(parameter) for parameter in group)
            if group_key in seen_groups:
                continue
            seen_groups.add(group_key)
            owners = {direct_owners.get(parameter) for parameter in group}
            if len(owners) > 1:
                self._multi_leaf_block_parameters.update(group)
        self._parameter_use_counts = _parameter_use_counts(model)
        self._has_reused_parameters = any(
            count > 1 for count in self._parameter_use_counts.values()
        )
        self._requires_graph_block_scope = _has_multi_leaf_parameter_block(
            model,
            self._block_parameters_by_parameter,
        )
        self._use_graph_tensors = (
            self._has_reused_parameters
            or self._requires_graph_block_scope
            or not _can_use_sequential_fast_path(model)
        )
        self._raw_graph_parameters = (
            _iter_raw_graph_parameters(model) if self._use_graph_tensors else ()
        )
        self._state = RuntimeState()

    def __enter__(self) -> "EagerHVPRuntime":
        if self._state.entered:
            raise RuntimeError("modular_hvp contexts are single-use")
        if EagerHVPRuntime._ACTIVE_RUNTIME is not None:
            raise RuntimeError("another modular_hvp context is already active")
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                raise RuntimeError("this model already has an active modular_hvp context")

        self._state.entered = True
        self._clear_hvp_slots()
        self._install_forward_wrappers()
        self._install_loss_patch()
        for module in self.model.modules():
            setattr(module, self._ACTIVE_ATTR, True)
        EagerHVPRuntime._ACTIVE_RUNTIME = self
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if EagerHVPRuntime._ACTIVE_RUNTIME is self:
            EagerHVPRuntime._ACTIVE_RUNTIME = None
        self._restore_raw_parameter_graph_sources()
        self._restore_loss_patch()
        self._restore_forward_wrappers()
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                delattr(module, self._ACTIVE_ATTR)
        self._state.loss_record = None
        self._state.primal_loss = None
        self._state.output_id = None
        self._state.output_node_id = None
        self._state.next_node_id = 0
        self._state.loss_factor = None
        self._state.output_curvature = None
        self._state.raw_parameter_patches.clear()
        self._state.raw_parameter_tangents_by_node.clear()
        self._state.forward_records.clear()
        self._state.active_graph = None
        self._state.same_block_input_tangent_node_ids_by_channel.clear()
        self._state.parameterized_frontier_by_tensor_node.clear()
        self._state.same_block_input_crossing_output_node_ids.clear()
        self._state.curvatures_by_node_id.clear()
        self._state.graph.clear()
        self._state.use_graph_curvature = False
        self._state.eager_backward_active = False
        return None

    def record_mse_loss(
        self,
        *,
        input_value: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        input_primal = self._unwrap_graph_value(input_value)
        target_primal = primal(target)
        loss_patch = self._state.loss_patch
        if loss_patch is None:
            raise RuntimeError("mse_loss was observed before the loss patch was installed")
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("mse_loss input must be a tensor")
        primal_loss = loss_patch.original_mse_loss(
            input_primal,
            target_primal,
            reduction=reduction,
        )
        if reduction not in {"mean", "sum"}:
            raise NotImplementedError(
                "modular_hvp currently supports mse_loss reductions 'mean' and 'sum'"
            )
        self._state.loss_record = MSELossRecord(
            input_numel=input_primal.numel(),
            device=input_primal.device,
            dtype=input_primal.dtype,
            reduction=reduction,
        )
        self._state.primal_loss = primal_loss
        self._state.output_id = id(primal_loss)
        if isinstance(input_value, GraphTensor):
            self._state.output_node_id = input_value.node_id
        primal_loss.register_hook(self._make_loss_hook())
        return primal_loss

    def record_cross_entropy_loss(
        self,
        *,
        input_value: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None,
        ignore_index: int,
        reduction: str,
        label_smoothing: float,
    ) -> torch.Tensor:
        if weight is not None:
            raise NotImplementedError(
                "modular_hvp currently supports cross_entropy without class weights"
            )
        if label_smoothing != 0.0:
            raise NotImplementedError(
                "modular_hvp currently supports cross_entropy with label_smoothing=0"
            )
        if reduction not in {"mean", "sum"}:
            raise NotImplementedError(
                "modular_hvp currently supports cross_entropy reductions 'mean' and 'sum'"
            )
        input_primal = self._unwrap_graph_value(input_value)
        target_primal = self._unwrap_graph_value(target)
        if not isinstance(input_primal, torch.Tensor) or not isinstance(
            target_primal,
            torch.Tensor,
        ):
            raise NotImplementedError("cross_entropy input and target must be tensors")
        loss_patch = self._state.loss_patch
        if loss_patch is None:
            raise RuntimeError("cross_entropy was observed before the loss patch was installed")
        primal_loss = loss_patch.original_cross_entropy(
            input_primal,
            target_primal,
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
        self._state.loss_record = CrossEntropyLossRecord(
            logits=input_primal.detach(),
            target=target_primal.detach(),
            reduction=reduction,
            ignore_index=ignore_index,
        )
        self._state.primal_loss = primal_loss
        self._state.output_id = id(primal_loss)
        input_node_id = self._node_id(input_value)
        if input_node_id is not None:
            self._state.output_node_id = input_node_id
        primal_loss.register_hook(self._make_loss_hook())
        return primal_loss

    def _clear_hvp_slots(self) -> None:
        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", None)
