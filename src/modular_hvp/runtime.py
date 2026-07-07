"""Scoped hook runtime for ModularHVP."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from modular_hvp.backend import DualBackend, LocalDualActivations
from modular_hvp.model_utils import (
    _can_use_sequential_fast_path,
    _is_supported_leaf_module,
)


@dataclass(frozen=True, slots=True)
class ParameterBlock:
    """Initial per-parameter block state."""

    name: str
    parameter: nn.Parameter
    tangent: torch.Tensor


@dataclass(frozen=True, slots=True)
class ForwardRecord:
    """Saved runtime state for one module call."""

    local_dual_activations: LocalDualActivations


@dataclass(slots=True)
class ForwardPatch:
    """Original forward state needed to restore a module after the context."""

    module: nn.Module
    original_forward: Any
    had_instance_forward: bool
    instance_forward: Any


@dataclass(slots=True)
class RuntimeState:
    """Mutable state owned by one context-manager invocation."""

    entered: bool = False
    forward_patches: list[ForwardPatch] = field(default_factory=list)
    backward_handles: list[RemovableHandle] = field(default_factory=list)
    records_by_module_id: dict[int, list[ForwardRecord]] = field(
        default_factory=lambda: defaultdict(list)
    )


class ModularHVPRuntime:
    """Context manager that wires a model into the ModularHVP runtime."""

    _ACTIVE_ATTR = "_modular_hvp_runtime_active"

    def __init__(
        self,
        *,
        model: nn.Module,
        tangents: Mapping[str | nn.Parameter, torch.Tensor],
        backend: DualBackend,
    ) -> None:
        self.model = model
        self.backend = backend
        self.parameter_blocks = _resolve_parameter_blocks(model, tangents)
        self._blocks_by_parameter = {
            block.parameter: block for block in self.parameter_blocks
        }
        self._module_tangents = _collect_module_tangents(model, self._blocks_by_parameter)
        self._state = RuntimeState()

    def __enter__(self) -> "ModularHVPRuntime":
        if self._state.entered:
            raise RuntimeError("modular_hvp contexts are single-use")
        self._state.entered = True

        self._ensure_model_is_not_already_active()
        self._clear_hvp_slots()
        self._mark_model_active()
        self._install_forward_wrappers()
        self._install_backward_hooks()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._remove_backward_hooks()
        self._restore_forward_wrappers()
        self._clear_active_flags()
        self._state.records_by_module_id.clear()
        return None

    def _ensure_model_is_not_already_active(self) -> None:
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                raise RuntimeError("this model already has an active modular_hvp context")

    def _clear_hvp_slots(self) -> None:
        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", None)

    def _mark_model_active(self) -> None:
        for module in self.model.modules():
            setattr(module, self._ACTIVE_ATTR, True)

    def _install_forward_wrappers(self) -> None:
        for module in self.model.modules():
            if id(module) not in self._module_tangents:
                continue
            original_forward = module.forward
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")

            def wrapped_forward(
                *args: Any,
                __module: nn.Module = module,
                __original_forward: Any = original_forward,
                **kwargs: Any,
            ) -> Any:
                return self._run_wrapped_forward(
                    module=__module,
                    original_forward=__original_forward,
                    args=args,
                    kwargs=kwargs,
                )

            self._state.forward_patches.append(
                ForwardPatch(
                    module=module,
                    original_forward=original_forward,
                    had_instance_forward=had_instance_forward,
                    instance_forward=instance_forward,
                )
            )
            module.forward = wrapped_forward  # type: ignore[method-assign]

    def _install_backward_hooks(self) -> None:
        for module in self.model.modules():
            if id(module) not in self._module_tangents:
                continue
            handle = module.register_full_backward_hook(self._make_backward_hook(module))
            self._state.backward_handles.append(handle)

    def _run_wrapped_forward(
        self,
        *,
        module: nn.Module,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        active_param_tangents = self._module_tangents.get(id(module), {})
        primal_output, local_dual_activations = self.backend.local_forward(
            module=module,
            original_forward=original_forward,
            args=args,
            kwargs=kwargs,
            active_param_tangents=active_param_tangents,
        )

        if active_param_tangents:
            self._state.records_by_module_id[id(module)].append(
                ForwardRecord(local_dual_activations=local_dual_activations)
            )
        return primal_output

    def _make_backward_hook(self, module: nn.Module) -> Any:
        def backward_hook(
            hooked_module: nn.Module,
            grad_input: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor | None, ...],
        ) -> None:
            self._run_backward_hook(
                module=hooked_module,
                grad_input=grad_input,
                grad_output=grad_output,
            )
            return None

        return backward_hook

    def _run_backward_hook(
        self,
        *,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        active_param_tangents = self._module_tangents[id(module)]
        records = self._state.records_by_module_id[id(module)]
        if not records:
            raise RuntimeError(
                f"backward for {module.__class__.__name__} has no saved local dual activations"
            )
        record = records.pop()

        hvp_by_parameter = self.backend.dual_backward(
            module=module,
            local_dual_activations=record.local_dual_activations,
            active_param_tangents=active_param_tangents,
            grad_input=grad_input,
            grad_output=grad_output,
        )
        self._accumulate_hvps(
            hvp_by_parameter,
            expected_parameters=active_param_tangents.keys(),
        )

    def _accumulate_hvps(
        self,
        hvp_by_parameter: Mapping[nn.Parameter, torch.Tensor],
        *,
        expected_parameters: Iterable[nn.Parameter],
    ) -> None:
        expected_by_id = {id(parameter): parameter for parameter in expected_parameters}
        returned_by_id = {id(parameter): parameter for parameter in hvp_by_parameter}
        if expected_by_id.keys() != returned_by_id.keys():
            missing = expected_by_id.keys() - returned_by_id.keys()
            extra = returned_by_id.keys() - expected_by_id.keys()
            details: list[str] = []
            if missing:
                missing_names = ", ".join(
                    repr(self._blocks_by_parameter[expected_by_id[parameter_id]].name)
                    for parameter_id in missing
                )
                details.append(f"missing {missing_names}")
            if extra:
                details.append(f"extra {len(extra)} inactive parameter(s)")
            raise RuntimeError(
                "backend returned HVPs for the wrong parameter set: "
                + "; ".join(details)
            )

        with torch.no_grad():
            for parameter, hvp in hvp_by_parameter.items():
                if parameter not in self._blocks_by_parameter:
                    raise RuntimeError("backend returned an HVP for an inactive parameter")
                if hvp.shape != parameter.shape:
                    raise RuntimeError(
                        f"backend returned HVP with shape {tuple(hvp.shape)} for "
                        f"parameter {self._blocks_by_parameter[parameter].name}; "
                        f"expected {tuple(parameter.shape)}"
                    )
                if hvp.device != parameter.device:
                    raise RuntimeError(
                        f"backend returned HVP on {hvp.device} for parameter "
                        f"{self._blocks_by_parameter[parameter].name}; "
                        f"expected {parameter.device}"
                    )
                if hvp.dtype != parameter.dtype:
                    raise RuntimeError(
                        f"backend returned HVP with dtype {hvp.dtype} for parameter "
                        f"{self._blocks_by_parameter[parameter].name}; "
                        f"expected {parameter.dtype}"
                    )

                hvp = hvp.detach()
                existing = getattr(parameter, "hvp", None)
                if existing is None:
                    setattr(parameter, "hvp", hvp.clone(memory_format=torch.preserve_format))
                else:
                    existing.add_(hvp)

    def _remove_backward_hooks(self) -> None:
        for handle in reversed(self._state.backward_handles):
            handle.remove()
        self._state.backward_handles.clear()

    def _restore_forward_wrappers(self) -> None:
        for patch in reversed(self._state.forward_patches):
            if patch.had_instance_forward:
                patch.module.forward = patch.instance_forward  # type: ignore[method-assign]
            else:
                delattr(patch.module, "forward")
        self._state.forward_patches.clear()

    def _clear_active_flags(self) -> None:
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                delattr(module, self._ACTIVE_ATTR)


def _resolve_parameter_blocks(
    model: nn.Module,
    tangents: Mapping[str | nn.Parameter, torch.Tensor],
) -> tuple[ParameterBlock, ...]:
    named_parameters = tuple(model.named_parameters())
    trainable_parameters = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    )
    parameters_by_name = {name: parameter for name, parameter in named_parameters}
    parameter_ids = {id(parameter): name for name, parameter in named_parameters}

    tangent_by_parameter: dict[nn.Parameter, torch.Tensor] = {}
    seen_keys: set[str | int] = set()
    for key, tangent in tangents.items():
        if not isinstance(tangent, torch.Tensor):
            raise TypeError(f"tangent for {key!r} must be a torch.Tensor")

        if isinstance(key, str):
            if key not in parameters_by_name:
                raise KeyError(f"unknown parameter name in tangent mapping: {key!r}")
            parameter = parameters_by_name[key]
            duplicate_key: str | int = key
        elif isinstance(key, nn.Parameter):
            if id(key) not in parameter_ids:
                raise KeyError("tangent mapping contains a parameter outside the model")
            parameter = key
            duplicate_key = id(key)
        else:
            raise TypeError(
                "tangent keys must be parameter names or torch.nn.Parameter objects"
            )

        if duplicate_key in seen_keys or parameter in tangent_by_parameter:
            name = parameter_ids[id(parameter)]
            raise ValueError(f"duplicate tangent provided for parameter {name!r}")
        seen_keys.add(duplicate_key)
        tangent_by_parameter[parameter] = tangent

    blocks: list[ParameterBlock] = []
    missing: list[str] = []
    for name, parameter in trainable_parameters:
        tangent = tangent_by_parameter.get(parameter)
        if tangent is None:
            missing.append(name)
            continue
        _validate_tangent(name=name, parameter=parameter, tangent=tangent)
        blocks.append(ParameterBlock(name=name, parameter=parameter, tangent=tangent))

    if missing:
        missing_text = ", ".join(repr(name) for name in missing)
        raise ValueError(f"missing tangents for trainable parameters: {missing_text}")

    extra_frozen = [
        parameter_ids[id(parameter)]
        for parameter in tangent_by_parameter
        if not parameter.requires_grad
    ]
    if extra_frozen:
        frozen_text = ", ".join(repr(name) for name in extra_frozen)
        raise ValueError(f"tangents were provided for frozen parameters: {frozen_text}")

    return tuple(blocks)


def _resolve_parameter_block_groups(
    model: nn.Module,
    parameter_blocks: Iterable[ParameterBlock],
    blocks: Mapping[Any, Iterable[str | nn.Parameter]] | None,
) -> dict[nn.Parameter, tuple[nn.Parameter, ...]]:
    """Resolve the user block partition into one group tuple per parameter."""

    parameter_blocks_tuple = tuple(parameter_blocks)
    trainable_parameters = tuple(block.parameter for block in parameter_blocks_tuple)
    if blocks is None:
        return {parameter: (parameter,) for parameter in trainable_parameters}

    parameters_by_name = dict(model.named_parameters())
    parameter_names_by_id = {
        id(parameter): name for name, parameter in parameters_by_name.items()
    }
    trainable_by_id = {id(parameter): parameter for parameter in trainable_parameters}

    block_groups: list[tuple[nn.Parameter, ...]] = []
    seen_parameter_ids: set[int] = set()
    for block_key, members in blocks.items():
        group: list[nn.Parameter] = []
        seen_in_group: set[int] = set()
        for member in members:
            parameter = _resolve_block_member_parameter(
                member,
                parameters_by_name=parameters_by_name,
                parameter_names_by_id=parameter_names_by_id,
            )
            parameter_id = id(parameter)
            if parameter_id not in trainable_by_id:
                name = parameter_names_by_id[parameter_id]
                raise ValueError(
                    f"block {block_key!r} includes frozen parameter {name!r}"
                )
            if parameter_id in seen_in_group:
                name = parameter_names_by_id[parameter_id]
                raise ValueError(
                    f"block {block_key!r} includes parameter {name!r} more than once"
                )
            if parameter_id in seen_parameter_ids:
                name = parameter_names_by_id[parameter_id]
                raise ValueError(
                    f"parameter {name!r} appears in more than one block"
                )
            seen_in_group.add(parameter_id)
            seen_parameter_ids.add(parameter_id)
            group.append(parameter)
        if not group:
            raise ValueError(f"block {block_key!r} is empty")
        block_groups.append(tuple(group))

    missing = [
        parameter_names_by_id[id(parameter)]
        for parameter in trainable_parameters
        if id(parameter) not in seen_parameter_ids
    ]
    if missing:
        missing_text = ", ".join(repr(name) for name in missing)
        raise ValueError(f"custom blocks do not cover trainable parameters: {missing_text}")

    _validate_first_pass_block_groups(model, block_groups, parameter_names_by_id)
    return {
        parameter: group
        for group in block_groups
        for parameter in group
    }


def _has_multi_leaf_parameter_block(
    model: nn.Module,
    block_parameters_by_parameter: Mapping[nn.Parameter, tuple[nn.Parameter, ...]],
) -> bool:
    """Return whether any custom block spans more than one direct leaf owner."""

    direct_owners = _direct_parameter_owners(model)
    seen_groups: set[tuple[int, ...]] = set()
    for group in block_parameters_by_parameter.values():
        group_key = tuple(id(parameter) for parameter in group)
        if group_key in seen_groups:
            continue
        seen_groups.add(group_key)
        owners = {direct_owners.get(parameter) for parameter in group}
        if len(owners) > 1:
            return True
    return False


def _resolve_block_member_parameter(
    member: str | nn.Parameter,
    *,
    parameters_by_name: Mapping[str, nn.Parameter],
    parameter_names_by_id: Mapping[int, str],
) -> nn.Parameter:
    if isinstance(member, str):
        if member not in parameters_by_name:
            raise KeyError(f"unknown parameter name in block mapping: {member!r}")
        return parameters_by_name[member]
    if isinstance(member, nn.Parameter):
        if id(member) not in parameter_names_by_id:
            raise KeyError("block mapping contains a parameter outside the model")
        return member
    raise TypeError("block members must be parameter names or torch.nn.Parameter objects")


def _validate_first_pass_block_groups(
    model: nn.Module,
    block_groups: Iterable[tuple[nn.Parameter, ...]],
    parameter_names_by_id: Mapping[int, str],
) -> None:
    """Validate supported custom block shapes."""

    grouped_blocks = [group for group in block_groups if len(group) > 1]
    if not grouped_blocks:
        return
    if not (_can_use_sequential_fast_path(model) or _is_supported_leaf_module(model)):
        raise NotImplementedError(
            "custom module-wise blocks currently require a supported nn.Sequential "
            "model or one supported leaf module"
        )

    direct_owners = _direct_parameter_owners(model)
    parameter_order = {
        parameter: index
        for index, parameter in enumerate(dict(model.named_parameters()).values())
    }

    for group in grouped_blocks:
        owners = {direct_owners.get(parameter) for parameter in group}
        if None in owners:
            names = ", ".join(
                repr(parameter_names_by_id[id(parameter)])
                for parameter in group
            )
            raise NotImplementedError(
                "custom module-wise blocks require directly owned parameters; "
                f"in the sequential first pass; got block {names}"
            )
        if len(owners) == 1:
            owner = next(iter(owners))
            if _is_supported_leaf_module(owner):
                continue
            names = ", ".join(
                repr(parameter_names_by_id[id(parameter)])
                for parameter in group
            )
            raise NotImplementedError(
                "custom module-wise blocks currently require grouped parameters to "
                f"belong to one supported leaf module; got block {names}"
            )
        if not _can_use_sequential_fast_path(model):
            names = ", ".join(
                repr(parameter_names_by_id[id(parameter)])
                for parameter in group
            )
            raise NotImplementedError(
                "blocks spanning multiple leaf modules currently require a supported "
                f"nn.Sequential model; got block {names}"
            )
        if not all(isinstance(owner, nn.Linear) for owner in owners):
            names = ", ".join(
                repr(parameter_names_by_id[id(parameter)])
                for parameter in group
            )
            raise NotImplementedError(
                "blocks spanning multiple leaf modules are currently supported only "
                f"for sequential Linear MLPs; got block {names}"
            )
        positions = sorted(parameter_order[parameter] for parameter in group)
        if positions != list(range(positions[0], positions[-1] + 1)):
            names = ", ".join(
                repr(parameter_names_by_id[id(parameter)])
                for parameter in group
            )
            raise NotImplementedError(
                "blocks spanning multiple leaf modules must form a contiguous "
                f"parameter span in the sequential model; got block {names}"
            )


def _direct_parameter_owners(model: nn.Module) -> dict[nn.Parameter, nn.Module]:
    direct_owners: dict[nn.Parameter, nn.Module] = {}
    for module in model.modules():
        for parameter in module.parameters(recurse=False):
            direct_owners.setdefault(parameter, module)
    return direct_owners


def _validate_tangent(
    *,
    name: str,
    parameter: nn.Parameter,
    tangent: torch.Tensor,
) -> None:
    if tangent.shape != parameter.shape:
        raise ValueError(
            f"tangent for parameter {name!r} has shape {tuple(tangent.shape)}; "
            f"expected {tuple(parameter.shape)}"
        )
    if tangent.device != parameter.device:
        raise ValueError(
            f"tangent for parameter {name!r} is on {tangent.device}; "
            f"expected {parameter.device}"
        )
    if tangent.dtype != parameter.dtype:
        raise ValueError(
            f"tangent for parameter {name!r} has dtype {tangent.dtype}; "
            f"expected {parameter.dtype}"
        )


def _collect_module_tangents(
    model: nn.Module,
    blocks_by_parameter: Mapping[nn.Parameter, ParameterBlock],
) -> dict[int, dict[nn.Parameter, torch.Tensor]]:
    module_tangents: dict[int, dict[nn.Parameter, torch.Tensor]] = {}
    for module in model.modules():
        direct_tangents: dict[nn.Parameter, torch.Tensor] = {}
        for parameter in module.parameters(recurse=False):
            block = blocks_by_parameter.get(parameter)
            if block is not None:
                direct_tangents[parameter] = block.tangent
        if direct_tangents:
            module_tangents[id(module)] = direct_tangents
    return module_tangents
