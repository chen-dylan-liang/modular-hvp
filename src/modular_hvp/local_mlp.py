"""Local-dual runtime for Sequential MLPs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from modular_hvp.dual import make_dual, tangent
from modular_hvp.runtime import ParameterBlock, _resolve_parameter_blocks


@dataclass(frozen=True, slots=True)
class LinearForwardRecord:
    module: nn.Linear
    input_primal: torch.Tensor
    local_dual_activations: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ReLUForwardRecord:
    mask: torch.Tensor


ForwardRecord = LinearForwardRecord | ReLUForwardRecord


@dataclass(slots=True)
class ForwardPatch:
    module: nn.Module
    original_forward: Any
    had_instance_forward: bool
    instance_forward: Any


@dataclass(slots=True)
class LossPatch:
    original_mse_loss: Callable[..., torch.Tensor]


@dataclass(slots=True)
class RuntimeState:
    entered: bool = False
    forward_patch: ForwardPatch | None = None
    loss_patch: LossPatch | None = None
    records: list[ForwardRecord] = field(default_factory=list)
    loss_curvature: tuple[str, float] | None = None
    backward_called: bool = False
    output_id: int | None = None


class LocalMLPHVPRuntime:
    """Default ModularHVP runtime for the current Linear/ReLU/MSE scope.

    This runtime consumes parameter tangents inside the model forward and stores
    only local output tangents for the module that owns each parameter. It does
    not propagate parameter tangent channels through later layers in forward.
    """

    _ACTIVE_RUNTIME: "LocalMLPHVPRuntime | None" = None
    _ACTIVE_ATTR = "_modular_hvp_local_mlp_active"

    def __init__(
        self,
        *,
        model: nn.Module,
        tangents: Mapping[str | nn.Parameter, torch.Tensor],
    ) -> None:
        if not _is_supported_mlp(model):
            raise NotImplementedError(
                "the default modular_hvp runtime currently supports "
                "nn.Sequential MLPs composed of nn.Linear and nn.ReLU modules"
            )
        self.model = model
        self.parameter_blocks = _resolve_parameter_blocks(model, tangents)
        self._blocks_by_parameter = {
            block.parameter: block for block in self.parameter_blocks
        }
        self._tangents_by_parameter = {
            block.parameter: block.tangent for block in self.parameter_blocks
        }
        self._state = RuntimeState()

    def __enter__(self) -> "LocalMLPHVPRuntime":
        if self._state.entered:
            raise RuntimeError("modular_hvp contexts are single-use")
        if LocalMLPHVPRuntime._ACTIVE_RUNTIME is not None:
            raise RuntimeError("another modular_hvp context is already active")
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                raise RuntimeError("this model already has an active modular_hvp context")

        self._state.entered = True
        self._clear_hvp_slots()
        self._install_forward_wrapper()
        self._install_loss_patch()
        for module in self.model.modules():
            setattr(module, self._ACTIVE_ATTR, True)
        LocalMLPHVPRuntime._ACTIVE_RUNTIME = self
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if LocalMLPHVPRuntime._ACTIVE_RUNTIME is self:
            LocalMLPHVPRuntime._ACTIVE_RUNTIME = None
        self._restore_loss_patch()
        self._restore_forward_wrapper()
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                delattr(module, self._ACTIVE_ATTR)
        self._state.records.clear()
        self._state.loss_curvature = None
        self._state.output_id = None
        return None

    def record_mse_loss(
        self,
        *,
        input_value: torch.Tensor,
        target: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        input_primal = input_value
        target_primal = _primal(target)
        loss_patch = self._state.loss_patch
        if loss_patch is None:
            raise RuntimeError("mse_loss was observed before the loss patch was installed")
        primal_loss = loss_patch.original_mse_loss(
            input_primal,
            target_primal,
            reduction=reduction,
        )
        if reduction == "mean":
            coefficient = 2.0 / input_primal.numel()
        elif reduction == "sum":
            coefficient = 2.0
        else:
            raise NotImplementedError(
                "modular_hvp currently supports mse_loss reductions 'mean' and 'sum'"
            )
        self._state.loss_curvature = ("scaled_identity", coefficient)
        primal_loss.register_hook(self._make_loss_hook())
        return primal_loss

    def compute_hvps_once(self) -> None:
        if self._state.backward_called:
            return
        self._state.backward_called = True
        self._compute_hvps()

    def _clear_hvp_slots(self) -> None:
        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", None)

    def _install_forward_wrapper(self) -> None:
        original_forward = self.model.forward
        had_instance_forward = "forward" in self.model.__dict__
        instance_forward = self.model.__dict__.get("forward")

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            return self._run_model_forward(args=args, kwargs=kwargs)

        self._state.forward_patch = ForwardPatch(
            module=self.model,
            original_forward=original_forward,
            had_instance_forward=had_instance_forward,
            instance_forward=instance_forward,
        )
        self.model.forward = wrapped_forward  # type: ignore[method-assign]

    def _install_loss_patch(self) -> None:
        original_mse_loss = F.mse_loss

        def wrapped_mse_loss(
            input: torch.Tensor,
            target: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            reduction = kwargs.get("reduction", "mean")
            if args:
                # Functional mse_loss positional args are size_average, reduce,
                # reduction. Keep unsupported legacy forms on the original path.
                return original_mse_loss(input, target, *args, **kwargs)
            if id(input) != self._state.output_id:
                return original_mse_loss(input, target, **kwargs)
            return self.record_mse_loss(
                input_value=input,
                target=target,
                reduction=reduction,
            )

        self._state.loss_patch = LossPatch(original_mse_loss=original_mse_loss)
        F.mse_loss = wrapped_mse_loss

    def _restore_loss_patch(self) -> None:
        patch = self._state.loss_patch
        if patch is None:
            return
        F.mse_loss = patch.original_mse_loss
        self._state.loss_patch = None

    def _restore_forward_wrapper(self) -> None:
        patch = self._state.forward_patch
        if patch is None:
            return
        if patch.had_instance_forward:
            patch.module.forward = patch.instance_forward  # type: ignore[method-assign]
        else:
            delattr(patch.module, "forward")
        self._state.forward_patch = None

    def _run_model_forward(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        if kwargs:
            raise NotImplementedError(
                "LocalMLPHVPRuntime currently supports positional model inputs only"
            )
        if len(args) != 1 or not isinstance(args[0], torch.Tensor):
            raise NotImplementedError(
                "LocalMLPHVPRuntime currently supports a single tensor model input"
            )

        self._state.records.clear()
        self._state.output_id = None
        output = args[0]
        for module in _iter_supported_modules(self.model):
            if isinstance(module, nn.Linear):
                output = self._run_linear_forward(module, output)
            elif isinstance(module, nn.ReLU):
                output = self._run_relu_forward(module, output)
            else:
                raise NotImplementedError(
                    f"unsupported module in Sequential MLP: {module.__class__.__name__}"
                )
        self._state.output_id = id(output)
        return output

    def _run_linear_forward(
        self,
        module: nn.Linear,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        input_primal = input_value.detach()
        local_duals: dict[nn.Parameter, torch.Tensor] = {}

        weight_tangent = self._tangents_by_parameter.get(module.weight)
        if weight_tangent is not None:
            output_hat = F.linear(
                input_primal,
                make_dual(module.weight.detach(), weight_tangent.detach()),
                module.bias.detach() if module.bias is not None else None,
            )
            local_duals[module.weight] = tangent(output_hat).detach()

        if module.bias is not None:
            bias_tangent = self._tangents_by_parameter.get(module.bias)
            if bias_tangent is not None:
                output_hat = F.linear(
                    input_primal,
                    module.weight.detach(),
                    make_dual(module.bias.detach(), bias_tangent.detach()),
                )
                local_duals[module.bias] = tangent(output_hat).detach()

        output = F.linear(input_value, module.weight, module.bias)
        self._state.records.append(
            LinearForwardRecord(
                module=module,
                input_primal=input_primal,
                local_dual_activations=local_duals,
            )
        )
        return output

    def _run_relu_forward(
        self,
        module: nn.ReLU,
        input_value: torch.Tensor,
    ) -> torch.Tensor:
        if module.inplace:
            raise NotImplementedError("LocalMLPHVPRuntime does not support inplace ReLU")
        mask = input_value.detach() > 0
        output = F.relu(input_value)
        self._state.records.append(ReLUForwardRecord(mask=mask))
        return output

    def _compute_hvps(self) -> None:
        loss_curvature = self._state.loss_curvature
        if loss_curvature is None:
            raise RuntimeError(
                "modular_hvp did not observe a supported scalar loss; "
                "currently use torch.nn.MSELoss or torch.nn.functional.mse_loss"
            )
        _, coefficient = loss_curvature

        def curvature_apply(vector: torch.Tensor) -> torch.Tensor:
            return coefficient * vector

        hvps: dict[nn.Parameter, torch.Tensor] = {}
        with torch.no_grad():
            for record in reversed(self._state.records):
                if isinstance(record, ReLUForwardRecord):
                    previous_curvature_apply = curvature_apply
                    mask = record.mask

                    def relu_curvature_apply(
                        vector: torch.Tensor,
                        *,
                        mask: torch.Tensor = mask,
                        previous_curvature_apply: Any = previous_curvature_apply,
                    ) -> torch.Tensor:
                        masked = torch.where(mask, vector, vector.new_zeros(()))
                        return torch.where(
                            mask,
                            previous_curvature_apply(masked),
                            vector.new_zeros(()),
                        )

                    curvature_apply = relu_curvature_apply
                    continue

                input_primal = record.input_primal
                for parameter, local_dual in record.local_dual_activations.items():
                    curved_dual = curvature_apply(local_dual)
                    if parameter is record.module.weight:
                        hvp = curved_dual.t().matmul(input_primal)
                    elif parameter is record.module.bias:
                        hvp = curved_dual.sum(dim=0)
                    else:
                        raise RuntimeError("local dual activation belongs to wrong module")
                    hvps[parameter] = hvp

                previous_curvature_apply = curvature_apply
                weight = record.module.weight.detach()

                def linear_curvature_apply(
                    vector: torch.Tensor,
                    *,
                    weight: torch.Tensor = weight,
                    previous_curvature_apply: Any = previous_curvature_apply,
                ) -> torch.Tensor:
                    output_tangent = vector.matmul(weight.t())
                    output_curvature = previous_curvature_apply(output_tangent)
                    return output_curvature.matmul(weight)

                curvature_apply = linear_curvature_apply

            for block in self.parameter_blocks:
                hvp = hvps.get(block.parameter)
                if hvp is None:
                    hvp = torch.zeros_like(
                        block.parameter,
                        memory_format=torch.preserve_format,
                    )
                setattr(block.parameter, "hvp", hvp.detach())

    def _make_loss_hook(self) -> Any:
        def loss_hook(grad: torch.Tensor) -> torch.Tensor:
            self.compute_hvps_once()
            return grad

        return loss_hook


def _is_supported_mlp(model: nn.Module) -> bool:
    if isinstance(model, nn.Linear):
        return True
    if isinstance(model, nn.Sequential):
        return all(isinstance(module, (nn.Linear, nn.ReLU)) for module in model.children())
    return False


def _iter_supported_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    if isinstance(model, nn.Linear):
        return (model,)
    if isinstance(model, nn.Sequential):
        return tuple(model.children())
    raise TypeError("unsupported model type")


def _primal(value: Any) -> Any:
    return value


def _normalize_backward_kwargs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    names = ("gradient", "retain_graph", "create_graph", "inputs")
    normalized = dict(kwargs)
    for name, value in zip(names, args, strict=False):
        if name in normalized:
            raise TypeError(f"backward() got multiple values for argument {name!r}")
        normalized[name] = value

    unsupported: dict[str, Any] = {}
    for name, value in normalized.items():
        if name in {"gradient", "retain_graph", "inputs"}:
            if value is not None:
                unsupported[name] = value
        elif name == "create_graph":
            if value:
                unsupported[name] = value
        else:
            unsupported[name] = value
    return unsupported
