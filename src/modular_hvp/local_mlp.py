"""Local-dual runtime for Sequential MLPs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from modular_hvp.dual import (
    TangentPayload,
    _extract_tangent,
    _make_multi_dual,
    make_dual,
    primal,
    tangent,
)
from modular_hvp.runtime import ParameterBlock, _resolve_parameter_blocks


@dataclass(frozen=True, slots=True)
class LinearForwardRecord:
    module: nn.Linear
    input_primal: torch.Tensor
    input_id: int
    output_id: int
    local_dual_activations: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ReLUForwardRecord:
    input_primal: torch.Tensor
    input_id: int
    output_id: int


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


@dataclass(frozen=True, slots=True)
class MSELossRecord:
    input_primal: torch.Tensor
    target_primal: torch.Tensor
    reduction: str


@dataclass(slots=True)
class RuntimeState:
    entered: bool = False
    forward_patch: ForwardPatch | None = None
    loss_patch: LossPatch | None = None
    records: list[ForwardRecord] = field(default_factory=list)
    loss_record: MSELossRecord | None = None
    primal_loss: torch.Tensor | None = None
    backward_called: bool = False
    output_id: int | None = None
    output_tangent: TangentPayload | None = None
    grad_tangents_by_tensor_id: dict[int, TangentPayload] = field(default_factory=dict)
    eager_backward_active: bool = False


class LocalMLPHVPRuntime:
    """Default ModularHVP runtime for the current Linear/ReLU/MSE scope.

    This runtime consumes parameter tangents inside the model forward and stores
    only local output tangents for the module that owns each parameter. It does
    not propagate parameter tangents through later layers in forward.
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
        self._parameter_use_counts = _parameter_use_counts(model)
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
        self._state.loss_record = None
        self._state.primal_loss = None
        self._state.output_id = None
        self._state.output_tangent = None
        self._state.grad_tangents_by_tensor_id.clear()
        self._state.eager_backward_active = False
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
        if reduction not in {"mean", "sum"}:
            raise NotImplementedError(
                "modular_hvp currently supports mse_loss reductions 'mean' and 'sum'"
            )
        self._state.loss_record = MSELossRecord(
            input_primal=input_primal.detach(),
            target_primal=target_primal.detach(),
            reduction=reduction,
        )
        self._state.primal_loss = primal_loss
        primal_loss.register_hook(self._make_loss_hook())
        return primal_loss

    def _clear_hvp_slots(self) -> None:
        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", None)

    def _initialize_hvp_accumulators(self) -> None:
        for block in self.parameter_blocks:
            setattr(
                block.parameter,
                "hvp",
                torch.zeros_like(
                    block.parameter,
                    memory_format=torch.preserve_format,
                ),
            )

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
        self._state.output_tangent = None
        self._state.grad_tangents_by_tensor_id.clear()
        self._state.backward_called = False
        self._state.eager_backward_active = False
        output: Any = args[0]
        for module in _iter_supported_modules(self.model):
            if isinstance(module, nn.Linear):
                output = self._run_linear_forward(module, output)
            elif isinstance(module, nn.ReLU):
                output = self._run_relu_forward(module, output)
            else:
                raise NotImplementedError(
                    f"unsupported module in Sequential MLP: {module.__class__.__name__}"
                )
        output_primal = primal(output)
        self._state.output_id = id(output_primal)
        self._state.output_tangent = tangent(output)
        return output_primal

    def _run_linear_forward(
        self,
        module: nn.Linear,
        input_value: Any,
    ) -> Any:
        input_primal_live = primal(input_value)
        input_primal = input_primal_live.detach()
        input_id = id(input_primal_live)
        local_duals: dict[nn.Parameter, torch.Tensor] = {}
        dual_parameters: dict[str, torch.Tensor] = {}

        weight_tangent = self._tangents_by_parameter.get(module.weight)
        if weight_tangent is not None:
            dual_parameters["weight"] = _make_multi_dual(
                module.weight,
                {module.weight: weight_tangent.detach()},
            )

        if module.bias is not None:
            bias_tangent = self._tangents_by_parameter.get(module.bias)
            if bias_tangent is not None:
                dual_parameters["bias"] = _make_multi_dual(
                    module.bias,
                    {module.bias: bias_tangent.detach()},
                )

        if dual_parameters:
            output_hat = self._run_module_with_dual_parameters(
                module,
                dual_parameters,
                input_value,
            )
            output = primal(output_hat)
            tangent_payload = tangent(output_hat)
            for parameter in self._active_module_parameters(module):
                if parameter in self._tangents_by_parameter:
                    local_duals[parameter] = _extract_tangent(
                        tangent_payload,
                        parameter,
                    ).detach()
        else:
            output_hat = self._call_module_forward(module, input_value)
            output = primal(output_hat)

        output_id = id(output)
        record = LinearForwardRecord(
            module=module,
            input_primal=input_primal,
            input_id=input_id,
            output_id=output_id,
            local_dual_activations=local_duals,
        )
        self._state.records.append(record)
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output_hat

    def _run_relu_forward(
        self,
        module: nn.ReLU,
        input_value: Any,
    ) -> Any:
        if module.inplace:
            raise NotImplementedError("LocalMLPHVPRuntime does not support inplace ReLU")
        input_primal_live = primal(input_value)
        input_primal = input_primal_live.detach()
        output_hat = self._call_module_forward(module, input_value)
        output = primal(output_hat)
        record = ReLUForwardRecord(
            input_primal=input_primal,
            input_id=id(input_primal_live),
            output_id=id(output),
        )
        self._state.records.append(record)
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output_hat

    def _call_module_forward(
        self,
        module: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        patch = self._state.forward_patch
        if patch is not None and patch.module is module:
            return patch.original_forward(*args, **kwargs)
        return module.forward(*args, **kwargs)

    def _run_module_with_dual_parameters(
        self,
        module: nn.Module,
        dual_parameters: Mapping[str, torch.Tensor],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        original_parameters: dict[str, torch.Tensor | None] = {}
        for parameter_name, dual_parameter in dual_parameters.items():
            original_parameter = module._parameters[parameter_name]
            if original_parameter is None:
                raise RuntimeError(f"parameter {parameter_name!r} is None")
            original_parameters[parameter_name] = original_parameter
            module._parameters[parameter_name] = dual_parameter
        try:
            return self._call_module_forward(module, *args, **kwargs)
        finally:
            for parameter_name, original_parameter in original_parameters.items():
                module._parameters[parameter_name] = original_parameter

    def _active_module_parameters(self, module: nn.Module) -> tuple[nn.Parameter, ...]:
        parameters: list[nn.Parameter] = []
        if isinstance(module, nn.Linear):
            parameters.append(module.weight)
            if module.bias is not None:
                parameters.append(module.bias)
        return tuple(parameters)

    def _start_eager_backward(self, grad: torch.Tensor) -> None:
        loss_record = self._state.loss_record
        if loss_record is None:
            raise RuntimeError(
                "modular_hvp did not observe a supported scalar loss; "
                "currently use torch.nn.MSELoss or torch.nn.functional.mse_loss"
            )
        if any(count > 1 for count in self._parameter_use_counts.values()):
            self._compute_reused_parameter_hvps_autodiff()
            return

        with torch.no_grad():
            self._initialize_hvp_accumulators()
            output_id = self._state.output_id
            output_tangent = self._state.output_tangent
            if output_id is None or output_tangent is None:
                raise RuntimeError("missing saved model-output tangent")
            scale = _mse_hessian_scale(loss_record)
            grad_scale = grad.detach()
            output_grad_tangent = _map_tangent_payload(
                lambda item: grad_scale * scale * item,
                output_tangent,
            )
            self._state.grad_tangents_by_tensor_id[output_id] = output_grad_tangent
            self._state.eager_backward_active = True

    def _make_forward_record_hook(self, record: ForwardRecord) -> Any:
        def forward_record_hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._state.eager_backward_active:
                return grad
            grad_tangent = self._state.grad_tangents_by_tensor_id.pop(
                record.output_id,
                None,
            )
            if grad_tangent is None:
                return grad
            with torch.no_grad():
                if isinstance(record, LinearForwardRecord):
                    self._consume_linear_backward_record(
                        record=record,
                        grad=grad,
                        grad_tangent=grad_tangent,
                    )
                else:
                    self._consume_relu_backward_record(
                        record=record,
                        grad=grad,
                        grad_tangent=grad_tangent,
                    )
            return grad

        return forward_record_hook

    def _consume_linear_backward_record(
        self,
        *,
        record: LinearForwardRecord,
        grad: torch.Tensor,
        grad_tangent: TangentPayload,
    ) -> None:
        local_parameters = set(record.local_dual_activations)
        for parameter in record.local_dual_activations:
            parameter_grad_tangent = _extract_tangent(grad_tangent, parameter)
            if parameter is record.module.weight:
                hvp = torch.ops.aten.mm.default(
                    torch.ops.aten.t.default(parameter_grad_tangent),
                    record.input_primal,
                )
            elif parameter is record.module.bias:
                hvp = torch.ops.aten.sum.dim_IntList(
                    parameter_grad_tangent,
                    [0],
                    False,
                )
            else:
                raise RuntimeError("local dual activation belongs to wrong module")
            self._accumulate_hvp(parameter, hvp)

        upstream_tangent = _drop_tangent_payload_keys(
            grad_tangent,
            local_parameters,
        )
        if upstream_tangent is None:
            return
        grad_output_hat = _make_dual_payload(torch.zeros_like(grad), upstream_tangent)
        input_tangent_hat = _linear_input_backward_program(
            record.module.weight.detach(),
            grad_output_hat,
        )
        self._accumulate_grad_tangent(
            record.input_id,
            tangent(input_tangent_hat),
        )

    def _consume_relu_backward_record(
        self,
        *,
        record: ReLUForwardRecord,
        grad: torch.Tensor,
        grad_tangent: TangentPayload,
    ) -> None:
        grad_output_hat = _make_dual_payload(torch.zeros_like(grad), grad_tangent)
        input_tangent_hat = _relu_backward_program(record.input_primal, grad_output_hat)
        self._accumulate_grad_tangent(record.input_id, tangent(input_tangent_hat))

    def _accumulate_grad_tangent(
        self,
        tensor_id: int,
        grad_tangent: TangentPayload | None,
    ) -> None:
        if grad_tangent is None:
            return
        existing = self._state.grad_tangents_by_tensor_id.get(tensor_id)
        if existing is None:
            self._state.grad_tangents_by_tensor_id[tensor_id] = grad_tangent
        else:
            self._state.grad_tangents_by_tensor_id[tensor_id] = _add_tangent_payloads(
                existing,
                grad_tangent,
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

    def _compute_reused_parameter_hvps_autodiff(self) -> None:
        primal_loss = self._state.primal_loss
        if primal_loss is None:
            raise RuntimeError("missing saved loss for reused-parameter HVP")

        with torch.enable_grad():
            for block in self.parameter_blocks:
                gradient = torch.autograd.grad(
                    primal_loss,
                    [block.parameter],
                    create_graph=True,
                    retain_graph=True,
                    materialize_grads=True,
                )[0]
                directional_gradient = (gradient * block.tangent).sum()
                hvp = torch.autograd.grad(
                    directional_gradient,
                    [block.parameter],
                    retain_graph=True,
                    materialize_grads=True,
                )[0]
                setattr(block.parameter, "hvp", hvp.detach())


def _is_supported_mlp(model: nn.Module) -> bool:
    if isinstance(model, nn.Linear):
        return True
    if isinstance(model, nn.Sequential):
        return all(
            isinstance(module, (nn.Linear, nn.ReLU))
            for module in model._modules.values()
        )
    return False


def _iter_supported_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    if isinstance(model, nn.Linear):
        return (model,)
    if isinstance(model, nn.Sequential):
        return tuple(model._modules.values())
    raise TypeError("unsupported model type")


def _linear_forward_program(
    input_value: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    output = torch.ops.aten.mm.default(
        input_value,
        torch.ops.aten.t.default(weight),
    )
    if bias is not None:
        output = torch.ops.aten.add.Tensor(output, bias)
    return output


def _linear_backward_program(
    input_value: torch.Tensor,
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_input = _linear_input_backward_program(weight, grad_output)
    grad_weight = torch.ops.aten.mm.default(
        torch.ops.aten.t.default(grad_output),
        input_value,
    )
    grad_bias = torch.ops.aten.sum.dim_IntList(grad_output, [0], False)
    return grad_input, grad_weight, grad_bias


def _linear_input_backward_program(
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.mm.default(grad_output, weight)


def _relu_backward_program(
    input_value: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.threshold_backward.default(grad_output, input_value, 0)


def _mse_input_backward_program(
    input_value: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    return torch.ops.aten.mse_loss_backward.default(
        grad_output,
        input_value,
        target,
        _mse_reduction_enum(reduction),
    )


def _mse_reduction_enum(reduction: str) -> int:
    if reduction == "none":
        return 0
    if reduction == "mean":
        return 1
    if reduction == "sum":
        return 2
    raise ValueError(f"unknown mse_loss reduction: {reduction!r}")


def _parameter_use_counts(model: nn.Module) -> dict[nn.Parameter, int]:
    counts: dict[nn.Parameter, int] = {}
    for module in _iter_supported_modules(model):
        if not isinstance(module, nn.Linear):
            continue
        counts[module.weight] = counts.get(module.weight, 0) + 1
        if module.bias is not None:
            counts[module.bias] = counts.get(module.bias, 0) + 1
    return counts


def _primal(value: Any) -> Any:
    return value


def _mse_hessian_scale(record: MSELossRecord) -> torch.Tensor:
    if record.reduction == "mean":
        return record.input_primal.new_tensor(2.0 / record.input_primal.numel())
    if record.reduction == "sum":
        return record.input_primal.new_tensor(2.0)
    raise ValueError(f"unknown mse_loss reduction: {record.reduction!r}")


def _make_dual_payload(
    primal_value: torch.Tensor,
    payload: TangentPayload,
) -> torch.Tensor:
    if isinstance(payload, dict):
        return _make_multi_dual(primal_value, payload)
    return make_dual(primal_value, payload)


def _map_tangent_payload(
    fn: Callable[[torch.Tensor], torch.Tensor],
    payload: TangentPayload,
) -> TangentPayload:
    if isinstance(payload, dict):
        return {key: fn(value) for key, value in payload.items()}
    return fn(payload)


def _add_tangent_payloads(
    left: TangentPayload,
    right: TangentPayload,
) -> TangentPayload:
    if isinstance(left, dict) or isinstance(right, dict):
        if not isinstance(left, dict):
            left = {key: torch.zeros_like(value) for key, value in right.items()}
        if not isinstance(right, dict):
            right = {key: torch.zeros_like(value) for key, value in left.items()}
        keys = left.keys() | right.keys()
        output: dict[object, torch.Tensor] = {}
        for key in keys:
            if key in left and key in right:
                output[key] = left[key] + right[key]
            elif key in left:
                output[key] = left[key]
            else:
                output[key] = right[key]
        return output
    return left + right


def _drop_tangent_payload_keys(
    payload: TangentPayload,
    keys: set[nn.Parameter],
) -> TangentPayload | None:
    if not isinstance(payload, dict):
        return None if keys else payload
    output = {key: value for key, value in payload.items() if key not in keys}
    return output or None


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
