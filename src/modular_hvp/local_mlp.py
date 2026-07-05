"""Local-dual runtime for supported sequential networks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from modular_hvp.dual import (
    make_dual,
    primal,
    tangent,
)
from modular_hvp.runtime import ParameterBlock, _resolve_parameter_blocks


@dataclass(frozen=True, slots=True)
class LinearForwardRecord:
    module: nn.Linear
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class Conv2dForwardRecord:
    module: nn.Conv2d
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class BatchNorm2dForwardRecord:
    module: nn.BatchNorm2d
    input_activation: SavedTensorRef
    local_output_tangents: dict[nn.Parameter, torch.Tensor]


@dataclass(frozen=True, slots=True)
class ReLUForwardRecord:
    output_activation: SavedTensorRef


@dataclass(frozen=True, slots=True)
class FlattenForwardRecord:
    input_shape: torch.Size
    start_dim: int
    end_dim: int


@dataclass(frozen=True, slots=True)
class AvgPool2dForwardRecord:
    module: nn.AvgPool2d
    input_activation: SavedTensorRef


@dataclass(frozen=True, slots=True)
class AdaptiveAvgPool2dForwardRecord:
    input_activation: SavedTensorRef
    output_size: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MaxPool2dForwardRecord:
    module: nn.MaxPool2d
    input_activation: SavedTensorRef
    indices: SavedTensorRef


ForwardRecord = (
    LinearForwardRecord
    | Conv2dForwardRecord
    | BatchNorm2dForwardRecord
    | ReLUForwardRecord
    | FlattenForwardRecord
    | AvgPool2dForwardRecord
    | AdaptiveAvgPool2dForwardRecord
    | MaxPool2dForwardRecord
)


@dataclass(slots=True)
class SavedTensorRef:
    """One-shot access to a tensor PyTorch autograd already saved."""

    grad_fn: Any | None
    saved_attrs: tuple[str, ...]
    expected_shape: torch.Size
    fallback: torch.Tensor | None = None

    def resolve_and_release(self) -> torch.Tensor:
        try:
            if self.grad_fn is not None:
                saved = _find_saved_tensor(
                    self.grad_fn,
                    saved_attrs=self.saved_attrs,
                    expected_shape=self.expected_shape,
                )
                if saved is not None:
                    return saved
            if self.fallback is not None:
                return self.fallback
            raise RuntimeError("could not resolve saved activation from autograd graph")
        finally:
            self.grad_fn = None
            self.fallback = None


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
    input_numel: int
    device: torch.device
    dtype: torch.dtype
    reduction: str


@dataclass(slots=True)
class RuntimeState:
    entered: bool = False
    forward_patch: ForwardPatch | None = None
    loss_patch: LossPatch | None = None
    loss_record: MSELossRecord | None = None
    primal_loss: torch.Tensor | None = None
    backward_called: bool = False
    output_id: int | None = None
    curvature: Callable[[torch.Tensor], torch.Tensor] | None = None
    eager_backward_active: bool = False


class LocalMLPHVPRuntime:
    """Default ModularHVP runtime for supported single-chain models.

    This runtime consumes parameter tangents inside the owning module forward,
    saves local dual activations, and returns only primal tensors to ordinary
    model execution. Backward hooks consume those local activations when
    PyTorch reaches the matching module and accumulate into the single public
    ``p.hvp`` slot.
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
                "single-chain nn.Sequential models composed of supported "
                "Linear/CNN leaf modules"
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
        self._state.loss_record = None
        self._state.primal_loss = None
        self._state.output_id = None
        self._state.curvature = None
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
        target_primal = primal(target)
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
            input_numel=input_primal.numel(),
            device=input_primal.device,
            dtype=input_primal.dtype,
            reduction=reduction,
        )
        self._state.primal_loss = primal_loss
        primal_loss.register_hook(self._make_loss_hook())
        return primal_loss

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

        self._state.output_id = None
        self._state.curvature = None
        self._state.backward_called = False
        self._state.eager_backward_active = False
        output: Any = args[0]
        for module in _iter_supported_modules(self.model):
            if isinstance(module, nn.Linear):
                output = self._run_linear_forward(module, output)
            elif isinstance(module, nn.Conv2d):
                output = self._run_conv2d_forward(module, output)
            elif isinstance(module, nn.BatchNorm2d):
                output = self._run_batch_norm2d_forward(module, output)
            elif isinstance(module, nn.ReLU):
                output = self._run_relu_forward(module, output)
            elif isinstance(module, nn.Flatten):
                output = self._run_flatten_forward(module, output)
            elif isinstance(module, nn.AvgPool2d):
                output = self._run_avg_pool2d_forward(module, output)
            elif isinstance(module, nn.AdaptiveAvgPool2d):
                output = self._run_adaptive_avg_pool2d_forward(module, output)
            elif isinstance(module, nn.MaxPool2d):
                output = self._run_max_pool2d_forward(module, output)
            else:
                raise NotImplementedError(
                    f"unsupported module in Sequential model: {module.__class__.__name__}"
                )
        output_primal = primal(output)
        self._state.output_id = id(output_primal)
        return output_primal

    def _run_linear_forward(
        self,
        module: nn.Linear,
        input_value: Any,
    ) -> Any:
        input_primal_live = primal(input_value)
        output: torch.Tensor | None = None
        local_output_tangents: dict[nn.Parameter, torch.Tensor] = {}

        weight_tangent = self._tangents_by_parameter.get(module.weight)
        if weight_tangent is not None:
            output_hat = self._run_module_with_dual_parameters(
                module,
                {"weight": make_dual(module.weight, weight_tangent)},
                input_primal_live,
            )
            output = primal(output_hat)
            local_output_tangents[module.weight] = tangent(output_hat)

        if module.bias is not None:
            bias_tangent = self._tangents_by_parameter.get(module.bias)
            if bias_tangent is not None:
                if output is None:
                    output_hat = self._run_module_with_dual_parameters(
                        module,
                        {"bias": make_dual(module.bias, bias_tangent)},
                        input_primal_live,
                    )
                    output = primal(output_hat)
                    local_output_tangents[module.bias] = tangent(output_hat)
                else:
                    local_output_tangents[module.bias] = torch.ops.aten.expand.default(
                        bias_tangent.detach(),
                        output.shape,
                    )

        if output is None:
            primal_output = self._call_module_forward(module, input_primal_live)
            output = primal(primal_output)

        record = LinearForwardRecord(
            module=module,
            input_activation=_make_linear_input_activation_ref(
                output,
                input_primal_live,
            ),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_conv2d_forward(
        self,
        module: nn.Conv2d,
        input_value: Any,
    ) -> Any:
        if module.padding_mode != "zeros":
            raise NotImplementedError(
                "LocalMLPHVPRuntime currently supports Conv2d padding_mode='zeros'"
            )
        input_primal_live = primal(input_value)
        output: torch.Tensor | None = None
        local_output_tangents: dict[nn.Parameter, torch.Tensor] = {}

        weight_tangent = self._tangents_by_parameter.get(module.weight)
        if weight_tangent is not None:
            output_hat = self._run_module_with_dual_parameters(
                module,
                {"weight": make_dual(module.weight, weight_tangent)},
                input_primal_live,
            )
            output = primal(output_hat)
            local_output_tangents[module.weight] = tangent(output_hat)

        if module.bias is not None:
            bias_tangent = self._tangents_by_parameter.get(module.bias)
            if bias_tangent is not None:
                if output is None:
                    output = primal(self._call_module_forward(module, input_primal_live))
                local_output_tangents[module.bias] = _reshape_channel_tangent(
                    bias_tangent.detach(),
                    output.dim(),
                ).expand_as(output)

        if output is None:
            output = primal(self._call_module_forward(module, input_primal_live))

        record = Conv2dForwardRecord(
            module=module,
            input_activation=_make_conv_input_activation_ref(output, input_primal_live),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_batch_norm2d_forward(
        self,
        module: nn.BatchNorm2d,
        input_value: Any,
    ) -> Any:
        if module.training:
            raise NotImplementedError(
                "LocalMLPHVPRuntime currently supports BatchNorm2d in eval mode only"
            )
        input_primal_live = primal(input_value)
        output: torch.Tensor | None = None
        local_output_tangents: dict[nn.Parameter, torch.Tensor] = {}

        if module.weight is not None:
            weight_tangent = self._tangents_by_parameter.get(module.weight)
            if weight_tangent is not None:
                output_hat = self._run_module_with_dual_parameters(
                    module,
                    {"weight": make_dual(module.weight, weight_tangent)},
                    input_primal_live,
                )
                output = primal(output_hat)
                local_output_tangents[module.weight] = tangent(output_hat)

        if module.bias is not None:
            bias_tangent = self._tangents_by_parameter.get(module.bias)
            if bias_tangent is not None:
                if output is None:
                    output = primal(self._call_module_forward(module, input_primal_live))
                local_output_tangents[module.bias] = _reshape_channel_tangent(
                    bias_tangent.detach(),
                    output.dim(),
                ).expand_as(output)

        if output is None:
            output = primal(self._call_module_forward(module, input_primal_live))

        record = BatchNorm2dForwardRecord(
            module=module,
            input_activation=_make_batch_norm_input_activation_ref(
                output,
                input_primal_live,
            ),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_relu_forward(
        self,
        module: nn.ReLU,
        input_value: Any,
    ) -> Any:
        if module.inplace:
            raise NotImplementedError("LocalMLPHVPRuntime does not support inplace ReLU")
        input_primal_live = primal(input_value)
        output_hat = self._call_module_forward(module, input_primal_live)
        output = primal(output_hat)
        record = ReLUForwardRecord(
            output_activation=_make_relu_output_activation_ref(output),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_flatten_forward(
        self,
        module: nn.Flatten,
        input_value: Any,
    ) -> Any:
        input_primal_live = primal(input_value)
        output = primal(self._call_module_forward(module, input_primal_live))
        record = FlattenForwardRecord(
            input_shape=input_primal_live.shape,
            start_dim=module.start_dim,
            end_dim=module.end_dim,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_avg_pool2d_forward(
        self,
        module: nn.AvgPool2d,
        input_value: Any,
    ) -> Any:
        input_primal_live = primal(input_value)
        output = primal(self._call_module_forward(module, input_primal_live))
        record = AvgPool2dForwardRecord(
            module=module,
            input_activation=_make_pool_input_activation_ref(output, input_primal_live),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_adaptive_avg_pool2d_forward(
        self,
        module: nn.AdaptiveAvgPool2d,
        input_value: Any,
    ) -> Any:
        input_primal_live = primal(input_value)
        output = primal(self._call_module_forward(module, input_primal_live))
        record = AdaptiveAvgPool2dForwardRecord(
            input_activation=_make_adaptive_pool_input_activation_ref(
                output,
                input_primal_live,
            ),
            output_size=_pair(module.output_size),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

    def _run_max_pool2d_forward(
        self,
        module: nn.MaxPool2d,
        input_value: Any,
    ) -> Any:
        if module.return_indices:
            raise NotImplementedError(
                "LocalMLPHVPRuntime currently supports MaxPool2d with return_indices=False"
            )
        input_primal_live = primal(input_value)
        output = primal(self._call_module_forward(module, input_primal_live))
        record = MaxPool2dForwardRecord(
            module=module,
            input_activation=_make_pool_input_activation_ref(output, input_primal_live),
            indices=_make_max_pool_indices_ref(output),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return output

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
            scale = _mse_hessian_scale(loss_record)
            factor = grad.detach() * scale

            def output_curvature(value: torch.Tensor) -> torch.Tensor:
                return _scale_tensor_for_loss(value, factor)

            self._state.curvature = output_curvature
            self._state.loss_record = None
            self._state.eager_backward_active = True

    def _make_forward_record_hook(self, record: ForwardRecord) -> Any:
        def forward_record_hook(grad: torch.Tensor) -> torch.Tensor:
            if not self._state.eager_backward_active:
                return grad
            if self._state.curvature is None:
                return grad
            with torch.no_grad():
                if isinstance(record, LinearForwardRecord):
                    self._consume_linear_backward_record(
                        record=record,
                        grad=grad,
                    )
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
                elif isinstance(record, ReLUForwardRecord):
                    self._consume_relu_backward_record(
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
                else:
                    raise TypeError(f"unknown forward record: {type(record).__name__}")
            return grad

        return forward_record_hook

    def _consume_linear_backward_record(
        self,
        *,
        record: LinearForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._state.curvature
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

        self._state.curvature = _make_linear_input_curvature(
            curvature,
            record.module.weight.detach(),
        )

    def _consume_conv2d_backward_record(
        self,
        *,
        record: Conv2dForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._state.curvature
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

        self._state.curvature = _make_conv2d_input_curvature(
            curvature,
            record.module,
            input_activation,
        )

    def _consume_batch_norm2d_backward_record(
        self,
        *,
        record: BatchNorm2dForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._state.curvature
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

        self._state.curvature = _make_batch_norm2d_input_curvature(
            curvature,
            record.module,
            input_activation,
        )

    def _consume_relu_backward_record(
        self,
        *,
        record: ReLUForwardRecord,
        grad: torch.Tensor,
    ) -> None:
        curvature = self._state.curvature
        if curvature is None:
            raise RuntimeError("missing output curvature for ReLU backward hook")
        relu_output = record.output_activation.resolve_and_release()
        self._state.curvature = _make_relu_input_curvature(curvature, relu_output)

    def _consume_flatten_backward_record(
        self,
        *,
        record: FlattenForwardRecord,
    ) -> None:
        curvature = self._state.curvature
        if curvature is None:
            raise RuntimeError("missing output curvature for Flatten backward hook")
        self._state.curvature = _make_flatten_input_curvature(curvature, record)

    def _consume_avg_pool2d_backward_record(
        self,
        *,
        record: AvgPool2dForwardRecord,
    ) -> None:
        curvature = self._state.curvature
        if curvature is None:
            raise RuntimeError("missing output curvature for AvgPool2d backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        self._state.curvature = _make_avg_pool2d_input_curvature(
            curvature,
            record.module,
            input_activation,
        )

    def _consume_adaptive_avg_pool2d_backward_record(
        self,
        *,
        record: AdaptiveAvgPool2dForwardRecord,
    ) -> None:
        curvature = self._state.curvature
        if curvature is None:
            raise RuntimeError(
                "missing output curvature for AdaptiveAvgPool2d backward hook"
            )
        input_activation = record.input_activation.resolve_and_release().detach()
        self._state.curvature = _make_adaptive_avg_pool2d_input_curvature(
            curvature,
            input_activation,
            record.output_size,
        )

    def _consume_max_pool2d_backward_record(
        self,
        *,
        record: MaxPool2dForwardRecord,
    ) -> None:
        curvature = self._state.curvature
        if curvature is None:
            raise RuntimeError("missing output curvature for MaxPool2d backward hook")
        input_activation = record.input_activation.resolve_and_release().detach()
        indices = record.indices.resolve_and_release().detach()
        self._state.curvature = _make_max_pool2d_input_curvature(
            curvature,
            record.module,
            input_activation,
            indices,
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
    if _is_supported_leaf_module(model):
        return True
    if isinstance(model, nn.Sequential):
        return all(
            _is_supported_leaf_module(module)
            for module in model._modules.values()
        )
    return False


def _is_supported_leaf_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Linear,
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.ReLU,
            nn.Flatten,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.MaxPool2d,
        ),
    )


def _iter_supported_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    if _is_supported_leaf_module(model):
        return (model,)
    if isinstance(model, nn.Sequential):
        return tuple(model._modules.values())
    raise TypeError("unsupported model type")


def _make_linear_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_mat1", "_saved_self"),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_conv_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_input",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_batch_norm_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_input",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_relu_output_activation_ref(output: torch.Tensor) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result",),
        expected_shape=output.shape,
        fallback=output,
    )


def _make_pool_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_self",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_adaptive_pool_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_self",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_max_pool_indices_ref(output: torch.Tensor) -> SavedTensorRef:
    if output.grad_fn is None:
        return SavedTensorRef(
            grad_fn=None,
            saved_attrs=("_saved_result1",),
            expected_shape=output.shape,
        )
    indices = _find_saved_tensor(
        output.grad_fn,
        saved_attrs=("_saved_result1",),
        expected_shape=output.shape,
    )
    if indices is None:
        raise RuntimeError("could not resolve MaxPool2d indices from autograd graph")
    return SavedTensorRef(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result1",),
        expected_shape=output.shape,
    )


def _make_saved_tensor_ref(
    *,
    grad_fn: Any | None,
    saved_attrs: tuple[str, ...],
    expected_shape: torch.Size,
    fallback: torch.Tensor,
) -> SavedTensorRef:
    ref = SavedTensorRef(
        grad_fn=grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
    )
    if grad_fn is not None and _find_saved_tensor(
        grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
    ) is not None:
        return ref
    return SavedTensorRef(
        grad_fn=grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
        fallback=fallback.detach(),
    )


def _find_saved_tensor(
    grad_fn: Any,
    *,
    saved_attrs: tuple[str, ...],
    expected_shape: torch.Size,
) -> torch.Tensor | None:
    seen: set[int] = set()
    stack = [grad_fn]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        node_id = id(node)
        if node_id in seen:
            continue
        seen.add(node_id)
        for attr in saved_attrs:
            if not hasattr(node, attr):
                continue
            try:
                value = getattr(node, attr)
            except RuntimeError:
                continue
            if torch.is_tensor(value) and value.shape == expected_shape:
                return value
        for next_node, _ in getattr(node, "next_functions", ()):
            if next_node is not None:
                stack.append(next_node)
    return None


def _linear_input_backward_program(
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.mm.default(grad_output, weight)


def _linear_weight_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.mm.default(
        torch.ops.aten.t.default(grad_output),
        input_activation,
    )


def _linear_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.sum.dim_IntList(grad_output, [0], False)


def _conv2d_input_backward_program(
    module: nn.Conv2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    grad_input, _, _ = torch.ops.aten.convolution_backward.default(
        grad_output,
        input_activation,
        module.weight.detach(),
        _conv_bias_sizes(module),
        module.stride,
        module.padding,
        module.dilation,
        False,
        (0, 0),
        module.groups,
        (True, False, False),
    )
    return grad_input


def _conv2d_weight_backward_program(
    module: nn.Conv2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _, grad_weight, _ = torch.ops.aten.convolution_backward.default(
        grad_output,
        input_activation,
        module.weight.detach(),
        _conv_bias_sizes(module),
        module.stride,
        module.padding,
        module.dilation,
        False,
        (0, 0),
        module.groups,
        (False, True, False),
    )
    return grad_weight


def _conv2d_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.sum.dim_IntList(grad_output, [0, 2, 3], False)


def _batch_norm2d_input_backward_program(
    module: nn.BatchNorm2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    grad_input, _, _ = torch.ops.aten.native_batch_norm_backward.default(
        grad_output,
        input_activation,
        module.weight.detach() if module.weight is not None else None,
        _batch_norm_running_mean(module),
        _batch_norm_running_var(module),
        None,
        None,
        False,
        module.eps,
        (True, False, False),
    )
    return grad_input


def _batch_norm2d_weight_backward_program(
    module: nn.BatchNorm2d,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    _, grad_weight, _ = torch.ops.aten.native_batch_norm_backward.default(
        grad_output,
        input_activation,
        module.weight.detach() if module.weight is not None else None,
        _batch_norm_running_mean(module),
        _batch_norm_running_var(module),
        None,
        None,
        False,
        module.eps,
        (False, True, False),
    )
    return grad_weight


def _batch_norm2d_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    return torch.ops.aten.sum.dim_IntList(grad_output, [0, 2, 3], False)


def _relu_backward_program(
    input_value: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.threshold_backward.default(grad_output, input_value, 0)


def _make_linear_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    weight: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    weight = weight.detach()
    weight_t = torch.ops.aten.t.default(weight)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten.mm.default(input_tangent, weight_t)
        output_grad_tangent = output_curvature(output_tangent)
        return _linear_input_backward_program(weight, output_grad_tangent)

    return input_curvature


def _make_conv2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.Conv2d,
    input_activation: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    weight = module.weight.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten.convolution.default(
            input_tangent,
            weight,
            None,
            module.stride,
            module.padding,
            module.dilation,
            False,
            (0, 0),
            module.groups,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return _conv2d_input_backward_program(
            module,
            input_activation,
            output_grad_tangent,
        )

    return input_curvature


def _make_batch_norm2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.BatchNorm2d,
    input_activation: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _batch_norm2d_input_jvp(module, input_tangent)
        output_grad_tangent = output_curvature(output_tangent)
        return _batch_norm2d_input_backward_program(
            module,
            input_activation,
            output_grad_tangent,
        )

    return input_curvature


def _make_relu_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    relu_output: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    relu_output = relu_output.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _relu_backward_program(relu_output, input_tangent)
        output_grad_tangent = output_curvature(output_tangent)
        return _relu_backward_program(relu_output, output_grad_tangent)

    return input_curvature


def _make_flatten_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    record: FlattenForwardRecord,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.flatten(
            input_tangent,
            start_dim=record.start_dim,
            end_dim=record.end_dim,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return output_grad_tangent.reshape(record.input_shape)

    return input_curvature


def _make_avg_pool2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.AvgPool2d,
    input_activation: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    kernel_size = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    padding = _pair(module.padding)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten.avg_pool2d.default(
            input_tangent,
            kernel_size,
            stride,
            padding,
            module.ceil_mode,
            module.count_include_pad,
            module.divisor_override,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return torch.ops.aten.avg_pool2d_backward.default(
            output_grad_tangent,
            input_activation,
            kernel_size,
            stride,
            padding,
            module.ceil_mode,
            module.count_include_pad,
            module.divisor_override,
        )

    return input_curvature


def _make_adaptive_avg_pool2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    input_activation: torch.Tensor,
    output_size: tuple[int, int],
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.ops.aten._adaptive_avg_pool2d.default(
            input_tangent,
            output_size,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return torch.ops.aten._adaptive_avg_pool2d_backward.default(
            output_grad_tangent,
            input_activation,
        )

    return input_curvature


def _make_max_pool2d_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.MaxPool2d,
    input_activation: torch.Tensor,
    indices: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    indices = indices.detach()
    kernel_size = _pair(module.kernel_size)
    stride = _pair(module.stride if module.stride is not None else module.kernel_size)
    padding = _pair(module.padding)
    dilation = _pair(module.dilation)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _max_pool2d_jvp(input_tangent, indices)
        output_grad_tangent = output_curvature(output_tangent)
        return torch.ops.aten.max_pool2d_with_indices_backward.default(
            output_grad_tangent,
            input_activation,
            kernel_size,
            stride,
            padding,
            dilation,
            module.ceil_mode,
            indices,
        )

    return input_curvature


def _parameter_use_counts(model: nn.Module) -> dict[nn.Parameter, int]:
    counts: dict[nn.Parameter, int] = {}
    for module in _iter_supported_modules(model):
        for parameter in module.parameters(recurse=False):
            counts[parameter] = counts.get(parameter, 0) + 1
    return counts


def _mse_hessian_scale(record: MSELossRecord) -> torch.Tensor:
    if record.reduction == "mean":
        return torch.tensor(
            2.0 / record.input_numel,
            device=record.device,
            dtype=record.dtype,
        )
    if record.reduction == "sum":
        return torch.tensor(2.0, device=record.device, dtype=record.dtype)
    raise ValueError(f"unknown mse_loss reduction: {record.reduction!r}")


def _scale_tensor_for_loss(value: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    if value.is_contiguous():
        value.mul_(factor)
        return value
    return value * factor


def _pair(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected pair, got {value!r}")
        return (int(value[0]), int(value[1]))
    return (int(value), int(value))


def _reshape_channel_tangent(value: torch.Tensor, ndim: int) -> torch.Tensor:
    if value.ndim != 1:
        return value
    return value.reshape((1, value.shape[0], *([1] * (ndim - 2))))


def _conv_bias_sizes(module: nn.Conv2d) -> list[int] | None:
    if module.bias is None:
        return None
    return list(module.bias.shape)


def _batch_norm_running_mean(module: nn.BatchNorm2d) -> torch.Tensor:
    if module.running_mean is None:
        raise NotImplementedError(
            "BatchNorm2d runtime support requires tracked running statistics"
        )
    return module.running_mean.detach()


def _batch_norm_running_var(module: nn.BatchNorm2d) -> torch.Tensor:
    if module.running_var is None:
        raise NotImplementedError(
            "BatchNorm2d runtime support requires tracked running statistics"
        )
    return module.running_var.detach()


def _batch_norm2d_input_jvp(
    module: nn.BatchNorm2d,
    input_tangent: torch.Tensor,
) -> torch.Tensor:
    scale = torch.rsqrt(_batch_norm_running_var(module) + module.eps)
    if module.weight is not None:
        scale = scale * module.weight.detach()
    return input_tangent * _reshape_channel_tangent(scale, input_tangent.dim())


def _max_pool2d_jvp(
    input_tangent: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if input_tangent.dim() == 3:
        channels, _, _ = input_tangent.shape
        flat_tangent = input_tangent.reshape(channels, -1)
        flat_indices = indices.reshape(channels, -1)
        return flat_tangent.gather(1, flat_indices).reshape(indices.shape)
    if input_tangent.dim() == 4:
        batch, channels, _, _ = input_tangent.shape
        flat_tangent = input_tangent.reshape(batch, channels, -1)
        flat_indices = indices.reshape(batch, channels, -1)
        return flat_tangent.gather(2, flat_indices).reshape(indices.shape)
    raise NotImplementedError("MaxPool2d tangent expects a 3D or 4D input")


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
