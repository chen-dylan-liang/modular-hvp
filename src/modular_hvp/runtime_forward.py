"""Forward wrapping and local-dual recording for the eager runtime."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from modular_hvp.dual import make_dual, primal, tangent
from modular_hvp.graph import RecordedForwardGraph
from modular_hvp.graph_tensor import GraphTensor
from modular_hvp.kernels import (
    _gelu_jvp,
    _layer_norm_input_jvp,
    _layer_norm_normalized_input_from_stats,
    _pair,
    _reshape_channel_tangent,
)
from modular_hvp.losses import LossPatch
from modular_hvp.model_utils import (
    _iter_unique_supported_leaf_modules,
    _is_supported_leaf_module,
    _should_wrap_graph_input_tensor,
)
from modular_hvp.records import (
    AdaptiveAvgPool2dForwardRecord,
    AvgPool2dForwardRecord,
    BatchNorm2dForwardRecord,
    Conv2dForwardRecord,
    DropoutForwardRecord,
    EmbeddingForwardRecord,
    FlattenForwardRecord,
    FunctionalLayerNormForwardRecord,
    GELUForwardRecord,
    LayerNormForwardRecord,
    LinearForwardRecord,
    MaxPool2dForwardRecord,
    ReLUForwardRecord,
    ScaledDotProductAttentionForwardRecord,
)
from modular_hvp.runtime_state import ForwardPatch, RawParameterPatch
from modular_hvp.saved_tensors import (
    _make_adaptive_pool_input_activation_ref,
    _make_batch_norm_input_activation_ref,
    _make_conv_input_activation_ref,
    _make_dropout_multiplier_ref,
    _make_exact_saved_tensor_ref,
    _make_functional_linear_input_activation_ref,
    _make_gelu_input_activation_ref,
    _make_layer_norm_input_activation_ref,
    _make_layer_norm_mean_ref,
    _make_layer_norm_rstd_ref,
    _make_linear_input_activation_ref,
    _make_max_pool_indices_ref,
    _make_pool_input_activation_ref,
    _make_relu_output_activation_ref,
)


class ForwardRuntimeMixin:
    def _install_forward_wrappers(self) -> None:
        original_forward = self.model.forward
        had_instance_forward = "forward" in self.model.__dict__
        instance_forward = self.model.__dict__.get("forward")

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            return self._run_model_forward(args=args, kwargs=kwargs)

        self._state.forward_patches.append(
            ForwardPatch(
                module=self.model,
                original_forward=original_forward,
                had_instance_forward=had_instance_forward,
                instance_forward=instance_forward,
            )
        )
        self.model.forward = wrapped_forward  # type: ignore[method-assign]

        for module in _iter_unique_supported_leaf_modules(self.model):
            if module is self.model:
                continue
            original_forward = module.forward
            had_instance_forward = "forward" in module.__dict__
            instance_forward = module.__dict__.get("forward")

            def wrapped_leaf_forward(
                *args: Any,
                __module: nn.Module = module,
                **kwargs: Any,
            ) -> Any:
                return self._run_supported_leaf_forward(
                    module=__module,
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
            module.forward = wrapped_leaf_forward  # type: ignore[method-assign]

    def _original_forward_for(self, module: nn.Module) -> Any:
        for patch in reversed(self._state.forward_patches):
            if patch.module is module:
                return patch.original_forward
        return module.forward

    def _restore_forward_wrappers(self) -> None:
        for patch in reversed(self._state.forward_patches):
            if patch.had_instance_forward:
                patch.module.forward = patch.instance_forward  # type: ignore[method-assign]
            else:
                delattr(patch.module, "forward")
        self._state.forward_patches.clear()

    def _wrap_graph_input(self, value: Any) -> Any:
        if isinstance(value, GraphTensor):
            return value
        if isinstance(value, torch.Tensor):
            if _should_wrap_graph_input_tensor(value):
                return GraphTensor(value, self, self._new_node_id())
            return value
        if isinstance(value, tuple):
            return tuple(self._wrap_graph_input(item) for item in value)
        if isinstance(value, list):
            return [self._wrap_graph_input(item) for item in value]
        if isinstance(value, dict):
            return {key: self._wrap_graph_input(item) for key, item in value.items()}
        return value

    def _unwrap_graph_value(self, value: Any) -> Any:
        if isinstance(value, GraphTensor):
            return value.primal
        if isinstance(value, tuple):
            return tuple(self._unwrap_graph_value(item) for item in value)
        if isinstance(value, list):
            return [self._unwrap_graph_value(item) for item in value]
        if isinstance(value, dict):
            return {key: self._unwrap_graph_value(item) for key, item in value.items()}
        return value

    def _node_id(self, value: Any) -> int | None:
        if isinstance(value, GraphTensor):
            return value.node_id
        if isinstance(value, torch.Tensor):
            return id(value)
        return None

    def _make_graph_output(self, output: torch.Tensor) -> GraphTensor:
        return GraphTensor(output, self, self._new_node_id())

    def _make_runtime_output(
        self,
        output: torch.Tensor,
    ) -> tuple[torch.Tensor | GraphTensor, int]:
        if self._use_graph_tensors:
            graph_output = self._make_graph_output(output)
            return graph_output, graph_output.node_id
        return output, id(output)

    def _new_node_id(self) -> int:
        node_id = self._state.next_node_id
        self._state.next_node_id += 1
        return node_id

    def _install_raw_parameter_graph_sources(self) -> None:
        if not self._use_graph_tensors:
            return
        for module, parameter_name, parameter in self._raw_graph_parameters:
            tangent_value = self._tangents_by_parameter.get(parameter)
            if tangent_value is None:
                continue
            node_id = self._new_node_id()
            graph_parameter = GraphTensor(parameter, self, node_id)
            self._state.raw_parameter_patches.append(
                RawParameterPatch(
                    module=module,
                    name=parameter_name,
                    parameter=parameter,
                    node_id=node_id,
                )
            )
            self._state.raw_parameter_tangents_by_node[node_id] = {
                parameter: tangent_value.detach()
            }
            module._parameters[parameter_name] = graph_parameter

    def _restore_raw_parameter_graph_sources(self) -> None:
        for patch in reversed(self._state.raw_parameter_patches):
            patch.module._parameters[patch.name] = patch.parameter
        self._state.raw_parameter_patches.clear()

    def _run_supported_leaf_forward(
        self,
        *,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | GraphTensor:
        if isinstance(module, nn.MultiheadAttention):
            return self._run_multihead_attention_forward(module, args, kwargs)
        if kwargs:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports positional module inputs only"
            )
        if len(args) != 1:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports single-input leaf modules"
            )
        input_value = args[0]
        if isinstance(module, nn.Embedding):
            return self._run_embedding_forward(module, input_value)
        if isinstance(module, nn.Linear):
            return self._run_linear_forward(module, input_value)
        if isinstance(module, nn.Conv2d):
            return self._run_conv2d_forward(module, input_value)
        if isinstance(module, nn.BatchNorm2d):
            return self._run_batch_norm2d_forward(module, input_value)
        if isinstance(module, nn.LayerNorm):
            return self._run_layer_norm_forward(module, input_value)
        if isinstance(module, nn.ReLU):
            return self._run_relu_forward(module, input_value)
        if isinstance(module, nn.GELU):
            return self._run_gelu_forward(module, input_value)
        if isinstance(module, nn.Flatten):
            return self._run_flatten_forward(module, input_value)
        if isinstance(module, nn.AvgPool2d):
            return self._run_avg_pool2d_forward(module, input_value)
        if isinstance(module, nn.AdaptiveAvgPool2d):
            return self._run_adaptive_avg_pool2d_forward(module, input_value)
        if isinstance(module, nn.MaxPool2d):
            return self._run_max_pool2d_forward(module, input_value)
        if isinstance(module, nn.Dropout):
            return self._run_dropout_forward(module, input_value)
        raise NotImplementedError(
            f"unsupported module in ModularHVP graph: {module.__class__.__name__}"
        )

    def _install_loss_patch(self) -> None:
        original_mse_loss = F.mse_loss
        original_cross_entropy = F.cross_entropy

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
            if not isinstance(input, GraphTensor) and id(input) != self._state.output_id:
                return original_mse_loss(input, target, **kwargs)
            return self.record_mse_loss(
                input_value=input,
                target=target,
                reduction=reduction,
            )

        def wrapped_cross_entropy(
            input: torch.Tensor,
            target: torch.Tensor,
            *args: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            if args:
                return original_cross_entropy(input, target, *args, **kwargs)
            if not isinstance(input, GraphTensor) and id(input) != self._state.output_id:
                return original_cross_entropy(input, target, **kwargs)
            return self.record_cross_entropy_loss(
                input_value=input,
                target=target,
                weight=kwargs.get("weight"),
                ignore_index=int(kwargs.get("ignore_index", -100)),
                reduction=kwargs.get("reduction", "mean"),
                label_smoothing=float(kwargs.get("label_smoothing", 0.0)),
            )

        self._state.loss_patch = LossPatch(
            original_mse_loss=original_mse_loss,
            original_cross_entropy=original_cross_entropy,
        )
        F.mse_loss = wrapped_mse_loss
        F.cross_entropy = wrapped_cross_entropy

    def _restore_loss_patch(self) -> None:
        patch = self._state.loss_patch
        if patch is None:
            return
        F.mse_loss = patch.original_mse_loss
        F.cross_entropy = patch.original_cross_entropy
        self._state.loss_patch = None

    def _run_model_forward(
        self,
        *,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        self._state.output_id = None
        self._state.output_node_id = None
        self._state.next_node_id = 0
        self._state.loss_factor = None
        self._state.output_curvature = None
        self._state.raw_parameter_patches.clear()
        self._state.raw_parameter_tangents_by_node.clear()
        self._state.forward_records.clear()
        self._state.active_graph = None
        self._state.curvatures_by_node_id.clear()
        self._state.graph.clear()
        self._state.use_graph_curvature = False
        self._state.backward_called = False
        self._state.eager_backward_active = False
        graph_args = self._wrap_graph_input(args) if self._use_graph_tensors else args
        graph_kwargs = self._wrap_graph_input(kwargs) if self._use_graph_tensors else kwargs
        self._install_raw_parameter_graph_sources()
        try:
            if _is_supported_leaf_module(self.model):
                output: Any = self._run_supported_leaf_forward(
                    module=self.model,
                    args=graph_args,
                    kwargs=graph_kwargs,
                )
            else:
                output = self._original_forward_for(self.model)(*graph_args, **graph_kwargs)
        finally:
            self._restore_raw_parameter_graph_sources()
        output_primal = self._unwrap_graph_value(output)
        if isinstance(output_primal, torch.Tensor):
            self._state.output_id = id(output_primal)
            output_node_id = self._node_id(output)
            if self._state.output_node_id is None:
                self._state.output_node_id = (
                    output_node_id if output_node_id is not None else id(output_primal)
                )
        return output_primal

    def _run_functional_layer_norm_forward(
        self,
        *,
        input_value: Any,
        normalized_shape: tuple[int, ...],
        weight_value: Any,
        bias_value: Any,
        eps: float,
    ) -> GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal = self._unwrap_graph_value(input_value)
        weight_primal = self._unwrap_graph_value(weight_value)
        bias_primal = self._unwrap_graph_value(bias_value)
        if input_node_id is None or not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("layer_norm graph operation expects tensor input")
        with torch._C.DisableTorchFunctionSubclass():
            output, mean, rstd = torch.ops.aten.native_layer_norm.default(
                input_primal,
                list(normalized_shape),
                weight_primal,
                bias_primal,
                eps,
            )
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = self._make_functional_layer_norm_record(
                input_value=input_value,
                input_primal=input_primal,
                weight_primal=weight_primal,
                bias_primal=bias_primal,
                normalized_shape=normalized_shape,
                eps=eps,
                output=output,
                output_node_id=graph_output.node_id,
                mean=mean,
                rstd=rstd,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _make_functional_layer_norm_record(
        self,
        *,
        input_value: Any,
        input_primal: torch.Tensor,
        weight_primal: Any,
        bias_primal: Any,
        normalized_shape: tuple[int, ...],
        eps: float,
        output: torch.Tensor,
        output_node_id: int,
        mean: torch.Tensor,
        rstd: torch.Tensor,
    ) -> FunctionalLayerNormForwardRecord:
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("layer_norm graph operation missing input node")
        local_output_tangents: dict[nn.Parameter, torch.Tensor] = {}
        with torch.no_grad():
            normalized_input = _layer_norm_normalized_input_from_stats(
                input_primal.detach(),
                mean.detach(),
                rstd.detach(),
            )
            if isinstance(weight_primal, nn.Parameter):
                weight_tangent = self._tangents_by_parameter.get(weight_primal)
                if weight_tangent is not None:
                    local_output_tangents[weight_primal] = (
                        normalized_input * weight_tangent.detach()
                    )
            if isinstance(bias_primal, nn.Parameter):
                bias_tangent = self._tangents_by_parameter.get(bias_primal)
                if bias_tangent is not None:
                    local_output_tangents[bias_primal] = bias_tangent.detach().expand_as(
                        output,
                    )
        return FunctionalLayerNormForwardRecord(
            weight=weight_primal if isinstance(weight_primal, torch.Tensor) else None,
            bias=bias_primal if isinstance(bias_primal, torch.Tensor) else None,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_layer_norm_input_activation_ref(
                output,
                input_primal,
            ),
            mean=_make_exact_saved_tensor_ref(mean),
            rstd=_make_exact_saved_tensor_ref(rstd),
            normalized_shape=normalized_shape,
            eps=eps,
            local_output_tangents=local_output_tangents,
        )

    def _run_dropout_function_forward(
        self,
        *,
        input_value: Any,
        p: float,
        training: bool,
        inplace: bool,
        func: Any,
    ) -> torch.Tensor | GraphTensor:
        if inplace:
            raise NotImplementedError("EagerHVPRuntime does not support inplace Dropout")
        if p < 0.0 or p > 1.0:
            raise ValueError(f"dropout probability has to be between 0 and 1, got {p}")
        input_node_id = self._node_id(input_value)
        input_primal = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("dropout graph operation expects tensor input")
        if not training or p == 0.0:
            return input_value
        if p == 1.0:
            raise NotImplementedError("EagerHVPRuntime does not support Dropout p=1")
        with torch._C.DisableTorchFunctionSubclass():
            output = func(input_primal, p=p, training=training, inplace=False)
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("dropout graph operation returned non-tensor")
        runtime_output, output_node_id = self._make_runtime_output(output)
        if output.requires_grad:
            record = DropoutForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output_node_id,
                multiplier=_make_dropout_multiplier_ref(output, input_primal),
            )
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_embedding_forward(
        self,
        module: nn.Embedding,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        if module.max_norm is not None:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports Embedding with max_norm=None"
            )
        if module.sparse:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports dense Embedding gradients"
            )
        indices = self._unwrap_graph_value(input_value)
        if not isinstance(indices, torch.Tensor):
            raise NotImplementedError("Embedding indices must be tensors")
        output: torch.Tensor | None = None
        local_output_tangents: dict[nn.Parameter, torch.Tensor] = {}

        weight_tangent = self._tangents_by_parameter.get(module.weight)
        if weight_tangent is not None:
            output_hat = self._run_module_with_dual_parameters(
                module,
                {"weight": make_dual(module.weight, weight_tangent)},
                indices,
            )
            output = primal(output_hat)
            local_output_tangents[module.weight] = tangent(output_hat)

        if output is None:
            output = primal(self._call_module_forward(module, indices))

        runtime_output, output_node_id = self._make_runtime_output(output)
        record = EmbeddingForwardRecord(
            module=module,
            output_node_id=output_node_id,
            indices=indices.detach(),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_linear_forward(
        self,
        module: nn.Linear,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("Linear inputs must be tensors")
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

        runtime_output, output_node_id = self._make_runtime_output(output)
        record = LinearForwardRecord(
            module=module,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_linear_input_activation_ref(
                output,
                input_primal_live,
            ),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_conv2d_forward(
        self,
        module: nn.Conv2d,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        if module.padding_mode != "zeros":
            raise NotImplementedError(
                "EagerHVPRuntime currently supports Conv2d padding_mode='zeros'"
            )
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("Conv2d inputs must be tensors")
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

        runtime_output, output_node_id = self._make_runtime_output(output)
        record = Conv2dForwardRecord(
            module=module,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_conv_input_activation_ref(output, input_primal_live),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_batch_norm2d_forward(
        self,
        module: nn.BatchNorm2d,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        if module.training:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports BatchNorm2d in eval mode only"
            )
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("BatchNorm2d inputs must be tensors")
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

        runtime_output, output_node_id = self._make_runtime_output(output)
        record = BatchNorm2dForwardRecord(
            module=module,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_batch_norm_input_activation_ref(
                output,
                input_primal_live,
            ),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_layer_norm_forward(
        self,
        module: nn.LayerNorm,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("LayerNorm inputs must be tensors")
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
                local_output_tangents[module.bias] = bias_tangent.detach().expand_as(
                    output,
                )

        if output is None:
            output = primal(self._call_module_forward(module, input_primal_live))

        runtime_output, output_node_id = self._make_runtime_output(output)
        record = LayerNormForwardRecord(
            module=module,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_layer_norm_input_activation_ref(
                output,
                input_primal_live,
            ),
            mean=_make_layer_norm_mean_ref(module, output, input_primal_live),
            rstd=_make_layer_norm_rstd_ref(module, output, input_primal_live),
            local_output_tangents=local_output_tangents,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_relu_forward(
        self,
        module: nn.ReLU,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        if module.inplace:
            raise NotImplementedError("EagerHVPRuntime does not support inplace ReLU")
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("ReLU inputs must be tensors")
        output_hat = self._call_module_forward(module, input_primal_live)
        output = primal(output_hat)
        runtime_output, output_node_id = self._make_runtime_output(output)
        record = ReLUForwardRecord(
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            output_activation=_make_relu_output_activation_ref(output),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_gelu_forward(
        self,
        module: nn.GELU,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("GELU inputs must be tensors")
        output = primal(self._call_module_forward(module, input_primal_live))
        runtime_output, output_node_id = self._make_runtime_output(output)
        record = GELUForwardRecord(
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_gelu_input_activation_ref(output, input_primal_live),
            approximate=module.approximate,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_flatten_forward(
        self,
        module: nn.Flatten,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("Flatten inputs must be tensors")
        output = primal(self._call_module_forward(module, input_primal_live))
        runtime_output, output_node_id = self._make_runtime_output(output)
        record = FlattenForwardRecord(
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_shape=input_primal_live.shape,
            start_dim=module.start_dim,
            end_dim=module.end_dim,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_dropout_forward(
        self,
        module: nn.Dropout,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        return self._run_dropout_function_forward(
            input_value=input_value,
            p=float(module.p),
            training=module.training,
            inplace=module.inplace,
            func=F.dropout,
        )

    def _run_multihead_attention_forward(
        self,
        module: nn.MultiheadAttention,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor | GraphTensor, torch.Tensor | GraphTensor | None]:
        if not module.batch_first:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports MultiheadAttention with batch_first=True"
            )
        if module.training and module.dropout != 0.0:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports MultiheadAttention dropout only in eval mode"
            )
        if module.in_proj_weight is None:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports packed MultiheadAttention projections"
            )
        query = args[0]
        key = args[1] if len(args) > 1 else query
        value = args[2] if len(args) > 2 else key
        if key is not query or value is not query:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports self-attention MultiheadAttention only"
            )
        if kwargs.get("key_padding_mask") is not None or kwargs.get("attn_mask") is not None:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports MultiheadAttention without masks"
            )
        need_weights = bool(kwargs.get("need_weights", True))
        average_attn_weights = bool(kwargs.get("average_attn_weights", True))
        is_causal = bool(kwargs.get("is_causal", False))
        query_primal = self._unwrap_graph_value(query)
        if not isinstance(query_primal, torch.Tensor):
            raise NotImplementedError("MultiheadAttention query must be a tensor")
        batch, tokens, embed_dim = query_primal.shape
        if embed_dim != module.embed_dim:
            raise ValueError("MultiheadAttention input embed_dim does not match module")

        projected = torch.ops.aten.linear.default(
            query,
            module.in_proj_weight,
            module.in_proj_bias,
        )
        q, k, v = projected.chunk(3, dim=-1)
        head_dim = embed_dim // module.num_heads
        q = q.view(batch, tokens, module.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch, tokens, module.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch, tokens, module.num_heads, head_dim).transpose(1, 2)

        if need_weights:
            scores = (q @ k.transpose(-2, -1)) / (head_dim**0.5)
            attention = torch.softmax(scores, dim=-1)
            attended = attention @ v
            weights_value = self._unwrap_graph_value(attention)
            if not isinstance(weights_value, torch.Tensor):
                raise RuntimeError("MultiheadAttention weights must be tensors")
            weights: torch.Tensor | GraphTensor | None = weights_value
            if average_attn_weights:
                weights = weights_value.mean(dim=1)
        else:
            attended = torch.ops.aten.scaled_dot_product_attention.default(
                q,
                k,
                v,
                None,
                0.0,
                is_causal,
            )
            weights = None

        attended = attended.transpose(1, 2).contiguous().view(batch, tokens, embed_dim)
        output = torch.ops.aten.linear.default(
            attended,
            module.out_proj.weight,
            module.out_proj.bias,
        )
        return output, weights

    def _run_avg_pool2d_forward(
        self,
        module: nn.AvgPool2d,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("AvgPool2d inputs must be tensors")
        output = primal(self._call_module_forward(module, input_primal_live))
        runtime_output, output_node_id = self._make_runtime_output(output)
        record = AvgPool2dForwardRecord(
            module=module,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_pool_input_activation_ref(output, input_primal_live),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_adaptive_avg_pool2d_forward(
        self,
        module: nn.AdaptiveAvgPool2d,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("AdaptiveAvgPool2d inputs must be tensors")
        output = primal(self._call_module_forward(module, input_primal_live))
        runtime_output, output_node_id = self._make_runtime_output(output)
        record = AdaptiveAvgPool2dForwardRecord(
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_adaptive_pool_input_activation_ref(
                output,
                input_primal_live,
            ),
            output_size=_pair(module.output_size),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _run_max_pool2d_forward(
        self,
        module: nn.MaxPool2d,
        input_value: Any,
    ) -> torch.Tensor | GraphTensor:
        if module.return_indices:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports MaxPool2d with return_indices=False"
            )
        input_node_id = self._node_id(input_value)
        input_primal_live = self._unwrap_graph_value(input_value)
        if input_node_id is None or not isinstance(input_primal_live, torch.Tensor):
            raise NotImplementedError("MaxPool2d inputs must be tensors")
        output = primal(self._call_module_forward(module, input_primal_live))
        runtime_output, output_node_id = self._make_runtime_output(output)
        record = MaxPool2dForwardRecord(
            module=module,
            input_node_id=input_node_id,
            output_node_id=output_node_id,
            input_activation=_make_pool_input_activation_ref(output, input_primal_live),
            indices=_make_max_pool_indices_ref(output),
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return runtime_output

    def _call_module_forward(
        self,
        module: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._original_forward_for(module)(*args, **kwargs)

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
