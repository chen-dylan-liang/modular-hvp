"""Local-dual runtime for supported eager tensor graphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F

from modular_hvp.dual import (
    make_dual,
    primal,
    tangent,
)
from modular_hvp.dispatch import (
    _get_aten_overload,
    _is_direct_graph_aten_op,
    _is_python_add,
    _is_python_cat,
    _is_python_chunk,
    _is_python_contiguous,
    _is_python_div,
    _is_python_dropout,
    _is_python_dtype_cast,
    _is_python_flatten,
    _is_python_layer_norm,
    _is_python_masked_fill,
    _is_python_matmul,
    _is_python_multi_head_attention_forward,
    _is_python_mul,
    _is_python_rms_norm,
    _is_python_scaled_dot_product_attention,
    _is_python_softmax,
    _is_python_sub,
    _is_python_transpose,
    _is_python_unary_elementwise,
    _is_python_view_or_reshape,
)
from modular_hvp.graph import (
    GraphTraversalState,
    RecordedForwardGraph,
    _local_parameter_use_counts,
    _record_has_backward_nonlinearity_tangents,
    _take_graph_local_parameters_by_output_node,
)
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
    FunctionalLayerNormForwardRecord,
    ForwardRecord,
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
    SavedTensorRef,
    ScaledDotProductAttentionForwardRecord,
    SelectForwardRecord,
    SliceForwardRecord,
    SoftmaxForwardRecord,
    TransposeForwardRecord,
    UnaryElementwiseForwardRecord,
    _find_saved_tensor,
    _record_input_node_ids,
    _record_local_output_tangents,
    _record_output_node_id,
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
    _broadcast_like,
    _canonicalize_index,
    _conv2d_bias_backward_program,
    _conv2d_weight_backward_program,
    _embedding_weight_backward_program,
    _gelu_backward_program,
    _gelu_derivative,
    _gelu_jvp,
    _layer_norm_bias_backward_program,
    _layer_norm_input_jvp,
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
    _normalize_dim,
    _normalize_select_index,
    _normalize_slice_end,
    _normalize_slice_start,
    _pair,
    _reshape_channel_tangent,
    _split_output_sizes,
    _unary_elementwise_kind,
    _value_shape,
)
from modular_hvp.losses import (
    CrossEntropyLossRecord,
    LossPatch,
    LossRecord,
    MSELossRecord,
    _make_loss_output_curvature,
)
from modular_hvp.model_utils import (
    _can_use_sequential_fast_path,
    _iter_raw_graph_parameters,
    _iter_unique_supported_leaf_modules,
    _is_supported_leaf_module,
    _parameter_use_counts,
    _should_wrap_graph_input_tensor,
    _validate_supported_model,
)
from modular_hvp.runtime import ParameterBlock, _resolve_parameter_blocks
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
    _make_saved_tensor_ref,
    _make_softmax_output_activation_ref,
)


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
    curvatures_by_node_id: dict[int, list[Callable[[torch.Tensor], torch.Tensor]]] = (
        field(default_factory=dict)
    )
    graph: GraphTraversalState = field(default_factory=GraphTraversalState)
    use_graph_curvature: bool = False
    eager_backward_active: bool = False


class GraphTensor(torch.Tensor):
    """Internal primal tensor wrapper used only to record eager graph edges."""

    __slots__ = ("_primal", "_runtime", "_node_id")

    @staticmethod
    def __new__(
        cls,
        primal_value: torch.Tensor,
        runtime: "EagerHVPRuntime",
        node_id: int,
    ) -> "GraphTensor":
        if primal_value.layout != torch.strided:
            raise NotImplementedError("GraphTensor currently supports strided tensors only")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            primal_value.shape,
            strides=primal_value.stride(),
            storage_offset=primal_value.storage_offset(),
            dtype=primal_value.dtype,
            layout=primal_value.layout,
            device=primal_value.device,
            requires_grad=primal_value.requires_grad,
        )

    def __init__(
        self,
        primal_value: torch.Tensor,
        runtime: "EagerHVPRuntime",
        node_id: int,
    ) -> None:
        self._primal = primal_value
        self._runtime = runtime
        self._node_id = node_id

    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        runtime = _graph_runtime_from_tree((args, kwargs))
        if runtime is None:
            return NotImplemented
        if _is_python_add(func):
            alpha = kwargs.get("alpha", 1)
            return runtime._dispatch_graph_linear_binary_function(
                func,
                args,
                kwargs,
                right_coefficient=float(alpha),
            )
        if _is_python_sub(func):
            alpha = kwargs.get("alpha", 1)
            return runtime._dispatch_graph_linear_binary_function(
                func,
                args,
                kwargs,
                right_coefficient=-float(alpha),
            )
        if _is_python_flatten(func):
            return runtime._dispatch_graph_shape_function(func, args, kwargs)
        if _is_python_chunk(func):
            return runtime._dispatch_graph_chunk_function(func, args, kwargs)
        if _is_python_view_or_reshape(func):
            return runtime._dispatch_graph_shape_function(func, args, kwargs)
        if _is_python_transpose(func):
            return runtime._dispatch_graph_transpose_function(func, args, kwargs)
        if _is_python_contiguous(func):
            return runtime._dispatch_graph_contiguous_function(func, args, kwargs)
        if _is_python_matmul(func):
            return runtime._dispatch_graph_matmul_function(func, args, kwargs)
        if _is_python_mul(func):
            return runtime._dispatch_graph_mul_function(func, args, kwargs)
        if _is_python_div(func):
            return runtime._dispatch_graph_div_function(func, args, kwargs)
        if _is_python_cat(func):
            return runtime._dispatch_graph_cat_function(func, args, kwargs)
        if _is_python_dtype_cast(func):
            return runtime._dispatch_graph_cast_function(func, args, kwargs)
        if _is_python_unary_elementwise(func):
            return runtime._dispatch_graph_unary_elementwise_function(func, args, kwargs)
        if _is_python_layer_norm(func):
            return runtime._dispatch_graph_layer_norm_function(func, args, kwargs)
        if _is_python_rms_norm(func):
            return runtime._dispatch_graph_rms_norm_function(func, args, kwargs)
        if _is_python_softmax(func):
            return runtime._dispatch_graph_softmax_function(func, args, kwargs)
        if _is_python_dropout(func):
            return runtime._dispatch_graph_dropout_function(func, args, kwargs)
        if _is_python_masked_fill(func):
            value = args[2] if len(args) > 2 else kwargs.get("value")
            op = (
                torch.ops.aten.masked_fill.Tensor
                if isinstance(value, torch.Tensor)
                else torch.ops.aten.masked_fill.Scalar
            )
            return runtime._dispatch_graph_op(op, args, kwargs)
        if _is_python_scaled_dot_product_attention(func):
            return runtime._dispatch_graph_op(
                torch.ops.aten.scaled_dot_product_attention.default,
                args,
                kwargs,
            )
        if _is_python_multi_head_attention_forward(func):
            return runtime._dispatch_graph_composite_function(func, args, kwargs)
        if _is_direct_graph_aten_op(func):
            return runtime._dispatch_graph_op(func, args, kwargs)
        raise NotImplementedError(f"ModularHVP graph traversal does not support {func}")

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if kwargs is None:
            kwargs = {}
        runtime = _graph_runtime_from_tree((args, kwargs))
        if runtime is None:
            raise RuntimeError("GraphTensor dispatch could not find its runtime")
        return runtime._dispatch_graph_op(func, args, kwargs)

    def __iadd__(self, other: Any) -> "GraphTensor":
        return torch.add(self, other)

    def __isub__(self, other: Any) -> "GraphTensor":
        return torch.sub(self, other)

    def __getitem__(self, index: Any) -> Any:
        return self._runtime._dispatch_graph_getitem(self, index)

    @property
    def primal(self) -> torch.Tensor:
        return self._primal

    @property
    def shape(self) -> torch.Size:
        return self._primal.shape

    @property
    def dtype(self) -> torch.dtype:
        return self._primal.dtype

    @property
    def device(self) -> torch.device:
        return self._primal.device

    @property
    def is_nested(self) -> bool:
        return self._primal.is_nested

    @property
    def ndim(self) -> int:
        return self._primal.ndim

    def size(self, *args: Any) -> Any:
        return self._primal.size(*args)

    def dim(self) -> int:
        return self._primal.dim()

    def to(self, *args: Any, **kwargs: Any) -> "GraphTensor":
        return torch.Tensor.to(self, *args, **kwargs)

    def float(self) -> "GraphTensor":
        return self.to(torch.float32)

    @property
    def node_id(self) -> int:
        return self._node_id

    def __repr__(self) -> str:
        return f"GraphTensor(primal={self._primal!r}, node_id={self._node_id})"


class EagerHVPRuntime:
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
    ) -> None:
        _validate_supported_model(model)
        self.model = model
        self.parameter_blocks = _resolve_parameter_blocks(model, tangents)
        self._blocks_by_parameter = {
            block.parameter: block for block in self.parameter_blocks
        }
        self._tangents_by_parameter = {
            block.parameter: block.tangent for block in self.parameter_blocks
        }
        self._parameter_use_counts = _parameter_use_counts(model)
        self._has_reused_parameters = any(
            count > 1 for count in self._parameter_use_counts.values()
        )
        self._use_graph_tensors = (
            self._has_reused_parameters or not _can_use_sequential_fast_path(model)
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

    def _dispatch_graph_getitem(self, value: GraphTensor, index: Any) -> Any:
        if not isinstance(index, tuple):
            index = (index,)
        items = _canonicalize_index(index, value.primal.dim())
        current: Any = value
        dim = 0
        for item in items:
            if item is None:
                current = torch.ops.aten.unsqueeze.default(current, dim)
                dim += 1
                continue
            if isinstance(item, slice):
                step = 1 if item.step is None else int(item.step)
                if step != 1:
                    raise NotImplementedError(
                        "ModularHVP graph traversal currently supports slice step=1"
                    )
                current_primal = self._unwrap_graph_value(current)
                if not isinstance(current_primal, torch.Tensor):
                    raise RuntimeError("getitem current value must be a tensor")
                start = 0 if item.start is None else int(item.start)
                end = current_primal.shape[dim] if item.stop is None else int(item.stop)
                current = torch.ops.aten.slice.Tensor(current, dim, start, end, step)
                dim += 1
                continue
            if isinstance(item, int):
                current = torch.ops.aten.select.int(current, dim, item)
                continue
            raise NotImplementedError(
                f"ModularHVP graph traversal does not support tensor index {item!r}"
            )
        return current

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

    def _dispatch_graph_op(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        primal_args = self._unwrap_graph_value(args)
        primal_kwargs = self._unwrap_graph_value(kwargs)
        output = func(*primal_args, **primal_kwargs)
        if not isinstance(output, torch.Tensor):
            if isinstance(output, tuple):
                graph_outputs: list[Any] = []
                for output_index, item in enumerate(output):
                    if not isinstance(item, torch.Tensor) or not _should_wrap_graph_input_tensor(item):
                        graph_outputs.append(item)
                        continue
                    graph_output = self._make_graph_output(item)
                    if item.requires_grad:
                        record = self._make_tuple_primitive_record(
                            func=func,
                            args=args,
                            kwargs=kwargs,
                            output=graph_output,
                            output_index=output_index,
                            primal_outputs=output,
                        )
                        if record is not None:
                            item.register_hook(self._make_forward_record_hook(record))
                    graph_outputs.append(graph_output)
                return tuple(graph_outputs)
            return output

        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = self._make_primitive_record(
                func=func,
                args=args,
                kwargs=kwargs,
                output=graph_output,
            )
            if record is not None:
                output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_shape_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("shape graph operation expects a tensor input")
        with torch._C.DisableTorchFunctionSubclass():
            output = func(*self._unwrap_graph_value(args), **self._unwrap_graph_value(kwargs))
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("shape graph operation returned a non-tensor")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("shape graph operation missing input node")
        graph_output = self._make_graph_output(output)
        record = ReshapeForwardRecord(
            input_node_id=input_node_id,
            output_node_id=graph_output.node_id,
            input_shape=input_primal.shape,
            output_shape=output.shape,
        )
        if output.requires_grad:
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_chunk_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, ...]:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("chunk graph operation expects a tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("chunk graph operation missing input node")
        dim = _normalize_dim(
            int(kwargs.get("dim", args[2] if len(args) > 2 else 0)),
            input_primal.dim(),
        )
        with torch._C.DisableTorchFunctionSubclass():
            outputs = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        sizes = [output.shape[dim] for output in outputs]
        graph_outputs: list[Any] = []
        start = 0
        for item, size in zip(outputs, sizes, strict=True):
            graph_output = self._make_graph_output(item)
            if item.requires_grad:
                record = SliceForwardRecord(
                    input_node_id=input_node_id,
                    output_node_id=graph_output.node_id,
                    input_shape=input_primal.shape,
                    dim=dim,
                    start=start,
                    end=start + size,
                )
                item.register_hook(self._make_forward_record_hook(record))
            graph_outputs.append(graph_output)
            start += size
        return tuple(graph_outputs)

    def _dispatch_graph_transpose_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("transpose graph operation expects a tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("transpose graph operation missing input node")
        dim0 = _normalize_dim(int(args[1]), input_primal.dim())
        dim1 = _normalize_dim(int(args[2]), input_primal.dim())
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = TransposeForwardRecord(
                input_node_id=input_node_id,
                output_node_id=graph_output.node_id,
                dim0=dim0,
                dim1=dim1,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_contiguous_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("contiguous graph operation expects a tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("contiguous graph operation missing input node")
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = ContiguousForwardRecord(
                input_node_id=input_node_id,
                output_node_id=graph_output.node_id,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_matmul_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        left, right = args[:2]
        left_primal = self._unwrap_graph_value(left)
        right_primal = self._unwrap_graph_value(right)
        if not isinstance(left_primal, torch.Tensor) or not isinstance(
            right_primal,
            torch.Tensor,
        ):
            raise NotImplementedError("matmul graph operation expects tensor inputs")
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = MatmulForwardRecord(
                left_node_id=self._node_id(left),
                right_node_id=self._node_id(right),
                output_node_id=graph_output.node_id,
                left_activation=_make_saved_tensor_ref(
                    grad_fn=output.grad_fn,
                    saved_attrs=("_saved_self", "_saved_mat1"),
                    expected_shape=left_primal.shape,
                    fallback=left_primal,
                    always_keep_fallback=True,
                ),
                right_activation=_make_saved_tensor_ref(
                    grad_fn=output.grad_fn,
                    saved_attrs=("_saved_other", "_saved_mat2"),
                    expected_shape=right_primal.shape,
                    fallback=right_primal,
                    always_keep_fallback=True,
                ),
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_mul_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        left, right = args[:2]
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("mul graph operation returned non-tensor")
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = MulForwardRecord(
                output_node_id=graph_output.node_id,
                left_node_id=self._node_id(left),
                right_node_id=self._node_id(right),
                left_value=self._unwrap_graph_value(left),
                right_value=self._unwrap_graph_value(right),
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_div_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        left, right = args[:2]
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("div graph operation returned non-tensor")
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = DivForwardRecord(
                output_node_id=graph_output.node_id,
                left_node_id=self._node_id(left),
                right_node_id=self._node_id(right),
                left_value=self._unwrap_graph_value(left),
                right_value=self._unwrap_graph_value(right),
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_cat_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        tensors = args[0]
        if not isinstance(tensors, (tuple, list)):
            raise NotImplementedError("cat graph operation expects a tensor sequence")
        primal_tensors = self._unwrap_graph_value(tensors)
        if not all(isinstance(item, torch.Tensor) for item in primal_tensors):
            raise NotImplementedError("cat graph operation expects tensors")
        dim_arg = kwargs.get("dim", args[1] if len(args) > 1 else 0)
        dim = _normalize_dim(int(dim_arg), primal_tensors[0].dim())
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                primal_tensors,
                *self._unwrap_graph_value(args[1:]),
                **self._unwrap_graph_value(kwargs),
            )
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = CatForwardRecord(
                output_node_id=graph_output.node_id,
                input_node_ids=tuple(self._node_id(item) for item in tensors),
                input_shapes=tuple(item.shape for item in primal_tensors),
                dim=dim,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_cast_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("cast graph operation expects tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("cast graph operation missing input node")
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("cast graph operation returned non-tensor")
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = CastForwardRecord(
                input_node_id=input_node_id,
                output_node_id=graph_output.node_id,
                input_dtype=input_primal.dtype,
                output_dtype=output.dtype,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_unary_elementwise_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("unary graph operation expects tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("unary graph operation missing input node")
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("unary graph operation returned non-tensor")
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            kind, scalar = _unary_elementwise_kind(func, args, kwargs)
            record = UnaryElementwiseForwardRecord(
                input_node_id=input_node_id,
                output_node_id=graph_output.node_id,
                input_activation=_make_exact_saved_tensor_ref(input_primal),
                output_activation=_make_exact_saved_tensor_ref(output)
                if kind in {"sigmoid", "tanh", "relu"}
                else None,
                kind=kind,
                scalar=scalar,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_layer_norm_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        normalized_shape_arg = (
            args[1] if len(args) > 1 else kwargs["normalized_shape"]
        )
        weight_value = args[2] if len(args) > 2 else kwargs.get("weight")
        bias_value = args[3] if len(args) > 3 else kwargs.get("bias")
        eps = float(args[4] if len(args) > 4 else kwargs.get("eps", 1e-5))
        return self._run_functional_layer_norm_forward(
            input_value=input_value,
            normalized_shape=tuple(int(item) for item in normalized_shape_arg),
            weight_value=weight_value,
            bias_value=bias_value,
            eps=eps,
        )

    def _dispatch_graph_rms_norm_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("rms_norm graph operation expects tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("rms_norm graph operation missing input node")
        normalized_shape = tuple(
            int(item) for item in (args[1] if len(args) > 1 else kwargs["normalized_shape"])
        )
        eps = float(kwargs.get("eps", args[2] if len(args) > 2 else 1e-5))
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("rms_norm graph operation returned non-tensor")
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = RMSNormForwardRecord(
                input_node_id=input_node_id,
                output_node_id=graph_output.node_id,
                input_activation=_make_exact_saved_tensor_ref(input_primal),
                normalized_shape=normalized_shape,
                eps=eps,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_softmax_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> GraphTensor:
        input_value = args[0]
        input_primal = self._unwrap_graph_value(input_value)
        if not isinstance(input_primal, torch.Tensor):
            raise NotImplementedError("softmax graph operation expects tensor input")
        input_node_id = self._node_id(input_value)
        if input_node_id is None:
            raise NotImplementedError("softmax graph operation missing input node")
        dim_arg = kwargs.get("dim", args[1] if len(args) > 1 else None)
        if dim_arg is None:
            raise NotImplementedError("softmax graph operation requires dim")
        dim = _normalize_dim(int(dim_arg), input_primal.dim())
        with torch._C.DisableTorchFunctionSubclass():
            output = func(
                *self._unwrap_graph_value(args),
                **self._unwrap_graph_value(kwargs),
            )
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            record = SoftmaxForwardRecord(
                input_node_id=input_node_id,
                output_node_id=graph_output.node_id,
                output_activation=_make_softmax_output_activation_ref(output),
                dim=dim,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _dispatch_graph_dropout_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> torch.Tensor | GraphTensor:
        input_value = args[0]
        p = float(args[1] if len(args) > 1 else kwargs.get("p", 0.5))
        training = bool(args[2] if len(args) > 2 else kwargs.get("training", True))
        inplace = bool(args[3] if len(args) > 3 else kwargs.get("inplace", False))
        return self._run_dropout_function_forward(
            input_value=input_value,
            p=p,
            training=training,
            inplace=inplace,
            func=func,
        )

    def _dispatch_graph_composite_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    def _dispatch_graph_linear_binary_function(
        self,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        *,
        right_coefficient: float,
    ) -> GraphTensor:
        with torch._C.DisableTorchFunctionSubclass():
            output = func(*self._unwrap_graph_value(args), **self._unwrap_graph_value(kwargs))
        if not isinstance(output, torch.Tensor):
            raise NotImplementedError("linear binary graph operation returned non-tensor")
        graph_output = self._make_graph_output(output)
        if output.requires_grad:
            left_primal = self._unwrap_graph_value(args[0]) if args else 0
            right_primal = self._unwrap_graph_value(args[1]) if len(args) > 1 else 0
            record = AddForwardRecord(
                output_node_id=graph_output.node_id,
                left_node_id=self._node_id(args[0]) if args else None,
                right_node_id=self._node_id(args[1]) if len(args) > 1 else None,
                alpha=right_coefficient,
                left_shape=_value_shape(left_primal),
                right_shape=_value_shape(right_primal),
                output_shape=output.shape,
            )
            output.register_hook(self._make_forward_record_hook(record))
        return graph_output

    def _make_primitive_record(
        self,
        *,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: GraphTensor,
    ) -> ForwardRecord | None:
        if func in {torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar}:
            alpha = kwargs.get("alpha", 1)
            left = self._unwrap_graph_value(args[0])
            right = self._unwrap_graph_value(args[1])
            return AddForwardRecord(
                output_node_id=output.node_id,
                left_node_id=self._node_id(args[0]),
                right_node_id=self._node_id(args[1]),
                alpha=float(alpha),
                left_shape=_value_shape(left),
                right_shape=_value_shape(right),
                output_shape=output.primal.shape,
            )
        if func in {torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar}:
            alpha = kwargs.get("alpha", 1)
            left = self._unwrap_graph_value(args[0])
            right = self._unwrap_graph_value(args[1]) if len(args) > 1 else 0
            return AddForwardRecord(
                output_node_id=output.node_id,
                left_node_id=self._node_id(args[0]),
                right_node_id=self._node_id(args[1]) if len(args) > 1 else None,
                alpha=-float(alpha),
                left_shape=_value_shape(left),
                right_shape=_value_shape(right),
                output_shape=output.primal.shape,
            )
        if func in {torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar}:
            left, right = args[:2]
            return MulForwardRecord(
                output_node_id=output.node_id,
                left_node_id=self._node_id(left),
                right_node_id=self._node_id(right),
                left_value=self._unwrap_graph_value(left),
                right_value=self._unwrap_graph_value(right),
            )
        if func is torch.ops.aten.cat.default:
            tensors = args[0]
            if not isinstance(tensors, (tuple, list)):
                return None
            primal_tensors = self._unwrap_graph_value(tensors)
            if not all(isinstance(item, torch.Tensor) for item in primal_tensors):
                return None
            dim_arg = kwargs.get("dim", args[1] if len(args) > 1 else 0)
            return CatForwardRecord(
                output_node_id=output.node_id,
                input_node_ids=tuple(self._node_id(item) for item in tensors),
                input_shapes=tuple(item.shape for item in primal_tensors),
                dim=_normalize_dim(int(dim_arg), primal_tensors[0].dim()),
            )
        if func is torch.ops.aten.linear.default:
            input_value, weight_value = args[:2]
            bias_value = args[2] if len(args) > 2 else None
            input_primal = self._unwrap_graph_value(input_value)
            weight_primal = self._unwrap_graph_value(weight_value)
            bias_primal = self._unwrap_graph_value(bias_value)
            if not isinstance(input_primal, torch.Tensor) or not isinstance(
                weight_primal,
                torch.Tensor,
            ):
                return None
            input_node_id = self._node_id(input_value)
            if input_node_id is None:
                return None
            local_output_tangents: dict[nn.Parameter, torch.Tensor] = {}
            if isinstance(weight_primal, nn.Parameter):
                weight_tangent = self._tangents_by_parameter.get(weight_primal)
                if weight_tangent is not None:
                    local_output_tangents[weight_primal] = torch.matmul(
                        input_primal.detach(),
                        weight_tangent.detach().t(),
                    )
            if isinstance(bias_primal, nn.Parameter):
                bias_tangent = self._tangents_by_parameter.get(bias_primal)
                if bias_tangent is not None:
                    local_output_tangents[bias_primal] = bias_tangent.detach().expand_as(
                        output.primal,
                    )
            return FunctionalLinearForwardRecord(
                weight=weight_primal,
                bias=bias_primal if isinstance(bias_primal, torch.Tensor) else None,
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_activation=_make_functional_linear_input_activation_ref(
                    output.primal,
                    input_primal,
                ),
                local_output_tangents=local_output_tangents,
            )
        if func in {
            torch.ops.aten.view.default,
            torch.ops.aten.reshape.default,
            torch.ops.aten._unsafe_view.default,
            torch.ops.aten.unsqueeze.default,
            torch.ops.aten.squeeze.default,
            torch.ops.aten.squeeze.dim,
        }:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            if not isinstance(input_primal, torch.Tensor):
                return None
            input_node_id = self._node_id(input_value)
            if input_node_id is None:
                return None
            return ReshapeForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_shape=input_primal.shape,
                output_shape=output.primal.shape,
            )
        if func is torch.ops.aten.slice.Tensor:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            input_node_id = self._node_id(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            dim = _normalize_dim(int(args[1]), input_primal.dim())
            start = _normalize_slice_start(args[2], input_primal.shape[dim])
            end = _normalize_slice_end(args[3], input_primal.shape[dim])
            step = int(args[4]) if len(args) > 4 else 1
            if step != 1:
                raise NotImplementedError(
                    "ModularHVP graph traversal currently supports slice step=1"
                )
            return SliceForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_shape=input_primal.shape,
                dim=dim,
                start=start,
                end=end,
            )
        if func is torch.ops.aten.select.int:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            input_node_id = self._node_id(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            dim = _normalize_dim(int(args[1]), input_primal.dim())
            index = _normalize_select_index(int(args[2]), input_primal.shape[dim])
            return SelectForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_shape=input_primal.shape,
                dim=dim,
                index=index,
            )
        if func is torch.ops.aten.transpose.int:
            input_value = args[0]
            input_node_id = self._node_id(input_value)
            input_primal = self._unwrap_graph_value(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            return TransposeForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                dim0=_normalize_dim(int(args[1]), input_primal.dim()),
                dim1=_normalize_dim(int(args[2]), input_primal.dim()),
            )
        if func is torch.ops.aten.contiguous.default:
            input_value = args[0]
            input_node_id = self._node_id(input_value)
            if input_node_id is None:
                return None
            return ContiguousForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
            )
        if func in {
            torch.ops.aten.to.dtype,
            torch.ops.aten.to.device,
            torch.ops.aten._to_copy.default,
        }:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            input_node_id = self._node_id(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            return CastForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_dtype=input_primal.dtype,
                output_dtype=output.primal.dtype,
            )
        if func in {
            torch.ops.aten.relu.default,
            torch.ops.aten.sigmoid.default,
            torch.ops.aten.tanh.default,
            torch.ops.aten.square.default,
            torch.ops.aten.pow.Tensor_Scalar,
        }:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            input_node_id = self._node_id(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            kind, scalar = _unary_elementwise_kind(func, args, kwargs)
            return UnaryElementwiseForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_activation=_make_exact_saved_tensor_ref(input_primal),
                output_activation=_make_exact_saved_tensor_ref(output.primal)
                if kind in {"sigmoid", "tanh", "relu"}
                else None,
                kind=kind,
                scalar=scalar,
            )
        rms_norm_op = _get_aten_overload("rms_norm", "default")
        native_rms_norm_op = _get_aten_overload("native_rms_norm", "default")
        if func in {rms_norm_op, native_rms_norm_op}:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            input_node_id = self._node_id(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            normalized_shape = tuple(int(item) for item in args[1])
            eps = float(kwargs.get("eps", args[2] if len(args) > 2 else 1e-5))
            return RMSNormForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_activation=_make_exact_saved_tensor_ref(input_primal),
                normalized_shape=normalized_shape,
                eps=eps,
            )
        if func in {
            torch.ops.aten.matmul.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.mm.default,
        }:
            left, right = args[:2]
            left_primal = self._unwrap_graph_value(left)
            right_primal = self._unwrap_graph_value(right)
            if not isinstance(left_primal, torch.Tensor) or not isinstance(
                right_primal,
                torch.Tensor,
            ):
                return None
            return MatmulForwardRecord(
                left_node_id=self._node_id(left),
                right_node_id=self._node_id(right),
                output_node_id=output.node_id,
                left_activation=_make_saved_tensor_ref(
                    grad_fn=output.primal.grad_fn,
                    saved_attrs=("_saved_self", "_saved_mat1"),
                    expected_shape=left_primal.shape,
                    fallback=left_primal,
                    always_keep_fallback=True,
                ),
                right_activation=_make_saved_tensor_ref(
                    grad_fn=output.primal.grad_fn,
                    saved_attrs=("_saved_other", "_saved_mat2"),
                    expected_shape=right_primal.shape,
                    fallback=right_primal,
                    always_keep_fallback=True,
                ),
            )
        if func in {torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar}:
            left, right = args[:2]
            return DivForwardRecord(
                output_node_id=output.node_id,
                left_node_id=self._node_id(left),
                right_node_id=self._node_id(right),
                left_value=self._unwrap_graph_value(left),
                right_value=self._unwrap_graph_value(right),
            )
        if func is torch.ops.aten._softmax.default:
            input_value = args[0]
            input_node_id = self._node_id(input_value)
            input_primal = self._unwrap_graph_value(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            dim = _normalize_dim(int(args[1]), input_primal.dim())
            return SoftmaxForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                output_activation=_make_softmax_output_activation_ref(output.primal),
                dim=dim,
            )
        if func in {
            torch.ops.aten.masked_fill.Scalar,
            torch.ops.aten.masked_fill.Tensor,
        }:
            input_value, mask = args[:2]
            input_node_id = self._node_id(input_value)
            mask_primal = self._unwrap_graph_value(mask)
            if input_node_id is None or not isinstance(mask_primal, torch.Tensor):
                return None
            if isinstance(mask, GraphTensor):
                raise NotImplementedError(
                    "ModularHVP graph traversal does not support dual masks"
                )
            return MaskedFillForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                mask=mask_primal.detach(),
            )
        if func is torch.ops.aten.scaled_dot_product_attention.default:
            query, key, value = args[:3]
            query_primal = self._unwrap_graph_value(query)
            key_primal = self._unwrap_graph_value(key)
            value_primal = self._unwrap_graph_value(value)
            if not all(
                isinstance(item, torch.Tensor)
                for item in (query_primal, key_primal, value_primal)
            ):
                return None
            dropout_p = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0.0)
            if dropout_p != 0.0:
                raise NotImplementedError(
                    "ModularHVP graph traversal currently requires attention dropout_p=0"
                )
            attn_mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
            if isinstance(attn_mask, GraphTensor):
                raise NotImplementedError(
                    "ModularHVP graph traversal does not support dual attention masks"
                )
            return ScaledDotProductAttentionForwardRecord(
                query_node_id=self._node_id(query),
                key_node_id=self._node_id(key),
                value_node_id=self._node_id(value),
                output_node_id=output.node_id,
                query_activation=_make_exact_saved_tensor_ref(query_primal),
                key_activation=_make_exact_saved_tensor_ref(key_primal),
                value_activation=_make_exact_saved_tensor_ref(value_primal),
                attn_mask=attn_mask.detach() if isinstance(attn_mask, torch.Tensor) else None,
                is_causal=bool(args[5] if len(args) > 5 else kwargs.get("is_causal", False)),
                scale=kwargs.get("scale"),
            )
        raise NotImplementedError(f"ModularHVP graph traversal does not support {func}")

    def _make_tuple_primitive_record(
        self,
        *,
        func: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: GraphTensor,
        output_index: int,
        primal_outputs: tuple[Any, ...],
    ) -> ForwardRecord | None:
        if func in {
            torch.ops.aten.split.Tensor,
            torch.ops.aten.split.sizes,
            torch.ops.aten.split_with_sizes.default,
        }:
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            if not isinstance(input_primal, torch.Tensor):
                return None
            input_node_id = self._node_id(input_value)
            if input_node_id is None:
                return None
            dim_arg = kwargs.get("dim", args[2] if len(args) > 2 else 0)
            dim = _normalize_dim(int(dim_arg), input_primal.dim())
            sizes = _split_output_sizes(
                input_primal.shape[dim],
                args[1],
                output_count=output_index + 1,
            )
            start = sum(sizes[:output_index])
            end = start + sizes[output_index]
            return SliceForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                input_shape=input_primal.shape,
                dim=dim,
                start=start,
                end=end,
            )
        if func is torch.ops.aten.native_layer_norm.default:
            if output_index != 0:
                return None
            input_value = args[0]
            input_primal = self._unwrap_graph_value(input_value)
            if not isinstance(input_primal, torch.Tensor):
                return None
            normalized_shape = tuple(int(item) for item in args[1])
            weight_primal = self._unwrap_graph_value(args[2])
            bias_primal = self._unwrap_graph_value(args[3])
            eps = float(args[4])
            mean = primal_outputs[1]
            rstd = primal_outputs[2]
            if not isinstance(mean, torch.Tensor) or not isinstance(rstd, torch.Tensor):
                return None
            return self._make_functional_layer_norm_record(
                input_value=input_value,
                input_primal=input_primal,
                weight_primal=weight_primal,
                bias_primal=bias_primal,
                normalized_shape=normalized_shape,
                eps=eps,
                output=output.primal,
                output_node_id=output.node_id,
                mean=mean,
                rstd=rstd,
            )
        if func is torch.ops.aten.native_dropout.default:
            if output_index != 0:
                return None
            input_value = args[0]
            input_node_id = self._node_id(input_value)
            input_primal = self._unwrap_graph_value(input_value)
            if input_node_id is None or not isinstance(input_primal, torch.Tensor):
                return None
            p = float(args[1])
            train = bool(args[2])
            mask = primal_outputs[1]
            if not isinstance(mask, torch.Tensor):
                return None
            multiplier = torch.ones_like(input_primal.detach())
            if train and p != 0.0:
                multiplier = mask.detach().to(dtype=input_primal.dtype) / (1.0 - p)
            return DropoutForwardRecord(
                input_node_id=input_node_id,
                output_node_id=output.node_id,
                multiplier=_make_exact_saved_tensor_ref(multiplier),
            )
        if func is torch.ops.aten._scaled_dot_product_flash_attention.default:
            if output_index != 0:
                return None
            query, key, value = args[:3]
            query_primal = self._unwrap_graph_value(query)
            key_primal = self._unwrap_graph_value(key)
            value_primal = self._unwrap_graph_value(value)
            if not all(
                isinstance(item, torch.Tensor)
                for item in (query_primal, key_primal, value_primal)
            ):
                return None
            dropout_p = args[3] if len(args) > 3 else kwargs.get("dropout_p", 0.0)
            if dropout_p != 0.0:
                raise NotImplementedError(
                    "ModularHVP graph traversal currently requires attention dropout_p=0"
                )
            return ScaledDotProductAttentionForwardRecord(
                query_node_id=self._node_id(query),
                key_node_id=self._node_id(key),
                value_node_id=self._node_id(value),
                output_node_id=output.node_id,
                query_activation=_make_exact_saved_tensor_ref(query_primal),
                key_activation=_make_exact_saved_tensor_ref(key_primal),
                value_activation=_make_exact_saved_tensor_ref(value_primal),
                attn_mask=None,
                is_causal=bool(args[4] if len(args) > 4 else kwargs.get("is_causal", False)),
                scale=kwargs.get("scale"),
            )
        raise NotImplementedError(f"ModularHVP graph traversal does not support {func}")

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

    def _start_eager_backward(self, grad: torch.Tensor) -> None:
        loss_record = self._state.loss_record
        if loss_record is None:
            raise RuntimeError(
                "modular_hvp did not observe a supported scalar loss; "
                "currently use torch.nn.MSELoss or torch.nn.functional.mse_loss"
            )

        with torch.no_grad():
            output_curvature = _make_loss_output_curvature(loss_record, grad.detach())
            self._state.output_curvature = output_curvature

            if self._state.output_node_id is None:
                raise RuntimeError("missing model output node for ModularHVP backward")
            if self._state.use_graph_curvature:
                graph = RecordedForwardGraph.from_records(
                    records=self._state.forward_records,
                    output_node_id=self._state.output_node_id,
                    retain_local_parameter_inputs=self._has_reused_parameters,
                )
                if graph.requires_hooked_primal_grads or self._has_reused_parameters:
                    self._state.active_graph = graph
                    self._initialize_graph_hvp_hook_state(graph)
                    self._state.loss_record = None
                    self._state.eager_backward_active = True
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

    def _make_forward_record_hook(self, record: ForwardRecord) -> Any:
        self._state.forward_records.append(record)
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
        output_tangents = tangents_by_node.pop(graph.output_node_id, {})
        output_grad_tangents: dict[nn.Parameter, torch.Tensor] = {}
        for parameter, tangent_value in output_tangents.items():
            output_grad_tangents[parameter] = output_curvature(tangent_value)
        output_tangents.clear()
        local_parameters_by_output_node = _take_graph_local_parameters_by_output_node(
            graph.records
        )
        self._state.graph.prepare_hooked_backward(
            output_node_id=graph.output_node_id,
            output_grad_tangents=output_grad_tangents,
            retained_forward_tangents_by_node=tangents_by_node,
            local_parameters_by_output_node=local_parameters_by_output_node,
            remaining_local_uses_by_parameter=_local_parameter_use_counts(
                local_parameters_by_output_node
            ),
            input_use_counts=graph.fresh_input_use_counts(),
        )
        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))

    def _compute_graph_forward_tangent_packets(
        self,
        graph: RecordedForwardGraph,
        *,
        retained_node_ids: set[int] | frozenset[int] | None = None,
    ) -> dict[int, dict[nn.Parameter, torch.Tensor]]:
        tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            node_id: packet.copy()
            for node_id, packet in self._state.raw_parameter_tangents_by_node.items()
        }
        retained_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {}
        if retained_node_ids is not None:
            for node_id, packet in tangents_by_node.items():
                if node_id in retained_node_ids:
                    retained_tangents_by_node[node_id] = packet
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
                for parameter, tangent_value in local_tangents.items():
                    _accumulate_parameter_tensor(
                        record_tangents,
                        parameter,
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
                if (
                    retained_node_ids is None
                    or _record_output_node_id(record) in retained_node_ids
                ):
                    retained_tangents_by_node[_record_output_node_id(record)] = (
                        node_tangents
                    )
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
            self._accumulate_graph_record_hvps(
                record,
                output_grad_tangents,
                primal_grad=graph_state.primal_grad(record),
                forward_tangents_by_node=graph_state.forward_tangents_by_node,
            )
            for parameter in local_parameters:
                if graph_state.consume_local_parameter_use(parameter) == 0:
                    output_grad_tangents.pop(parameter, None)

            if output_grad_tangents or _record_has_backward_nonlinearity_tangents(
                record,
                graph_state.forward_tangents_by_node,
            ):
                _propagate_backward_tangent_packet_with_grad(
                    record,
                    graph_state.primal_grad(record),
                    output_grad_tangents,
                    graph_state.forward_tangents_by_node,
                    graph_state.grad_tangents_by_node,
                    self._tangents_by_parameter,
                )

            graph_state.finish_record(graph, record)

        if graph_state.is_complete(graph):
            self._accumulate_raw_parameter_graph_hvps(graph_state.grad_tangents_by_node)
            for block in self.parameter_blocks:
                if getattr(block.parameter, "hvp", None) is None:
                    setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))
            self._state.active_graph = None
            self._state.eager_backward_active = False

    def _compute_graph_hvps_single_pass(self, graph: RecordedForwardGraph) -> None:
        output_curvature = self._state.output_curvature
        if output_curvature is None:
            raise RuntimeError("graph HVP requested before loss initialization")

        tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            node_id: packet.copy()
            for node_id, packet in self._state.raw_parameter_tangents_by_node.items()
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
                for parameter, tangent_value in local_tangents.items():
                    _accumulate_parameter_tensor(
                        record_tangents,
                        parameter,
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

        output_tangents = tangents_by_node.get(graph.output_node_id, {})
        tangents_by_node.clear()
        grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]] = {
            graph.output_node_id: {
                parameter: output_curvature(tangent_value)
                for parameter, tangent_value in output_tangents.items()
            }
        }

        for record in graph.reverse_records:
            output_grad_tangents = grad_tangents_by_node.pop(
                _record_output_node_id(record),
                None,
            )
            if not output_grad_tangents:
                _record_local_output_tangents(record).clear()
                continue

            local_parameters = tuple(_record_local_output_tangents(record))
            self._accumulate_graph_record_hvps(record, output_grad_tangents)
            for parameter in local_parameters:
                output_grad_tangents.pop(parameter, None)
            if not output_grad_tangents:
                continue

            _propagate_backward_tangent_packet(
                record,
                output_grad_tangents,
                grad_tangents_by_node,
            )

        self._accumulate_raw_parameter_graph_hvps(grad_tangents_by_node)

        for block in self.parameter_blocks:
            if getattr(block.parameter, "hvp", None) is None:
                setattr(block.parameter, "hvp", torch.zeros_like(block.parameter))

    def _accumulate_raw_parameter_graph_hvps(
        self,
        grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
    ) -> None:
        for node_id, local_packet in self._state.raw_parameter_tangents_by_node.items():
            output_packet = grad_tangents_by_node.pop(node_id, {})
            for parameter in local_packet:
                hvp = output_packet.get(parameter)
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
                    grad_tangent = output_grad_tangents.get(parameter)
                    input_tangent = _local_record_input_tangent(
                        record,
                        parameter,
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
                    grad_tangent = output_grad_tangents.get(parameter)
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
                    grad_tangent = output_grad_tangents.get(parameter)
                    input_tangent = _local_record_input_tangent(
                        record,
                        parameter,
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
                    grad_tangent = output_grad_tangents.get(parameter)
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
                    grad_tangent = output_grad_tangents.get(parameter)
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
                    grad_tangent = output_grad_tangents.get(parameter)
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
                    grad_tangent = output_grad_tangents.get(parameter)
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

    def _consume_embedding_backward_record(
        self,
        *,
        record: EmbeddingForwardRecord,
    ) -> None:
        curvature = self._take_node_curvature(record.output_node_id)
        if curvature is None:
            raise RuntimeError("missing output curvature for Embedding backward hook")
        try:
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
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
            for parameter, local_output_tangent in record.local_output_tangents.items():
                parameter_grad_tangent = curvature(local_output_tangent)
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


def _graph_runtime_from_tree(value: Any) -> "EagerHVPRuntime | None":
    if isinstance(value, GraphTensor):
        return value._runtime
    if isinstance(value, tuple | list):
        for item in value:
            runtime = _graph_runtime_from_tree(item)
            if runtime is not None:
                return runtime
    if isinstance(value, dict):
        for item in value.values():
            runtime = _graph_runtime_from_tree(item)
            if runtime is not None:
                return runtime
    return None
