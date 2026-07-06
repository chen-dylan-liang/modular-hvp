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
from modular_hvp.graph import (
    GraphTraversalState,
    RecordedForwardGraph,
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
    EmbeddingForwardRecord,
    FlattenForwardRecord,
    ForwardRecord,
    FunctionalLinearForwardRecord,
    GELUForwardRecord,
    LayerNormForwardRecord,
    LinearForwardRecord,
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
from modular_hvp.runtime import ParameterBlock, _resolve_parameter_blocks


@dataclass(slots=True)
class ForwardPatch:
    module: nn.Module
    original_forward: Any
    had_instance_forward: bool
    instance_forward: Any


@dataclass(slots=True)
class LossPatch:
    original_mse_loss: Callable[..., torch.Tensor]
    original_cross_entropy: Callable[..., torch.Tensor]


@dataclass(frozen=True, slots=True)
class MSELossRecord:
    input_numel: int
    device: torch.device
    dtype: torch.dtype
    reduction: str


@dataclass(frozen=True, slots=True)
class CrossEntropyLossRecord:
    logits: torch.Tensor
    target: torch.Tensor
    reduction: str
    ignore_index: int


LossRecord = MSELossRecord | CrossEntropyLossRecord


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
        if _is_python_rms_norm(func):
            return runtime._dispatch_graph_rms_norm_function(func, args, kwargs)
        if _is_python_softmax(func):
            return runtime._dispatch_graph_softmax_function(func, args, kwargs)
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
        self._use_graph_tensors = not _can_use_sequential_fast_path(model)
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
                    if not isinstance(item, torch.Tensor) or item.dtype == torch.long:
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
            record = AddForwardRecord(
                output_node_id=graph_output.node_id,
                left_node_id=self._node_id(args[0]) if args else None,
                right_node_id=self._node_id(args[1]) if len(args) > 1 else None,
                alpha=right_coefficient,
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
            return AddForwardRecord(
                output_node_id=output.node_id,
                left_node_id=self._node_id(args[0]),
                right_node_id=self._node_id(args[1]),
                alpha=float(alpha),
            )
        if func in {torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar}:
            alpha = kwargs.get("alpha", 1)
            return AddForwardRecord(
                output_node_id=output.node_id,
                left_node_id=self._node_id(args[0]),
                right_node_id=self._node_id(args[1]) if len(args) > 1 else None,
                alpha=-float(alpha),
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
    ) -> torch.Tensor:
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
        if not isinstance(output_primal, torch.Tensor):
            raise NotImplementedError("EagerHVPRuntime currently expects tensor output")
        self._state.output_id = id(output_primal)
        output_node_id = self._node_id(output)
        if self._state.output_node_id is None:
            self._state.output_node_id = (
                output_node_id if output_node_id is not None else id(output_primal)
            )
        return output_primal

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
        if module.training:
            raise NotImplementedError(
                "EagerHVPRuntime currently supports Dropout in eval mode only"
            )
        return input_value

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
        if any(count > 1 for count in self._parameter_use_counts.values()):
            self._compute_reused_parameter_hvps_autodiff()
            return

        with torch.no_grad():
            output_curvature = _make_loss_output_curvature(loss_record, grad.detach())
            self._state.output_curvature = output_curvature

            if self._state.output_node_id is None:
                raise RuntimeError("missing model output node for ModularHVP backward")
            if self._state.use_graph_curvature:
                graph = RecordedForwardGraph.from_records(
                    records=self._state.forward_records,
                    output_node_id=self._state.output_node_id,
                )
                if graph.requires_hooked_primal_grads:
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
        if isinstance(
            record,
            (
                FunctionalLinearForwardRecord,
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
        self._state.graph.prepare_hooked_backward(
            output_node_id=graph.output_node_id,
            output_grad_tangents=output_grad_tangents,
            retained_forward_tangents_by_node=tangents_by_node,
            local_parameters_by_output_node=(
                _take_graph_local_parameters_by_output_node(graph.records)
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
            self._accumulate_graph_record_hvps(record, output_grad_tangents)
            for parameter in local_parameters:
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
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.module.weight:
                        hvp = _linear_weight_backward_program(
                            input_activation,
                            grad_tangent,
                        )
                    elif parameter is record.module.bias:
                        hvp = _linear_bias_backward_program(grad_tangent)
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
                    if grad_tangent is None:
                        hvp = torch.zeros_like(parameter)
                    elif parameter is record.weight:
                        hvp = _linear_weight_backward_program(
                            input_activation,
                            grad_tangent,
                        )
                    elif parameter is record.bias:
                        hvp = _linear_bias_backward_program(grad_tangent)
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


def _is_python_flatten(func: Any) -> bool:
    return func is torch.flatten or (
        getattr(func, "__name__", None) == "flatten"
        and getattr(func, "__module__", "").startswith("torch")
    )


def _is_python_chunk(func: Any) -> bool:
    return getattr(func, "__name__", None) == "chunk"


def _is_python_view_or_reshape(func: Any) -> bool:
    return getattr(func, "__name__", None) in {
        "view",
        "reshape",
        "unsqueeze",
        "squeeze",
    }


def _is_python_transpose(func: Any) -> bool:
    return getattr(func, "__name__", None) == "transpose"


def _is_python_contiguous(func: Any) -> bool:
    return getattr(func, "__name__", None) == "contiguous"


def _is_python_matmul(func: Any) -> bool:
    return getattr(func, "__name__", None) == "matmul"


def _is_python_mul(func: Any) -> bool:
    return getattr(func, "__name__", None) == "mul"


def _is_python_div(func: Any) -> bool:
    return getattr(func, "__name__", None) == "div"


def _is_python_cat(func: Any) -> bool:
    return getattr(func, "__name__", None) == "cat"


def _is_python_dtype_cast(func: Any) -> bool:
    return getattr(func, "__name__", None) in {"to", "float"}


def _is_python_unary_elementwise(func: Any) -> bool:
    return getattr(func, "__name__", None) in {
        "relu",
        "sigmoid",
        "tanh",
        "square",
        "pow",
    }


def _is_python_rms_norm(func: Any) -> bool:
    return getattr(func, "__name__", None) == "rms_norm"


def _is_python_softmax(func: Any) -> bool:
    return getattr(func, "__name__", None) == "softmax"


def _is_python_multi_head_attention_forward(func: Any) -> bool:
    return (
        getattr(func, "__name__", None) == "multi_head_attention_forward"
        and getattr(func, "__module__", None) == "torch.nn.functional"
    )


def _get_aten_overload(name: str, overload: str) -> Any | None:
    packet = getattr(torch.ops.aten, name, None)
    if packet is None or overload not in packet.overloads():
        return None
    return getattr(packet, overload)


_DIRECT_GRAPH_ATEN_OPS = {
    torch.ops.aten.linear.default,
    torch.ops.aten.relu.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.add.Scalar,
    torch.ops.aten.sub.Tensor,
    torch.ops.aten.sub.Scalar,
    torch.ops.aten.sigmoid.default,
    torch.ops.aten.tanh.default,
    torch.ops.aten.square.default,
    torch.ops.aten.pow.Tensor_Scalar,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.mul.Scalar,
    torch.ops.aten.cat.default,
    torch.ops.aten.unsqueeze.default,
    torch.ops.aten.squeeze.default,
    torch.ops.aten.squeeze.dim,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.select.int,
    torch.ops.aten.to.dtype,
    torch.ops.aten.to.device,
    torch.ops.aten._to_copy.default,
    torch.ops.aten.scaled_dot_product_attention.default,
}

_DIRECT_GRAPH_ATEN_OPS.update(
    op
    for op in (
        _get_aten_overload("rms_norm", "default"),
        _get_aten_overload("native_rms_norm", "default"),
    )
    if op is not None
)


def _is_direct_graph_aten_op(func: Any) -> bool:
    return func in _DIRECT_GRAPH_ATEN_OPS


def _is_python_add(func: Any) -> bool:
    return getattr(func, "__name__", None) == "add"


def _is_python_sub(func: Any) -> bool:
    return getattr(func, "__name__", None) == "sub"


def _validate_supported_model(model: nn.Module) -> None:
    unsupported: list[str] = []
    for name, module in model.named_modules(remove_duplicate=True):
        if module is model and not _is_leaf_module(module):
            continue
        if not _is_leaf_module(module):
            continue
        if _is_supported_leaf_module(module) or _is_transparent_leaf_module(module):
            continue
        qualified_name = name or "<root>"
        unsupported.append(f"{qualified_name}: {module.__class__.__name__}")
    if unsupported:
        raise NotImplementedError(
            "the default modular_hvp runtime does not support these leaf modules: "
            + ", ".join(unsupported)
        )


def _is_leaf_module(module: nn.Module) -> bool:
    return not any(module.children())


def _is_supported_leaf_module(module: nn.Module) -> bool:
    return isinstance(
        module,
        (
            nn.Linear,
            nn.Embedding,
            nn.MultiheadAttention,
            nn.Conv2d,
            nn.BatchNorm2d,
            nn.LayerNorm,
            nn.ReLU,
            nn.GELU,
            nn.Flatten,
            nn.AvgPool2d,
            nn.AdaptiveAvgPool2d,
            nn.MaxPool2d,
            nn.Dropout,
        ),
    )


def _is_transparent_leaf_module(module: nn.Module) -> bool:
    return isinstance(module, nn.Identity)


def _can_use_sequential_fast_path(model: nn.Module) -> bool:
    if not isinstance(model, nn.Sequential):
        return False
    for module in model.modules():
        if module is model:
            continue
        if isinstance(module, nn.Sequential):
            continue
        if _is_leaf_module(module) and (
            _is_supported_leaf_module(module) or _is_transparent_leaf_module(module)
        ):
            continue
        return False
    return True


def _should_wrap_graph_input_tensor(value: torch.Tensor) -> bool:
    return value.is_floating_point() or value.is_complex()


def _iter_unique_supported_leaf_modules(model: nn.Module) -> tuple[nn.Module, ...]:
    return tuple(
        module
        for module in model.modules()
        if _is_supported_leaf_module(module)
    )


def _iter_supported_leaf_modules_with_duplicates(model: nn.Module) -> tuple[nn.Module, ...]:
    return tuple(
        module
        for _, module in model.named_modules(remove_duplicate=False)
        if _is_supported_leaf_module(module)
    )


def _iter_raw_graph_parameters(
    model: nn.Module,
) -> tuple[tuple[nn.Module, str, nn.Parameter], ...]:
    raw_parameters: list[tuple[nn.Module, str, nn.Parameter]] = []
    for module in model.modules():
        if _is_supported_leaf_module(module):
            continue
        for parameter_name, parameter in module._parameters.items():
            if parameter is None or not parameter.requires_grad:
                continue
            raw_parameters.append((module, parameter_name, parameter))
    return tuple(raw_parameters)


def _make_linear_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_mat1", "_saved_self"),
        expected_shape=input_value.shape,
        fallback=input_value,
        always_keep_fallback=input_value.dim() != 2,
    )


def _make_functional_linear_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_mat1", "_saved_self"),
        expected_shape=input_value.shape,
        fallback=input_value,
        always_keep_fallback=input_value.dim() != 2,
    )


def _make_exact_saved_tensor_ref(value: torch.Tensor) -> SavedTensorRef:
    return SavedTensorRef(
        grad_fn=None,
        saved_attrs=(),
        expected_shape=value.shape,
        fallback=value.detach(),
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
        always_keep_fallback=True,
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
        always_keep_fallback=True,
    )


def _make_layer_norm_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_input",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_layer_norm_mean_ref(
    module: nn.LayerNorm,
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    dims = _layer_norm_dims(input_value.dim(), module.normalized_shape)
    expected_shape = _layer_norm_stat_shape(input_value.shape, module.normalized_shape)
    with torch.no_grad():
        mean = input_value.detach().mean(dim=dims, keepdim=True)
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result1",),
        expected_shape=expected_shape,
        fallback=mean,
    )


def _make_layer_norm_rstd_ref(
    module: nn.LayerNorm,
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    dims = _layer_norm_dims(input_value.dim(), module.normalized_shape)
    expected_shape = _layer_norm_stat_shape(input_value.shape, module.normalized_shape)
    with torch.no_grad():
        mean = input_value.detach().mean(dim=dims, keepdim=True)
        var = (input_value.detach() - mean).pow(2).mean(dim=dims, keepdim=True)
        rstd = torch.rsqrt(var + module.eps)
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result2",),
        expected_shape=expected_shape,
        fallback=rstd,
    )


def _make_relu_output_activation_ref(output: torch.Tensor) -> SavedTensorRef:
    return SavedTensorRef(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result",),
        expected_shape=output.shape,
        fallback=output.detach(),
    )


def _make_gelu_input_activation_ref(
    output: torch.Tensor,
    input_value: torch.Tensor,
) -> SavedTensorRef:
    return _make_saved_tensor_ref(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_self",),
        expected_shape=input_value.shape,
        fallback=input_value,
    )


def _make_softmax_output_activation_ref(output: torch.Tensor) -> SavedTensorRef:
    return SavedTensorRef(
        grad_fn=output.grad_fn,
        saved_attrs=("_saved_result",),
        expected_shape=output.shape,
        fallback=output.detach(),
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
    always_keep_fallback: bool = False,
) -> SavedTensorRef:
    fallback_value = fallback.detach() if always_keep_fallback else None
    ref = SavedTensorRef(
        grad_fn=grad_fn,
        saved_attrs=saved_attrs,
        expected_shape=expected_shape,
        fallback=fallback_value,
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


def _linear_input_backward_program(
    weight: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(grad_output, weight)


def _linear_weight_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    if input_activation.dim() > 2:
        input_activation = input_activation.reshape(-1, input_activation.shape[-1])
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
    return torch.ops.aten.mm.default(
        torch.ops.aten.t.default(grad_output),
        input_activation,
    )


def _linear_bias_backward_program(grad_output: torch.Tensor) -> torch.Tensor:
    reduction_dims = tuple(range(grad_output.dim() - 1))
    if not reduction_dims:
        return grad_output
    return torch.ops.aten.sum.dim_IntList(grad_output, list(reduction_dims), False)


def _embedding_weight_backward_program(
    module: nn.Embedding,
    indices: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    padding_idx = -1 if module.padding_idx is None else module.padding_idx
    return torch.ops.aten.embedding_dense_backward.default(
        grad_output,
        indices,
        module.weight.shape[0],
        padding_idx,
        module.scale_grad_by_freq,
    )


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


def _layer_norm_input_backward_program(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    grad_input, _, _ = torch.ops.aten.native_layer_norm_backward.default(
        grad_output,
        input_activation,
        list(module.normalized_shape),
        mean,
        rstd,
        module.weight.detach() if module.weight is not None else None,
        module.bias.detach() if module.bias is not None else None,
        (True, False, False),
    )
    return grad_input


def _layer_norm_weight_backward_program(
    normalized_input: torch.Tensor,
    grad_output: torch.Tensor,
    parameter_shape: torch.Size,
) -> torch.Tensor:
    reduction_dims = _leading_reduction_dims(grad_output.dim(), len(parameter_shape))
    if not reduction_dims:
        return grad_output * normalized_input
    return torch.ops.aten.sum.dim_IntList(
        grad_output * normalized_input,
        list(reduction_dims),
        False,
    )


def _layer_norm_bias_backward_program(
    grad_output: torch.Tensor,
    parameter_shape: torch.Size,
) -> torch.Tensor:
    reduction_dims = _leading_reduction_dims(grad_output.dim(), len(parameter_shape))
    if not reduction_dims:
        return grad_output
    return torch.ops.aten.sum.dim_IntList(grad_output, list(reduction_dims), False)


def _layer_norm_normalized_input(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
) -> torch.Tensor:
    dims = _layer_norm_dims(input_activation.dim(), module.normalized_shape)
    mean = input_activation.mean(dim=dims, keepdim=True)
    centered = input_activation - mean
    var = (centered * centered).mean(dim=dims, keepdim=True)
    return centered * torch.rsqrt(var + module.eps)


def _layer_norm_input_jvp(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
) -> torch.Tensor:
    dims = _layer_norm_dims(input_activation.dim(), module.normalized_shape)
    mean = input_activation.mean(dim=dims, keepdim=True)
    centered = input_activation - mean
    var = (centered * centered).mean(dim=dims, keepdim=True)
    rstd = torch.rsqrt(var + module.eps)
    mean_tangent = input_tangent.mean(dim=dims, keepdim=True)
    centered_tangent = input_tangent - mean_tangent
    var_tangent = (2.0 * centered * centered_tangent).mean(dim=dims, keepdim=True)
    rstd_tangent = -0.5 * rstd.pow(3) * var_tangent
    normalized_tangent = centered_tangent * rstd + centered * rstd_tangent
    if module.weight is not None:
        normalized_tangent = normalized_tangent * module.weight.detach()
    return normalized_tangent


def _layer_norm_input_backward_jvp(
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
) -> torch.Tensor | None:
    if grad_output_tangent is None and input_tangent is None:
        return None
    dims = _layer_norm_dims(input_activation.dim(), module.normalized_shape)
    normalized_size = 1
    for size in module.normalized_shape:
        normalized_size *= int(size)
    weight = module.weight.detach() if module.weight is not None else None
    centered = input_activation - mean
    normalized = centered * rstd
    scaled_grad = grad_output if weight is None else grad_output * weight
    scaled_grad_tangent = (
        None
        if grad_output_tangent is None
        else grad_output_tangent
        if weight is None
        else grad_output_tangent * weight
    )

    if input_tangent is None:
        normalized_tangent = torch.zeros_like(input_activation)
        rstd_tangent = torch.zeros_like(rstd)
    else:
        mean_tangent = input_tangent.mean(dim=dims, keepdim=True)
        centered_tangent = input_tangent - mean_tangent
        var_tangent = (2.0 * centered * centered_tangent).mean(dim=dims, keepdim=True)
        rstd_tangent = -0.5 * rstd.pow(3) * var_tangent
        normalized_tangent = centered_tangent * rstd + centered * rstd_tangent

    sum_grad = scaled_grad.sum(dim=dims, keepdim=True)
    sum_grad_norm = (scaled_grad * normalized).sum(dim=dims, keepdim=True)
    inner = (
        normalized_size * scaled_grad
        - sum_grad
        - normalized * sum_grad_norm
    )
    inner_tangent = None
    if scaled_grad_tangent is not None:
        sum_grad_tangent = scaled_grad_tangent.sum(dim=dims, keepdim=True)
        sum_grad_norm_tangent = (scaled_grad_tangent * normalized).sum(
            dim=dims,
            keepdim=True,
        )
        inner_tangent = (
            normalized_size * scaled_grad_tangent
            - sum_grad_tangent
            - normalized * sum_grad_norm_tangent
        )
    if input_tangent is not None:
        sum_grad_norm_tangent = (scaled_grad * normalized_tangent).sum(
            dim=dims,
            keepdim=True,
        )
        input_inner_tangent = (
            -normalized_tangent * sum_grad_norm
            - normalized * sum_grad_norm_tangent
        )
        inner_tangent = (
            input_inner_tangent
            if inner_tangent is None
            else inner_tangent + input_inner_tangent
        )

    result = None
    if input_tangent is not None:
        result = (rstd_tangent / normalized_size) * inner
    if inner_tangent is not None:
        term = (rstd / normalized_size) * inner_tangent
        result = term if result is None else result + term
    return result


def _rms_norm_dims(
    input_ndim: int,
    normalized_shape: tuple[int, ...],
) -> tuple[int, ...]:
    normalized_ndim = len(normalized_shape)
    return tuple(range(input_ndim - normalized_ndim, input_ndim))


def _rms_norm_stats(
    input_activation: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> tuple[torch.Tensor, tuple[int, ...], int]:
    dims = _rms_norm_dims(input_activation.dim(), normalized_shape)
    square_mean = (input_activation * input_activation).mean(dim=dims, keepdim=True)
    rstd = torch.rsqrt(square_mean + eps)
    normalized_size = 1
    for size in normalized_shape:
        normalized_size *= int(size)
    return rstd, dims, normalized_size


def _rms_norm_jvp(
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    rstd, dims, _ = _rms_norm_stats(input_activation, normalized_shape, eps)
    square_mean_tangent = (2.0 * input_activation * input_tangent).mean(
        dim=dims,
        keepdim=True,
    )
    rstd_tangent = -0.5 * rstd.pow(3) * square_mean_tangent
    return input_tangent * rstd + input_activation * rstd_tangent


def _rms_norm_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor:
    rstd, dims, normalized_size = _rms_norm_stats(input_activation, normalized_shape, eps)
    inner = (grad_output * input_activation).sum(dim=dims, keepdim=True)
    return rstd * grad_output - input_activation * rstd.pow(3) * inner / normalized_size


def _rms_norm_backward_jvp(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
    normalized_shape: tuple[int, ...],
    eps: float,
) -> torch.Tensor | None:
    if grad_output_tangent is None and input_tangent is None:
        return None
    rstd, dims, normalized_size = _rms_norm_stats(input_activation, normalized_shape, eps)
    inner = (grad_output * input_activation).sum(dim=dims, keepdim=True)
    result = None
    if grad_output_tangent is not None:
        inner_tangent = (grad_output_tangent * input_activation).sum(
            dim=dims,
            keepdim=True,
        )
        result = (
            rstd * grad_output_tangent
            - input_activation * rstd.pow(3) * inner_tangent / normalized_size
        )
    if input_tangent is not None:
        square_mean_tangent = (2.0 * input_activation * input_tangent).mean(
            dim=dims,
            keepdim=True,
        )
        rstd_tangent = -0.5 * rstd.pow(3) * square_mean_tangent
        inner_tangent = (grad_output * input_tangent).sum(dim=dims, keepdim=True)
        term = (
            rstd_tangent * grad_output
            - input_tangent * rstd.pow(3) * inner / normalized_size
            - input_activation * 3.0 * rstd.pow(2) * rstd_tangent * inner / normalized_size
            - input_activation * rstd.pow(3) * inner_tangent / normalized_size
        )
        result = term if result is None else result + term
    return result


def _relu_backward_program(
    input_value: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.aten.threshold_backward.default(grad_output, input_value, 0)


def _unary_elementwise_kind(
    func: Any,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> tuple[str, float | None]:
    if func is torch.ops.aten.relu.default or getattr(func, "__name__", None) == "relu":
        return "relu", None
    if func is torch.ops.aten.sigmoid.default or getattr(func, "__name__", None) == "sigmoid":
        return "sigmoid", None
    if func is torch.ops.aten.tanh.default or getattr(func, "__name__", None) == "tanh":
        return "tanh", None
    if func is torch.ops.aten.square.default or getattr(func, "__name__", None) == "square":
        return "pow", 2.0
    if func is torch.ops.aten.pow.Tensor_Scalar or getattr(func, "__name__", None) == "pow":
        exponent = args[1] if len(args) > 1 else kwargs.get("exponent")
        if not isinstance(exponent, (int, float)):
            raise NotImplementedError(
                "ModularHVP graph traversal currently supports scalar pow exponents"
            )
        return "pow", float(exponent)
    raise NotImplementedError(f"ModularHVP graph traversal does not support {func}")


def _unary_elementwise_jvp(
    kind: str,
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    return _unary_elementwise_derivative(
        kind,
        input_activation,
        output_activation,
        scalar,
    ) * input_tangent


def _unary_elementwise_backward_program(
    kind: str,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    return _unary_elementwise_derivative(
        kind,
        input_activation,
        output_activation,
        scalar,
    ) * grad_output


def _unary_elementwise_backward_jvp(
    kind: str,
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    input_tangent: torch.Tensor | None,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor | None:
    derivative = _unary_elementwise_derivative(
        kind,
        input_activation,
        output_activation,
        scalar,
    )
    result = None
    if grad_output_tangent is not None:
        result = derivative * grad_output_tangent
    if input_tangent is not None:
        second = _unary_elementwise_second_derivative(
            kind,
            input_activation,
            output_activation,
            scalar,
        )
        term = grad_output * second * input_tangent
        result = term if result is None else result + term
    return result


def _unary_elementwise_derivative(
    kind: str,
    input_activation: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    if kind == "relu":
        value = output_activation if output_activation is not None else input_activation
        return (value > 0).to(dtype=input_activation.dtype)
    if kind == "sigmoid":
        output = (
            output_activation
            if output_activation is not None
            else torch.sigmoid(input_activation)
        )
        return output * (1.0 - output)
    if kind == "tanh":
        output = (
            output_activation if output_activation is not None else torch.tanh(input_activation)
        )
        return 1.0 - output * output
    if kind == "pow":
        exponent = 1.0 if scalar is None else scalar
        return exponent * input_activation.pow(exponent - 1.0)
    raise ValueError(f"unknown unary elementwise kind: {kind!r}")


def _unary_elementwise_second_derivative(
    kind: str,
    input_activation: torch.Tensor,
    output_activation: torch.Tensor | None,
    scalar: float | None,
) -> torch.Tensor:
    if kind == "relu":
        return torch.zeros_like(input_activation)
    if kind == "sigmoid":
        output = (
            output_activation
            if output_activation is not None
            else torch.sigmoid(input_activation)
        )
        derivative = output * (1.0 - output)
        return derivative * (1.0 - 2.0 * output)
    if kind == "tanh":
        output = (
            output_activation if output_activation is not None else torch.tanh(input_activation)
        )
        derivative = 1.0 - output * output
        return -2.0 * output * derivative
    if kind == "pow":
        exponent = 1.0 if scalar is None else scalar
        if exponent == 0.0:
            return torch.zeros_like(input_activation)
        return exponent * (exponent - 1.0) * input_activation.pow(exponent - 2.0)
    raise ValueError(f"unknown unary elementwise kind: {kind!r}")


def _gelu_jvp(
    input_activation: torch.Tensor,
    input_tangent: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    derivative = _gelu_derivative(input_activation, approximate)
    return derivative * input_tangent


def _gelu_backward_program(
    input_activation: torch.Tensor,
    grad_output: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    derivative = _gelu_derivative(input_activation, approximate)
    return grad_output * derivative


def _gelu_derivative(
    input_activation: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    if approximate == "tanh":
        coeff = (2.0 / torch.pi) ** 0.5
        inner = coeff * (input_activation + 0.044715 * input_activation.pow(3))
        tanh_inner = torch.tanh(inner)
        return 0.5 * (1 + tanh_inner) + 0.5 * input_activation * (
            1 - tanh_inner.pow(2)
        ) * coeff * (1 + 3 * 0.044715 * input_activation.pow(2))
    normal_cdf = 0.5 * (1 + torch.erf(input_activation / (2.0**0.5)))
    normal_pdf = torch.exp(-0.5 * input_activation.pow(2)) / (2.0 * torch.pi) ** 0.5
    return normal_cdf + input_activation * normal_pdf


def _gelu_second_derivative(
    input_activation: torch.Tensor,
    approximate: str,
) -> torch.Tensor:
    if approximate == "tanh":
        raise NotImplementedError(
            "GELU tanh second-derivative runtime rule is not implemented yet"
        )
    normal_pdf = torch.exp(-0.5 * input_activation.pow(2)) / (2.0 * torch.pi) ** 0.5
    return (2.0 - input_activation.pow(2)) * normal_pdf


def _make_linear_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    weight: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    weight = weight.detach()
    weight_t = torch.ops.aten.t.default(weight)

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = torch.matmul(input_tangent, weight_t)
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


def _make_layer_norm_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    module: nn.LayerNorm,
    input_activation: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()
    mean = mean.detach()
    rstd = rstd.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _layer_norm_input_jvp(
            module,
            input_activation,
            input_tangent,
        )
        output_grad_tangent = output_curvature(output_tangent)
        return _layer_norm_input_backward_program(
            module,
            input_activation,
            mean,
            rstd,
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


def _make_gelu_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    input_activation: torch.Tensor,
    approximate: str,
) -> Callable[[torch.Tensor], torch.Tensor]:
    input_activation = input_activation.detach()

    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = _gelu_jvp(input_activation, input_tangent, approximate)
        output_grad_tangent = output_curvature(output_tangent)
        return _gelu_backward_program(
            input_activation,
            output_grad_tangent,
            approximate,
        )

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


def _make_reshape_input_curvature(
    output_curvature: Callable[[torch.Tensor], torch.Tensor],
    record: ReshapeForwardRecord,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def input_curvature(input_tangent: torch.Tensor) -> torch.Tensor:
        output_tangent = input_tangent.reshape(record.output_shape)
        output_grad_tangent = output_curvature(output_tangent)
        return output_grad_tangent.reshape(record.input_shape)

    return input_curvature


def _forward_record_tangent_packet(
    record: ForwardRecord,
    tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
) -> dict[nn.Parameter, torch.Tensor] | None:
    if isinstance(record, LinearForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        weight_t = torch.ops.aten.t.default(record.module.weight.detach())
        return {
            parameter: torch.matmul(input_tangent, weight_t)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, EmbeddingForwardRecord):
        return None

    if isinstance(record, FunctionalLinearForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        weight_t = torch.ops.aten.t.default(record.weight.detach())
        return {
            parameter: torch.matmul(input_tangent, weight_t)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, Conv2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        weight = record.module.weight.detach()
        return {
            parameter: torch.ops.aten.convolution.default(
                input_tangent,
                weight,
                None,
                record.module.stride,
                record.module.padding,
                record.module.dilation,
                False,
                (0, 0),
                record.module.groups,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, BatchNorm2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        scale = _batch_norm2d_input_scale(record.module)
        return {
            parameter: input_tangent * _reshape_channel_tangent(
                scale,
                input_tangent.dim(),
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, LayerNormForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        return {
            parameter: _layer_norm_input_jvp(
                record.module,
                input_activation,
                input_tangent,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, RMSNormForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        return {
            parameter: _rms_norm_jvp(
                input_activation,
                input_tangent,
                record.normalized_shape,
                record.eps,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, ReLUForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        relu_output = record.output_activation.resolve().detach()
        return {
            parameter: _relu_backward_program(relu_output, input_tangent)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, GELUForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        return {
            parameter: _gelu_jvp(
                input_activation,
                input_tangent,
                record.approximate,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, FlattenForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: torch.flatten(
                input_tangent,
                start_dim=record.start_dim,
                end_dim=record.end_dim,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, AvgPool2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        kernel_size = _pair(record.module.kernel_size)
        stride = _pair(
            record.module.stride
            if record.module.stride is not None
            else record.module.kernel_size
        )
        padding = _pair(record.module.padding)
        return {
            parameter: torch.ops.aten.avg_pool2d.default(
                input_tangent,
                kernel_size,
                stride,
                padding,
                record.module.ceil_mode,
                record.module.count_include_pad,
                record.module.divisor_override,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, AdaptiveAvgPool2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: torch.ops.aten._adaptive_avg_pool2d.default(
                input_tangent,
                record.output_size,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, MaxPool2dForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        indices = record.indices.resolve().detach()
        return {
            parameter: _max_pool2d_jvp(input_tangent, indices)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, AddForwardRecord):
        result: dict[nn.Parameter, torch.Tensor] = {}
        if record.left_node_id is not None:
            left_packet = tangents_by_node.get(record.left_node_id)
            if left_packet:
                result.update(left_packet)
        if record.right_node_id is not None:
            right_packet = tangents_by_node.get(record.right_node_id)
            if right_packet:
                alpha = record.alpha
                for parameter, right_tangent in right_packet.items():
                    value = right_tangent if alpha == 1.0 else alpha * right_tangent
                    _accumulate_parameter_tensor(result, parameter, value)
        return result or None

    if isinstance(record, MulForwardRecord):
        left_packet = (
            tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        if not left_packet and not right_packet:
            return None
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        result: dict[nn.Parameter, torch.Tensor] = {}
        if left_packet:
            for parameter, left_tangent in left_packet.items():
                _accumulate_parameter_tensor(result, parameter, left_tangent * right)
        if right_packet:
            for parameter, right_tangent in right_packet.items():
                _accumulate_parameter_tensor(result, parameter, left * right_tangent)
        return result or None

    if isinstance(record, CatForwardRecord):
        packets = [
            tangents_by_node.get(node_id) if node_id is not None else None
            for node_id in record.input_node_ids
        ]
        parameters = set().union(*(set(packet) for packet in packets if packet))
        if not parameters:
            return None
        result: dict[nn.Parameter, torch.Tensor] = {}
        for parameter in parameters:
            reference = next(packet[parameter] for packet in packets if packet and parameter in packet)
            pieces = [
                packet[parameter]
                if packet is not None and parameter in packet
                else torch.zeros(
                    shape,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                for packet, shape in zip(packets, record.input_shapes, strict=True)
            ]
            result[parameter] = torch.cat(pieces, dim=record.dim)
        return result

    if isinstance(record, ReshapeForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.reshape(record.output_shape)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, SliceForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        length = record.end - record.start
        return {
            parameter: input_tangent.narrow(record.dim, record.start, length)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, SelectForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.select(record.dim, record.index)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, TransposeForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.transpose(record.dim0, record.dim1)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, ContiguousForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.contiguous()
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, CastForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        return {
            parameter: input_tangent.to(dtype=record.output_dtype)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, UnaryElementwiseForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        input_activation = record.input_activation.resolve().detach()
        output_activation = (
            None
            if record.output_activation is None
            else record.output_activation.resolve().detach()
        )
        return {
            parameter: _unary_elementwise_jvp(
                record.kind,
                input_activation,
                input_tangent,
                output_activation,
                record.scalar,
            )
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, MatmulForwardRecord):
        left_packet = (
            tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        if not left_packet and not right_packet:
            return None
        left = record.left_activation.resolve().detach()
        right = record.right_activation.resolve().detach()
        result: dict[nn.Parameter, torch.Tensor] = {}
        if left_packet:
            for parameter, left_tangent in left_packet.items():
                _accumulate_parameter_tensor(
                    result,
                    parameter,
                    torch.matmul(left_tangent, right),
                )
        if right_packet:
            for parameter, right_tangent in right_packet.items():
                _accumulate_parameter_tensor(
                    result,
                    parameter,
                    torch.matmul(left, right_tangent),
                )
        return result or None

    if isinstance(record, DivForwardRecord):
        left_packet = (
            tangents_by_node.get(record.left_node_id)
            if record.left_node_id is not None
            else None
        )
        right_packet = (
            tangents_by_node.get(record.right_node_id)
            if record.right_node_id is not None
            else None
        )
        if not left_packet and not right_packet:
            return None
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        result: dict[nn.Parameter, torch.Tensor] = {}
        if left_packet:
            for parameter, left_tangent in left_packet.items():
                _accumulate_parameter_tensor(result, parameter, left_tangent / right)
        if right_packet:
            for parameter, right_tangent in right_packet.items():
                _accumulate_parameter_tensor(
                    result,
                    parameter,
                    -(left * right_tangent) / (right * right),
                )
        return result or None

    if isinstance(record, SoftmaxForwardRecord):
        input_packet = tangents_by_node.get(record.input_node_id)
        if not input_packet:
            return None
        output = record.output_activation.resolve().detach()
        return {
            parameter: _softmax_jvp(output, input_tangent, record.dim)
            for parameter, input_tangent in input_packet.items()
        }

    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query_packet = (
            tangents_by_node.get(record.query_node_id)
            if record.query_node_id is not None
            else None
        )
        key_packet = (
            tangents_by_node.get(record.key_node_id)
            if record.key_node_id is not None
            else None
        )
        value_packet = (
            tangents_by_node.get(record.value_node_id)
            if record.value_node_id is not None
            else None
        )
        if not query_packet and not key_packet and not value_packet:
            return None
        query = record.query_activation.resolve().detach()
        key = record.key_activation.resolve().detach()
        value = record.value_activation.resolve().detach()
        attention = _attention_weights(
            query,
            key,
            record.attn_mask,
            record.is_causal,
            record.scale,
        )
        result: dict[nn.Parameter, torch.Tensor] = {}
        parameters = set(query_packet or ()) | set(key_packet or ()) | set(value_packet or ())
        for parameter in parameters:
            score_tangent = _attention_score_tangent(
                query,
                (query_packet or {}).get(parameter),
                key,
                (key_packet or {}).get(parameter),
                record.scale,
            )
            output_tangent = None
            if score_tangent is not None:
                output_tangent = torch.matmul(
                    _softmax_jvp(attention, score_tangent, -1),
                    value,
                )
            value_tangent = (value_packet or {}).get(parameter)
            if value_tangent is not None:
                term = torch.matmul(attention, value_tangent)
                output_tangent = term if output_tangent is None else output_tangent + term
            if output_tangent is not None:
                result[parameter] = output_tangent
        return result or None

    raise TypeError(f"unknown forward record: {type(record).__name__}")


def _propagate_backward_tangent_packet(
    record: ForwardRecord,
    output_grad_tangents: Mapping[nn.Parameter, torch.Tensor],
    grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
) -> None:
    if isinstance(record, LinearForwardRecord):
        weight = record.module.weight.detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _linear_input_backward_program(weight, output_grad_tangent),
            )
        return

    if isinstance(record, EmbeddingForwardRecord):
        return

    if isinstance(record, FunctionalLinearForwardRecord):
        weight = record.weight.detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _linear_input_backward_program(weight, output_grad_tangent),
            )
        return

    if isinstance(record, Conv2dForwardRecord):
        input_activation = _empty_from_saved_ref(
            record.input_activation,
            next(iter(output_grad_tangents.values())),
        )
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _conv2d_input_backward_program(
                    record.module,
                    input_activation,
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, BatchNorm2dForwardRecord):
        input_activation = _empty_from_saved_ref(
            record.input_activation,
            next(iter(output_grad_tangents.values())),
        )
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _batch_norm2d_input_backward_program(
                    record.module,
                    input_activation,
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, LayerNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        mean = record.mean.resolve().detach()
        rstd = record.rstd.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _layer_norm_input_backward_program(
                    record.module,
                    input_activation,
                    mean,
                    rstd,
                    output_grad_tangent,
                ),
            )
        return

    if isinstance(record, RMSNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _rms_norm_backward_program(
                    input_activation,
                    output_grad_tangent,
                    record.normalized_shape,
                    record.eps,
                ),
            )
        return

    if isinstance(record, ReLUForwardRecord):
        relu_output = record.output_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _relu_backward_program(relu_output, output_grad_tangent),
            )
        return

    if isinstance(record, GELUForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _gelu_backward_program(
                    input_activation,
                    output_grad_tangent,
                    record.approximate,
                ),
            )
        return

    if isinstance(record, FlattenForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.reshape(record.input_shape),
            )
        return

    if isinstance(record, AvgPool2dForwardRecord):
        like = next(iter(output_grad_tangents.values()))
        input_activation = _empty_from_saved_ref(record.input_activation, like)
        kernel_size = _pair(record.module.kernel_size)
        stride = _pair(
            record.module.stride
            if record.module.stride is not None
            else record.module.kernel_size
        )
        padding = _pair(record.module.padding)
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.ops.aten.avg_pool2d_backward.default(
                    output_grad_tangent,
                    input_activation,
                    kernel_size,
                    stride,
                    padding,
                    record.module.ceil_mode,
                    record.module.count_include_pad,
                    record.module.divisor_override,
                ),
            )
        return

    if isinstance(record, AdaptiveAvgPool2dForwardRecord):
        like = next(iter(output_grad_tangents.values()))
        input_activation = _empty_from_saved_ref(record.input_activation, like)
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.ops.aten._adaptive_avg_pool2d_backward.default(
                    output_grad_tangent,
                    input_activation,
                ),
            )
        return

    if isinstance(record, MaxPool2dForwardRecord):
        like = next(iter(output_grad_tangents.values()))
        input_activation = _empty_from_saved_ref(record.input_activation, like)
        module = record.module
        kernel_size = _pair(module.kernel_size)
        stride = _pair(module.stride if module.stride is not None else module.kernel_size)
        padding = _pair(module.padding)
        dilation = _pair(module.dilation)
        indices = record.indices.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                torch.ops.aten.max_pool2d_with_indices_backward.default(
                    output_grad_tangent,
                    input_activation,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    module.ceil_mode,
                    indices,
                ),
            )
        return

    if isinstance(record, SliceForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        length = record.end - record.start
        for parameter, output_grad_tangent in output_grad_tangents.items():
            input_grad_tangent = torch.zeros(
                record.input_shape,
                device=output_grad_tangent.device,
                dtype=output_grad_tangent.dtype,
            )
            input_grad_tangent.narrow(record.dim, record.start, length).copy_(
                output_grad_tangent,
            )
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                input_grad_tangent,
            )
        return

    if isinstance(record, TransposeForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.transpose(record.dim0, record.dim1),
            )
        return

    if isinstance(record, ContiguousForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent,
            )
        return

    if isinstance(record, MatmulForwardRecord):
        left = record.left_activation.resolve().detach()
        right = record.right_activation.resolve().detach()
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    _matmul_left_backward_program(output_grad_tangent, right),
                )
        if record.right_node_id is not None:
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    _matmul_right_backward_program(left, output_grad_tangent),
                )
        return

    if isinstance(record, DivForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    output_grad_tangent / right,
                )
        if record.right_node_id is not None:
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    -(output_grad_tangent * left) / (right * right),
                )
        return

    if isinstance(record, SoftmaxForwardRecord):
        output = record.output_activation.resolve().detach()
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _softmax_backward_program(output, output_grad_tangent, record.dim),
            )
        return

    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query = record.query_activation.resolve().detach()
        key = record.key_activation.resolve().detach()
        value = record.value_activation.resolve().detach()
        attention = _attention_weights(
            query,
            key,
            record.attn_mask,
            record.is_causal,
            record.scale,
        )
        scale = _attention_scale(query, record.scale)
        if record.query_node_id is not None:
            query_packet = grad_tangents_by_node.setdefault(record.query_node_id, {})
        else:
            query_packet = None
        if record.key_node_id is not None:
            key_packet = grad_tangents_by_node.setdefault(record.key_node_id, {})
        else:
            key_packet = None
        if record.value_node_id is not None:
            value_packet = grad_tangents_by_node.setdefault(record.value_node_id, {})
        else:
            value_packet = None
        for parameter, output_grad_tangent in output_grad_tangents.items():
            value_grad_tangent, score_grad_tangent = _attention_value_and_score_grads(
                attention,
                value,
                output_grad_tangent,
            )
            if value_packet is not None:
                _accumulate_parameter_tensor(
                    value_packet,
                    parameter,
                    value_grad_tangent,
                )
            if query_packet is not None:
                _accumulate_parameter_tensor(
                    query_packet,
                    parameter,
                    torch.matmul(score_grad_tangent, key) * scale,
                )
            if key_packet is not None:
                _accumulate_parameter_tensor(
                    key_packet,
                    parameter,
                    torch.matmul(score_grad_tangent.transpose(-2, -1), query) * scale,
                )
        return

    if isinstance(record, AddForwardRecord):
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    output_grad_tangent,
                )
        if record.right_node_id is not None:
            alpha = record.alpha
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    output_grad_tangent if alpha == 1.0 else alpha * output_grad_tangent,
                )
        return

    if isinstance(record, MulForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        if record.left_node_id is not None:
            left_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            left_shape = _value_shape(left)
            for parameter, output_grad_tangent in output_grad_tangents.items():
                value = output_grad_tangent * right
                _accumulate_parameter_tensor(
                    left_packet,
                    parameter,
                    _unbroadcast_like(value, left_shape),
                )
        if record.right_node_id is not None:
            right_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            right_shape = _value_shape(right)
            for parameter, output_grad_tangent in output_grad_tangents.items():
                value = left * output_grad_tangent
                _accumulate_parameter_tensor(
                    right_packet,
                    parameter,
                    _unbroadcast_like(value, right_shape),
                )
        return

    if isinstance(record, CatForwardRecord):
        sizes = [shape[record.dim] for shape in record.input_shapes]
        starts = [0]
        for size in sizes[:-1]:
            starts.append(starts[-1] + size)
        for input_node_id, shape, start, size in zip(
            record.input_node_ids,
            record.input_shapes,
            starts,
            sizes,
            strict=True,
        ):
            if input_node_id is None:
                continue
            node_packet = grad_tangents_by_node.setdefault(input_node_id, {})
            for parameter, output_grad_tangent in output_grad_tangents.items():
                _accumulate_parameter_tensor(
                    node_packet,
                    parameter,
                    output_grad_tangent.narrow(record.dim, start, size).reshape(shape),
                )
        return

    if isinstance(record, ReshapeForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.reshape(record.input_shape),
            )
        return

    if isinstance(record, SelectForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            input_grad_tangent = torch.zeros(
                record.input_shape,
                device=output_grad_tangent.device,
                dtype=output_grad_tangent.dtype,
            )
            input_grad_tangent.select(record.dim, record.index).copy_(output_grad_tangent)
            _accumulate_parameter_tensor(node_packet, parameter, input_grad_tangent)
        return

    if isinstance(record, CastForwardRecord):
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                output_grad_tangent.to(dtype=record.input_dtype),
            )
        return

    if isinstance(record, UnaryElementwiseForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        output_activation = (
            None
            if record.output_activation is None
            else record.output_activation.resolve().detach()
        )
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter, output_grad_tangent in output_grad_tangents.items():
            _accumulate_parameter_tensor(
                node_packet,
                parameter,
                _unary_elementwise_backward_program(
                    record.kind,
                    input_activation,
                    output_grad_tangent,
                    output_activation,
                    record.scalar,
                ),
            )
        return

    raise TypeError(f"unknown forward record: {type(record).__name__}")


def _propagate_backward_tangent_packet_with_grad(
    record: ForwardRecord,
    grad: torch.Tensor,
    output_grad_tangents: Mapping[nn.Parameter, torch.Tensor],
    forward_tangents_by_node: Mapping[int, Mapping[nn.Parameter, torch.Tensor]],
    grad_tangents_by_node: dict[int, dict[nn.Parameter, torch.Tensor]],
) -> None:
    if isinstance(record, MatmulForwardRecord):
        left = record.left_activation.resolve().detach()
        right = record.right_activation.resolve().detach()
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id, {})
            if record.left_node_id is not None
            else {}
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id, {})
            if record.right_node_id is not None
            else {}
        )
        parameters = set(output_grad_tangents) | set(left_packet) | set(right_packet)
        if record.left_node_id is not None:
            left_grad_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                right_tangent = right_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = _matmul_left_backward_program(grad_tangent, right)
                if right_tangent is not None:
                    term = _matmul_left_backward_program(grad, right_tangent)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(left_grad_packet, parameter, value)
        if record.right_node_id is not None:
            right_grad_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                left_tangent = left_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = _matmul_right_backward_program(left, grad_tangent)
                if left_tangent is not None:
                    term = _matmul_right_backward_program(left_tangent, grad)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(right_grad_packet, parameter, value)
        return

    if isinstance(record, MulForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id, {})
            if record.left_node_id is not None
            else {}
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id, {})
            if record.right_node_id is not None
            else {}
        )
        parameters = set(output_grad_tangents) | set(left_packet) | set(right_packet)
        if record.left_node_id is not None:
            left_grad_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            left_shape = _value_shape(left)
            for parameter in parameters:
                value = None
                grad_tangent = output_grad_tangents.get(parameter)
                right_tangent = right_packet.get(parameter)
                if grad_tangent is not None:
                    value = grad_tangent * right
                if right_tangent is not None:
                    term = grad * right_tangent
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(
                        left_grad_packet,
                        parameter,
                        _unbroadcast_like(value, left_shape),
                    )
        if record.right_node_id is not None:
            right_grad_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            right_shape = _value_shape(right)
            for parameter in parameters:
                value = None
                grad_tangent = output_grad_tangents.get(parameter)
                left_tangent = left_packet.get(parameter)
                if grad_tangent is not None:
                    value = left * grad_tangent
                if left_tangent is not None:
                    term = left_tangent * grad
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(
                        right_grad_packet,
                        parameter,
                        _unbroadcast_like(value, right_shape),
                    )
        return

    if isinstance(record, DivForwardRecord):
        left = _detach_value(record.left_value)
        right = _detach_value(record.right_value)
        left_packet = (
            forward_tangents_by_node.get(record.left_node_id, {})
            if record.left_node_id is not None
            else {}
        )
        right_packet = (
            forward_tangents_by_node.get(record.right_node_id, {})
            if record.right_node_id is not None
            else {}
        )
        parameters = set(output_grad_tangents) | set(left_packet) | set(right_packet)
        if record.left_node_id is not None:
            left_grad_packet = grad_tangents_by_node.setdefault(record.left_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                right_tangent = right_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = grad_tangent / right
                if right_tangent is not None:
                    term = -(grad * right_tangent) / (right * right)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(left_grad_packet, parameter, value)
        if record.right_node_id is not None:
            right_grad_packet = grad_tangents_by_node.setdefault(record.right_node_id, {})
            for parameter in parameters:
                grad_tangent = output_grad_tangents.get(parameter)
                left_tangent = left_packet.get(parameter)
                right_tangent = right_packet.get(parameter)
                value = None
                if grad_tangent is not None:
                    value = -(grad_tangent * left) / (right * right)
                if left_tangent is not None:
                    term = -(grad * left_tangent) / (right * right)
                    value = term if value is None else value + term
                if right_tangent is not None:
                    term = 2.0 * grad * left * right_tangent / (right * right * right)
                    value = term if value is None else value + term
                if value is not None:
                    _accumulate_parameter_tensor(right_grad_packet, parameter, value)
        return

    if isinstance(record, UnaryElementwiseForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        output_activation = (
            None
            if record.output_activation is None
            else record.output_activation.resolve().detach()
        )
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _unary_elementwise_backward_jvp(
                record.kind,
                input_activation,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
                output_activation,
                record.scalar,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, GELUForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        derivative = _gelu_derivative(input_activation, record.approximate)
        second = _gelu_second_derivative(input_activation, record.approximate)
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = None
            grad_tangent = output_grad_tangents.get(parameter)
            input_tangent = input_packet.get(parameter)
            if grad_tangent is not None:
                value = derivative * grad_tangent
            if input_tangent is not None:
                term = grad * second * input_tangent
                value = term if value is None else value + term
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, LayerNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        mean = record.mean.resolve().detach()
        rstd = record.rstd.resolve().detach()
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _layer_norm_input_backward_jvp(
                record.module,
                input_activation,
                mean,
                rstd,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, RMSNormForwardRecord):
        input_activation = record.input_activation.resolve().detach()
        input_packet = forward_tangents_by_node.get(record.input_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(input_packet):
            value = _rms_norm_backward_jvp(
                input_activation,
                grad,
                output_grad_tangents.get(parameter),
                input_packet.get(parameter),
                record.normalized_shape,
                record.eps,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, SoftmaxForwardRecord):
        output = record.output_activation.resolve().detach()
        output_packet = forward_tangents_by_node.get(record.output_node_id, {})
        node_packet = grad_tangents_by_node.setdefault(record.input_node_id, {})
        for parameter in set(output_grad_tangents) | set(output_packet):
            value = _softmax_backward_jvp(
                output,
                grad,
                output_grad_tangents.get(parameter),
                output_packet.get(parameter),
                record.dim,
            )
            if value is not None:
                _accumulate_parameter_tensor(node_packet, parameter, value)
        return

    if isinstance(record, ScaledDotProductAttentionForwardRecord):
        query = record.query_activation.resolve().detach()
        key = record.key_activation.resolve().detach()
        value = record.value_activation.resolve().detach()
        query_packet = (
            forward_tangents_by_node.get(record.query_node_id, {})
            if record.query_node_id is not None
            else {}
        )
        key_packet = (
            forward_tangents_by_node.get(record.key_node_id, {})
            if record.key_node_id is not None
            else {}
        )
        value_packet = (
            forward_tangents_by_node.get(record.value_node_id, {})
            if record.value_node_id is not None
            else {}
        )
        query_grad_packet = (
            grad_tangents_by_node.setdefault(record.query_node_id, {})
            if record.query_node_id is not None
            else None
        )
        key_grad_packet = (
            grad_tangents_by_node.setdefault(record.key_node_id, {})
            if record.key_node_id is not None
            else None
        )
        value_grad_packet = (
            grad_tangents_by_node.setdefault(record.value_node_id, {})
            if record.value_node_id is not None
            else None
        )
        parameters = (
            set(output_grad_tangents)
            | set(query_packet)
            | set(key_packet)
            | set(value_packet)
        )
        for parameter in parameters:
            query_grad_tangent, key_grad_tangent, value_grad_tangent = (
                _attention_backward_jvp(
                    query,
                    query_packet.get(parameter),
                    key,
                    key_packet.get(parameter),
                    value,
                    value_packet.get(parameter),
                    grad,
                    output_grad_tangents.get(parameter),
                    record.attn_mask,
                    record.is_causal,
                    record.scale,
                )
            )
            if query_grad_packet is not None and query_grad_tangent is not None:
                _accumulate_parameter_tensor(
                    query_grad_packet,
                    parameter,
                    query_grad_tangent,
                )
            if key_grad_packet is not None and key_grad_tangent is not None:
                _accumulate_parameter_tensor(
                    key_grad_packet,
                    parameter,
                    key_grad_tangent,
                )
            if value_grad_packet is not None and value_grad_tangent is not None:
                _accumulate_parameter_tensor(
                    value_grad_packet,
                    parameter,
                    value_grad_tangent,
                )
        return

    _propagate_backward_tangent_packet(
        record,
        output_grad_tangents,
        grad_tangents_by_node,
    )


def _accumulate_parameter_tensor(
    values_by_parameter: dict[nn.Parameter, torch.Tensor],
    parameter: nn.Parameter,
    value: torch.Tensor,
) -> None:
    existing = values_by_parameter.get(parameter)
    if existing is None:
        values_by_parameter[parameter] = value
    else:
        values_by_parameter[parameter] = existing + value


def _empty_from_saved_ref(
    ref: SavedTensorRef,
    like: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        ref.expected_shape,
        device=like.device,
        dtype=like.dtype,
    )


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
    for module in _iter_supported_leaf_modules_with_duplicates(model):
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


def _make_loss_output_curvature(
    record: LossRecord,
    grad: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    if isinstance(record, MSELossRecord):
        factor = grad * _mse_hessian_scale(record)

        def mse_output_curvature(value: torch.Tensor) -> torch.Tensor:
            return _scale_tensor_for_loss(value, factor)

        return mse_output_curvature

    if isinstance(record, CrossEntropyLossRecord):
        logits = record.logits
        target = record.target
        probabilities = torch.softmax(logits, dim=-1)
        valid = target != record.ignore_index
        if record.reduction == "mean":
            normalizer = valid.sum().clamp_min(1).to(dtype=logits.dtype)
        elif record.reduction == "sum":
            normalizer = torch.ones((), device=logits.device, dtype=logits.dtype)
        else:
            raise ValueError(f"unknown cross_entropy reduction: {record.reduction!r}")
        grad_factor = grad.to(device=logits.device, dtype=logits.dtype) / normalizer

        def cross_entropy_output_curvature(value: torch.Tensor) -> torch.Tensor:
            value = value.to(dtype=logits.dtype)
            weighted = (probabilities * value).sum(dim=-1, keepdim=True)
            result = probabilities * (value - weighted)
            return torch.where(valid.unsqueeze(-1), result, torch.zeros_like(result)) * grad_factor

        return cross_entropy_output_curvature

    raise TypeError(f"unknown loss record: {type(record).__name__}")


def _scale_tensor_for_loss(value: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    if value.is_contiguous():
        value.mul_(factor)
        return value
    return value * factor


def _normalize_dim(dim: int, ndim: int) -> int:
    if dim < 0:
        dim += ndim
    if dim < 0 or dim >= ndim:
        raise IndexError(f"dimension out of range: {dim}")
    return dim


def _normalize_slice_start(start: Any, dim_size: int) -> int:
    if start is None:
        return 0
    value = int(start)
    if value < 0:
        value += dim_size
    return max(0, min(value, dim_size))


def _normalize_slice_end(end: Any, dim_size: int) -> int:
    if end is None:
        return dim_size
    value = int(end)
    if value < 0:
        value += dim_size
    return max(0, min(value, dim_size))


def _normalize_select_index(index: int, dim_size: int) -> int:
    if index < 0:
        index += dim_size
    if index < 0 or index >= dim_size:
        raise IndexError(f"select index out of range: {index}")
    return index


def _canonicalize_index(index: tuple[Any, ...], ndim: int) -> tuple[Any, ...]:
    ellipsis_count = sum(1 for item in index if item is Ellipsis)
    if ellipsis_count > 1:
        raise IndexError("an index can only have a single ellipsis")
    consumed_dims = sum(1 for item in index if item is not Ellipsis and item is not None)
    if consumed_dims > ndim:
        raise IndexError("too many indices for tensor")
    result: list[Any] = []
    for item in index:
        if item is Ellipsis:
            result.extend([slice(None)] * (ndim - consumed_dims))
        else:
            result.append(item)
    if ellipsis_count == 0:
        result.extend([slice(None)] * (ndim - consumed_dims))
    return tuple(result)


def _value_shape(value: torch.Tensor | float | int) -> torch.Size:
    if isinstance(value, torch.Tensor):
        return value.shape
    return torch.Size(())


def _unbroadcast_like(value: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    if value.shape == target_shape:
        return value
    if len(target_shape) == 0:
        return value.sum()
    result = value
    while result.dim() > len(target_shape):
        result = result.sum(dim=0)
    for dim, target_size in enumerate(target_shape):
        if target_size == 1 and result.shape[dim] != 1:
            result = result.sum(dim=dim, keepdim=True)
    return result.reshape(target_shape)


def _split_output_sizes(
    dim_size: int,
    split_spec: Any,
    *,
    output_count: int,
) -> list[int]:
    if isinstance(split_spec, (tuple, list)):
        return [int(size) for size in split_spec]
    split_size = int(split_spec)
    if split_size <= 0:
        raise ValueError("split size must be positive")
    sizes: list[int] = []
    remaining = dim_size
    while remaining > 0 and len(sizes) < output_count:
        size = min(split_size, remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def _leading_reduction_dims(ndim: int, parameter_ndim: int) -> tuple[int, ...]:
    return tuple(range(ndim - parameter_ndim))


def _layer_norm_dims(
    input_ndim: int,
    normalized_shape: tuple[int, ...] | list[int] | torch.Size,
) -> tuple[int, ...]:
    normalized_ndim = len(tuple(normalized_shape))
    return tuple(range(input_ndim - normalized_ndim, input_ndim))


def _layer_norm_stat_shape(
    input_shape: torch.Size,
    normalized_shape: tuple[int, ...] | list[int] | torch.Size,
) -> torch.Size:
    normalized_ndim = len(tuple(normalized_shape))
    leading = tuple(input_shape[: len(input_shape) - normalized_ndim])
    return torch.Size((*leading, *([1] * normalized_ndim)))


def _detach_value(value: torch.Tensor | float | int) -> torch.Tensor | float | int:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _transpose_last_two(value: torch.Tensor) -> torch.Tensor:
    return value.transpose(-2, -1)


def _matmul_left_backward_program(
    grad_output: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(grad_output, _transpose_last_two(right))


def _matmul_right_backward_program(
    left: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    return torch.matmul(_transpose_last_two(left), grad_output)


def _softmax_jvp(
    softmax_output: torch.Tensor,
    input_tangent: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    weighted = (softmax_output * input_tangent).sum(dim=dim, keepdim=True)
    return softmax_output * (input_tangent - weighted)


def _softmax_backward_program(
    softmax_output: torch.Tensor,
    grad_output: torch.Tensor,
    dim: int,
) -> torch.Tensor:
    centered = grad_output - (grad_output * softmax_output).sum(dim=dim, keepdim=True)
    return softmax_output * centered


def _softmax_backward_jvp(
    softmax_output: torch.Tensor,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    softmax_output_tangent: torch.Tensor | None,
    dim: int,
) -> torch.Tensor | None:
    if grad_output_tangent is None and softmax_output_tangent is None:
        return None
    centered = grad_output - (grad_output * softmax_output).sum(dim=dim, keepdim=True)
    result = None
    if grad_output_tangent is not None:
        centered_tangent = grad_output_tangent - (
            grad_output_tangent * softmax_output
        ).sum(dim=dim, keepdim=True)
        result = softmax_output * centered_tangent
    if softmax_output_tangent is not None:
        centered_tangent = -(
            grad_output * softmax_output_tangent
        ).sum(dim=dim, keepdim=True)
        term = softmax_output_tangent * centered + softmax_output * centered_tangent
        result = term if result is None else result + term
    return result


def _attention_scale(query: torch.Tensor, scale: float | None) -> float:
    return (query.size(-1) ** -0.5) if scale is None else float(scale)


def _attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    scores = torch.matmul(query, key.transpose(-2, -1)) * _attention_scale(query, scale)
    if is_causal:
        query_len = query.size(-2)
        key_len = key.size(-2)
        causal_mask = torch.ones(
            query_len,
            key_len,
            dtype=torch.bool,
            device=query.device,
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    return scores


def _attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    return torch.softmax(
        _attention_scores(query, key, attn_mask, is_causal, scale),
        dim=-1,
    )


def _attention_score_tangent(
    query: torch.Tensor,
    query_tangent: torch.Tensor | None,
    key: torch.Tensor,
    key_tangent: torch.Tensor | None,
    scale: float | None,
) -> torch.Tensor | None:
    scale_value = _attention_scale(query, scale)
    return _sum_optional_tensors(
        torch.matmul(query_tangent, key.transpose(-2, -1)) * scale_value
        if query_tangent is not None
        else None,
        torch.matmul(query, key_tangent.transpose(-2, -1)) * scale_value
        if key_tangent is not None
        else None,
    )


def _attention_value_and_score_grads(
    attention: torch.Tensor,
    value: torch.Tensor,
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    value_grad = torch.matmul(attention.transpose(-2, -1), grad_output)
    attention_grad = torch.matmul(grad_output, value.transpose(-2, -1))
    score_grad = _softmax_backward_program(attention, attention_grad, -1)
    return value_grad, score_grad


def _attention_backward_jvp(
    query: torch.Tensor,
    query_tangent: torch.Tensor | None,
    key: torch.Tensor,
    key_tangent: torch.Tensor | None,
    value: torch.Tensor,
    value_tangent: torch.Tensor | None,
    grad_output: torch.Tensor,
    grad_output_tangent: torch.Tensor | None,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    scale: float | None,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    attention = _attention_weights(query, key, attn_mask, is_causal, scale)
    score_tangent = _attention_score_tangent(
        query,
        query_tangent,
        key,
        key_tangent,
        scale,
    )
    attention_tangent = (
        _softmax_jvp(attention, score_tangent, -1)
        if score_tangent is not None
        else None
    )
    _, score_grad = _attention_value_and_score_grads(
        attention,
        value,
        grad_output,
    )
    attention_grad = torch.matmul(grad_output, value.transpose(-2, -1))

    value_grad_tangent = None
    if attention_tangent is not None:
        value_grad_tangent = torch.matmul(
            attention_tangent.transpose(-2, -1),
            grad_output,
        )
    if grad_output_tangent is not None:
        term = torch.matmul(attention.transpose(-2, -1), grad_output_tangent)
        value_grad_tangent = (
            term if value_grad_tangent is None else value_grad_tangent + term
        )

    attention_grad_tangent = None
    if grad_output_tangent is not None:
        attention_grad_tangent = torch.matmul(
            grad_output_tangent,
            value.transpose(-2, -1),
        )
    if value_tangent is not None:
        term = torch.matmul(grad_output, value_tangent.transpose(-2, -1))
        attention_grad_tangent = (
            term if attention_grad_tangent is None else attention_grad_tangent + term
        )

    score_grad_tangent = _softmax_backward_jvp(
        attention,
        attention_grad,
        attention_grad_tangent,
        attention_tangent,
        -1,
    )
    scale_value = _attention_scale(query, scale)
    query_grad_tangent = None
    if score_grad_tangent is not None:
        query_grad_tangent = torch.matmul(score_grad_tangent, key) * scale_value
    if key_tangent is not None:
        term = torch.matmul(score_grad, key_tangent) * scale_value
        query_grad_tangent = (
            term if query_grad_tangent is None else query_grad_tangent + term
        )

    key_grad_tangent = None
    if score_grad_tangent is not None:
        key_grad_tangent = (
            torch.matmul(score_grad_tangent.transpose(-2, -1), query) * scale_value
        )
    if query_tangent is not None:
        term = torch.matmul(score_grad.transpose(-2, -1), query_tangent) * scale_value
        key_grad_tangent = (
            term if key_grad_tangent is None else key_grad_tangent + term
        )

    return query_grad_tangent, key_grad_tangent, value_grad_tangent


def _sum_optional_tensors(
    left: torch.Tensor | None,
    right: torch.Tensor | None,
) -> torch.Tensor | None:
    if left is None:
        return right
    if right is None:
        return left
    return left + right


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
    scale = _batch_norm2d_input_scale(module)
    return input_tangent * _reshape_channel_tangent(scale, input_tangent.dim())


def _batch_norm2d_input_scale(module: nn.BatchNorm2d) -> torch.Tensor:
    scale = torch.rsqrt(_batch_norm_running_var(module) + module.eps)
    if module.weight is not None:
        scale = scale * module.weight.detach()
    return scale


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


# Backward-compatible internal name used by earlier milestones/tests.
LocalMLPHVPRuntime = EagerHVPRuntime
