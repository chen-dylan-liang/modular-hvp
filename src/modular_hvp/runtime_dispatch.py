"""GraphTensor dispatch and forward-record construction."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from modular_hvp.dispatch import _get_aten_overload
from modular_hvp.graph_tensor import GraphTensor
from modular_hvp.kernels import (
    _broadcast_like,
    _canonicalize_index,
    _gelu_jvp,
    _layer_norm_input_jvp,
    _layer_norm_normalized_input_from_stats,
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
from modular_hvp.records import (
    AddForwardRecord,
    CastForwardRecord,
    CatForwardRecord,
    ContiguousForwardRecord,
    DivForwardRecord,
    DropoutForwardRecord,
    FunctionalLayerNormForwardRecord,
    FunctionalLinearForwardRecord,
    MatmulForwardRecord,
    MaskedFillForwardRecord,
    MulForwardRecord,
    RMSNormForwardRecord,
    ReshapeForwardRecord,
    ScaledDotProductAttentionForwardRecord,
    SelectForwardRecord,
    SliceForwardRecord,
    SoftmaxForwardRecord,
    TransposeForwardRecord,
    UnaryElementwiseForwardRecord,
    ForwardRecord,
)
from modular_hvp.saved_tensors import (
    _make_exact_saved_tensor_ref,
    _make_functional_linear_input_activation_ref,
    _make_saved_tensor_ref,
    _make_softmax_output_activation_ref,
)


class GraphDispatchMixin:
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
                            self._register_forward_record_hook(record, item)
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
                self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
                self._register_forward_record_hook(record, item)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
                ),
                right_activation=_make_saved_tensor_ref(
                    grad_fn=output.grad_fn,
                    saved_attrs=("_saved_other", "_saved_mat2"),
                    expected_shape=right_primal.shape,
                    fallback=right_primal,
                ),
            )
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
            self._register_forward_record_hook(record, output)
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
                local_output_tangents=self._share_local_output_tangents_by_block(
                    local_output_tangents,
                ),
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
                ),
                right_activation=_make_saved_tensor_ref(
                    grad_fn=output.primal.grad_fn,
                    saved_attrs=("_saved_other", "_saved_mat2"),
                    expected_shape=right_primal.shape,
                    fallback=right_primal,
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
