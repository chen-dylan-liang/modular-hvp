"""Block-channel dual tensor runtime for the public ``modular_hvp`` API."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from modular_hvp.runtime import _resolve_parameter_blocks


class BlockDualTensor(torch.Tensor):
    """Tensor wrapper carrying one independent tangent channel per parameter."""

    __slots__ = ("_primal", "_tangents")

    @staticmethod
    def __new__(
        cls,
        primal: torch.Tensor,
        tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> "BlockDualTensor":
        if primal.layout != torch.strided:
            raise NotImplementedError("BlockDualTensor supports strided tensors only")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            primal.shape,
            strides=primal.stride(),
            storage_offset=primal.storage_offset(),
            dtype=primal.dtype,
            layout=primal.layout,
            device=primal.device,
            requires_grad=primal.requires_grad,
        )

    def __init__(
        self,
        primal: torch.Tensor,
        tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> None:
        self._primal = primal
        self._tangents = dict(tangents)

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
        if _has_out_argument(kwargs):
            raise NotImplementedError(f"BlockDualTensor rule not implemented for {func}")
        rule = _TORCH_FUNCTION_RULES.get(_function_name(func))
        if rule is None:
            raise NotImplementedError(f"BlockDualTensor rule not implemented for {func}")
        with torch._C.DisableTorchFunctionSubclass():
            return rule(func, args, kwargs)

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
        if _has_out_argument(kwargs):
            raise NotImplementedError(f"BlockDualTensor rule not implemented for {func}")
        rule = _ATEN_RULES.get(func)
        if rule is None:
            raise NotImplementedError(f"BlockDualTensor rule not implemented for {func}")
        return rule(func, args, kwargs)

    @property
    def primal(self) -> torch.Tensor:
        return self._primal

    @property
    def tangents(self) -> dict[nn.Parameter, torch.Tensor]:
        return self._tangents

    def __repr__(self) -> str:
        return (
            f"BlockDualTensor(primal={self._primal!r}, "
            f"channels={len(self._tangents)})"
        )


@dataclass(slots=True)
class _ParameterPatch:
    module: nn.Module
    leaf_name: str
    original: nn.Parameter


class BlockDualHVPRuntime:
    """Context manager backing the default ``modular_hvp`` implementation."""

    _ACTIVE_RUNTIME: "BlockDualHVPRuntime | None" = None
    _ACTIVE_ATTR = "_modular_hvp_block_dual_active"

    def __init__(
        self,
        *,
        model: nn.Module,
        tangents: Mapping[str | nn.Parameter, torch.Tensor],
    ) -> None:
        self.model = model
        self.parameter_blocks = _resolve_parameter_blocks(model, tangents)
        self._patches: list[_ParameterPatch] = []
        self._entered = False
        self._backward_called = False

    def __enter__(self) -> "BlockDualHVPRuntime":
        if self._entered:
            raise RuntimeError("modular_hvp contexts are single-use")
        self._entered = True
        if BlockDualHVPRuntime._ACTIVE_RUNTIME is not None:
            raise RuntimeError("another modular_hvp context is already active")
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                raise RuntimeError("this model already has an active modular_hvp context")

        for block in self.parameter_blocks:
            setattr(block.parameter, "hvp", None)
        self._patch_parameters()
        for module in self.model.modules():
            setattr(module, self._ACTIVE_ATTR, True)
        BlockDualHVPRuntime._ACTIVE_RUNTIME = self
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if BlockDualHVPRuntime._ACTIVE_RUNTIME is self:
            BlockDualHVPRuntime._ACTIVE_RUNTIME = None
        self._restore_parameters()
        for module in self.model.modules():
            if getattr(module, self._ACTIVE_ATTR, False):
                delattr(module, self._ACTIVE_ATTR)
        return None

    def backward(
        self,
        loss: BlockDualTensor,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        if self._backward_called:
            raise RuntimeError("loss.backward() was already called in this context")
        self._backward_called = True
        kwargs = _normalize_backward_kwargs(args, kwargs)
        if kwargs:
            raise NotImplementedError(
                "BlockDualHVPRuntime currently supports loss.backward() without "
                "extra arguments"
            )
        if loss.primal.ndim != 0:
            raise RuntimeError("modular_hvp currently requires a scalar loss")

        for block in self.parameter_blocks:
            tangent_loss = loss.tangents.get(block.parameter)
            if tangent_loss is None:
                hvp = torch.zeros_like(
                    block.parameter,
                    memory_format=torch.preserve_format,
                )
            else:
                hvp = torch.autograd.grad(
                    tangent_loss,
                    block.parameter,
                    retain_graph=True,
                    create_graph=False,
                    materialize_grads=True,
                )[0]
            setattr(block.parameter, "hvp", hvp.detach())

        loss.primal.backward()

    def _patch_parameters(self) -> None:
        for block in self.parameter_blocks:
            module, leaf_name = _resolve_parameter_owner(self.model, block.name)
            original = module._parameters[leaf_name]
            if original is not block.parameter:
                raise RuntimeError(f"parameter {block.name!r} changed before entry")
            self._patches.append(
                _ParameterPatch(module=module, leaf_name=leaf_name, original=original)
            )
            module._parameters[leaf_name] = make_block_dual(
                block.parameter,
                {block.parameter: block.tangent},
            )

    def _restore_parameters(self) -> None:
        for patch in reversed(self._patches):
            patch.module._parameters[patch.leaf_name] = patch.original
        self._patches.clear()


def make_block_dual(
    primal: torch.Tensor,
    tangents: Mapping[nn.Parameter, torch.Tensor],
) -> BlockDualTensor:
    for parameter, tangent in tangents.items():
        if tangent.shape != primal.shape and parameter is primal:
            raise ValueError(
                f"block tangent has shape {tuple(tangent.shape)}; "
                f"expected {tuple(primal.shape)}"
            )
        if tangent.device != primal.device:
            raise ValueError(
                f"block tangent is on {tangent.device}; expected {primal.device}"
            )
        if tangent.dtype != primal.dtype:
            raise ValueError(
                f"block tangent has dtype {tangent.dtype}; expected {primal.dtype}"
            )
    return BlockDualTensor(primal, tangents)


def is_block_dual(value: object) -> bool:
    return isinstance(value, BlockDualTensor)


Rule = Callable[[Any, tuple[Any, ...], dict[str, Any]], Any]
_TORCH_FUNCTION_RULES: dict[str, Rule] = {}
_ATEN_RULES: dict[Any, Rule] = {}


def _resolve_parameter_owner(model: nn.Module, param_name: str) -> tuple[nn.Module, str]:
    module_path, _, leaf_name = param_name.rpartition(".")
    module = model.get_submodule(module_path) if module_path else model
    return module, leaf_name


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


def _has_out_argument(kwargs: dict[str, Any]) -> bool:
    out = kwargs.get("out")
    if out is None:
        return False
    if isinstance(out, tuple):
        return any(item is not None for item in out)
    return True


def _function_name(func: Any) -> str:
    return getattr(func, "__name__", repr(func))


def _tree_map(fn: Callable[[Any], Any], value: Any) -> Any:
    if is_block_dual(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [_tree_map(fn, item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(fn, item) for key, item in value.items()}
    return value


def _primal(value: Any) -> Any:
    return value.primal if is_block_dual(value) else value


def _tree_primal(value: Any) -> Any:
    return _tree_map(_primal, value)


def _tangents(value: Any) -> Mapping[nn.Parameter, torch.Tensor]:
    return value.tangents if is_block_dual(value) else {}


def _all_channels(*values: Any) -> set[nn.Parameter]:
    channels: set[nn.Parameter] = set()

    def visit(value: Any) -> None:
        if is_block_dual(value):
            channels.update(value.tangents)
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for value in values:
        visit(value)
    return channels


def _zero_scalar_like(value: torch.Tensor) -> torch.Tensor:
    return value.new_zeros(())


def _wrap(
    primal_output: torch.Tensor,
    tangents: Mapping[nn.Parameter, torch.Tensor],
) -> BlockDualTensor:
    normalized: dict[nn.Parameter, torch.Tensor] = {}
    for parameter, tangent in tangents.items():
        if tangent.shape != primal_output.shape:
            tangent = torch.broadcast_to(tangent, primal_output.shape)
        normalized[parameter] = tangent
    return make_block_dual(primal_output, normalized)


def _add_terms(*terms: torch.Tensor | None) -> torch.Tensor | None:
    active = [term for term in terms if term is not None]
    if not active:
        return None
    result = active[0]
    for term in active[1:]:
        result = result + term
    return result


def _binary_parts(
    args: tuple[Any, ...],
) -> tuple[Any, Any, Any, Any, Mapping[nn.Parameter, torch.Tensor], Mapping[nn.Parameter, torch.Tensor]]:
    left, right = args[:2]
    left_p = _primal(left)
    right_p = _primal(right)
    return left, right, left_p, right_p, _tangents(left), _tangents(right)


def _add_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    alpha = kwargs.get("alpha", 1)
    left, right, left_p, right_p, left_t, right_t = _binary_parts(args)
    primal_output = func(left_p, right_p, **kwargs)
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel in _all_channels(left, right):
        terms = [left_t.get(channel)]
        right_term = right_t.get(channel)
        if right_term is not None:
            terms.append(alpha * right_term)
        tangent = _add_terms(*terms)
        if tangent is not None:
            tangents[channel] = tangent
    return _wrap(primal_output, tangents)


def _sub_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    alpha = kwargs.get("alpha", 1)
    left, right, left_p, right_p, left_t, right_t = _binary_parts(args)
    primal_output = func(left_p, right_p, **kwargs)
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel in _all_channels(left, right):
        terms = [left_t.get(channel)]
        right_term = right_t.get(channel)
        if right_term is not None:
            terms.append(-alpha * right_term)
        tangent = _add_terms(*terms)
        if tangent is not None:
            tangents[channel] = tangent
    return _wrap(primal_output, tangents)


def _mul_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    left, right, left_p, right_p, left_t, right_t = _binary_parts(args)
    primal_output = func(left_p, right_p, **kwargs)
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel in _all_channels(left, right):
        terms: list[torch.Tensor | None] = []
        if channel in left_t:
            terms.append(left_t[channel] * right_p)
        if channel in right_t:
            terms.append(left_p * right_t[channel])
        tangent = _add_terms(*terms)
        if tangent is not None:
            tangents[channel] = tangent
    return _wrap(primal_output, tangents)


def _div_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    left, right, left_p, right_p, left_t, right_t = _binary_parts(args)
    primal_output = func(left_p, right_p, **kwargs)
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel in _all_channels(left, right):
        left_term = left_t.get(channel)
        right_term = right_t.get(channel)
        terms: list[torch.Tensor | None] = []
        if left_term is not None:
            terms.append(left_term / right_p)
        if right_term is not None:
            terms.append(-(left_p * right_term) / right_p.pow(2))
        tangent = _add_terms(*terms)
        if tangent is not None:
            tangents[channel] = tangent
    return _wrap(primal_output, tangents)


def _pow_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    base = args[0]
    exponent = args[1]
    base_p = _primal(base)
    exponent_p = _primal(exponent)
    primal_output = func(base_p, exponent_p, **kwargs)
    if is_block_dual(exponent):
        raise NotImplementedError("BlockDualTensor tensor exponents are not supported")
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel, tangent in _tangents(base).items():
        tangents[channel] = exponent_p * base_p.pow(exponent_p - 1) * tangent
    return _wrap(primal_output, tangents)


def _neg_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0]
    primal_output = func(_primal(value), **kwargs)
    return _wrap(
        primal_output,
        {channel: -tangent for channel, tangent in _tangents(value).items()},
    )


def _linear_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    input_value = args[0]
    weight = args[1]
    bias = args[2] if len(args) > 2 else kwargs.get("bias")
    input_p = _primal(input_value)
    weight_p = _primal(weight)
    bias_p = _primal(bias) if bias is not None else None
    primal_output = F.linear(input_p, weight_p, bias_p)

    input_t = _tangents(input_value)
    weight_t = _tangents(weight)
    bias_t = _tangents(bias)
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel in _all_channels(input_value, weight, bias):
        terms: list[torch.Tensor | None] = []
        if channel in input_t:
            terms.append(F.linear(input_t[channel], weight_p, None))
        if channel in weight_t:
            terms.append(F.linear(input_p, weight_t[channel], None))
        if channel in bias_t:
            terms.append(bias_t[channel])
        tangent = _add_terms(*terms)
        if tangent is not None:
            tangents[channel] = tangent
    return _wrap(primal_output, tangents)


def _relu_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0]
    value_p = _primal(value)
    primal_output = func(value_p, **kwargs)
    return _wrap(
        primal_output,
        {
            channel: torch.where(value_p > 0, tangent, _zero_scalar_like(tangent))
            for channel, tangent in _tangents(value).items()
        },
    )


def _mse_loss_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    input_value = args[0]
    target = args[1]
    reduction = kwargs.get("reduction", "mean")
    input_p = _primal(input_value)
    target_p = _primal(target)
    primal_output = func(input_p, target_p, **kwargs)
    diff = input_p - target_p
    input_t = _tangents(input_value)
    target_t = _tangents(target)
    tangents: dict[nn.Parameter, torch.Tensor] = {}
    for channel in _all_channels(input_value, target):
        input_term = input_t.get(channel)
        target_term = target_t.get(channel)
        if input_term is None and target_term is None:
            continue
        if input_term is None:
            diff_t = -target_term
        elif target_term is None:
            diff_t = input_term
        else:
            diff_t = input_term - target_term
        unreduced = 2 * diff * diff_t
        if reduction == "mean":
            tangents[channel] = unreduced.mean()
        elif reduction == "sum":
            tangents[channel] = unreduced.sum()
        elif reduction == "none":
            tangents[channel] = unreduced
        else:
            raise ValueError(f"unsupported mse_loss reduction: {reduction!r}")
    return _wrap(primal_output, tangents)


def _sum_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0]
    primal_output = func(_primal(value), *args[1:], **kwargs)
    return _wrap(
        primal_output,
        {
            channel: func(tangent, *args[1:], **kwargs)
            for channel, tangent in _tangents(value).items()
        },
    )


def _mean_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0]
    primal_output = func(_primal(value), *args[1:], **kwargs)
    return _wrap(
        primal_output,
        {
            channel: func(tangent, *args[1:], **kwargs)
            for channel, tangent in _tangents(value).items()
        },
    )


def _same_shape_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    value = args[0]
    primal_output = func(_primal(value), *args[1:], **kwargs)
    return _wrap(
        primal_output,
        {
            channel: func(tangent, *args[1:], **kwargs)
            for channel, tangent in _tangents(value).items()
        },
    )


def _backward_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    loss = args[0]
    runtime = BlockDualHVPRuntime._ACTIVE_RUNTIME
    if runtime is None:
        raise RuntimeError("BlockDualTensor.backward() requires an active modular_hvp")
    runtime.backward(loss, args[1:], kwargs)
    return None


def _register_function_rule(names: tuple[str, ...], rule: Rule) -> None:
    for name in names:
        _TORCH_FUNCTION_RULES[name] = rule


_register_function_rule(("add", "__add__", "__radd__"), _add_rule)
_register_function_rule(("sub", "__sub__"), _sub_rule)
_register_function_rule(("mul", "__mul__", "__rmul__"), _mul_rule)
_register_function_rule(("div", "__truediv__"), _div_rule)
_register_function_rule(("pow", "__pow__"), _pow_rule)
_register_function_rule(("neg", "__neg__"), _neg_rule)
_register_function_rule(("linear",), _linear_rule)
_register_function_rule(("relu",), _relu_rule)
_register_function_rule(("mse_loss",), _mse_loss_rule)
_register_function_rule(("sum",), _sum_rule)
_register_function_rule(("mean",), _mean_rule)
_register_function_rule(
    (
        "view",
        "reshape",
        "flatten",
        "transpose",
        "t",
        "permute",
        "squeeze",
        "unsqueeze",
        "contiguous",
        "clone",
        "detach",
    ),
    _same_shape_rule,
)
_register_function_rule(("backward",), _backward_rule)
