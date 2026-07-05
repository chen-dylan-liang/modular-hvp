"""DualTensor primitive operator backend."""

from __future__ import annotations

import math
from contextvars import ContextVar
from contextlib import contextmanager
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

TangentValue = torch.Tensor
_OUTER_GRAD_ENABLED: ContextVar[bool | None] = ContextVar(
    "_OUTER_GRAD_ENABLED",
    default=None,
)
_IN_TANGENT_EVAL: ContextVar[bool] = ContextVar(
    "_IN_TANGENT_EVAL",
    default=False,
)


class DualTensor(torch.Tensor):
    """Tensor wrapper representing ``primal + eps * tangent``."""

    __slots__ = ("_primal", "_tangent", "_primal_is_zero")

    @staticmethod
    def __new__(
        cls,
        primal: torch.Tensor | None,
        tangent: TangentValue,
        *,
        primal_is_zero: bool = False,
    ) -> "DualTensor":
        metadata = tangent if primal_is_zero else primal
        if metadata is None:
            raise TypeError("primal must be a torch.Tensor unless primal_is_zero=True")
        if metadata.layout != torch.strided:
            raise NotImplementedError("DualTensor currently supports strided tensors only")
        return torch.Tensor._make_wrapper_subclass(
            cls,
            metadata.shape,
            strides=metadata.stride(),
            storage_offset=metadata.storage_offset(),
            dtype=metadata.dtype,
            layout=metadata.layout,
            device=metadata.device,
            requires_grad=metadata.requires_grad,
        )

    def __init__(
        self,
        primal: torch.Tensor | None,
        tangent: TangentValue,
        *,
        primal_is_zero: bool = False,
    ) -> None:
        tangent = _detach_tangent(tangent)
        if primal_is_zero:
            primal = None
        else:
            if primal is None:
                raise TypeError("primal must be a torch.Tensor")
            _validate_dual_parts(primal, tangent)
        _assert_tangent_is_graph_free(tangent)
        self._primal = primal
        self._tangent = tangent
        self._primal_is_zero = primal_is_zero

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
            raise NotImplementedError(f"DualTensor rule not implemented for {func}")
        if not _IN_TANGENT_EVAL.get() and _tree_has_dual((args, kwargs)):
            with torch._C.DisableTorchFunctionSubclass():
                primal_output = func(*_tree_primal(args), **_tree_primal(kwargs))
            tangent_token = _IN_TANGENT_EVAL.set(True)
            grad_token = _OUTER_GRAD_ENABLED.set(torch.is_grad_enabled())
            try:
                tangent_output = func(*args, **kwargs)
            finally:
                _OUTER_GRAD_ENABLED.reset(grad_token)
                _IN_TANGENT_EVAL.reset(tangent_token)
            return _replace_primal(tangent_output, primal_output)

        token = _OUTER_GRAD_ENABLED.set(torch.is_grad_enabled())
        try:
            rule = _RULES.get(func)
            if rule is not None:
                return rule(func, args, kwargs)
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        finally:
            _OUTER_GRAD_ENABLED.reset(token)

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
            raise NotImplementedError(f"DualTensor rule not implemented for {func}")
        rule = _RULES.get(func)
        if rule is None:
            raise NotImplementedError(f"DualTensor rule not implemented for {func}")
        return rule(func, args, kwargs)

    @property
    def primal(self) -> torch.Tensor:
        if self._primal_is_zero:
            return torch.zeros_like(self._tangent, memory_format=torch.preserve_format)
        assert self._primal is not None
        return self._primal

    @property
    def tangent(self) -> TangentValue:
        return self._tangent

    def __repr__(self) -> str:
        if self._primal_is_zero:
            return f"DualTensor(primal=ZeroTensor(shape={tuple(self.shape)}), tangent={self._tangent!r})"
        return f"DualTensor(primal={self._primal!r}, tangent={self._tangent!r})"


def make_dual(primal: torch.Tensor, tangent: torch.Tensor) -> DualTensor:
    """Create a dual tensor after validating primal and tangent compatibility."""

    if not isinstance(primal, torch.Tensor):
        raise TypeError("primal must be a torch.Tensor")
    if not isinstance(tangent, torch.Tensor):
        raise TypeError("tangent must be a torch.Tensor")
    return DualTensor(primal, tangent)


def _make_zero_dual(tangent_value: TangentValue) -> DualTensor:
    return DualTensor(None, tangent_value, primal_is_zero=True)


@contextmanager
def _tangent_eval_only() -> Any:
    token = _IN_TANGENT_EVAL.set(True)
    try:
        yield
    finally:
        _IN_TANGENT_EVAL.reset(token)


def is_dual(x: object) -> bool:
    return isinstance(x, DualTensor)


def unpack_dual(x: Any) -> tuple[Any, TangentValue | None]:
    if is_dual(x):
        return x.primal, x.tangent
    return x, None


def primal(x: Any) -> Any:
    return x.primal if is_dual(x) else x


def tangent(x: Any) -> Any:
    if is_dual(x):
        return x.tangent
    return None


def run_with_dual_parameter(
    model: nn.Module,
    param_name: str,
    tangent_value: torch.Tensor,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Temporarily replace one parameter with a DualTensor and run the model.

    This helper is intended for backend tests and examples. It mutates
    ``module._parameters`` only inside a try/finally block and always restores
    the original parameter object.
    """

    module, leaf_name = _resolve_parameter_owner(model, param_name)
    original = module._parameters[leaf_name]
    if original is None:
        raise ValueError(f"parameter {param_name!r} is None")

    module._parameters[leaf_name] = make_dual(original, tangent_value)
    try:
        return model(*args, **kwargs)
    finally:
        module._parameters[leaf_name] = original


Rule = Callable[[Any, tuple[Any, ...], dict[str, Any]], Any]
_RULES: dict[Any, Rule] = {}


def _detach_tangent(value: TangentValue) -> TangentValue:
    return value.detach()


def _assert_tangent_is_graph_free(value: TangentValue) -> None:
    assert not value.requires_grad
    assert value.grad_fn is None


def _validate_dual_parts(
    primal_value: torch.Tensor,
    tangent_value: TangentValue,
) -> None:
    if primal_value.shape != tangent_value.shape:
        raise ValueError(
            f"dual tangent has shape {tuple(tangent_value.shape)}; "
            f"expected {tuple(primal_value.shape)}"
        )
    if primal_value.device != tangent_value.device:
        raise ValueError(
            f"dual tangent is on {tangent_value.device}; expected {primal_value.device}"
        )
    if primal_value.dtype != tangent_value.dtype:
        raise ValueError(
            f"dual tangent has dtype {tangent_value.dtype}; expected {primal_value.dtype}"
        )


def _resolve_parameter_owner(model: nn.Module, param_name: str) -> tuple[nn.Module, str]:
    if not param_name:
        raise ValueError("param_name must not be empty")
    module_path, _, leaf_name = param_name.rpartition(".")
    module = model.get_submodule(module_path) if module_path else model
    if leaf_name not in module._parameters:
        raise KeyError(f"unknown parameter: {param_name!r}")
    return module, leaf_name


def _has_out_argument(kwargs: dict[str, Any]) -> bool:
    out = kwargs.get("out")
    if out is None:
        return False
    if isinstance(out, tuple):
        return any(item is not None for item in out)
    return True


def _tree_map(fn: Callable[[Any], Any], value: Any) -> Any:
    if is_dual(value):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(_tree_map(fn, item) for item in value)
    if isinstance(value, list):
        return [_tree_map(fn, item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_map(fn, item) for key, item in value.items()}
    return value


def _tree_has_dual(value: Any) -> bool:
    if is_dual(value):
        return True
    if isinstance(value, tuple | list):
        return any(_tree_has_dual(item) for item in value)
    if isinstance(value, dict):
        return any(_tree_has_dual(item) for item in value.values())
    return False


def _tree_primal(value: Any) -> Any:
    return _tree_map(primal, value)


def _tree_detach(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_tree_detach(item) for item in value)
    if isinstance(value, list):
        return [_tree_detach(item) for item in value]
    if isinstance(value, dict):
        return {key: _tree_detach(item) for key, item in value.items()}
    return value


def _call_primal(func: Any, *args: Any, **kwargs: Any) -> Any:
    if _IN_TANGENT_EVAL.get():
        with torch.no_grad():
            return func(*_tree_detach(args), **_tree_detach(kwargs))
    grad_enabled = _OUTER_GRAD_ENABLED.get()
    if grad_enabled is None:
        grad_enabled = torch.is_grad_enabled()
    with torch.set_grad_enabled(grad_enabled):
        return func(*args, **kwargs)


def _replace_primal(value: Any, primal_value: Any) -> Any:
    if is_dual(value):
        if not isinstance(primal_value, torch.Tensor):
            return primal_value
        return DualTensor(primal_value, value.tangent)
    if isinstance(value, tuple):
        return tuple(
            _replace_primal(item, primal_item)
            for item, primal_item in zip(value, primal_value, strict=True)
        )
    if isinstance(value, list):
        return [
            _replace_primal(item, primal_item)
            for item, primal_item in zip(value, primal_value, strict=True)
        ]
    return primal_value


def _split(value: Any) -> tuple[Any, Any, bool]:
    if is_dual(value):
        return value.primal, value.tangent, True
    return value, None, False


def _is_zero_primal(value: Any) -> bool:
    return is_dual(value) and value._primal_is_zero


def _ng(value: Any) -> Any:
    return value.detach() if isinstance(value, torch.Tensor) else value


def _add_terms(*terms: Any) -> Any:
    active_terms = [term for term in terms if term is not None]
    if not active_terms:
        return None
    result = active_terms[0]
    for term in active_terms[1:]:
        result = _add_tangents(result, term)
    return result


def _add_tangents(left: Any, right: Any) -> Any:
    return left + right


def _map_tangent(fn: Callable[[torch.Tensor], torch.Tensor], value: Any) -> Any:
    return fn(value)


def _zero_matmul_tangent(left: Any, right: Any) -> Any:
    return torch.zeros_like(left @ right)


def _wrap(primal_output: Any, tangent_output: Any) -> Any:
    if not isinstance(primal_output, torch.Tensor):
        return primal_output
    return _make_rule_output(primal_output, tangent_output)


def _make_rule_output(
    primal_output: torch.Tensor,
    tangent_output: Any,
) -> DualTensor:
    if tangent_output.shape != primal_output.shape:
        tangent_output = torch.broadcast_to(tangent_output, primal_output.shape)
    return make_dual(primal_output, tangent_output)


def _make_zero_rule_output(tangent_output: torch.Tensor) -> DualTensor:
    return _make_zero_dual(tangent_output)


def _zero_scalar_like(value: torch.Tensor) -> torch.Tensor:
    return value.new_zeros(())


def _apply_same_unary_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args and _is_zero_primal(args[0]):
        with torch.no_grad():
            value_t = tangent(args[0])
            other_args = args[1:]
            tangent_output = _map_tangent(
                lambda item: func(item, *_tree_detach(other_args), **kwargs),
                value_t,
            )
            return _make_zero_rule_output(tangent_output)
    value = args[0]
    value_p, value_t, _ = _split(value)
    other_args = args[1:]
    primal_kwargs = _tree_primal(kwargs)
    primal_args = (value_p, *_tree_primal(other_args))
    primal_kwargs = _tree_primal(kwargs)
    primal_output = _call_primal(func, *primal_args, **primal_kwargs)
    with torch.no_grad():
        tangent_output = (
            torch.zeros_like(primal_output.detach())
            if value_t is None
            else _map_tangent(
                lambda item: func(item, *_tree_detach(other_args), **primal_kwargs),
                value_t,
            )
        )
    return _wrap(primal_output, tangent_output)


def _add_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    left, right = args[:2]
    alpha = kwargs.get("alpha", 1)
    left_p, left_t, left_is_dual = _split(left)
    right_p, right_t, right_is_dual = _split(right)
    primal_output = _call_primal(func, left_p, right_p, **kwargs)

    with torch.no_grad():
        tangent_output = _add_terms(
            left_t if left_is_dual else None,
            _map_tangent(lambda item: alpha * item, right_t)
            if right_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _sub_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    left, right = args[:2]
    alpha = kwargs.get("alpha", 1)
    left_p, left_t, left_is_dual = _split(left)
    right_p, right_t, right_is_dual = _split(right)
    primal_output = _call_primal(func, left_p, right_p, **kwargs)

    with torch.no_grad():
        tangent_output = _add_terms(
            left_t if left_is_dual else None,
            _map_tangent(lambda item: -alpha * item, right_t)
            if right_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _rsub_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    right, left = args[:2]
    right_p, right_t, right_is_dual = _split(right)
    left_p, left_t, left_is_dual = _split(left)
    primal_output = _call_primal(func, right_p, left_p, **kwargs)
    with torch.no_grad():
        tangent_output = _add_terms(
            left_t if left_is_dual else None,
            _map_tangent(lambda item: -item, right_t) if right_is_dual else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _neg_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    value = args[0]
    value_p, value_t, _ = _split(value)
    primal_output = _call_primal(func, value_p, **kwargs)
    with torch.no_grad():
        tangent_output = (
            torch.zeros_like(primal_output.detach())
            if value_t is None
            else _map_tangent(lambda item: -item, value_t)
        )
    return _make_rule_output(primal_output, tangent_output)


def _mul_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    left, right = args[:2]
    left_p, left_t, left_is_dual = _split(left)
    right_p, right_t, right_is_dual = _split(right)
    primal_output = _call_primal(func, left_p, right_p, **kwargs)

    with torch.no_grad():
        left_ng = _ng(left_p)
        right_ng = _ng(right_p)
        tangent_output = _add_terms(
            _map_tangent(lambda item: item * right_ng, left_t)
            if left_is_dual
            else None,
            _map_tangent(lambda item: left_ng * item, right_t)
            if right_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _div_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    left, right = args[:2]
    left_p, left_t, left_is_dual = _split(left)
    right_p, right_t, right_is_dual = _split(right)
    primal_output = _call_primal(func, left_p, right_p, **kwargs)

    with torch.no_grad():
        left_ng = _ng(left_p)
        right_ng = _ng(right_p)
        tangent_output = _add_terms(
            _map_tangent(lambda item: item / right_ng, left_t)
            if left_is_dual
            else None,
            _map_tangent(
                lambda item: -(left_ng * item) / (right_ng * right_ng),
                right_t,
            )
            if right_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _rdiv_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    denominator, numerator = args[:2]
    denominator_p, denominator_t, denominator_is_dual = _split(denominator)
    numerator_p, numerator_t, numerator_is_dual = _split(numerator)
    primal_output = _call_primal(func, denominator_p, numerator_p, **kwargs)
    with torch.no_grad():
        denominator_ng = _ng(denominator_p)
        numerator_ng = _ng(numerator_p)
        tangent_output = _add_terms(
            _map_tangent(lambda item: item / denominator_ng, numerator_t)
            if numerator_is_dual
            else None,
            _map_tangent(
                lambda item: -(numerator_ng * item) / (denominator_ng * denominator_ng),
                denominator_t,
            )
            if denominator_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _reciprocal_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    value = args[0]
    value_p, value_t, _ = _split(value)
    primal_output = _call_primal(func, value_p, **kwargs)
    with torch.no_grad():
        if value_t is None:
            tangent_output = torch.zeros_like(primal_output.detach())
        else:
            value_ng = value_p.detach()
            tangent_output = _map_tangent(
                lambda item: -item / (value_ng * value_ng),
                value_t,
            )
    return _make_rule_output(primal_output, tangent_output)


def _pow_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    base, exponent = args[:2]
    base_p, base_t, base_is_dual = _split(base)
    exponent_p, exponent_t, exponent_is_dual = _split(exponent)
    primal_output = _call_primal(func, base_p, exponent_p, **kwargs)

    with torch.no_grad():
        base_ng = _ng(base_p)
        exponent_ng = _ng(exponent_p)
        primal_ng = torch.pow(base_ng, exponent_ng)
        tangent_output = _add_terms(
            _map_tangent(
                lambda item: exponent_ng * torch.pow(base_ng, exponent_ng - 1) * item,
                base_t,
            )
            if base_is_dual
            else None,
            _map_tangent(
                lambda item: primal_ng * torch.log(base_ng) * item,
                exponent_t,
            )
            if exponent_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _scalar_pow_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    base, exponent = args[:2]
    exponent_p, exponent_t, exponent_is_dual = _split(exponent)
    primal_output = _call_primal(func, base, exponent_p, **kwargs)
    with torch.no_grad():
        if not exponent_is_dual:
            tangent_output = torch.zeros_like(primal_output.detach())
        else:
            primal_ng = torch.pow(
                torch.as_tensor(
                    base,
                    dtype=primal_output.dtype,
                    device=primal_output.device,
                ),
                _ng(exponent_p),
            )
            tangent_output = _map_tangent(
                lambda item: primal_ng * math.log(base) * item,
                exponent_t,
            )
    return _make_rule_output(primal_output, tangent_output)


def _matmul_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    left, right = args[:2]
    left_is_zero = _is_zero_primal(left)
    right_is_zero = _is_zero_primal(right)
    if left_is_zero or right_is_zero:
        with torch.no_grad():
            if left_is_zero and right_is_zero:
                tangent_output = _zero_matmul_tangent(
                    tangent(left),
                    tangent(right),
                )
            elif left_is_zero:
                right_ng = primal(right).detach()
                tangent_output = _map_tangent(
                    lambda item: item @ right_ng,
                    tangent(left),
                )
            else:
                left_ng = primal(left).detach()
                tangent_output = _map_tangent(
                    lambda item: left_ng @ item,
                    tangent(right),
                )
        return _make_zero_rule_output(tangent_output)

    left_p, left_t, left_is_dual = _split(left)
    right_p, right_t, right_is_dual = _split(right)
    primal_output = _call_primal(func, left_p, right_p, **kwargs)
    with torch.no_grad():
        left_ng = left_p.detach() if isinstance(left_p, torch.Tensor) else left_p
        right_ng = right_p.detach() if isinstance(right_p, torch.Tensor) else right_p
        tangent_output = _add_terms(
            _map_tangent(lambda item: item @ right_ng, left_t)
            if left_is_dual
            else None,
            _map_tangent(lambda item: left_ng @ item, right_t)
            if right_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _rmatmul_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    right, left = args[:2]
    right_p, right_t, right_is_dual = _split(right)
    left_p, left_t, left_is_dual = _split(left)
    primal_output = _call_primal(func, right_p, left_p, **kwargs)
    with torch.no_grad():
        left_ng = left_p.detach() if isinstance(left_p, torch.Tensor) else left_p
        right_ng = right_p.detach() if isinstance(right_p, torch.Tensor) else right_p
        tangent_output = _add_terms(
            _map_tangent(lambda item: item @ right_ng, left_t)
            if left_is_dual
            else None,
            _map_tangent(lambda item: left_ng @ item, right_t)
            if right_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _addmm_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    input_value, mat1, mat2 = args[:3]
    beta = kwargs.get("beta", 1)
    alpha = kwargs.get("alpha", 1)
    input_p, input_t, input_is_dual = _split(input_value)
    mat1_p, mat1_t, mat1_is_dual = _split(mat1)
    mat2_p, mat2_t, mat2_is_dual = _split(mat2)
    primal_output = _call_primal(func, input_p, mat1_p, mat2_p, **kwargs)
    with torch.no_grad():
        mat1_ng = mat1_p.detach()
        mat2_ng = mat2_p.detach()
        tangent_output = _add_terms(
            _map_tangent(lambda item: beta * item, input_t)
            if input_is_dual
            else None,
            _map_tangent(lambda item: alpha * (item @ mat2_ng), mat1_t)
            if mat1_is_dual
            else None,
            _map_tangent(lambda item: alpha * (mat1_ng @ item), mat2_t)
            if mat2_is_dual
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _linear_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    input_value, weight = args[:2]
    bias = args[2] if len(args) > 2 else None
    input_p, input_t, input_is_dual = _split(input_value)
    weight_p, weight_t, weight_is_dual = _split(weight)
    bias_p, bias_t, bias_is_dual = _split(bias)
    primal_output = _call_primal(func, input_p, weight_p, bias_p, **kwargs)

    with torch.no_grad():
        input_ng = input_p.detach()
        weight_ng = weight_p.detach()
        tangent_output = _add_terms(
            _map_tangent(lambda item: item.matmul(weight_ng.t()), input_t)
            if input_is_dual
            else None,
            _map_tangent(lambda item: input_ng.matmul(item.t()), weight_t)
            if weight_is_dual
            else None,
            bias_t if bias_is_dual else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _reduction_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    value = args[0]
    if _is_zero_primal(value):
        return _make_zero_rule_output(
            _map_tangent(
                lambda item: func(item, *args[1:], **kwargs),
                tangent(value),
            ),
        )
    value_p, value_t, _ = _split(value)
    other_args = args[1:]
    primal_output = _call_primal(func, value_p, *other_args, **kwargs)
    with torch.no_grad():
        tangent_output = (
            torch.zeros_like(primal_output.detach())
            if value_t is None
            else _map_tangent(
                lambda item: func(item, *other_args, **kwargs),
                value_t,
            )
        )
    return _make_rule_output(primal_output, tangent_output)


def _relu_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    value = args[0]
    value_p, value_t, _ = _split(value)
    primal_output = _call_primal(func, value_p, **kwargs)
    with torch.no_grad():
        value_ng = value_p.detach()
        tangent_output = (
            torch.zeros_like(primal_output.detach())
            if value_t is None
            else _map_tangent(
                lambda item: torch.where(value_ng > 0, item, _zero_scalar_like(item)),
                value_t,
            )
        )
    return _make_rule_output(primal_output, tangent_output)


def _gelu_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    value = args[0]
    approximate = kwargs.get("approximate", "none")
    value_p, value_t, _ = _split(value)
    primal_output = _call_primal(func, value_p, **kwargs)

    with torch.no_grad():
        if value_t is None:
            tangent_output = torch.zeros_like(primal_output.detach())
        else:
            value_ng = value_p.detach()
            if approximate == "tanh":
                coeff = math.sqrt(2.0 / math.pi)
                inner = coeff * (value_ng + 0.044715 * value_ng.pow(3))
                tanh_inner = torch.tanh(inner)
                derivative = 0.5 * (1 + tanh_inner) + 0.5 * value_ng * (
                    1 - tanh_inner.pow(2)
                ) * coeff * (1 + 3 * 0.044715 * value_ng.pow(2))
            else:
                normal_cdf = 0.5 * (1 + torch.erf(value_ng / math.sqrt(2.0)))
                normal_pdf = torch.exp(-0.5 * value_ng.pow(2)) / math.sqrt(
                    2.0 * math.pi
                )
                derivative = normal_cdf + value_ng * normal_pdf
            tangent_output = _map_tangent(
                lambda item: derivative * item,
                value_t,
            )

    return _make_rule_output(primal_output, tangent_output)


def _threshold_backward_rule(
    func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> DualTensor:
    grad_output, input_value, threshold = args[:3]
    input_p = primal(input_value)
    if _is_zero_primal(grad_output):
        with torch.no_grad():
            input_ng = input_p.detach()
            tangent_output = _map_tangent(
                lambda item: torch.where(
                    input_ng > threshold,
                    item,
                    _zero_scalar_like(item),
                ),
                tangent(grad_output),
            )
        return _make_zero_rule_output(tangent_output)

    grad_p, grad_t, grad_is_dual = _split(grad_output)
    primal_output = _call_primal(func, grad_p, input_p, threshold, **kwargs)
    with torch.no_grad():
        input_ng = input_p.detach()
        tangent_output = (
            _map_tangent(
                lambda item: torch.where(
                    input_ng > threshold,
                    item,
                    _zero_scalar_like(item),
                ),
                grad_t,
            )
            if grad_is_dual
            else torch.zeros_like(primal_output.detach())
        )
    return _make_rule_output(primal_output, tangent_output)


def _mse_loss_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> DualTensor:
    input_value, target = args[:2]
    reduction = args[2] if len(args) > 2 else kwargs.get("reduction", 1)
    input_p, input_t, input_is_dual = _split(input_value)
    target_p, target_t, target_is_dual = _split(target)
    if len(args) > 2:
        primal_output = _call_primal(func, input_p, target_p, reduction, **kwargs)
    else:
        primal_output = _call_primal(func, input_p, target_p, **kwargs)

    with torch.no_grad():
        diff = input_p.detach() - target_p.detach()
        diff_tangent = _add_terms(
            input_t if input_is_dual else None,
            _map_tangent(lambda item: -item, target_t) if target_is_dual else None,
        )
        if diff_tangent is None:
            tangent_unreduced = torch.zeros_like(diff)
        else:
            tangent_unreduced = _map_tangent(
                lambda item: 2 * diff * item,
                diff_tangent,
            )

        if reduction in (0, "none"):
            tangent_output = tangent_unreduced
        elif reduction in (1, "mean"):
            tangent_output = _map_tangent(lambda item: item.mean(), tangent_unreduced)
        elif reduction in (2, "sum"):
            tangent_output = _map_tangent(lambda item: item.sum(), tangent_unreduced)
        else:
            raise ValueError(f"unknown mse_loss reduction: {reduction!r}")
    return _make_rule_output(primal_output, tangent_output)


def _mse_loss_backward_rule(
    func: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> DualTensor:
    grad_output, input_value, target, reduction = args[:4]
    grad_p, grad_t, grad_is_dual = _split(grad_output)
    input_p, input_t, input_is_dual = _split(input_value)
    target_p, target_t, target_is_dual = _split(target)
    primal_output = _call_primal(func, grad_p, input_p, target_p, reduction, **kwargs)

    with torch.no_grad():
        grad_ng = grad_p.detach()
        diff = input_p.detach() - target_p.detach()
        diff_tangent = _add_terms(
            input_t if input_is_dual else None,
            _map_tangent(lambda item: -item, target_t) if target_is_dual else None,
        )
        if reduction in (0, "none"):
            scale = 2.0
        elif reduction in (1, "mean"):
            scale = 2.0 / input_p.numel()
        elif reduction in (2, "sum"):
            scale = 2.0
        else:
            raise ValueError(f"unknown mse_loss reduction: {reduction!r}")

        tangent_output = _add_terms(
            _map_tangent(lambda item: scale * item * diff, grad_t)
            if grad_is_dual
            else None,
            _map_tangent(lambda item: scale * grad_ng * item, diff_tangent)
            if diff_tangent is not None
            else None,
        )
        if tangent_output is None:
            tangent_output = torch.zeros_like(primal_output.detach())
    return _make_rule_output(primal_output, tangent_output)


def _comparison_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    with torch.no_grad():
        return func(
            *_tree_detach(_tree_primal(args)),
            **_tree_detach(_tree_primal(kwargs)),
        )


def _where_rule(func: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if len(args) == 1:
        raise NotImplementedError(f"DualTensor rule not implemented for {func}")

    condition, left, right = args[:3]
    condition_p = primal(condition)
    left_p, left_t, left_is_dual = _split(left)
    right_p, right_t, right_is_dual = _split(right)
    primal_output = _call_primal(func, condition_p, left_p, right_p, **kwargs)
    with torch.no_grad():
        condition_ng = (
            condition_p.detach()
            if isinstance(condition_p, torch.Tensor)
            else condition_p
        )
        left_tangent = (
            left_t if left_is_dual else torch.zeros_like(primal_output.detach())
        )
        right_tangent = (
            right_t if right_is_dual else torch.zeros_like(primal_output.detach())
        )
        tangent_output = _where_tangent(func, condition_ng, left_tangent, right_tangent)
    return _make_rule_output(primal_output, tangent_output)


def _where_tangent(func: Any, condition: Any, left: Any, right: Any) -> Any:
    return func(condition, left, right)


def _register(name: str, overload: str, rule: Rule) -> None:
    packet = getattr(torch.ops.aten, name, None)
    if packet is None or overload not in packet.overloads():
        return
    _RULES[getattr(packet, overload)] = rule


for _name, _overload in (
    ("add", "Tensor"),
    ("add", "Scalar"),
):
    _register(_name, _overload, _add_rule)

for _name, _overload in (
    ("sub", "Tensor"),
    ("sub", "Scalar"),
):
    _register(_name, _overload, _sub_rule)

for _name, _overload in (
    ("rsub", "Tensor"),
    ("rsub", "Scalar"),
):
    _register(_name, _overload, _rsub_rule)

_register("neg", "default", _neg_rule)
_register("mul", "Tensor", _mul_rule)
_register("mul", "Scalar", _mul_rule)
_register("div", "Tensor", _div_rule)
_register("div", "Scalar", _div_rule)
_register("reciprocal", "default", _reciprocal_rule)
_register("pow", "Tensor_Scalar", _pow_rule)
_register("pow", "Tensor_Tensor", _pow_rule)
_register("pow", "Scalar", _scalar_pow_rule)

for _name in ("mm", "matmul", "bmm"):
    _register(_name, "default", _matmul_rule)
_register("addmm", "default", _addmm_rule)
_register("linear", "default", _linear_rule)

for _name, _overload in (
    ("sum", "default"),
    ("sum", "dim_IntList"),
    ("mean", "default"),
    ("mean", "dim"),
):
    _register(_name, _overload, _reduction_rule)

for _name, _overload in (
    ("view", "default"),
    ("reshape", "default"),
    ("_unsafe_view", "default"),
    ("flatten", "using_ints"),
    ("transpose", "int"),
    ("t", "default"),
    ("permute", "default"),
    ("squeeze", "default"),
    ("squeeze", "dim"),
    ("squeeze", "dims"),
    ("unsqueeze", "default"),
    ("expand", "default"),
    ("contiguous", "default"),
    ("clone", "default"),
    ("to", "device"),
    ("to", "dtype"),
    ("to", "other"),
    ("to", "dtype_layout"),
    ("_to_copy", "default"),
    ("detach", "default"),
):
    _register(_name, _overload, _apply_same_unary_rule)

_register("relu", "default", _relu_rule)
_register("threshold_backward", "default", _threshold_backward_rule)
_register("gelu", "default", _gelu_rule)
_register("mse_loss", "default", _mse_loss_rule)
_register("mse_loss_backward", "default", _mse_loss_backward_rule)

for _name in ("gt", "ge", "lt", "le", "eq", "ne"):
    _register(_name, "Tensor", _comparison_rule)
    _register(_name, "Scalar", _comparison_rule)
_register("where", "self", _where_rule)
