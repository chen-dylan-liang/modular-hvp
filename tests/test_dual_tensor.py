from __future__ import annotations

import pytest
import torch
from torch import nn

from modular_hvp import (
    is_dual,
    make_dual,
    primal,
    run_with_dual_parameter,
    tangent,
    unpack_dual,
)


def _central_difference(
    fn,
    values: tuple[torch.Tensor, ...],
    directions: tuple[torch.Tensor, ...],
    *,
    h: float = 1e-6,
) -> torch.Tensor:
    plus_values = tuple(value + h * direction for value, direction in zip(values, directions))
    minus_values = tuple(value - h * direction for value, direction in zip(values, directions))
    return (fn(*plus_values) - fn(*minus_values)) / (2 * h)


def test_basic_arithmetic_matches_finite_difference() -> None:
    torch.manual_seed(0)
    a = torch.randn(5, 4, dtype=torch.float64)
    b = torch.randn(5, 4, dtype=torch.float64) + 3.0
    a_dot = torch.randn_like(a)
    b_dot = torch.randn_like(b)

    def fn(a_value: torch.Tensor, b_value: torch.Tensor) -> torch.Tensor:
        return ((a_value * b_value + a_value / (b_value + 2.0)).mean())

    output = fn(make_dual(a, a_dot), make_dual(b, b_dot))
    fd = _central_difference(fn, (a, b), (a_dot, b_dot))

    assert is_dual(output)
    assert torch.allclose(primal(output), fn(a, b))
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)


def test_matmul_matches_finite_difference() -> None:
    torch.manual_seed(1)
    a = torch.randn(4, 3, dtype=torch.float64)
    b = torch.randn(3, 5, dtype=torch.float64)
    a_dot = torch.randn_like(a)
    b_dot = torch.randn_like(b)

    def fn(a_value: torch.Tensor, b_value: torch.Tensor) -> torch.Tensor:
        return (a_value @ b_value).sum()

    output = fn(make_dual(a, a_dot), make_dual(b, b_dot))
    fd = _central_difference(fn, (a, b), (a_dot, b_dot))

    assert torch.allclose(primal(output), fn(a, b))
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)


def test_relu_matches_finite_difference_away_from_kinks() -> None:
    torch.manual_seed(2)
    x = torch.randn(6, 7, dtype=torch.float64)
    x = torch.where(x.abs() < 0.25, x.sign().clamp(min=0) + 0.5, x)
    x_dot = torch.randn_like(x)

    def fn(x_value: torch.Tensor) -> torch.Tensor:
        return torch.relu(x_value).sum()

    output = fn(make_dual(x, x_dot))
    fd = _central_difference(fn, (x,), (x_dot,))

    assert torch.allclose(primal(output), fn(x))
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)


def test_linear_weight_dual_matches_finite_difference() -> None:
    torch.manual_seed(3)
    layer = nn.Linear(4, 3).double()
    x = torch.randn(5, 4, dtype=torch.float64)
    weight_dot = torch.randn_like(layer.weight)

    base = layer(x)
    output = run_with_dual_parameter(layer, "weight", weight_dot, x)

    h = 1e-6
    with torch.no_grad():
        layer.weight.add_(h * weight_dot)
        plus = layer(x)
        layer.weight.add_(-2 * h * weight_dot)
        minus = layer(x)
        layer.weight.add_(h * weight_dot)
    fd = (plus - minus) / (2 * h)

    assert isinstance(layer.weight, nn.Parameter)
    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)


def test_toy_mlp_single_dual_parameter_matches_finite_difference() -> None:
    torch.manual_seed(4)
    model = nn.Sequential(
        nn.Linear(3, 6),
        nn.ReLU(),
        nn.Linear(6, 2),
    ).double()

    for _ in range(100):
        x = torch.randn(4, 3, dtype=torch.float64)
        with torch.no_grad():
            hidden = model[0](x)
        if torch.all(hidden.abs() > 1e-3):
            break
    else:
        raise AssertionError("could not sample an input away from ReLU kinks")

    weight_dot = torch.randn_like(model[0].weight)
    base = model(x)
    output = run_with_dual_parameter(model, "0.weight", weight_dot, x)

    h = 1e-6
    with torch.no_grad():
        model[0].weight.add_(h * weight_dot)
        plus = model(x)
        model[0].weight.add_(-2 * h * weight_dot)
        minus = model(x)
        model[0].weight.add_(h * weight_dot)
    fd = (plus - minus) / (2 * h)

    assert isinstance(model[0].weight, nn.Parameter)
    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-5, atol=1e-6)


def test_backward_like_tensor_program_matches_finite_difference() -> None:
    torch.manual_seed(5)
    x = torch.randn(4, 3, dtype=torch.float64)
    weight = torch.randn(2, 3, dtype=torch.float64)
    grad_out = torch.randn(4, 2, dtype=torch.float64)
    x_dot = torch.randn_like(x)
    weight_dot = torch.randn_like(weight)
    grad_out_dot = torch.randn_like(grad_out)

    def linear_backward_program(
        x_value: torch.Tensor,
        weight_value: torch.Tensor,
        grad_out_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        grad_x = grad_out_value @ weight_value
        grad_weight = grad_out_value.t() @ x_value
        grad_bias = grad_out_value.sum(dim=0)
        return grad_x, grad_weight, grad_bias

    outputs = linear_backward_program(
        make_dual(x, x_dot),
        make_dual(weight, weight_dot),
        make_dual(grad_out, grad_out_dot),
    )
    h = 1e-6
    plus_outputs = linear_backward_program(
        x + h * x_dot,
        weight + h * weight_dot,
        grad_out + h * grad_out_dot,
    )
    minus_outputs = linear_backward_program(
        x - h * x_dot,
        weight - h * weight_dot,
        grad_out - h * grad_out_dot,
    )
    fd_outputs = tuple(
        (plus - minus) / (2 * h)
        for plus, minus in zip(plus_outputs, minus_outputs, strict=True)
    )

    for output, fd in zip(outputs, fd_outputs, strict=True):
        assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)


def test_mse_loss_matches_finite_difference() -> None:
    torch.manual_seed(6)
    x = torch.randn(5, 2, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    x_dot = torch.randn_like(x)

    def fn(x_value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(x_value, target, reduction="mean")

    output = fn(make_dual(x, x_dot))
    fd = _central_difference(fn, (x,), (x_dot,))

    assert torch.allclose(primal(output), fn(x))
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)


def test_reverse_scalar_and_matmul_rules() -> None:
    x = torch.tensor([2.0, 4.0], dtype=torch.float64)
    x_dot = torch.tensor([0.5, -1.0], dtype=torch.float64)
    dual = make_dual(x, x_dot)

    rsub = 3.0 - dual
    assert torch.allclose(primal(rsub), 3.0 - x)
    assert torch.allclose(tangent(rsub), -x_dot)

    rdiv = 3.0 / dual
    assert torch.allclose(primal(rdiv), 3.0 / x)
    assert torch.allclose(tangent(rdiv), -(3.0 * x_dot) / x.pow(2))

    left = torch.randn(2, 3, dtype=torch.float64)
    right = torch.randn(3, 4, dtype=torch.float64)
    right_dot = torch.randn_like(right)
    output = left @ make_dual(right, right_dot)

    assert torch.allclose(primal(output), left @ right)
    assert torch.allclose(tangent(output), left @ right_dot)


def test_unpack_dual_and_unsupported_ops() -> None:
    x = torch.randn(3, dtype=torch.float64)
    x_dot = torch.randn_like(x)
    dual = make_dual(x, x_dot)

    primal_value, tangent_value = unpack_dual(dual)
    assert primal_value is x
    assert tangent_value is x_dot
    with pytest.raises(TypeError, match="expected a DualTensor"):
        unpack_dual(x)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="DualTensor rule not implemented"):
        torch.sin(dual)
