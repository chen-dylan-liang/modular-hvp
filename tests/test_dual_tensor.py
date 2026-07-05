from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import modular_hvp.dual as dual_backend
from modular_hvp import (
    is_dual,
    make_dual,
    primal,
    run_with_dual_parameter,
    tangent,
    unpack_dual,
)


def _assert_graph_free(value: torch.Tensor) -> None:
    assert not value.requires_grad
    assert value.grad_fn is None


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


def test_backend_registers_no_python_level_dual_rules() -> None:
    assert not hasattr(dual_backend, "_PY_RULES")


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


def test_matmul_primal_keeps_graph_and_tangent_is_graph_free() -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 4, requires_grad=True)
    w = torch.randn(4, 4, requires_grad=True)
    x_dot = torch.randn_like(x, requires_grad=True)
    w_dot = torch.randn_like(w, requires_grad=True)

    y_hat = make_dual(x, x_dot) @ make_dual(w, w_dot)

    assert primal(y_hat).requires_grad
    assert primal(y_hat).grad_fn is not None
    _assert_graph_free(tangent(y_hat))

    primal(y_hat).sum().backward()

    assert x.grad is not None
    assert w.grad is not None
    assert x_dot.grad is None
    assert w_dot.grad is None


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


def test_relu_primal_keeps_graph_and_tangent_is_graph_free() -> None:
    torch.manual_seed(12)
    x = torch.randn(16, requires_grad=True)
    with torch.no_grad():
        x.copy_(x + 0.1 * torch.sign(x))
    x_dot = torch.randn_like(x, requires_grad=True)

    y_hat = torch.relu(make_dual(x, x_dot))

    assert primal(y_hat).requires_grad
    assert primal(y_hat).grad_fn is not None
    _assert_graph_free(tangent(y_hat))

    h = 1e-4
    with torch.no_grad():
        fd = (torch.relu(x + h * x_dot) - torch.relu(x)) / h
    torch.testing.assert_close(tangent(y_hat), fd, rtol=1e-2, atol=1e-3)


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
    assert primal(output).requires_grad
    assert primal(output).grad_fn is not None
    _assert_graph_free(tangent(output))


def test_conv2d_weight_dual_matches_finite_difference() -> None:
    torch.manual_seed(8)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1).double()
    x = torch.randn(4, 2, 5, 5, dtype=torch.float64)
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

    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)
    assert primal(output).requires_grad
    assert primal(output).grad_fn is not None
    _assert_graph_free(tangent(output))


def test_conv2d_bias_dual_matches_finite_difference() -> None:
    torch.manual_seed(14)
    layer = nn.Conv2d(2, 3, kernel_size=3, padding=1).double()
    x = torch.randn(4, 2, 5, 5, dtype=torch.float64)
    bias_dot = torch.randn_like(layer.bias)

    base = layer(x)
    output = run_with_dual_parameter(layer, "bias", bias_dot, x)

    h = 1e-6
    with torch.no_grad():
        layer.bias.add_(h * bias_dot)
        plus = layer(x)
        layer.bias.add_(-2 * h * bias_dot)
        minus = layer(x)
        layer.bias.add_(h * bias_dot)
    fd = (plus - minus) / (2 * h)

    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)
    _assert_graph_free(tangent(output))


def test_batch_norm2d_training_input_dual_matches_finite_difference() -> None:
    torch.manual_seed(15)
    x = torch.randn(4, 3, 5, 5, dtype=torch.float64)
    x_dot = torch.randn_like(x)
    weight = torch.randn(3, dtype=torch.float64)
    bias = torch.randn(3, dtype=torch.float64)
    eps = 1e-5

    def fn(x_value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.native_batch_norm.default(
            x_value,
            weight,
            bias,
            None,
            None,
            True,
            0.1,
            eps,
        )[0]

    output = fn(make_dual(x, x_dot))
    fd = _central_difference(fn, (x,), (x_dot,))

    assert torch.allclose(primal(output), fn(x))
    assert torch.allclose(tangent(output), fd, rtol=1e-5, atol=1e-6)
    _assert_graph_free(tangent(output))


def test_batch_norm2d_training_weight_dual_matches_finite_difference() -> None:
    torch.manual_seed(9)
    baseline = nn.BatchNorm2d(3).double().train()
    model = nn.BatchNorm2d(3).double().train()
    initial_state = copy.deepcopy(baseline.state_dict())
    model.load_state_dict(initial_state)
    x = torch.randn(4, 3, 5, 5, dtype=torch.float64)
    weight_dot = torch.randn_like(model.weight)

    base = baseline(x)
    output = run_with_dual_parameter(model, "weight", weight_dot, x)

    def forward_with_weight_offset(offset: float) -> torch.Tensor:
        candidate = nn.BatchNorm2d(3).double().train()
        candidate.load_state_dict(initial_state)
        with torch.no_grad():
            candidate.weight.add_(offset * weight_dot)
            return candidate(x)

    h = 1e-6
    fd = (forward_with_weight_offset(h) - forward_with_weight_offset(-h)) / (2 * h)

    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)
    assert torch.allclose(model.running_mean, baseline.running_mean)
    assert torch.allclose(model.running_var, baseline.running_var)
    _assert_graph_free(tangent(output))


def test_max_pool2d_dual_uses_primal_argmax_locations() -> None:
    x = torch.arange(1, 17, dtype=torch.float64).reshape(1, 1, 4, 4)
    x_dot = torch.randn_like(x)

    def fn(x_value: torch.Tensor) -> torch.Tensor:
        return F.max_pool2d(x_value, kernel_size=2, stride=2).sum()

    output = fn(make_dual(x, x_dot))
    fd = _central_difference(fn, (x,), (x_dot,))

    assert torch.allclose(primal(output), fn(x))
    assert torch.allclose(tangent(output), fd, rtol=1e-6, atol=1e-6)
    _assert_graph_free(tangent(output))


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
    assert primal(output).requires_grad
    assert primal(output).grad_fn is not None
    _assert_graph_free(tangent(output))


def test_cnn_forward_dual_composes_through_primitives() -> None:
    torch.manual_seed(10)
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(4, 5, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(5, 2),
    ).double()
    model.eval()
    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
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

    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-5, atol=1e-6)
    assert primal(output).requires_grad
    assert primal(output).grad_fn is not None
    _assert_graph_free(tangent(output))


def test_resnet_like_forward_dual_composes_through_residual_add() -> None:
    class TinyResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(4, 3, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm2d(3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + residual)

    torch.manual_seed(13)
    model = TinyResidualBlock().double().eval()
    x = torch.randn(2, 3, 6, 6, dtype=torch.float64)
    weight_dot = torch.randn_like(model.conv2.weight)

    base = model(x)
    output = run_with_dual_parameter(model, "conv2.weight", weight_dot, x)

    h = 1e-6
    with torch.no_grad():
        model.conv2.weight.add_(h * weight_dot)
        plus = model(x)
        model.conv2.weight.add_(-2 * h * weight_dot)
        minus = model(x)
        model.conv2.weight.add_(h * weight_dot)
    fd = (plus - minus) / (2 * h)

    assert torch.allclose(primal(output), base)
    assert torch.allclose(tangent(output), fd, rtol=1e-5, atol=1e-6)
    _assert_graph_free(tangent(output))


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


def test_mse_loss_backward_matches_finite_difference() -> None:
    torch.manual_seed(7)
    x = torch.randn(5, 2, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    x_dot = torch.randn_like(x)
    grad_output = torch.ones((), dtype=torch.float64)

    def fn(x_value: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten.mse_loss_backward.default(grad_output, x_value, target, 1)

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
    x_dot = torch.randn_like(x, requires_grad=True)
    dual = make_dual(x, x_dot)

    primal_value, tangent_value = unpack_dual(dual)
    assert primal_value is x
    assert tangent_value is not x_dot
    assert torch.allclose(tangent_value, x_dot)
    _assert_graph_free(tangent_value)

    primal_value, tangent_value = unpack_dual(x)
    assert primal_value is x
    assert tangent_value is None

    with pytest.raises(NotImplementedError, match="DualTensor rule not implemented"):
        torch.sin(dual)
