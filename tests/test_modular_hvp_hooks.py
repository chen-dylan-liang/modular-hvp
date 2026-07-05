from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn

import modular_hvp.local_mlp as local_mlp
from modular_hvp import FakeDualBackend, LocalDualActivations, is_dual, modular_hvp


def _tangents_by_name(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _block_autodiff_hvps(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    tangents: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    loss = loss_fn(model(x), target)
    hvps: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        gradient = torch.autograd.grad(
            loss,
            [parameter],
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )[0]
        directional_gradient = (gradient * tangents[name]).sum()
        hvps[name] = torch.autograd.grad(
            directional_gradient,
            [parameter],
            retain_graph=True,
            materialize_grads=True,
        )[0].detach()
    return hvps


class RecordingBackend(FakeDualBackend):
    def __init__(self) -> None:
        self.local_forward_modules: list[str] = []
        self.backward_modules: list[str] = []

    def local_forward(
        self,
        *,
        module: nn.Module,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> tuple[Any, LocalDualActivations]:
        self.local_forward_modules.append(module.__class__.__name__)
        return super().local_forward(
            module=module,
            original_forward=original_forward,
            args=args,
            kwargs=kwargs,
            active_param_tangents=active_param_tangents,
        )

    def dual_backward(
        self,
        *,
        module: nn.Module,
        local_dual_activations: LocalDualActivations,
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
        grad_input: Sequence[torch.Tensor | None],
        grad_output: Sequence[torch.Tensor | None],
    ) -> Mapping[nn.Parameter, torch.Tensor]:
        self.backward_modules.append(module.__class__.__name__)
        return super().dual_backward(
            module=module,
            local_dual_activations=local_dual_activations,
            active_param_tangents=active_param_tangents,
            grad_input=grad_input,
            grad_output=grad_output,
        )


class CountingBackend(FakeDualBackend):
    def __init__(self) -> None:
        self.calls_by_parameter_id: dict[int, int] = {}

    def dual_backward(
        self,
        *,
        module: nn.Module,
        local_dual_activations: LocalDualActivations,
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
        grad_input: Sequence[torch.Tensor | None],
        grad_output: Sequence[torch.Tensor | None],
    ) -> Mapping[nn.Parameter, torch.Tensor]:
        for parameter in active_param_tangents:
            self.calls_by_parameter_id[id(parameter)] = (
                self.calls_by_parameter_id.get(id(parameter), 0) + 1
            )
        return super().dual_backward(
            module=module,
            local_dual_activations=local_dual_activations,
            active_param_tangents=active_param_tangents,
            grad_input=grad_input,
            grad_output=grad_output,
        )


def test_default_modular_hvp_matches_block_autodiff_on_mlp() -> None:
    torch.manual_seed(0)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    baseline_loss = criterion(baseline(x), target)
    baseline_loss.backward()
    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    assert torch.allclose(loss.detach(), baseline_loss.detach())
    for (_, expected), (_, actual) in zip(
        baseline.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.allclose(actual.grad, expected.grad)
        assert actual.hvp is not None
    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_matches_block_autodiff_on_sequential_cnn() -> None:
    torch.manual_seed(20)
    baseline = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 4 * 4, 3),
    ).double()
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 4 * 4, 3),
    ).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(4, 1, 8, 8, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_matches_block_autodiff_on_eval_cnn_with_batchnorm_and_pooling() -> None:
    torch.manual_seed(21)
    baseline = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(12, 2),
    ).double().eval()
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(12, 2),
    ).double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(3, 1, 8, 8, dtype=torch.float64)
    target = torch.randn(3, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_uses_original_module_forward() -> None:
    class OffsetLinear(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input) + 1.25

    torch.manual_seed(0)
    baseline = nn.Sequential(OffsetLinear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(OffsetLinear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    baseline_output = baseline(x)
    baseline_loss = criterion(baseline_output, target)
    baseline_loss.backward()
    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    with modular_hvp(model, tangents):
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

    assert torch.allclose(output.detach(), baseline_output.detach())
    assert torch.allclose(loss.detach(), baseline_loss.detach())
    for (_, expected), (_, actual) in zip(
        baseline.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.allclose(actual.grad, expected.grad)
        assert actual.hvp is not None
    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_runtime_uses_autograd_saved_activation_refs() -> None:
    class OffsetLinear(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input) + 1.25

    torch.manual_seed(0)
    layer = OffsetLinear(3, 4).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    linear_output = layer(x)
    linear_ref = local_mlp._make_linear_input_activation_ref(linear_output, x)

    assert linear_ref.fallback is None
    assert torch.equal(linear_ref.resolve_and_release(), x)
    assert linear_ref.grad_fn is None

    relu_output = torch.relu(linear_output)
    relu_ref = local_mlp._make_relu_output_activation_ref(relu_output)

    assert relu_ref.fallback is None
    assert torch.equal(relu_ref.resolve_and_release(), relu_output)
    assert relu_ref.grad_fn is None


def test_default_modular_hvp_consumes_local_dual_parameter_in_forward() -> None:
    class InspectLinear(nn.Linear):
        calls: int
        dual_weight_calls: int
        dual_bias_calls: int
        primal_calls: int

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__(in_features, out_features)
            self.calls = 0
            self.dual_weight_calls = 0
            self.dual_bias_calls = 0
            self.primal_calls = 0
            self.dual_input_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            weight_is_dual = is_dual(self.weight)
            bias_is_dual = is_dual(self.bias)
            self.dual_input_calls += int(is_dual(input))
            self.dual_weight_calls += int(weight_is_dual)
            self.dual_bias_calls += int(bias_is_dual)
            self.primal_calls += int(not weight_is_dual and not bias_is_dual)
            return super().forward(input)

    torch.manual_seed(0)
    model = nn.Sequential(InspectLinear(3, 2)).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    with modular_hvp(model, tangents):
        criterion(model(x), target).backward()

    layer = model[0]
    assert layer.calls == 1
    assert layer.dual_weight_calls == 1
    assert layer.dual_bias_calls == 0
    assert layer.dual_input_calls == 0
    assert layer.primal_calls == 0
    for parameter in model.parameters():
        assert parameter.hvp is not None


def test_default_modular_hvp_does_not_export_dual_activations() -> None:
    class InspectReLU(nn.ReLU):
        def __init__(self) -> None:
            super().__init__()
            self.dual_input_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.dual_input_calls += int(is_dual(input))
            return super().forward(input)

    class InspectLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__(in_features, out_features)
            self.dual_input_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.dual_input_calls += int(is_dual(input))
            return super().forward(input)

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(3, 5),
        InspectReLU(),
        InspectLinear(5, 4),
        InspectReLU(),
        InspectLinear(4, 2),
    ).double()
    x = torch.randn(6, 3, dtype=torch.float64)
    target = torch.randn(6, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    with modular_hvp(model, tangents):
        criterion(model(x), target).backward()

    for module in model:
        if isinstance(module, (InspectReLU, InspectLinear)):
            assert module.dual_input_calls == 0
    for parameter in model.parameters():
        assert parameter.hvp is not None


def test_default_modular_hvp_accumulates_reused_parameter_hvps() -> None:
    torch.manual_seed(1)

    def make_model() -> nn.Sequential:
        shared = nn.Linear(3, 3).double()
        return nn.Sequential(shared, nn.ReLU(), shared)

    baseline = make_model()
    model = make_model()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    with modular_hvp(model, tangents):
        criterion(model(x), target).backward()

    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_explicit_hook_backend_preserves_primal_forward_and_gradients() -> None:
    torch.manual_seed(0)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3)
    target = torch.randn(5, 2)
    criterion = nn.MSELoss()

    baseline_loss = criterion(baseline(x), target)
    baseline_loss.backward()

    backend = RecordingBackend()
    with modular_hvp(model, _tangents_by_name(model), backend=backend):
        loss = criterion(model(x), target)
        loss.backward()

    assert torch.allclose(loss.detach(), baseline_loss.detach())
    for (_, expected), (_, actual) in zip(
        baseline.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.allclose(actual.grad, expected.grad)
        assert actual.hvp is not None
        assert torch.equal(actual.hvp, torch.zeros_like(actual))
    assert backend.local_forward_modules == ["Linear", "Linear"]
    assert backend.backward_modules == ["Linear", "Linear"]


def test_default_context_restores_forward_and_removes_active_flags() -> None:
    model = nn.Linear(3, 2)
    original_forward = model.forward

    with modular_hvp(model, _tangents_by_name(model)):
        assert model.forward is not original_forward

    assert "forward" not in model.__dict__
    assert not hasattr(model, "_modular_hvp_local_mlp_active")
    assert model.forward.__func__ is original_forward.__func__


def test_explicit_hook_context_restores_forward_and_removes_active_flags() -> None:
    model = nn.Linear(3, 2)
    original_forward = model.forward

    with modular_hvp(model, _tangents_by_name(model), backend=FakeDualBackend()):
        assert model.forward is not original_forward

    assert "forward" not in model.__dict__
    assert not hasattr(model, "_modular_hvp_runtime_active")
    assert model.forward.__func__ is original_forward.__func__


def test_parameter_object_tangent_keys_are_supported() -> None:
    model = nn.Linear(3, 2)
    tangents = {
        parameter: torch.ones_like(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    with modular_hvp(model, tangents):
        output = model(torch.randn(4, 3))
        assert type(output) is torch.Tensor
        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

    for parameter in model.parameters():
        assert parameter.hvp is not None
        assert parameter.hvp.shape == parameter.shape


def test_missing_tangent_is_rejected() -> None:
    model = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="missing tangents"):
        modular_hvp(model, {"weight": torch.ones_like(model.weight)})


def test_shape_mismatch_is_rejected() -> None:
    model = nn.Linear(3, 2)
    tangents = _tangents_by_name(model)
    tangents["weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="shape"):
        modular_hvp(model, tangents)


def test_duplicate_name_and_object_tangent_is_rejected() -> None:
    model = nn.Linear(3, 2)
    tangents: dict[str | nn.Parameter, torch.Tensor] = {
        "weight": torch.ones_like(model.weight),
        model.weight: torch.ones_like(model.weight),
        "bias": torch.ones_like(model.bias),
    }

    with pytest.raises(ValueError, match="duplicate"):
        modular_hvp(model, tangents)


class ReusedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.linear(x)


def test_reused_module_consumes_forward_records_lifo_and_accumulates_hvp() -> None:
    model = ReusedLinear()
    backend = CountingBackend()

    with modular_hvp(model, _tangents_by_name(model), backend=backend):
        model(torch.randn(2, 3)).sum().backward()

    for parameter in model.parameters():
        assert backend.calls_by_parameter_id[id(parameter)] == 2
        assert parameter.hvp is not None
        assert torch.equal(parameter.hvp, torch.zeros_like(parameter))


def test_existing_stale_hvp_is_cleared_on_entry() -> None:
    model = nn.Linear(3, 2)
    model.weight.hvp = torch.full_like(model.weight, 17.0)

    with modular_hvp(model, _tangents_by_name(model)):
        output = model(torch.randn(4, 3))
        assert type(output) is torch.Tensor
        nn.MSELoss()(output, torch.zeros_like(output)).backward()

    assert model.weight.hvp is not None
    assert not torch.equal(model.weight.hvp, torch.full_like(model.weight, 17.0))
