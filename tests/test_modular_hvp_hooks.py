from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn

from modular_hvp import FakeDualBackend, LocalDualActivations, modular_hvp


def _tangents_by_name(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


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


def test_modular_hvp_preserves_primal_forward_and_gradients() -> None:
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


def test_context_restores_forward_and_removes_active_flags() -> None:
    model = nn.Linear(3, 2)
    original_forward = model.forward

    with modular_hvp(model, _tangents_by_name(model)):
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
        model(torch.randn(4, 3)).sum().backward()

    for parameter in model.parameters():
        assert parameter.hvp is not None
        assert torch.equal(parameter.hvp, torch.zeros_like(parameter))


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
        model(torch.randn(4, 3)).sum().backward()

    assert torch.equal(model.weight.hvp, torch.zeros_like(model.weight))
