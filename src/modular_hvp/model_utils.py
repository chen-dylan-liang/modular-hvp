"""Model-structure helpers for the eager runtime."""

from __future__ import annotations

import torch
from torch import nn


def _validate_supported_model(model: nn.Module) -> None:
    del model


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


def _parameter_use_counts(model: nn.Module) -> dict[nn.Parameter, int]:
    counts: dict[nn.Parameter, int] = {}
    for _, module in model.named_modules(remove_duplicate=False):
        for parameter in module._parameters.values():
            if parameter is None or not parameter.requires_grad:
                continue
            counts[parameter] = counts.get(parameter, 0) + 1
    return counts
