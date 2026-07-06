from __future__ import annotations

import pytest

from benchmarks.compare_resnet20 import (
    ResNet20Config,
    backpack_autodiff_block_hvp,
    backpack_hmp_resnet20_block_hvp,
    compare_results,
    make_problem,
    modular_hvp_block_hvp,
)


def test_resnet20_block_hvps_match_backpack_autodiff() -> None:
    config = ResNet20Config(
        seed=11,
        batch_size=1,
        image_size=8,
        width=1,
        d_out=2,
        dtype="float64",
    )

    model, loss_fn, x, target, vectors = make_problem(config)
    ours = modular_hvp_block_hvp(model, loss_fn, x, target, vectors)

    model, loss_fn, x, target, vectors = make_problem(config)
    autodiff = backpack_autodiff_block_hvp(model, loss_fn, x, target, vectors)

    errors = compare_results(ours, autodiff)
    assert errors["max_abs"] < 1e-10
    assert errors["max_rel"] < 1e-8


def test_resnet20_backpack_hmp_is_unsupported_for_residual_graph() -> None:
    config = ResNet20Config(
        seed=12,
        batch_size=1,
        image_size=8,
        width=1,
        d_out=2,
        dtype="float32",
    )

    from backpack import extend

    model, loss_fn, x, target, vectors = make_problem(config)
    model = extend(model, use_converter=True)
    loss_fn = extend(type(loss_fn)(reduction=loss_fn.reduction))

    with pytest.raises(NotImplementedError, match="Extension saving to hmp"):
        backpack_hmp_resnet20_block_hvp(model, loss_fn, x, target, vectors)
