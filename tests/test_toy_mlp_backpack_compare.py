from __future__ import annotations

from benchmarks.compare_toy_mlp import (
    ToyMLPConfig,
    backpack_autodiff_block_hvp,
    backpack_hmp_block_hvp,
    compare_results,
    make_problem,
    modular_hvp_block_hvp,
)


def test_toy_mlp_block_hvps_match_backpack_baselines() -> None:
    config = ToyMLPConfig(
        seed=7,
        batch_size=8,
        d_in=4,
        d_hidden=6,
        d_out=3,
        dtype="float64",
    )

    model, loss_fn, x, target, vectors = make_problem(config)
    ours = modular_hvp_block_hvp(model, loss_fn, x, target, vectors)

    model, loss_fn, x, target, vectors = make_problem(config)
    hmp = backpack_hmp_block_hvp(model, loss_fn, x, target, vectors)

    model, loss_fn, x, target, vectors = make_problem(config)
    autodiff = backpack_autodiff_block_hvp(model, loss_fn, x, target, vectors)

    hmp_errors = compare_results(ours, hmp)
    autodiff_errors = compare_results(ours, autodiff)

    assert hmp_errors["max_abs"] < 1e-10
    assert hmp_errors["max_rel"] < 1e-9
    assert autodiff_errors["max_abs"] < 1e-10
    assert autodiff_errors["max_rel"] < 1e-9
