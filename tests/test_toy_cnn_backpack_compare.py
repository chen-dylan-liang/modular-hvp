from __future__ import annotations

from benchmarks.compare_toy_cnn import (
    ToyCNNConfig,
    backpack_autodiff_block_hvp,
    backpack_hmp_prepared_block_hvp,
    compare_results,
    make_problem,
    modular_hvp_block_hvp,
)


def test_toy_cnn_block_hvps_match_backpack_baselines() -> None:
    config = ToyCNNConfig(
        seed=5,
        batch_size=4,
        channels=1,
        image_size=8,
        width=2,
        d_out=3,
        dtype="float32",
    )

    model, loss_fn, x, target, vectors = make_problem(config)
    ours = modular_hvp_block_hvp(model, loss_fn, x, target, vectors)

    model, loss_fn, x, target, vectors = make_problem(config)
    from backpack import extend

    model = extend(model)
    loss_fn = extend(type(loss_fn)(reduction=loss_fn.reduction))
    hmp = backpack_hmp_prepared_block_hvp(model, loss_fn, x, target, vectors)

    model, loss_fn, x, target, vectors = make_problem(config)
    autodiff = backpack_autodiff_block_hvp(model, loss_fn, x, target, vectors)

    hmp_errors = compare_results(ours, hmp)
    autodiff_errors = compare_results(ours, autodiff)

    assert hmp_errors["max_abs"] < 1e-5
    assert hmp_errors["max_rel"] < 1e-5
    assert autodiff_errors["max_abs"] < 1e-5
    assert autodiff_errors["max_rel"] < 1e-5
