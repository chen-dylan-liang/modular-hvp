"""Compare toy-MLP block HVPs against BackPACK baselines.

This script benchmarks three per-parameter block-HVP paths plus a standard
PyTorch backward pass baseline:

1. ModularHVP's current DualTensor backend plus autograd on the dual loss tangent.
2. BackPACK HMP, which stores a per-parameter Hessian-matrix-product closure.
3. BackPACK's reverse-over-reverse hessian_vector_product utility.
4. Standard PyTorch ``loss.backward()`` for ordinary gradients.

The ModularHVP path here is still a toy backend comparison, not the final
``with modular_hvp(...): loss.backward()`` integration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch
from torch import nn

from modular_hvp import make_dual, run_with_dual_parameter, tangent


@dataclass(frozen=True)
class ToyMLPConfig:
    seed: int = 0
    batch_size: int = 64
    d_in: int = 16
    d_hidden: int = 32
    hidden_layers: int = 1
    d_out: int = 8
    dtype: str = "float64"
    device: str = "cpu"


@dataclass(frozen=True)
class MethodStats:
    method: str
    mean_time_s: float
    min_time_s: float
    max_time_s: float
    median_peak_rss_delta_bytes: int
    peak_rss_delta_bytes: int
    peak_python_alloc_bytes: int
    peak_cuda_alloc_bytes: int | None


BlockHVPFn = Callable[
    [nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]
ProblemFactory = Callable[
    [ToyMLPConfig],
    tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: BlockHVPFn
    make_problem: ProblemFactory
    compare_hvp: bool = True


def make_toy_mlp(config: ToyMLPConfig) -> nn.Sequential:
    if config.hidden_layers < 1:
        raise ValueError("hidden_layers must be at least 1")

    layers: list[nn.Module] = []
    in_features = config.d_in
    for _ in range(config.hidden_layers):
        layers.append(nn.Linear(in_features, config.d_hidden))
        layers.append(nn.ReLU())
        in_features = config.d_hidden
    layers.append(nn.Linear(in_features, config.d_out))
    return nn.Sequential(*layers).to(device=config.device, dtype=_dtype(config.dtype))


def make_problem(
    config: ToyMLPConfig,
) -> tuple[nn.Sequential, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    dtype = _dtype(config.dtype)
    model = make_toy_mlp(config)
    loss_fn = nn.MSELoss(reduction="mean")
    x = torch.randn(config.batch_size, config.d_in, device=config.device, dtype=dtype)
    target = torch.randn(
        config.batch_size, config.d_out, device=config.device, dtype=dtype
    )
    vectors = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return model, loss_fn, x, target, vectors


def modular_dual_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute one ``H_pp @ v_p`` per parameter using the DualTensor backend."""

    params = dict(model.named_parameters())
    hvps: dict[str, torch.Tensor] = {}

    for param in params.values():
        setattr(param, "hvp", None)

    original_requires_grad = {
        name: param.requires_grad for name, param in params.items()
    }
    try:
        for param in params.values():
            param.requires_grad_(False)

        if isinstance(model, nn.Sequential):
            hvps.update(
                _modular_dual_sequential_mlp_block_hvp(
                    model,
                    loss_fn,
                    x,
                    target,
                    vectors,
                    params,
                    original_requires_grad,
                )
            )
        else:
            hvps.update(
                _modular_dual_full_forward_block_hvp(
                    model,
                    loss_fn,
                    x,
                    target,
                    vectors,
                    params,
                    original_requires_grad,
                )
            )
    finally:
        for name, param in params.items():
            param.requires_grad_(original_requires_grad[name])

    return hvps


def _modular_dual_full_forward_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    original_requires_grad: dict[str, bool],
) -> dict[str, torch.Tensor]:
    hvps: dict[str, torch.Tensor] = {}
    for name, param in params.items():
        if not original_requires_grad[name]:
            continue
        param.requires_grad_(True)
        output_hat = run_with_dual_parameter(model, name, vectors[name], x)
        hvp = _hvp_from_dual_output(loss_fn, output_hat, target, param)
        param.requires_grad_(False)
        setattr(param, "hvp", hvp)
        hvps[name] = hvp
    return hvps


def _modular_dual_sequential_mlp_block_hvp(
    model: nn.Sequential,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
    params: dict[str, torch.Tensor],
    original_requires_grad: dict[str, bool],
) -> dict[str, torch.Tensor]:
    module_entries = tuple(model._modules.items())
    suffix_modules = tuple(model.children())
    prefix_activation = x
    hvps: dict[str, torch.Tensor] = {}

    for module_index, (module_name, module) in enumerate(module_entries):
        if not isinstance(module, nn.Linear):
            with torch.no_grad():
                prefix_activation = module(prefix_activation)
            continue

        for leaf_name in ("weight", "bias"):
            param_name = f"{module_name}.{leaf_name}"
            param = params.get(param_name)
            if param is None or not original_requires_grad[param_name]:
                continue

            tangent_value = vectors[param_name]
            param.requires_grad_(True)
            output_hat = _run_sequential_suffix_with_dual_parameter(
                model,
                start_index=module_index,
                suffix_modules=suffix_modules,
                param_name=param_name,
                tangent_value=tangent_value,
                input_value=prefix_activation.detach(),
            )
            hvp = _hvp_from_dual_output(loss_fn, output_hat, target, param)
            param.requires_grad_(False)
            setattr(param, "hvp", hvp)
            hvps[param_name] = hvp

        with torch.no_grad():
            prefix_activation = module(prefix_activation)

    return hvps


def _run_sequential_suffix_with_dual_parameter(
    model: nn.Sequential,
    *,
    start_index: int,
    suffix_modules: tuple[nn.Module, ...],
    param_name: str,
    tangent_value: torch.Tensor,
    input_value: torch.Tensor,
) -> torch.Tensor:
    module_path, _, leaf_name = param_name.rpartition(".")
    owner_module = model.get_submodule(module_path)
    original = owner_module._parameters[leaf_name]
    if original is None:
        raise ValueError(f"parameter {param_name!r} is None")

    owner_module._parameters[leaf_name] = make_dual(original, tangent_value)
    try:
        output = input_value
        for suffix_module in suffix_modules[start_index:]:
            output = suffix_module(output)
        return output
    finally:
        owner_module._parameters[leaf_name] = original


def _hvp_from_dual_output(
    loss_fn: nn.Module,
    output_hat: torch.Tensor,
    target: torch.Tensor,
    param: torch.Tensor,
) -> torch.Tensor:
    loss_hat = loss_fn(output_hat, target)
    return torch.autograd.grad(
        tangent(loss_hat),
        param,
        create_graph=False,
        retain_graph=False,
        materialize_grads=True,
    )[0].detach()


def backpack_hmp_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute block HVPs with BackPACK's HMP extension."""

    from backpack import extend

    config = _config_from_model_and_data(model, x, target)
    bp_model = extend(make_toy_mlp(config))
    bp_model.load_state_dict(model.state_dict())
    bp_loss_fn = extend(type(loss_fn)(reduction=loss_fn.reduction))

    return _backpack_hmp_prepared_block_hvp(bp_model, bp_loss_fn, x, target, vectors)


def _make_backpack_hmp_problem(
    config: ToyMLPConfig,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Create a problem with BackPACK extension outside timed benchmark scope."""

    from backpack import extend

    model, loss_fn, x, target, vectors = make_problem(config)
    return (
        extend(model),
        extend(type(loss_fn)(reduction=loss_fn.reduction)),
        x,
        target,
        vectors,
    )


def _backpack_hmp_prepared_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute HMP block HVPs on a model/loss already prepared by BackPACK."""

    from backpack import backpack
    from backpack.extensions import HMP

    loss = loss_fn(model(x), target)
    with backpack(HMP()):
        loss.backward()

    hvps: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        hvp = param.hmp(vectors[name].unsqueeze(0))[0].detach()
        setattr(param, "hvp", hvp)
        hvps[name] = hvp
    return hvps


def backpack_autodiff_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute block HVPs with BackPACK's reverse-over-reverse utility."""

    from backpack.hessianfree.hvp import hessian_vector_product

    params = dict(model.named_parameters())
    loss = loss_fn(model(x), target)
    hvps: dict[str, torch.Tensor] = {}

    for name, param in params.items():
        hvp = hessian_vector_product(loss, [param], [vectors[name]])[0].detach()
        setattr(param, "hvp", hvp)
        hvps[name] = hvp
    return hvps


def torch_backward_pass(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Run a standard forward/loss/backward pass and return ordinary gradients."""

    del vectors
    loss = loss_fn(model(x), target)
    loss.backward()
    return {
        name: param.grad.detach()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def compare_results(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, float]:
    max_abs = 0.0
    max_rel = 0.0
    for name, ref in reference.items():
        other = candidate[name]
        abs_error = (ref - other).abs().max().item()
        denom = ref.abs().max().item() + 1e-12
        max_abs = max(max_abs, abs_error)
        max_rel = max(max_rel, abs_error / denom)
    return {"max_abs": max_abs, "max_rel": max_rel}


def benchmark_method(
    name: str,
    method: BlockHVPFn,
    config: ToyMLPConfig,
    *,
    problem_factory: ProblemFactory = make_problem,
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> MethodStats:
    times: list[float] = []
    peak_rss_deltas: list[int] = []
    peak_python_allocs: list[int] = []
    peak_cuda_allocs: list[int] = []

    for index in range(warmup + repeats):
        model, loss_fn, x, target, vectors = problem_factory(config)
        if config.device.startswith("cuda"):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        with _PeakRSSDeltaSampler(rss_sample_interval_s) as rss_sampler:
            tracemalloc.start()
            start = time.perf_counter()
            method(model, loss_fn, x, target, vectors)
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            _, peak_python_alloc = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        if index >= warmup:
            times.append(elapsed)
            peak_rss_deltas.append(rss_sampler.peak_delta_bytes)
            peak_python_allocs.append(peak_python_alloc)
            if config.device.startswith("cuda"):
                peak_cuda_allocs.append(torch.cuda.max_memory_allocated())

    return MethodStats(
        method=name,
        mean_time_s=statistics.mean(times),
        min_time_s=min(times),
        max_time_s=max(times),
        median_peak_rss_delta_bytes=int(statistics.median(peak_rss_deltas)),
        peak_rss_delta_bytes=max(peak_rss_deltas),
        peak_python_alloc_bytes=max(peak_python_allocs),
        peak_cuda_alloc_bytes=max(peak_cuda_allocs) if peak_cuda_allocs else None,
    )


class _PeakRSSDeltaSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self.process = psutil.Process()
        self.start_rss_bytes = 0
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def peak_delta_bytes(self) -> int:
        return max(0, self.peak_rss_bytes - self.start_rss_bytes)

    def __enter__(self) -> "_PeakRSSDeltaSampler":
        self.start_rss_bytes = self.process.memory_info().rss
        self.peak_rss_bytes = self.start_rss_bytes
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.peak_rss_bytes = max(self.peak_rss_bytes, self.process.memory_info().rss)

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.peak_rss_bytes = max(
                self.peak_rss_bytes,
                self.process.memory_info().rss,
            )


def _config_from_model_and_data(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
) -> ToyMLPConfig:
    linear_layers = [module for module in model if isinstance(module, nn.Linear)]
    if len(linear_layers) < 2:
        raise TypeError("expected an MLP with at least two Linear layers")
    hidden_widths = [layer.out_features for layer in linear_layers[:-1]]
    if len(set(hidden_widths)) != 1:
        raise TypeError("benchmark model reconstruction expects uniform hidden width")
    return ToyMLPConfig(
        batch_size=x.shape[0],
        d_in=linear_layers[0].in_features,
        d_hidden=hidden_widths[0],
        hidden_layers=len(linear_layers) - 1,
        d_out=linear_layers[-1].out_features,
        dtype=str(x.dtype).removeprefix("torch."),
        device=str(x.device),
    )


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name!r}")


def _methods() -> dict[str, MethodSpec]:
    return {
        "modular_dual": MethodSpec(
            "modular_dual",
            modular_dual_block_hvp,
            make_problem,
        ),
        "backpack_hmp": MethodSpec(
            "backpack_hmp",
            _backpack_hmp_prepared_block_hvp,
            _make_backpack_hmp_problem,
        ),
        "backpack_autodiff": MethodSpec(
            "backpack_autodiff",
            backpack_autodiff_block_hvp,
            make_problem,
        ),
        "torch_backward": MethodSpec(
            "torch_backward",
            torch_backward_pass,
            make_problem,
            compare_hvp=False,
        ),
    }


def _format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "n/a"
    units = ("B", "KiB", "MiB", "GiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def _print_results(
    comparisons: dict[str, dict[str, float]],
    stats: list[MethodStats],
) -> None:
    print("HVP correctness vs modular_dual")
    for method, errors in comparisons.items():
        print(
            f"  {method:17s} max_abs={errors['max_abs']:.3e} "
            f"max_rel={errors['max_rel']:.3e}"
        )

    print("\nBenchmark")
    header = (
        f"{'method':17s} {'mean ms':>10s} {'min ms':>10s} {'max ms':>10s} "
        f"{'med RSS delta':>16s} {'max RSS delta':>16s} "
        f"{'py alloc peak':>15s} {'cuda peak':>12s}"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        print(
            f"{item.method:17s} "
            f"{item.mean_time_s * 1e3:10.3f} "
            f"{item.min_time_s * 1e3:10.3f} "
            f"{item.max_time_s * 1e3:10.3f} "
            f"{_format_bytes(item.median_peak_rss_delta_bytes):>16s} "
            f"{_format_bytes(item.peak_rss_delta_bytes):>16s} "
            f"{_format_bytes(item.peak_python_alloc_bytes):>15s} "
            f"{_format_bytes(item.peak_cuda_alloc_bytes):>12s}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("toy", "mnist-mlp"),
        default="toy",
        help="Use toy dimensions or a synthetic MNIST-shaped MLP.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--d-in", type=int, default=None)
    parser.add_argument("--d-hidden", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--d-out", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=1.0)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of a text table.",
    )
    return parser.parse_args()


def _preset_config(name: str) -> ToyMLPConfig:
    if name == "toy":
        return ToyMLPConfig()
    if name == "mnist-mlp":
        return ToyMLPConfig(
            batch_size=256,
            d_in=28 * 28,
            d_hidden=256,
            hidden_layers=3,
            d_out=10,
            dtype="float32",
        )
    raise ValueError(f"unknown preset: {name!r}")


def main() -> None:
    args = parse_args()
    preset = _preset_config(args.preset)
    config = ToyMLPConfig(
        seed=args.seed,
        batch_size=args.batch_size if args.batch_size is not None else preset.batch_size,
        d_in=args.d_in if args.d_in is not None else preset.d_in,
        d_hidden=args.d_hidden if args.d_hidden is not None else preset.d_hidden,
        hidden_layers=(
            args.hidden_layers if args.hidden_layers is not None else preset.hidden_layers
        ),
        d_out=args.d_out if args.d_out is not None else preset.d_out,
        dtype=args.dtype if args.dtype is not None else preset.dtype,
        device=args.device,
    )

    methods = _methods()
    reference_spec = methods["modular_dual"]
    model, loss_fn, x, target, vectors = reference_spec.make_problem(config)
    reference = reference_spec.fn(model, loss_fn, x, target, vectors)

    comparisons: dict[str, dict[str, float]] = {}
    for name, spec in methods.items():
        if not spec.compare_hvp:
            continue
        model, loss_fn, x, target, vectors = spec.make_problem(config)
        candidate = spec.fn(model, loss_fn, x, target, vectors)
        comparisons[name] = compare_results(reference, candidate)

    stats = [
        benchmark_method(
            spec.name,
            spec.fn,
            config,
            problem_factory=spec.make_problem,
            warmup=args.warmup,
            repeats=args.repeats,
            rss_sample_interval_s=args.rss_sample_interval_ms / 1000,
        )
        for spec in methods.values()
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "config": asdict(config),
                    "comparisons": comparisons,
                    "stats": [asdict(item) for item in stats],
                },
                indent=2,
            )
        )
    else:
        _print_results(comparisons, stats)


if __name__ == "__main__":
    main()
