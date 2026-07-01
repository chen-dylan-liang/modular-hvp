"""Compare toy-MLP block HVPs against BackPACK baselines.

This script benchmarks three per-parameter block-HVP paths:

1. ModularHVP's current DualTensor backend plus autograd on the dual loss tangent.
2. BackPACK HMP, which stores a per-parameter Hessian-matrix-product closure.
3. BackPACK's reverse-over-reverse hessian_vector_product utility.

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

from modular_hvp import run_with_dual_parameter, tangent


@dataclass(frozen=True)
class ToyMLPConfig:
    seed: int = 0
    batch_size: int = 64
    d_in: int = 16
    d_hidden: int = 32
    d_out: int = 8
    dtype: str = "float64"
    device: str = "cpu"


@dataclass(frozen=True)
class MethodStats:
    method: str
    mean_time_s: float
    min_time_s: float
    max_time_s: float
    peak_rss_delta_bytes: int
    peak_python_alloc_bytes: int
    peak_cuda_alloc_bytes: int | None


BlockHVPFn = Callable[
    [nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]


def make_toy_mlp(config: ToyMLPConfig) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.d_in, config.d_hidden),
        nn.ReLU(),
        nn.Linear(config.d_hidden, config.d_out),
    ).to(device=config.device, dtype=_dtype(config.dtype))


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

    for name, param in params.items():
        output_hat = run_with_dual_parameter(model, name, vectors[name], x)
        loss_hat = loss_fn(output_hat, target)
        hvp = torch.autograd.grad(
            tangent(loss_hat),
            param,
            create_graph=False,
            retain_graph=False,
            materialize_grads=True,
        )[0].detach()
        setattr(param, "hvp", hvp)
        hvps[name] = hvp

    return hvps


def backpack_hmp_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute block HVPs with BackPACK's HMP extension."""

    from backpack import backpack, extend
    from backpack.extensions import HMP

    config = _config_from_model_and_data(model, x, target)
    bp_model = extend(make_toy_mlp(config))
    bp_model.load_state_dict(model.state_dict())
    bp_loss_fn = extend(type(loss_fn)(reduction=loss_fn.reduction))

    loss = bp_loss_fn(bp_model(x), target)
    with backpack(HMP()):
        loss.backward()

    hvps: dict[str, torch.Tensor] = {}
    for name, param in bp_model.named_parameters():
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
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> MethodStats:
    times: list[float] = []
    peak_rss_deltas: list[int] = []
    peak_python_allocs: list[int] = []
    peak_cuda_allocs: list[int] = []

    for index in range(warmup + repeats):
        model, loss_fn, x, target, vectors = make_problem(config)
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
    first_linear = model[0]
    second_linear = model[2]
    if not isinstance(first_linear, nn.Linear) or not isinstance(second_linear, nn.Linear):
        raise TypeError("expected nn.Sequential(Linear, ReLU, Linear)")
    return ToyMLPConfig(
        batch_size=x.shape[0],
        d_in=first_linear.in_features,
        d_hidden=first_linear.out_features,
        d_out=second_linear.out_features,
        dtype=str(x.dtype).removeprefix("torch."),
        device=str(x.device),
    )


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name!r}")


def _methods() -> dict[str, BlockHVPFn]:
    return {
        "modular_dual": modular_dual_block_hvp,
        "backpack_hmp": backpack_hmp_block_hvp,
        "backpack_autodiff": backpack_autodiff_block_hvp,
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
    print("Correctness vs modular_dual")
    for method, errors in comparisons.items():
        print(
            f"  {method:17s} max_abs={errors['max_abs']:.3e} "
            f"max_rel={errors['max_rel']:.3e}"
        )

    print("\nBenchmark")
    header = (
        f"{'method':17s} {'mean ms':>10s} {'min ms':>10s} {'max ms':>10s} "
        f"{'peak RSS delta':>16s} {'py alloc peak':>15s} {'cuda peak':>12s}"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        print(
            f"{item.method:17s} "
            f"{item.mean_time_s * 1e3:10.3f} "
            f"{item.min_time_s * 1e3:10.3f} "
            f"{item.max_time_s * 1e3:10.3f} "
            f"{_format_bytes(item.peak_rss_delta_bytes):>16s} "
            f"{_format_bytes(item.peak_python_alloc_bytes):>15s} "
            f"{_format_bytes(item.peak_cuda_alloc_bytes):>12s}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-in", type=int, default=16)
    parser.add_argument("--d-hidden", type=int, default=32)
    parser.add_argument("--d-out", type=int, default=8)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
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


def main() -> None:
    args = parse_args()
    config = ToyMLPConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        d_in=args.d_in,
        d_hidden=args.d_hidden,
        d_out=args.d_out,
        dtype=args.dtype,
        device=args.device,
    )

    methods = _methods()
    model, loss_fn, x, target, vectors = make_problem(config)
    reference = methods["modular_dual"](model, loss_fn, x, target, vectors)

    comparisons: dict[str, dict[str, float]] = {}
    for name, method in methods.items():
        model, loss_fn, x, target, vectors = make_problem(config)
        candidate = method(model, loss_fn, x, target, vectors)
        comparisons[name] = compare_results(reference, candidate)

    stats = [
        benchmark_method(
            name,
            method,
            config,
            warmup=args.warmup,
            repeats=args.repeats,
            rss_sample_interval_s=args.rss_sample_interval_ms / 1000,
        )
        for name, method in methods.items()
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
