"""Compare ModularHVP and reverse-over-reverse for ResNet20 module blocks."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import statistics
import sys
import time
import tracemalloc
import traceback
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_resnet20 import ResNet20Config, make_problem  # noqa: E402
from benchmarks.compare_toy_mlp import (  # noqa: E402
    MethodStats,
    _format_bytes,
    _ProcessRSSSampler,
    compare_results,
    torch_backward_pass,
)
from modular_hvp import modular_hvp  # noqa: E402


BlockHVPFn = Callable[
    [nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: BlockHVPFn
    compare_hvp: bool = True


def conv_linear_blocks(model: nn.Module) -> dict[str, tuple[str, ...]]:
    """Group Conv2d/Linear direct parameters; keep other parameters singleton."""

    named_parameters = dict(model.named_parameters())
    assigned: set[str] = set()
    blocks: dict[str, tuple[str, ...]] = {}

    for module_name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue
        prefix = f"{module_name}." if module_name else ""
        names = tuple(
            full_name
            for parameter_name, parameter in module.named_parameters(recurse=False)
            if parameter.requires_grad
            for full_name in (prefix + parameter_name,)
            if full_name in named_parameters
        )
        if names:
            blocks[module_name or "<root>"] = names
            assigned.update(names)

    for name, parameter in named_parameters.items():
        if parameter.requires_grad and name not in assigned:
            blocks[name] = (name,)

    return blocks


def modular_resnet20_conv_linear_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with modular_hvp(model, vectors, blocks=conv_linear_blocks(model)):
        loss = loss_fn(model(x), target)
        loss.backward()
    return _collect_hvps(model)


def ror_resnet20_conv_linear_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return _custom_block_ror_hvps(
        loss=loss_fn(model(x), target),
        model=model,
        vectors=vectors,
        blocks=conv_linear_blocks(model),
    )


def _custom_block_ror_hvps(
    *,
    loss: torch.Tensor,
    model: nn.Module,
    vectors: Mapping[str, torch.Tensor],
    blocks: Mapping[str, tuple[str, ...]],
) -> dict[str, torch.Tensor]:
    named_parameters = dict(model.named_parameters())
    hvps: dict[str, torch.Tensor] = {}
    for block_names in blocks.values():
        parameters = [named_parameters[name] for name in block_names]
        block_vectors = [vectors[name] for name in block_names]
        gradients = torch.autograd.grad(
            loss,
            parameters,
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )
        directional_gradient = sum(
            (gradient * tangent).sum()
            for gradient, tangent in zip(gradients, block_vectors, strict=True)
        )
        block_hvps = torch.autograd.grad(
            directional_gradient,
            parameters,
            retain_graph=True,
            materialize_grads=True,
        )
        for name, hvp in zip(block_names, block_hvps, strict=True):
            hvps[name] = hvp.detach()
    return hvps


def _collect_hvps(model: nn.Module) -> dict[str, torch.Tensor]:
    hvps: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        hvp = getattr(parameter, "hvp", None)
        if hvp is None:
            raise RuntimeError(f"missing HVP for parameter {name!r}")
        hvps[name] = hvp.detach()
    return hvps


def benchmark_method(
    name: str,
    config: ResNet20Config,
    *,
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> MethodStats:
    result = _run_method_benchmark_in_child(
        name=name,
        config=config,
        iterations=warmup + repeats,
        warmup=warmup,
        rss_sample_interval_s=rss_sample_interval_s,
    )
    return MethodStats(
        method=name,
        mean_time_s=statistics.mean(result["times"]),
        min_time_s=min(result["times"]),
        max_time_s=max(result["times"]),
        median_peak_rss_bytes=int(statistics.median(result["peak_rss_values"])),
        peak_rss_bytes=max(result["peak_rss_values"]),
        median_avg_rss_bytes=int(statistics.median(result["avg_rss_values"])),
        peak_python_alloc_bytes=max(result["peak_python_allocs"]),
        peak_cuda_alloc_bytes=(
            max(result["peak_cuda_allocs"]) if result["peak_cuda_allocs"] else None
        ),
    )


def _run_method_benchmark_in_child(
    *,
    name: str,
    config: ResNet20Config,
    iterations: int,
    warmup: int,
    rss_sample_interval_s: float,
) -> dict[str, list[Any]]:
    times: list[float] = []
    peak_rss_values: list[int] = []
    avg_rss_values: list[int] = []
    peak_python_allocs: list[int] = []
    peak_cuda_allocs: list[int] = []

    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=True)
    process = context.Process(
        target=_child_method_benchmark,
        args=(name, config, iterations, child_conn),
    )
    process.start()
    child_conn.close()
    psutil_process = psutil.Process(process.pid)

    try:
        for expected_index in range(iterations):
            ready_message = parent_conn.recv()
            if ready_message["event"] == "error":
                raise RuntimeError(ready_message["traceback"])
            if ready_message["event"] != "ready":
                raise RuntimeError(f"unexpected child message: {ready_message!r}")

            with _ProcessRSSSampler(psutil_process, rss_sample_interval_s) as sampler:
                parent_conn.send({"event": "start", "index": expected_index})
                done_message = parent_conn.recv()

            if done_message["event"] == "error":
                raise RuntimeError(done_message["traceback"])
            if done_message["event"] != "done":
                raise RuntimeError(f"unexpected child message: {done_message!r}")

            if expected_index >= warmup:
                times.append(done_message["elapsed_s"])
                peak_rss_values.append(sampler.peak_rss_bytes)
                avg_rss_values.append(sampler.avg_rss_bytes)
                peak_python_allocs.append(done_message["peak_python_alloc_bytes"])
                if done_message["peak_cuda_alloc_bytes"] is not None:
                    peak_cuda_allocs.append(done_message["peak_cuda_alloc_bytes"])
    finally:
        process.join()
        parent_conn.close()

    if process.exitcode != 0:
        raise RuntimeError(f"child process for {name!r} exited with {process.exitcode}")
    return {
        "times": times,
        "peak_rss_values": peak_rss_values,
        "avg_rss_values": avg_rss_values,
        "peak_python_allocs": peak_python_allocs,
        "peak_cuda_allocs": peak_cuda_allocs,
    }


def _child_method_benchmark(
    name: str,
    config: ResNet20Config,
    iterations: int,
    conn: Connection,
) -> None:
    try:
        spec = _methods()[name]
        for index in range(iterations):
            model, loss_fn, x, target, vectors = make_problem(config)
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            conn.send({"event": "ready", "index": index})
            start_message = conn.recv()
            if start_message.get("event") != "start":
                raise RuntimeError(f"unexpected parent message: {start_message!r}")

            tracemalloc.start()
            start = time.perf_counter()
            result = spec.fn(model, loss_fn, x, target, vectors)
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            _, peak_python_alloc = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_cuda_alloc = (
                torch.cuda.max_memory_allocated()
                if config.device.startswith("cuda")
                else None
            )
            conn.send(
                {
                    "event": "done",
                    "index": index,
                    "elapsed_s": elapsed,
                    "peak_python_alloc_bytes": peak_python_alloc,
                    "peak_cuda_alloc_bytes": peak_cuda_alloc,
                }
            )
            del result, model, loss_fn, x, target, vectors
            gc.collect()
    except BaseException:
        conn.send({"event": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        conn.close()


def compare_hvps(config: ResNet20Config) -> dict[str, dict[str, float]]:
    reference_model, loss_fn, x, target, vectors = make_problem(config)
    reference = ror_resnet20_conv_linear_block_hvp(
        reference_model,
        loss_fn,
        x,
        target,
        vectors,
    )
    comparisons: dict[str, dict[str, float]] = {}
    for name, spec in _methods().items():
        if not spec.compare_hvp:
            continue
        model, loss_fn, x, target, vectors = make_problem(config)
        comparisons[name] = compare_results(
            reference,
            spec.fn(model, loss_fn, x, target, vectors),
        )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=8)
    parser.add_argument("--width", type=int, default=2)
    parser.add_argument("--d-out", type=int, default=3)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=1.0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/resnet20_conv_linear_blocks.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ResNet20Config(
        seed=args.seed,
        batch_size=args.batch_size,
        image_size=args.image_size,
        width=args.width,
        d_out=args.d_out,
        dtype=args.dtype,
        device=args.device,
    )
    comparisons = compare_hvps(config)
    for name, errors in comparisons.items():
        print(
            f"{name}: max_abs={errors['max_abs']:.3e} "
            f"max_rel={errors['max_rel']:.3e}",
            flush=True,
        )
    stats = [
        benchmark_method(
            name,
            config,
            warmup=args.warmup,
            repeats=args.repeats,
            rss_sample_interval_s=args.rss_sample_interval_ms / 1000,
        )
        for name in _methods()
    ]
    _print_results(stats)
    payload = {
        "config": asdict(config)
        | {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rss_sample_interval_ms": args.rss_sample_interval_ms,
        },
        "block_partition": "Conv2d/Linear direct parameters grouped; all others singleton",
        "comparisons": comparisons,
        "stats": [asdict(item) for item in stats],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}", flush=True)


def _methods() -> dict[str, MethodSpec]:
    return {
        "modular_conv_linear": MethodSpec(
            "modular_conv_linear",
            modular_resnet20_conv_linear_block_hvp,
        ),
        "ror_conv_linear": MethodSpec(
            "ror_conv_linear",
            ror_resnet20_conv_linear_block_hvp,
        ),
        "torch_backward": MethodSpec(
            "torch_backward",
            torch_backward_pass,
            compare_hvp=False,
        ),
    }


def _print_results(stats: list[MethodStats]) -> None:
    reference = next(item for item in stats if item.method == "modular_conv_linear")
    print("\nBenchmark")
    header = (
        f"{'method':20s} {'mean ms':>10s} {'time vs modular':>16s} "
        f"{'med peak RSS':>14s} {'RSS vs modular':>14s}"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        print(
            f"{item.method:20s} "
            f"{item.mean_time_s * 1e3:10.3f} "
            f"{item.mean_time_s / reference.mean_time_s:16.2f}x "
            f"{_format_bytes(item.median_peak_rss_bytes):>14s} "
            f"{item.median_peak_rss_bytes / reference.median_peak_rss_bytes:14.2f}x"
        )


if __name__ == "__main__":
    main()
