"""Compare per-parameter and per-Linear-module MLP block HVPs."""

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_toy_mlp import (  # noqa: E402
    MethodStats,
    ToyMLPConfig,
    _format_bytes,
    _ProcessRSSSampler,
    compare_results,
    make_problem,
    modular_hvp_block_hvp,
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


def linear_module_blocks(model: nn.Module) -> dict[str, tuple[str, ...]]:
    """Return one custom block per Linear module: weight and optional bias."""

    named_parameters = dict(model.named_parameters())
    blocks: dict[str, tuple[str, ...]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
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
    return blocks


def modular_hvp_linear_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute one ModularHVP block per Linear module."""

    with modular_hvp(model, vectors, blocks=linear_module_blocks(model)):
        output = model(x)
        loss = loss_fn(output, target)
        loss.backward()

    hvps: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        hvp = getattr(parameter, "hvp", None)
        if hvp is None:
            raise RuntimeError(f"missing HVP for parameter {name!r}")
        hvps[name] = hvp.detach()
    return hvps


def backpack_autodiff_linear_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute one reverse-over-reverse HVP per Linear-module block."""

    from backpack.hessianfree.hvp import hessian_vector_product

    loss = loss_fn(model(x), target)
    named_parameters = dict(model.named_parameters())
    hvps: dict[str, torch.Tensor] = {}
    for block_names in linear_module_blocks(model).values():
        block_parameters = [named_parameters[name] for name in block_names]
        block_vectors = [vectors[name] for name in block_names]
        block_hvps = hessian_vector_product(loss, block_parameters, block_vectors)
        for name, hvp in zip(block_names, block_hvps, strict=True):
            hvps[name] = hvp.detach()
    return hvps


def _methods() -> dict[str, MethodSpec]:
    return {
        "modular_per_parameter": MethodSpec(
            "modular_per_parameter",
            modular_hvp_block_hvp,
        ),
        "modular_linear_blocks": MethodSpec(
            "modular_linear_blocks",
            modular_hvp_linear_block_hvp,
        ),
        "backpack_autodiff_linear_blocks": MethodSpec(
            "backpack_autodiff_linear_blocks",
            backpack_autodiff_linear_block_hvp,
        ),
        "torch_backward": MethodSpec("torch_backward", torch_backward_pass),
    }


def benchmark_method(
    name: str,
    config: ToyMLPConfig,
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
    config: ToyMLPConfig,
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
    config: ToyMLPConfig,
    iterations: int,
    conn: Connection,
) -> None:
    try:
        spec = _methods()[name]
        if name == "backpack_autodiff_linear_blocks":
            from backpack.hessianfree import hvp as _hvp  # noqa: F401

        for index in range(iterations):
            iteration_config = replace(config, seed=config.seed + index)
            model, loss_fn, x, target, vectors = make_problem(iteration_config)
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


def compare_block_outputs(config: ToyMLPConfig) -> dict[str, dict[str, float]]:
    reference_model, loss_fn, x, target, vectors = make_problem(config)
    reference = backpack_autodiff_linear_block_hvp(
        reference_model,
        loss_fn,
        x,
        target,
        vectors,
    )

    module_model, loss_fn, x, target, vectors = make_problem(config)
    module_hvps = modular_hvp_linear_block_hvp(
        module_model,
        loss_fn,
        x,
        target,
        vectors,
    )

    parameter_model, loss_fn, x, target, vectors = make_problem(config)
    parameter_hvps = modular_hvp_block_hvp(
        parameter_model,
        loss_fn,
        x,
        target,
        vectors,
    )

    return {
        "modular_linear_blocks_vs_autodiff_linear_blocks": compare_results(
            reference,
            module_hvps,
        ),
        "modular_per_parameter_vs_linear_blocks": compare_results(
            reference,
            parameter_hvps,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=int, nargs="+", default=[4, 50])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--d-in", type=int, default=784)
    parser.add_argument("--d-hidden", type=int, default=256)
    parser.add_argument("--d-out", type=int, default=10)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=1.0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/mlp_linear_blocks.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: list[dict[str, Any]] = []
    comparisons_by_depth: dict[str, dict[str, dict[str, float]]] = {}

    for depth in args.depths:
        config = ToyMLPConfig(
            seed=args.seed,
            batch_size=args.batch_size,
            d_in=args.d_in,
            d_hidden=args.d_hidden,
            hidden_layers=depth,
            d_out=args.d_out,
            dtype=args.dtype,
            device=args.device,
        )
        print(f"depth={depth}: checking grouped HVP correctness", flush=True)
        comparisons = compare_block_outputs(config)
        comparisons_by_depth[str(depth)] = comparisons
        for name, errors in comparisons.items():
            print(
                f"  {name}: max_abs={errors['max_abs']:.3e} "
                f"max_rel={errors['max_rel']:.3e}",
                flush=True,
            )

        for method_name in _methods():
            print(f"depth={depth}: benchmarking {method_name}", flush=True)
            stats = benchmark_method(
                method_name,
                config,
                warmup=args.warmup,
                repeats=args.repeats,
                rss_sample_interval_s=args.rss_sample_interval_ms / 1000,
            )
            record = {"depth": depth, **asdict(stats)}
            records.append(record)

        _print_depth_summary(depth, records)

    payload = {
        "config": {
            "depths": args.depths,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "d_in": args.d_in,
            "d_hidden": args.d_hidden,
            "d_out": args.d_out,
            "dtype": args.dtype,
            "device": args.device,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "repeat_seeds": [
                args.seed + index
                for index in range(args.warmup, args.warmup + args.repeats)
            ],
            "rss_sample_interval_ms": args.rss_sample_interval_ms,
        },
        "comparisons": comparisons_by_depth,
        "records": records,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}", flush=True)


def _print_depth_summary(depth: int, records: Sequence[Mapping[str, Any]]) -> None:
    depth_records = [record for record in records if record["depth"] == depth]
    base = next(
        record for record in depth_records if record["method"] == "modular_linear_blocks"
    )
    print(f"\ndepth={depth} benchmark")
    header = (
        f"{'method':33s} {'mean ms':>10s} {'time vs grouped':>15s} "
        f"{'med peak RSS':>14s} {'RSS vs grouped':>14s}"
    )
    print(header)
    print("-" * len(header))
    for record in depth_records:
        time_ratio = record["mean_time_s"] / base["mean_time_s"]
        rss_ratio = record["median_peak_rss_bytes"] / base["median_peak_rss_bytes"]
        print(
            f"{record['method']:33s} "
            f"{record['mean_time_s'] * 1e3:10.3f} "
            f"{time_ratio:15.2f}x "
            f"{_format_bytes(record['median_peak_rss_bytes']):>14s} "
            f"{rss_ratio:14.2f}x"
        )
    print()


if __name__ == "__main__":
    main()
