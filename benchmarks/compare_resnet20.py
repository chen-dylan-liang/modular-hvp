"""Compare public ``modular_hvp(...)`` block HVPs on a ResNet20-shaped CNN."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import statistics
import sys
import threading
import time
import tracemalloc
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

import psutil
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_toy_mlp import (  # noqa: E402
    backpack_autodiff_block_hvp,
    compare_results,
    modular_hvp_block_hvp,
    torch_backward_pass,
)


@dataclass(frozen=True)
class ResNet20Config:
    seed: int = 0
    batch_size: int = 2
    image_size: int = 8
    width: int = 2
    d_out: int = 3
    dtype: str = "float32"
    device: str = "cpu"


@dataclass(frozen=True)
class MethodStats:
    method: str
    mean_time_s: float
    min_time_s: float
    max_time_s: float
    median_peak_rss_bytes: int
    peak_rss_bytes: int
    median_avg_rss_bytes: int
    peak_python_alloc_bytes: int
    peak_cuda_alloc_bytes: int | None


@dataclass(frozen=True)
class MethodFailure:
    method: str
    error_type: str
    message: str


BlockHVPFn = Callable[
    [nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    dict[str, torch.Tensor],
]
ProblemFactory = Callable[
    [ResNet20Config],
    tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: BlockHVPFn
    make_problem: ProblemFactory
    compare_hvp: bool = True


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu2(out)


class ResNet20(nn.Module):
    def __init__(self, *, width: int, num_classes: int, image_size: int) -> None:
        super().__init__()
        if image_size % 4 != 0:
            raise ValueError("image_size must be divisible by 4")
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.stage1 = self._make_stage(width, width, blocks=3, stride=1)
        self.stage2 = self._make_stage(width, 2 * width, blocks=3, stride=2)
        self.stage3 = self._make_stage(2 * width, 4 * width, blocks=3, stride=2)
        self.pool = nn.AvgPool2d(kernel_size=image_size // 4)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * width, num_classes)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            BasicBlock(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = self.flatten(out)
        return self.fc(out)


def make_resnet20(config: ResNet20Config) -> ResNet20:
    return ResNet20(
        width=config.width,
        num_classes=config.d_out,
        image_size=config.image_size,
    ).to(device=config.device, dtype=_dtype(config.dtype)).eval()


def make_problem(
    config: ResNet20Config,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    dtype = _dtype(config.dtype)
    model = make_resnet20(config)
    loss_fn = nn.MSELoss(reduction="mean")
    x = torch.randn(
        config.batch_size,
        3,
        config.image_size,
        config.image_size,
        device=config.device,
        dtype=dtype,
    )
    target = torch.randn(config.batch_size, config.d_out, device=config.device, dtype=dtype)
    vectors = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return model, loss_fn, x, target, vectors


def _make_backpack_hmp_problem(
    config: ResNet20Config,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    from backpack import extend

    model, loss_fn, x, target, vectors = make_problem(config)
    return (
        extend(model, use_converter=True),
        extend(type(loss_fn)(reduction=loss_fn.reduction)),
        x,
        target,
        vectors,
    )


def _make_backpack_autodiff_problem(
    config: ResNet20Config,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    from backpack.hessianfree import hvp as _hvp  # noqa: F401

    return make_problem(config)


def backpack_hmp_resnet20_block_hvp(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    from backpack import backpack
    from backpack.extensions import HMP

    loss = loss_fn(model(x), target)
    with backpack(HMP()):
        loss.backward()
    hvps: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name not in vectors:
            raise RuntimeError(
                "BackPACK converter changed parameter names; cannot compare HMP fairly"
            )
        hvps[name] = parameter.hmp(vectors[name].unsqueeze(0))[0].detach()
    return hvps


def benchmark_method(
    name: str,
    config: ResNet20Config,
    *,
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> MethodStats | MethodFailure:
    result = _run_method_benchmark_in_child(
        name=name,
        config=config,
        warmup=warmup,
        repeats=repeats,
        rss_sample_interval_s=rss_sample_interval_s,
    )
    if "failure" in result:
        failure = result["failure"]
        return MethodFailure(
            method=name,
            error_type=failure["error_type"],
            message=failure["message"],
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
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> dict[str, Any]:
    times: list[float] = []
    peak_rss_values: list[int] = []
    avg_rss_values: list[int] = []
    peak_python_allocs: list[int] = []
    peak_cuda_allocs: list[int] = []

    context = mp.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=True)
    process = context.Process(
        target=_child_method_benchmark,
        args=(name, config, warmup + repeats, child_conn),
    )
    process.start()
    child_conn.close()
    psutil_process = psutil.Process(process.pid)

    try:
        for expected_index in range(warmup + repeats):
            ready_message = parent_conn.recv()
            if ready_message["event"] == "error":
                process.join()
                parent_conn.close()
                return {"failure": ready_message}
            if ready_message["event"] != "ready":
                raise RuntimeError(f"unexpected child message: {ready_message!r}")

            with _ProcessRSSSampler(psutil_process, rss_sample_interval_s) as sampler:
                parent_conn.send({"event": "start", "index": expected_index})
                done_message = parent_conn.recv()

            if done_message["event"] == "error":
                process.join()
                parent_conn.close()
                return {"failure": done_message}
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
            iteration_config = replace(config, seed=config.seed + index)
            model, loss_fn, x, target, vectors = spec.make_problem(iteration_config)
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
    except BaseException as exc:
        conn.send(
            {
                "event": "error",
                "error_type": type(exc).__name__,
                "message": str(exc).splitlines()[0],
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        conn.close()


class _ProcessRSSSampler:
    def __init__(self, process: psutil.Process, interval_s: float) -> None:
        self.interval_s = interval_s
        self.process = process
        self.peak_rss_bytes = 0
        self._rss_sum_bytes = 0
        self._sample_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def avg_rss_bytes(self) -> int:
        if self._sample_count == 0:
            return self.peak_rss_bytes
        return int(self._rss_sum_bytes / self._sample_count)

    def __enter__(self) -> "_ProcessRSSSampler":
        self._record_sample()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._record_sample()

    def _sample(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._record_sample()

    def _record_sample(self) -> None:
        try:
            rss = self.process.memory_info().rss
        except psutil.Error:
            return
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
        self._rss_sum_bytes += rss
        self._sample_count += 1


def _methods() -> dict[str, MethodSpec]:
    return {
        "modular_hvp": MethodSpec(
            "modular_hvp",
            modular_hvp_block_hvp,
            make_problem,
        ),
        "backpack_hmp": MethodSpec(
            "backpack_hmp",
            backpack_hmp_resnet20_block_hvp,
            _make_backpack_hmp_problem,
        ),
        "backpack_autodiff": MethodSpec(
            "backpack_autodiff",
            backpack_autodiff_block_hvp,
            _make_backpack_autodiff_problem,
        ),
        "torch_backward": MethodSpec(
            "torch_backward",
            torch_backward_pass,
            make_problem,
            compare_hvp=False,
        ),
    }


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name!r}")


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
    comparisons: dict[str, dict[str, float] | MethodFailure],
    stats: list[MethodStats | MethodFailure],
) -> None:
    print("HVP correctness vs modular_hvp")
    for method, result in comparisons.items():
        if isinstance(result, MethodFailure):
            print(f"  {method:17s} unsupported: {result.error_type}: {result.message}")
        else:
            print(
                f"  {method:17s} max_abs={result['max_abs']:.3e} "
                f"max_rel={result['max_rel']:.3e}"
            )

    print("\nBenchmark")
    header = (
        f"{'method':17s} {'mean ms':>10s} {'min ms':>10s} {'max ms':>10s} "
        f"{'med avg RSS':>14s} {'med peak RSS':>14s} {'max peak RSS':>14s} "
        f"{'py alloc peak':>15s} {'cuda peak':>12s}"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        if isinstance(item, MethodFailure):
            message = f"{item.error_type}: {item.message}"
            print(f"{item.method:17s} {'unsupported':>10s} {message}")
            continue
        print(
            f"{item.method:17s} "
            f"{item.mean_time_s * 1e3:10.3f} "
            f"{item.min_time_s * 1e3:10.3f} "
            f"{item.max_time_s * 1e3:10.3f} "
            f"{_format_bytes(item.median_avg_rss_bytes):>14s} "
            f"{_format_bytes(item.median_peak_rss_bytes):>14s} "
            f"{_format_bytes(item.peak_rss_bytes):>14s} "
            f"{_format_bytes(item.peak_python_alloc_bytes):>15s} "
            f"{_format_bytes(item.peak_cuda_alloc_bytes):>12s}"
        )


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
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run_candidate(
    spec: MethodSpec,
    config: ResNet20Config,
) -> dict[str, torch.Tensor] | MethodFailure:
    try:
        model, loss_fn, x, target, vectors = spec.make_problem(config)
        return spec.fn(model, loss_fn, x, target, vectors)
    except Exception as exc:
        return MethodFailure(
            method=spec.name,
            error_type=type(exc).__name__,
            message=str(exc).splitlines()[0],
        )


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

    methods = _methods()
    reference_spec = methods["modular_hvp"]
    reference = _run_candidate(reference_spec, config)
    if isinstance(reference, MethodFailure):
        raise RuntimeError(f"modular_hvp failed: {reference.message}")

    comparisons: dict[str, dict[str, float] | MethodFailure] = {}
    for name, spec in methods.items():
        if not spec.compare_hvp:
            continue
        candidate = _run_candidate(spec, config)
        comparisons[name] = (
            candidate
            if isinstance(candidate, MethodFailure)
            else compare_results(reference, candidate)
        )

    stats = [
        benchmark_method(
            name,
            config,
            warmup=args.warmup,
            repeats=args.repeats,
            rss_sample_interval_s=args.rss_sample_interval_ms / 1000.0,
        )
        for name in methods
    ]

    if args.json:
        print(
            json.dumps(
                {
                    "config": asdict(config)
                    | {
                        "warmup": args.warmup,
                        "repeats": args.repeats,
                        "repeat_seeds": [
                            config.seed + index
                            for index in range(args.warmup, args.warmup + args.repeats)
                        ],
                    },
                    "comparisons": {
                        name: asdict(result)
                        if isinstance(result, MethodFailure)
                        else result
                        for name, result in comparisons.items()
                    },
                    "stats": [asdict(item) for item in stats],
                },
                indent=2,
            )
        )
    else:
        _print_results(comparisons, stats)


if __name__ == "__main__":
    main()
