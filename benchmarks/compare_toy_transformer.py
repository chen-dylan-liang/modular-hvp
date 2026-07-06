"""Compare public ``modular_hvp(...)`` block HVPs on a toy Transformer."""

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
from dataclasses import asdict, dataclass
from multiprocessing.connection import Connection
from pathlib import Path

import psutil
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_toy_mlp import (
    backpack_autodiff_block_hvp,
    compare_results,
    modular_hvp_block_hvp,
    torch_backward_pass,
)


@dataclass(frozen=True)
class TransformerConfig:
    seed: int = 0
    batch_size: int = 4
    seq_len: int = 8
    d_model: int = 32
    n_heads: int = 4
    layers: int = 2
    mlp_ratio: int = 4
    attention: str = "unfused"
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
    [TransformerConfig],
    tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: BlockHVPFn
    make_problem: ProblemFactory
    compare_hvp: bool = True


class ToyTransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        mlp_ratio: int,
        need_weights: bool,
    ) -> None:
        super().__init__()
        self.need_weights = need_weights
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, mlp_ratio * d_model)
        self.gelu = nn.GELU()
        self.ff2 = nn.Linear(mlp_ratio * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=self.need_weights)
        x = x + attn_out
        return x + self.ff2(self.gelu(self.ff1(self.ln2(x))))


class ToyTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        need_weights = config.attention == "unfused"
        self.blocks = nn.Sequential(
            *[
                ToyTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    mlp_ratio=config.mlp_ratio,
                    need_weights=need_weights,
                )
                for _ in range(config.layers)
            ]
        )
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(self.blocks(x))


def make_toy_transformer(config: TransformerConfig) -> nn.Module:
    if config.d_model % config.n_heads != 0:
        raise ValueError("d_model must be divisible by n_heads")
    if config.attention not in {"unfused", "fused"}:
        raise ValueError("attention must be 'unfused' or 'fused'")
    return ToyTransformer(config).to(device=config.device, dtype=_dtype(config.dtype)).eval()


def make_problem(
    config: TransformerConfig,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    dtype = _dtype(config.dtype)
    model = make_toy_transformer(config)
    loss_fn = nn.MSELoss(reduction="mean")
    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device=config.device,
        dtype=dtype,
    )
    target = torch.randn_like(x)
    vectors = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return model, loss_fn, x, target, vectors


def _make_backpack_hmp_problem(
    config: TransformerConfig,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    from backpack import extend

    model, loss_fn, x, target, vectors = make_problem(config)
    return (
        extend(model),
        extend(type(loss_fn)(reduction=loss_fn.reduction)),
        x,
        target,
        vectors,
    )


def _make_backpack_autodiff_problem(
    config: TransformerConfig,
) -> tuple[nn.Module, nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    from backpack.hessianfree import hvp as _hvp  # noqa: F401

    return make_problem(config)


def backpack_hmp_transformer_block_hvp(
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
    config: TransformerConfig,
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
    config: TransformerConfig,
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> dict[str, object]:
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
                failure = ready_message["failure"]
                process.join()
                return {"failure": failure}
            if ready_message["event"] != "ready":
                raise RuntimeError(f"unexpected child message: {ready_message!r}")

            with _ProcessRSSSampler(psutil_process, rss_sample_interval_s) as sampler:
                parent_conn.send({"event": "start", "index": expected_index})
                done_message = parent_conn.recv()

            if done_message["event"] == "error":
                failure = done_message["failure"]
                process.join()
                return {"failure": failure}
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

    if process.exitcode not in {0, None}:
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
    config: TransformerConfig,
    iterations: int,
    conn: Connection,
) -> None:
    try:
        spec = _methods()[name]
        for index in range(iterations):
            model, loss_fn, x, target, vectors = spec.make_problem(config)
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
    except Exception as exc:
        conn.send(
            {
                "event": "error",
                "failure": {
                    "method": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc).splitlines()[0],
                    "traceback": traceback.format_exc(),
                },
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
            backpack_hmp_transformer_block_hvp,
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--attention", choices=("unfused", "fused"), default="unfused")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _run_candidate(
    spec: MethodSpec,
    config: TransformerConfig,
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
    config = TransformerConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
        mlp_ratio=args.mlp_ratio,
        attention=args.attention,
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
                    "config": asdict(config),
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
