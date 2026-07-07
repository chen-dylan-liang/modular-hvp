"""Compare ModularHVP and reverse-over-reverse for q/k/v attention grouping."""

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
from torch.nn import functional as F

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_toy_mlp import (  # noqa: E402
    MethodStats,
    _format_bytes,
    _ProcessRSSSampler,
    compare_results,
)
from modular_hvp import modular_hvp  # noqa: E402


@dataclass(frozen=True)
class QKVConfig:
    seed: int = 0
    batch_size: int = 4
    seq_len: int = 8
    vocab_size: int = 31
    d_model: int = 32
    n_heads: int = 4
    dtype: str = "float32"
    device: str = "cpu"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: Callable[
        [nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]
    compare_hvp: bool = True


class NanoChatQKVAttention(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        d_model: int,
        n_heads: int,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool)).view(
                1,
                1,
                seq_len,
                seq_len,
            ),
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch, tokens = idx.shape
        positions = torch.arange(tokens, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        h = self.ln(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        q = q.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        scores = scores.masked_fill(
            ~self.causal_mask[:, :, :tokens, :tokens],
            float("-inf"),
        )
        attention = torch.softmax(scores, dim=-1)
        y = (attention @ v).transpose(1, 2).contiguous()
        y = y.view(batch, tokens, self.n_heads * self.d_head)
        x = x + self.out_proj(y)
        logits = self.head(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )


def make_problem(
    config: QKVConfig,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    dtype = _dtype(config.dtype)
    model = NanoChatQKVAttention(
        vocab_size=config.vocab_size,
        seq_len=config.seq_len,
        d_model=config.d_model,
        n_heads=config.n_heads,
    ).to(device=config.device, dtype=dtype)
    idx = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.seq_len),
        device=config.device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.seq_len),
        device=config.device,
        dtype=torch.long,
    )
    targets[0, -1] = -1
    vectors = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    return model, idx, targets, vectors


def qkv_blocks(model: nn.Module) -> dict[str, tuple[str, ...]]:
    qkv_names = (
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
    )
    named_parameters = dict(model.named_parameters())
    blocks: dict[str, tuple[str, ...]] = {"qkv": qkv_names}
    for name in named_parameters:
        if name not in qkv_names:
            blocks[name] = (name,)
    return blocks


def modular_qkv_hvp(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with modular_hvp(model, vectors, blocks=qkv_blocks(model)):
        loss = model(idx, targets)
        loss.backward()
    return _collect_hvps(model)


def ror_qkv_hvp(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return _custom_block_ror_hvps(
        loss=model(idx, targets),
        model=model,
        vectors=vectors,
        blocks=qkv_blocks(model),
    )


def torch_backward(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    del vectors
    loss = model(idx, targets)
    loss.backward()
    return {
        name: parameter.grad.detach()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


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
    config: QKVConfig,
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
    config: QKVConfig,
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
    config: QKVConfig,
    iterations: int,
    conn: Connection,
) -> None:
    try:
        spec = _methods()[name]
        for index in range(iterations):
            model, idx, targets, vectors = make_problem(config)
            if config.device.startswith("cuda"):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            conn.send({"event": "ready", "index": index})
            start_message = conn.recv()
            if start_message.get("event") != "start":
                raise RuntimeError(f"unexpected parent message: {start_message!r}")
            tracemalloc.start()
            start = time.perf_counter()
            result = spec.fn(model, idx, targets, vectors)
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
            del result, model, idx, targets, vectors
            gc.collect()
    except BaseException:
        conn.send({"event": "error", "traceback": traceback.format_exc()})
        raise
    finally:
        conn.close()


def compare_hvps(config: QKVConfig) -> dict[str, dict[str, float]]:
    reference_model, idx, targets, vectors = make_problem(config)
    reference = ror_qkv_hvp(reference_model, idx, targets, vectors)
    comparisons: dict[str, dict[str, float]] = {}
    for name, spec in _methods().items():
        if not spec.compare_hvp:
            continue
        model, idx, targets, vectors = make_problem(config)
        comparisons[name] = compare_results(reference, spec.fn(model, idx, targets, vectors))
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=31)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=1.0)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/qkv_grouping.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = QKVConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
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
    _print_results(comparisons, stats)
    payload = {
        "config": asdict(config)
        | {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rss_sample_interval_ms": args.rss_sample_interval_ms,
        },
        "comparisons": comparisons,
        "stats": [asdict(item) for item in stats],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}", flush=True)


def _methods() -> dict[str, MethodSpec]:
    return {
        "modular_qkv": MethodSpec("modular_qkv", modular_qkv_hvp),
        "ror_qkv": MethodSpec("ror_qkv", ror_qkv_hvp),
        "torch_backward": MethodSpec("torch_backward", torch_backward, compare_hvp=False),
    }


def _print_results(
    comparisons: dict[str, dict[str, float]],
    stats: list[MethodStats],
) -> None:
    reference = next(item for item in stats if item.method == "modular_qkv")
    print("\nBenchmark")
    header = (
        f"{'method':16s} {'mean ms':>10s} {'time vs modular':>16s} "
        f"{'med peak RSS':>14s} {'RSS vs modular':>14s}"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        print(
            f"{item.method:16s} "
            f"{item.mean_time_s * 1e3:10.3f} "
            f"{item.mean_time_s / reference.mean_time_s:16.2f}x "
            f"{_format_bytes(item.median_peak_rss_bytes):>14s} "
            f"{item.median_peak_rss_bytes / reference.median_peak_rss_bytes:14.2f}x"
        )


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name!r}")


if __name__ == "__main__":
    main()
