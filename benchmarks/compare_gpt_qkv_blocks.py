"""Compare per-parameter and qkv-block HVPs on a GPT-shaped transformer."""

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
class GPTConfig:
    seed: int = 0
    batch_size: int = 2
    seq_len: int = 128
    vocab_size: int = 4096
    d_model: int = 512
    n_heads: int = 8
    layers: int = 4
    mlp_ratio: int = 4
    dtype: str = "float32"
    device: str = "cpu"


@dataclass(frozen=True)
class MethodFailure:
    method: str
    error_type: str
    message: str


@dataclass(frozen=True)
class MethodSpec:
    name: str
    fn: Callable[
        [nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
        dict[str, torch.Tensor],
    ]
    compare_hvp: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.seq_len, config.seq_len, dtype=torch.bool)).view(
                1,
                1,
                config.seq_len,
                config.seq_len,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
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
        return self.out_proj(y)


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.fc = nn.Linear(config.d_model, config.mlp_ratio * config.d_model)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(config.mlp_ratio * config.d_model, config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.proj(self.gelu(self.fc(self.ln_2(x))))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.d_model)
        self.wpe = nn.Embedding(config.seq_len, config.d_model)
        self.blocks = nn.ModuleList(GPTBlock(config) for _ in range(config.layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        _, tokens = idx.shape
        positions = torch.arange(tokens, device=idx.device)
        x = self.wte(idx) + self.wpe(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )


def make_problem(
    config: GPTConfig,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    dtype = _dtype(config.dtype)
    model = GPTLanguageModel(config).to(device=config.device, dtype=dtype).eval()
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
    blocks: dict[str, tuple[str, ...]] = {}
    for layer_idx, _ in enumerate(model.blocks):
        prefix = f"blocks.{layer_idx}.attn"
        blocks[f"qkv_{layer_idx}"] = (
            f"{prefix}.q_proj.weight",
            f"{prefix}.q_proj.bias",
            f"{prefix}.k_proj.weight",
            f"{prefix}.k_proj.bias",
            f"{prefix}.v_proj.weight",
            f"{prefix}.v_proj.bias",
        )
    return blocks


def modular_per_parameter_hvp(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    with modular_hvp(model, vectors):
        loss = model(idx, targets)
        loss.backward()
    return _collect_hvps(model)


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


def ror_per_parameter_hvp(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    blocks = {name: (name,) for name, parameter in model.named_parameters() if parameter.requires_grad}
    return _custom_block_ror_hvps(model(idx, targets), model, vectors, blocks)


def ror_qkv_hvp(
    model: nn.Module,
    idx: torch.Tensor,
    targets: torch.Tensor,
    vectors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return _custom_block_ror_hvps(model(idx, targets), model, vectors, qkv_blocks(model))


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
    config: GPTConfig,
    *,
    warmup: int,
    repeats: int,
    rss_sample_interval_s: float,
) -> MethodStats | MethodFailure:
    result = _run_method_benchmark_in_child(
        name=name,
        config=config,
        iterations=warmup + repeats,
        warmup=warmup,
        rss_sample_interval_s=rss_sample_interval_s,
    )
    if "failure" in result:
        return result["failure"]
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
    config: GPTConfig,
    iterations: int,
    warmup: int,
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
        args=(name, config, iterations, child_conn),
    )
    process.start()
    child_conn.close()
    psutil_process = psutil.Process(process.pid)

    try:
        for expected_index in range(iterations):
            ready_message = parent_conn.recv()
            if ready_message["event"] == "error":
                failure = ready_message["failure"]
                process.join()
                return {"failure": MethodFailure(**failure)}
            if ready_message["event"] != "ready":
                raise RuntimeError(f"unexpected child message: {ready_message!r}")

            with _ProcessRSSSampler(psutil_process, rss_sample_interval_s) as sampler:
                parent_conn.send({"event": "start", "index": expected_index})
                done_message = parent_conn.recv()

            if done_message["event"] == "error":
                failure = done_message["failure"]
                process.join()
                return {"failure": MethodFailure(**failure)}
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
    config: GPTConfig,
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
    except BaseException as exc:
        conn.send(
            {
                "event": "error",
                "failure": {
                    "method": name,
                    "error_type": type(exc).__name__,
                    "message": str(exc).splitlines()[0],
                },
            }
        )
    finally:
        conn.close()


def compare_hvps(config: GPTConfig) -> dict[str, dict[str, float] | MethodFailure]:
    comparisons: dict[str, dict[str, float] | MethodFailure] = {}
    comparison_groups = {
        "per_parameter": ("ror_per_parameter", "modular_per_parameter"),
        "qkv": ("ror_qkv", "modular_qkv"),
    }
    for group_name, (reference_name, candidate_name) in comparison_groups.items():
        try:
            reference_model, idx, targets, vectors = make_problem(config)
            reference = _methods()[reference_name].fn(
                reference_model,
                idx,
                targets,
                vectors,
            )
            model, idx, targets, vectors = make_problem(config)
            candidate = _methods()[candidate_name].fn(model, idx, targets, vectors)
            comparisons[group_name] = compare_results(reference, candidate)
        except BaseException as exc:
            comparisons[group_name] = MethodFailure(
                method=group_name,
                error_type=type(exc).__name__,
                message=str(exc).splitlines()[0],
            )
    return comparisons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rss-sample-interval-ms", type=float, default=1.0)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmarks/results/gpt_qkv_blocks.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GPTConfig(
        seed=args.seed,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        layers=args.layers,
        mlp_ratio=args.mlp_ratio,
        dtype=args.dtype,
        device=args.device,
    )

    comparisons = {} if args.skip_correctness else compare_hvps(config)
    for name, result in comparisons.items():
        if isinstance(result, MethodFailure):
            print(f"{name}: unsupported {result.error_type}: {result.message}", flush=True)
        else:
            print(
                f"{name}: max_abs={result['max_abs']:.3e} "
                f"max_rel={result['max_rel']:.3e}",
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
            "skip_correctness": args.skip_correctness,
        },
        "comparisons": {
            key: asdict(value) if isinstance(value, MethodFailure) else value
            for key, value in comparisons.items()
        },
        "stats": [
            asdict(item) if isinstance(item, (MethodStats, MethodFailure)) else item
            for item in stats
        ],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.json_out}", flush=True)


def _methods() -> dict[str, MethodSpec]:
    return {
        "modular_per_parameter": MethodSpec(
            "modular_per_parameter",
            modular_per_parameter_hvp,
        ),
        "ror_per_parameter": MethodSpec("ror_per_parameter", ror_per_parameter_hvp),
        "modular_qkv": MethodSpec("modular_qkv", modular_qkv_hvp),
        "ror_qkv": MethodSpec("ror_qkv", ror_qkv_hvp),
        "torch_backward": MethodSpec("torch_backward", torch_backward, compare_hvp=False),
    }


def _print_results(stats: list[MethodStats | MethodFailure]) -> None:
    stats_by_name = {
        item.method: item for item in stats if isinstance(item, MethodStats)
    }
    modular_parameter = stats_by_name.get("modular_per_parameter")
    modular_qkv = stats_by_name.get("modular_qkv")
    print("\nBenchmark")
    header = (
        f"{'method':22s} {'mean ms':>10s} {'time vs modular':>16s} "
        f"{'med peak RSS':>14s} {'cuda peak':>12s}"
    )
    print(header)
    print("-" * len(header))
    for item in stats:
        if isinstance(item, MethodFailure):
            print(f"{item.method:22s} {'unsupported':>10s} {item.error_type}: {item.message}")
            continue
        reference = modular_qkv if "qkv" in item.method else modular_parameter
        ratio = item.mean_time_s / reference.mean_time_s if reference is not None else float("nan")
        print(
            f"{item.method:22s} "
            f"{item.mean_time_s * 1e3:10.3f} "
            f"{ratio:16.2f}x "
            f"{_format_bytes(item.median_peak_rss_bytes):>14s} "
            f"{_format_bytes(item.peak_cuda_alloc_bytes):>12s}"
        )


def _dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported dtype: {name!r}")


if __name__ == "__main__":
    main()
