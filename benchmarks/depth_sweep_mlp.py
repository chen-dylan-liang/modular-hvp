"""Run and plot an MLP depth sweep for block-HVP methods."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_toy_mlp import (
    ToyMLPConfig,
    _methods,
    benchmark_method,
    compare_results,
)


HVP_METHODS = ("modular_hvp", "backpack_hmp", "backpack_autodiff")
METHOD_LABELS = {
    "modular_hvp": "ModularHVP",
    "backpack_hmp": "BackPACK HMP",
    "backpack_autodiff": "BackPACK autodiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[10, 25, 50, 75, 100],
        help="Hidden-layer depths to benchmark.",
    )
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
        default=Path("benchmarks/results/depth_sweep_width256.json"),
    )
    parser.add_argument(
        "--figure-out",
        type=Path,
        default=Path("benchmarks/results/depth_sweep_width256.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    methods = _methods()
    records: list[dict[str, Any]] = []

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
        print(f"depth={depth}: checking HVP correctness", flush=True)
        comparisons = _compare_hvp_methods(methods, config)

        for method_name in HVP_METHODS:
            print(f"depth={depth}: benchmarking {method_name}", flush=True)
            stats = benchmark_method(
                method_name,
                config,
                warmup=args.warmup,
                repeats=args.repeats,
                rss_sample_interval_s=args.rss_sample_interval_ms / 1000,
            )
            errors = comparisons[method_name]
            record = {
                "depth": depth,
                "method": method_name,
                "max_abs_error": errors["max_abs"],
                "max_rel_error": errors["max_rel"],
                **asdict(stats),
            }
            records.append(record)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.figure_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep": {
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
            "rss_sample_interval_ms": args.rss_sample_interval_ms,
        },
        "records": records,
    }
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _plot_depth_sweep(records, args.figure_out)
    print(f"wrote {args.json_out}", flush=True)
    print(f"wrote {args.figure_out}", flush=True)


def _compare_hvp_methods(
    methods: dict[str, Any],
    config: ToyMLPConfig,
) -> dict[str, dict[str, float]]:
    reference_spec = methods["modular_hvp"]
    model, loss_fn, x, target, vectors = reference_spec.make_problem(config)
    reference = reference_spec.fn(model, loss_fn, x, target, vectors)

    comparisons: dict[str, dict[str, float]] = {}
    for method_name in HVP_METHODS:
        spec = methods[method_name]
        model, loss_fn, x, target, vectors = spec.make_problem(config)
        candidate = spec.fn(model, loss_fn, x, target, vectors)
        comparisons[method_name] = compare_results(reference, candidate)
    return comparisons


def _plot_depth_sweep(records: list[dict[str, Any]], output_path: Path) -> None:
    depths = sorted({record["depth"] for record in records})
    by_method = {
        method_name: [
            next(
                record
                for record in records
                if record["method"] == method_name and record["depth"] == depth
            )
            for depth in depths
        ]
        for method_name in HVP_METHODS
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for method_name, method_records in by_method.items():
        label = METHOD_LABELS[method_name]
        axes[0].plot(
            depths,
            [record["mean_time_s"] for record in method_records],
            marker="o",
            linewidth=2,
            label=label,
        )
        axes[1].plot(
            depths,
            [
                record["median_peak_rss_bytes"] / (1024 * 1024)
                for record in method_records
            ],
            marker="o",
            linewidth=2,
            label=label,
        )

    axes[0].set_title("Runtime vs depth")
    axes[0].set_xlabel("Hidden Linear/ReLU blocks")
    axes[0].set_ylabel("Mean time (s)")
    axes[0].grid(True, alpha=0.25)

    axes[1].set_title("Process RSS vs depth")
    axes[1].set_xlabel("Hidden Linear/ReLU blocks")
    axes[1].set_ylabel("Median peak RSS (MiB)")
    axes[1].grid(True, alpha=0.25)

    for axis in axes:
        axis.legend(frameon=False)

    fig.suptitle("Width-256 MLP block-HVP depth sweep", fontsize=13)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
