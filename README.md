# ModularHVP

ModularHVP is an eager PyTorch-compatible runtime for computing block-scoped
Hessian-vector products during ordinary backward execution.

The target public API is:

```python
from modular_hvp import modular_hvp

with modular_hvp(model, v):
    out = model(x)
    loss = criterion(out, y)
    loss.backward()

for name, p in model.named_parameters():
    p.grad
    p.hvp
```

Current implementation status:

- The `modular_hvp(...)` context manager provides the per-parameter hook
  runtime, state management, and `p.grad`/`p.hvp` lifecycle.
- The primitive `DualTensor` backend implements the operator-overloading layer
  needed by MLP-style tensor programs.
- The current numerical HVP benchmark uses the `DualTensor` backend directly.
  Wiring that backend into the default `modular_hvp(...)` context is the next
  integration step.

The primitive dual-tensor backend is available independently for forward-mode
operator tests:

```python
from modular_hvp import make_dual, primal, tangent

x_hat = make_dual(x, x_dot)
y_hat = torch.relu(x_hat @ weight)

primal(y_hat)
tangent(y_hat)
```

This layer overloads selected ATen primitives and raises `NotImplementedError`
when a `DualTensor` reaches an unsupported operation.

## Toy MLP Comparison

The BackPACK comparison script checks per-parameter block HVPs on MLPs against
BackPACK HMP and BackPACK's reverse-over-reverse HVP utility. It also reports a
standard PyTorch `loss.backward()` pass as a first-order baseline:

```bash
uv run python benchmarks/compare_toy_mlp.py
```

For less noisy memory measurements, use the synthetic MNIST-shaped MLP preset:

```bash
uv run python benchmarks/compare_toy_mlp.py --preset mnist-mlp
```

The script reports max absolute/relative HVP error against the ModularHVP
DualTensor path plus wall-clock time, median/max sampled RSS delta, Python
allocation peak, and CUDA allocation peak when running on CUDA. The
`torch_backward` row is a timing and memory baseline only; it computes ordinary
gradients, not HVPs.

`RSS delta` is the sampled increase in the process's resident set size during a
method run. It is a coarse process-level measurement, so tiny toy runs can be
noisy; the MNIST-shaped and larger stress settings make the memory differences
more visible.

Recorded CPU runs:

| Setting | Command | Shape |
| --- | --- | --- |
| Production-shaped MNIST MLP preset | `uv run python benchmarks/compare_toy_mlp.py --preset mnist-mlp --warmup 2 --repeats 8` | batch 256, input 784, hidden width 256, 3 hidden Linear/ReLU blocks, output 10, float32 |
| Larger stress run | `uv run python benchmarks/compare_toy_mlp.py --batch-size 512 --d-in 784 --d-hidden 512 --hidden-layers 4 --d-out 10 --dtype float32 --warmup 1 --repeats 3` | batch 512, input 784, hidden width 512, 4 hidden Linear/ReLU blocks, output 10, float32 |

Latest local results:

For `backpack_hmp`, BackPACK's `extend(...)` setup is performed before the
timed region. The measured region contains the forward pass, BackPACK HMP
backward pass, and one `param.hmp(...)` application per parameter.

The current `modular_dual` benchmark path computes per-parameter block HVPs by
running the relevant suffix computation for each active parameter block. It is
therefore a backend milestone benchmark, not yet the expected cost profile of
the final integrated `modular_hvp(...): loss.backward()` runtime.

| Setting | Method | Max abs error | Max rel error | Mean time | Median RSS delta | Max RSS delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST preset | `modular_dual` | 0.000e+00 | 0.000e+00 | 14.358 ms | 44.00 KiB | 116.00 KiB |
| MNIST preset | `backpack_hmp` | 3.725e-09 | 3.351e-07 | 25.653 ms | 5.41 MiB | 5.57 MiB |
| MNIST preset | `backpack_autodiff` | 3.725e-09 | 3.351e-07 | 13.411 ms | 1.28 MiB | 1.43 MiB |
| MNIST preset | `torch_backward` | n/a | n/a | 1.773 ms | 16.00 KiB | 20.00 KiB |
| Larger stress | `modular_dual` | 0.000e+00 | 0.000e+00 | 54.253 ms | 128.00 KiB | 204.00 KiB |
| Larger stress | `backpack_hmp` | 3.725e-09 | 4.425e-07 | 77.157 ms | 28.19 MiB | 28.26 MiB |
| Larger stress | `backpack_autodiff` | 3.725e-09 | 3.035e-07 | 78.760 ms | 11.59 MiB | 11.64 MiB |
| Larger stress | `torch_backward` | n/a | n/a | 8.309 ms | 16.00 KiB | 20.00 KiB |
