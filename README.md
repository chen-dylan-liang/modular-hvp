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

- The default `modular_hvp(...)` context computes per-parameter block HVPs for
  Linear/ReLU MLPs with MSE loss using local dual activations.
- The primitive `DualTensor` backend implements the operator-overloading layer
  used by lower-level forward-mode tests.
- The original hook-plumbing runtime remains available as an internal backend
  extension point for future optimized dualized-backward integration.

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

The BackPACK comparison script checks the public `modular_hvp(...)` interface
against BackPACK HMP and BackPACK's reverse-over-reverse HVP utility. It also
reports a standard PyTorch `loss.backward()` pass as a first-order baseline:

```bash
uv run python benchmarks/compare_toy_mlp.py
```

For less noisy memory measurements, use the synthetic MNIST-shaped MLP preset:

```bash
uv run python benchmarks/compare_toy_mlp.py --preset mnist-mlp
```

The script reports max absolute/relative HVP error against `modular_hvp` plus
wall-clock time, median/max sampled RSS delta, Python allocation peak, and CUDA
allocation peak when running on CUDA. The `torch_backward` row is a timing and
memory baseline only; it computes ordinary gradients, not HVPs.

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

The current `modular_hvp` implementation is correct on these MLP benchmarks,
and follows the per-parameter locality rules: each parameter tangent is consumed
inside its owning Linear forward, the normal model receives only the primal
activation, and the saved local dual activation is consumed later to write the
single public `p.hvp`. This first local runtime is still scoped to
Linear/ReLU/MSE and uses explicit local backward primitives for that scope; the
next step is to route more of that backward dualization through the generic
primitive operator backend.

| Setting | Method | Max abs error | Max rel error | Mean time | Median RSS delta | Max RSS delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST preset | `modular_hvp` | 0.000e+00 | 0.000e+00 | 13.232 ms | 1.86 MiB | 1.89 MiB |
| MNIST preset | `backpack_hmp` | 3.725e-09 | 3.351e-07 | 24.834 ms | 4.94 MiB | 4.98 MiB |
| MNIST preset | `backpack_autodiff` | 3.725e-09 | 3.351e-07 | 16.231 ms | 1.78 MiB | 1.81 MiB |
| MNIST preset | `torch_backward` | n/a | n/a | 1.845 ms | 16.00 KiB | 24.00 KiB |
| Larger stress | `modular_hvp` | 0.000e+00 | 0.000e+00 | 50.622 ms | 7.11 MiB | 7.11 MiB |
| Larger stress | `backpack_hmp` | 3.725e-09 | 4.425e-07 | 77.961 ms | 27.24 MiB | 27.25 MiB |
| Larger stress | `backpack_autodiff` | 3.725e-09 | 3.035e-07 | 84.831 ms | 11.57 MiB | 11.68 MiB |
| Larger stress | `torch_backward` | n/a | n/a | 9.083 ms | 16.00 KiB | 16.00 KiB |
