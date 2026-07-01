# ModularHVP

ModularHVP is an eager PyTorch-compatible runtime for computing block-scoped
Hessian-vector products during ordinary backward execution.

The initial implementation provides the hook-plumbing milestone for
per-parameter blocks:

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

The current `modular_hvp(...)` context backend is intentionally fake: it
preserves the runtime shape and writes zero HVP tensors. Integration between
the hook runtime and primitive dual tensor rules is a later milestone.

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

The BackPACK comparison script checks per-parameter block HVPs on a small MLP
against BackPACK HMP and BackPACK's reverse-over-reverse HVP utility:

```bash
uv run python benchmarks/compare_toy_mlp.py
```

For less noisy memory measurements, use the synthetic MNIST-shaped MLP preset:

```bash
uv run python benchmarks/compare_toy_mlp.py --preset mnist-mlp
```

The script reports max absolute/relative error against the ModularHVP
DualTensor path plus wall-clock time, median/max sampled RSS delta, Python
allocation peak, and CUDA allocation peak when running on CUDA.

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

| Setting | Method | Max abs error | Max rel error | Mean time | Median RSS delta | Max RSS delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| MNIST preset | `modular_dual` | 0.000e+00 | 0.000e+00 | 14.075 ms | 40.00 KiB | 92.00 KiB |
| MNIST preset | `backpack_hmp` | 3.725e-09 | 3.351e-07 | 29.447 ms | 5.46 MiB | 5.56 MiB |
| MNIST preset | `backpack_autodiff` | 3.725e-09 | 3.351e-07 | 14.306 ms | 1.21 MiB | 1.32 MiB |
| Larger stress | `modular_dual` | 0.000e+00 | 0.000e+00 | 52.223 ms | 120.00 KiB | 264.00 KiB |
| Larger stress | `backpack_hmp` | 3.725e-09 | 4.425e-07 | 80.536 ms | 28.17 MiB | 28.19 MiB |
| Larger stress | `backpack_autodiff` | 3.725e-09 | 3.035e-07 | 80.883 ms | 11.59 MiB | 11.61 MiB |
