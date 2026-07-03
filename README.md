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
  used by lower-level forward-mode tests and by the current local backward
  tensor programs.
- `DualTensor.primal` preserves ordinary PyTorch autograd graph construction.
  `DualTensor.tangent` is detached at construction and every primitive tangent
  rule runs as a no-grad side channel using detached primal values.
- The runtime keeps the user-visible forward path ordinary. Internally, it uses
  a keyed side-channel of `DualTensor` tangents: each parameter block has its
  own key, keys are never summed into a global tangent, and only primal tensors
  are returned to user code.
- During backward, tensor hooks at saved activation boundaries consume the
  keyed side-channel as PyTorch's ordinary backward reaches that block. HVPs are
  written into the single public `p.hvp` field during that same backward pass.
- The generic hook-plumbing runtime remains available as an internal extension
  point for future optimized dualized-backward integration.

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

The script reports max absolute/relative HVP error against `modular_hvp` plus
wall-clock time, median/max sampled RSS delta, Python allocation peak, and CUDA
allocation peak when running on CUDA. The `torch_backward` row is a timing and
memory baseline only; it computes ordinary gradients, not HVPs. Returned HVP or
gradient tensors are kept alive until the RSS sampler exits, so methods are not
credited for immediately dropping their outputs.

`RSS delta` is the sampled increase in the process's resident set size during a
method run. It is a coarse process-level measurement, so tiny toy runs can be
noisy; deeper stress settings make the memory differences more visible.

Recorded CPU runs:

| Setting | Command | Shape |
| --- | --- | --- |
| Toy 4-layer MLP | `uv run python benchmarks/compare_toy_mlp.py --batch-size 512 --d-in 784 --d-hidden 512 --hidden-layers 4 --d-out 10 --dtype float32 --warmup 3 --repeats 20` | batch 512, input 784, hidden width 512, 4 hidden Linear/ReLU blocks, output 10, float32 |
| Deep stress 50-layer MLP | `uv run python benchmarks/compare_toy_mlp.py --batch-size 256 --d-in 784 --d-hidden 256 --hidden-layers 50 --d-out 10 --dtype float32 --warmup 1 --repeats 3` | batch 256, input 784, hidden width 256, 50 hidden Linear/ReLU blocks, output 10, float32 |

Latest local results:

For `backpack_hmp`, BackPACK's `extend(...)` setup is performed before the
timed region. The measured region contains the forward pass, BackPACK HMP
backward pass, and one `param.hmp(...)` application per parameter.

The current `modular_hvp` implementation is correct on these MLP benchmarks,
and follows the block-scoped invariant: each parameter tangent is keyed to its
own parameter block, never summed with neighboring blocks, and contributes only
to that parameter's public `p.hvp`. Reused parameters accumulate into that same
`p.hvp`; the current shared-parameter path uses a correctness fallback.

The local runtime does not compute a full HVP and slice it. It uses a
block-keyed side-channel so different parameter epsilons can pass through the
same primitive tensor program without being combined into one model-wide
tangent.

The forward side does not reimplement module math and does not replay once per
parameter. For an active owning module, the runtime calls the module's original
`forward` once with local `DualTensor` parameters. PyTorch decomposes that code
to ATen as usual; the `DualTensor` backend only changes behavior when an ATen
primitive receives a dual input. The resulting tangent payload is keyed by
parameter, so `dy_weight` and `dy_bias` remain separate local epsilons instead
of being summed into a global tangent.

The current backward side is still the narrow MLP/MSE milestone, not the final
general backward-hook runtime. It no longer runs repeated per-parameter suffix
applications from the loss hook. Instead, the loss hook initializes the
model-output curvature side-channel, and activation hooks consume and propagate
that side-channel as ordinary PyTorch backward reaches each block. The tensor
programs are expressed through ATen primitives and run through the `DualTensor`
registry:

- Linear backward-side pieces use `aten.mm`, `aten.t`, and `aten.sum`.
- ReLU dispatches as `aten.relu.default`; its backward-side program uses
  `aten.threshold_backward.default`.
- MSE loss dispatches as ordinary PyTorch `mse_loss`; the local runtime only
  uses the scalar Hessian scale at the model-output boundary for reductions
  `mean` and `sum`.

The backend has explicit graph-isolation tests: primal outputs keep
`requires_grad` and `grad_fn`, while tangent outputs have
`requires_grad == False` and `grad_fn is None`, even when the input tangent was
created with `requires_grad=True`.

| Setting | Method | Max abs error | Max rel error | Mean time | Median RSS delta | Max RSS delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Toy 4-layer | `modular_hvp` | 0.000e+00 | 0.000e+00 | 101.255 ms | 10.32 MiB | 10.48 MiB |
| Toy 4-layer | `backpack_hmp` | 3.725e-09 | 4.425e-07 | 113.975 ms | 27.19 MiB | 27.29 MiB |
| Toy 4-layer | `backpack_autodiff` | 3.725e-09 | 3.035e-07 | 131.983 ms | 11.55 MiB | 11.64 MiB |
| Toy 4-layer | `torch_backward` | n/a | n/a | 13.737 ms | 16.00 KiB | 20.00 KiB |
| Deep stress 50-layer | `modular_hvp` | 0.000e+00 | 0.000e+00 | 4519.322 ms | 72.35 MiB | 72.58 MiB |
| Deep stress 50-layer | `backpack_hmp` | 1.490e-08 | 8.969e-07 | 6600.170 ms | 110.85 MiB | 111.02 MiB |
| Deep stress 50-layer | `backpack_autodiff` | 1.490e-08 | 8.969e-07 | 5802.694 ms | 73.83 MiB | 73.83 MiB |
| Deep stress 50-layer | `torch_backward` | n/a | n/a | 36.219 ms | 9.78 MiB | 9.78 MiB |
