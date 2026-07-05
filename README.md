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
  `DualTensor.tangent` is exactly one tensor, detached at construction. Every
  primitive tangent rule runs as a no-grad side channel using detached primal
  values.
- Python-level wrappers such as `linear`, `relu`, and `matmul` are not
  registered as separate backend rules. They are allowed to lower into ATen,
  where the primitive `DualTensor` rules run.
- The runtime keeps the user-visible forward path ordinary. For each active
  parameter block, the owning module consumes that parameter's tangent locally,
  saves the resulting local dual activation, and returns only the primal output
  to downstream modules.
- Local epsilons are not represented as keyed `DualTensor` channels and are not
  exported across forward block boundaries.
- During backward, tensor hooks at saved activation boundaries consume the
  saved local dual activations as PyTorch's ordinary backward reaches the
  matching block. HVPs are written into the single public `p.hvp` field during
  that same backward pass.
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
wall-clock time, sampled per-process RSS, Python allocation peak, and CUDA
allocation peak when running on CUDA. The `torch_backward` row is a timing and
memory baseline only; it computes ordinary gradients, not HVPs.

For fair absolute RSS measurement, each method is benchmarked in its own spawned
process. Method-specific setup and imports happen before warmup/measurement, so
RSS includes the method's process footprint while timing focuses on the measured
forward/backward/HVP computation. Returned HVP or gradient tensors are kept
alive until the RSS sampler exits, so methods are not credited for immediately
dropping their outputs.

RSS is a coarse process-level measurement, so tiny toy runs can be noisy; deeper
stress settings make the memory differences more visible. The table reports
median average RSS, median peak RSS, and max peak RSS across measured repeats.

For `backpack_hmp`, BackPACK's `extend(...)` setup is performed before the
timed region. The measured region contains the forward pass, BackPACK HMP
backward pass, and one `param.hmp(...)` application per parameter.

The current `modular_hvp` implementation follows the block-scoped forward
locality invariant: parameter tangents are consumed inside the owning module,
the local dual activation is saved, and only the primal output is passed to the
rest of the model. Reused parameters accumulate into the same `p.hvp`; the
current shared-parameter path uses a correctness fallback.

The local runtime does not compute a full HVP and slice it. It also no longer
uses keyed `DualTensor` payloads to carry multiple epsilons through the forward
program. The primitive backend remains the only place where tensor operations
are overloaded.

The current backward side is still the narrow MLP/MSE milestone, not the final
general backward-hook runtime. The loss hook initializes the model-output
curvature action for MSE, and activation hooks apply that action to each saved
local dual activation as ordinary PyTorch backward reaches the matching block.
The narrow MLP path still has Linear/ReLU/MSE-specific hook plumbing; the
primitive `DualTensor` backend itself remains single-tangent and ATen-scoped.

- Linear backward-side pieces use `aten.mm`, `aten.t`, and `aten.sum`.
- ReLU dispatches as `aten.relu.default`; its backward-side program uses
  `aten.threshold_backward.default`.
- MSE loss dispatches as ordinary PyTorch `mse_loss`; the local runtime only
  uses the scalar Hessian scale at the model-output boundary for reductions
  `mean` and `sum`.

For Linear and ReLU activation values needed by the local backward-side tensor
programs, the MLP runtime resolves tensors already saved by PyTorch autograd and
releases the autograd-node reference inside the hook. It falls back to an
explicit detached activation only when the expected saved tensor is unavailable.
Saved local output tangents are cleared immediately after their owning backward
hook consumes them.

The backend has explicit graph-isolation tests: primal outputs keep
`requires_grad` and `grad_fn`, while tangent outputs have
`requires_grad == False` and `grad_fn is None`, even when the input tangent was
created with `requires_grad=True`.

Current 4-layer toy MLP result:

```bash
uv run python benchmarks/compare_toy_mlp.py \
  --batch-size 512 --d-in 784 --d-hidden 512 --hidden-layers 4 --d-out 10 \
  --dtype float32 --warmup 2 --repeats 8
```

Ratio columns compare each method against `modular_hvp`; values above `1.0x`
mean the method is slower or uses more peak RSS than `modular_hvp`.

| Method | Max abs error | Max rel error | Mean time | Time vs `modular_hvp` | Median peak RSS | Peak RSS vs `modular_hvp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modular_hvp` | 0.000e+00 | 0.000e+00 | 64.577 ms | 1.00x | 200.91 MiB | 1.00x |
| `backpack_hmp` | 3.725e-09 | 4.425e-07 | 99.989 ms | 1.55x | 366.73 MiB | 1.83x |
| `backpack_autodiff` | 3.725e-09 | 3.035e-07 | 110.823 ms | 1.72x | 256.54 MiB | 1.28x |
| `torch_backward` | n/a | n/a | 13.342 ms | 0.21x | 182.04 MiB | 0.91x |

## Depth Sweep

The depth sweep fixes width at 256 and compares the three HVP methods across
10, 25, 50, 75, and 100 hidden Linear/ReLU blocks:

```bash
uv run python benchmarks/depth_sweep_mlp.py \
  --depths 10 25 50 75 100 \
  --batch-size 256 --d-in 784 --d-hidden 256 --d-out 10 \
  --dtype float32 --warmup 1 --repeats 3
```

The script writes raw data and a trend figure to `benchmarks/results/`. No
checked-in depth sweep result is kept right now because the runtime semantics
changed from keyed forward channels to strict local forward activation records.

## Primitive Coverage Roadmap

Architecture support should be added by extending ATen-level `DualTensor`
rules, not by special-casing architecture motifs. For example, ResNet residual
connections are just ordinary tensor addition (`x + f(x)`), so `aten.add` is
the primitive rule; there should be no residual-specific rule.

Expected next primitive families:

- ResNet/CNN forward: `aten.convolution`, `aten.native_batch_norm`,
  pooling ops such as max/adaptive average pool, plus existing `aten.add`,
  `aten.relu`, shape/view ops, and linear ops.
- ResNet/CNN backward-like programs: convolution backward primitives,
  batch-norm backward primitives, pooling backward primitives, and existing
  activation/linear backward primitives.
- Transformer unfused path: existing linear/matmul/bmm/view/transpose/add/mul/div
  rules, plus `aten._softmax`, softmax backward, `aten.native_layer_norm`,
  layer-norm backward, GELU backward, dropout primitives, and indexing/select
  backward where they appear.
- Transformer fused path: if PyTorch dispatch reaches fused ATen ops such as
  scaled-dot-product attention or transformer fastpath kernels, treat those as
  atomic primitives only when they actually appear at dispatch. Do not replace
  them with module-level attention or transformer rules.

Each added primitive should get a focused finite-difference test and then a
small composition test for the target architecture.
