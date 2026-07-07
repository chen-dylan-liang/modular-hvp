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
  supported eager MLP/CNN/ResNet/Transformer graphs with MSE loss using local
  dual activations. Supported leaf modules currently include `Linear`,
  `Conv2d`, eval-mode `BatchNorm2d`, `LayerNorm`, `ReLU`, `GELU`, eval-mode
  `Dropout`, `Flatten`, `AvgPool2d`, `AdaptiveAvgPool2d`, `MaxPool2d`, and
  limited `MultiheadAttention`.
- The eager runtime also accepts an optional `blocks=` partition for the first
  module-wise milestone. Grouped parameters may belong to one supported leaf
  module in a sequential model, for example a `Linear` module's `weight` and
  `bias`. Contiguous multi-`Linear` blocks in sequential MLPs are also
  supported, including the full-MLP block. Each grouped block shares one local
  epsilon, so within-block Hessian cross terms are retained while cross-block
  terms are still ignored.
- The primitive `DualTensor` backend implements the operator-overloading layer
  used by lower-level forward-mode tests and by the current local backward
  tensor programs.
- The default runtime supports residual/DAG graphs composed from the supported
  leaf modules plus primitive tensor addition, split/slice, view/reshape,
  transpose/contiguous, matmul/division, softmax, and scaled-dot-product
  attention edges. Residual paths are handled as ordinary graph `add` edges;
  there is no residual-specific public API or residual-specific HVP rule.
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

## Custom Blocks

By default, every trainable parameter is its own block:

```python
with modular_hvp(model, v):
    loss = criterion(model(x), y)
    loss.backward()
```

For the first module-wise milestone, a sequential model can pass an explicit
partition:

```python
blocks = {
    model[0]: ("0.weight", "0.bias"),
    model[2]: ("2.weight", "2.bias"),
}

with modular_hvp(model, v, blocks=blocks):
    loss = criterion(model(x), y)
    loss.backward()
```

Every trainable parameter must appear in exactly one block. Parameters in the
same block share one local epsilon, so for block `B` the public values are:

```text
p.hvp = sum_{q in B} H[p, q] v_q
```

The current implementation intentionally rejects multi-leaf custom blocks
outside contiguous sequential Linear MLP spans. DAG-aware custom block
validation is the next graph-scoping step; the primitive `DualTensor` backend
does not need to change for that work.

## Runtime Structure

The default public runtime is now split by responsibility:

- `modular_hvp.dual`: primitive `DualTensor` operator-overloading backend.
- `modular_hvp.eager`: thin public eager-runtime coordinator.
- `modular_hvp.runtime_forward`: forward wrapping and local-dual recording.
- `modular_hvp.runtime_backward`: backward scheduling and HVP accumulation.
- `modular_hvp.runtime_dispatch`: `GraphTensor` dispatch and forward-record
  construction.
- `modular_hvp.graph_tensor`: primal graph-edge wrapper used by the eager
  recorder.
- `modular_hvp.runtime_state`: eager runtime state records.
- `modular_hvp.records`: forward-record dataclasses and saved-tensor
  references.
- `modular_hvp.graph`: recorded graph topology, use-count analysis, retained
  tangent-node analysis, and reverse traversal readiness state.

Runtime/coordinator files are kept below roughly 2k lines for reviewability.
`modular_hvp.dual` is the exception: it is the primitive ATen overload backend
and may grow with operator coverage.

`modular_hvp.local_mlp` remains only as a compatibility shim for older internal
imports. The default API imports the architecture-agnostic eager runtime.

## Graph Algorithm View

The public runtime treats one `modular_hvp(...): loss.backward()` execution as
a graph dataflow problem.

During the primal forward pass, the runtime records graph edges. At backward
startup, those records become a `RecordedForwardGraph`: an immutable topology
snapshot with forward order, reverse order, input-use counts, retained-tangent
node ids, and the model-output node id. Each supported module or primitive
tensor operation contributes one record with input node ids, an output node id,
local parameter ownership if any, and the minimal primal values needed by that
operation's local backward-side tensor program. The model still receives only
primal tensors at block boundaries; the local dual activation for a parameter
block is saved internally and is not exported as a global epsilon channel.

At backward startup, the DAG path initializes `GraphTraversalState`:

- `forward_tangents_by_node` stores only the retained tangent packets needed by
  nonlinear or bilinear backward rules.
- `grad_tangents_by_node` stores reverse-flowing tangent packets for the
  ordinary PyTorch gradient signal.
- `primal_grads_by_node` is filled by tensor hooks when PyTorch reaches a
  recorded output node.
- `pending_consumers_by_node` is the reverse-topological readiness counter.
- `local_parameters_by_output_node` records which parameter blocks own each
  local dual activation after the activation tensors themselves have been
  released.

The reverse traversal drains a record once all downstream consumers of that
record's output node have been processed and PyTorch has exposed the ordinary
primal gradient for that node. The record then consumes its local tangent packet,
accumulates into the single public `p.hvp`, propagates any remaining tangent
packet to its input nodes, and releases retained state for that output node.
The hooked DAG path uses a ready stack keyed by output node id, so it no longer
rescans the full reverse record list every time a tensor hook fires.

Sequential models are optimized as a chain special case, but the generic
contract is graph-based. New architectures should usually require only new
ATen/primitive records and tangent rules; residual joins, attention branches,
and future DAG motifs should be handled by the same graph dataflow traversal
rather than by public branch-specific quantities.

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
rest of the model. Reused and tied parameters use the graph path: each local
use-site contributes to the same public `p.hvp`, without model replay or a
reverse-over-reverse fallback.

The eager runtime does not compute a full HVP and slice it. It also no longer
uses keyed `DualTensor` payloads to carry multiple epsilons through the forward
program. The primitive backend remains the only place where tensor operations
are overloaded.

The current backward side has two internal paths. Single-chain graphs use the
fast composed-curvature path. DAG graphs use graph-indexed tangent packets: the
forward pass records supported leaf-module and primitive tensor edges, PyTorch
still performs one ordinary backward pass, and tensor hooks drain records in
reverse graph order once downstream tangent packets are ready. This preserves
cross terms at residual joins and attention branches without introducing
branch-specific public quantities. The primitive `DualTensor` backend itself
remains single-tangent and ATen-scoped.

- Linear backward-side pieces use `aten.mm`, `aten.t`, and `aten.sum`.
- Conv2d backward-side pieces use `aten.convolution` and
  `aten.convolution_backward`.
- Eval BatchNorm2d backward-side pieces use `aten.native_batch_norm_backward`.
- LayerNorm backward-side pieces use `aten.native_layer_norm_backward`.
- Pooling backward-side pieces use ATen pool backward primitives.
- ReLU dispatches as `aten.relu.default`; its backward-side program uses
  `aten.threshold_backward.default`.
- Attention backward-side pieces use the primitive matmul, softmax backward
  JVP, and scaled-dot-product attention JVP/VJP-linearization formulas.
- MSE loss dispatches as ordinary PyTorch `mse_loss`; the eager runtime only
  uses the scalar Hessian scale at the model-output boundary for reductions
  `mean` and `sum`.
- Token-model loss can also be seeded from in-forward
  `torch.nn.functional.cross_entropy` with unweighted `mean`/`sum` reduction.
  The output curvature uses the logits softmax Hessian-vector formula.
- `nn.Embedding` is supported as a local parameter source with dense embedding
  gradients. Token indices remain ordinary integer tensors; no tangent is
  propagated into them.
- Directly used floating parameters in composite module forwards are wrapped as
  graph source nodes for the duration of that one forward call, then restored.
  This covers learned scalar parameters such as residual/gating lambdas without
  changing the public API or exporting a global tangent.

For Linear activation values needed by the local backward-side tensor programs,
the runtime resolves tensors already saved by PyTorch autograd and releases the
autograd-node reference inside the hook. For Conv2d/BatchNorm2d local
parameter-gradient formulas and ReLU masks in residual/DAG graphs, the runtime
keeps a detached direct reference to avoid ambiguous same-shaped saved tensors
and freed autograd internals. Saved local output tangents are cleared
immediately after their owning backward hook consumes them.

The backend has explicit graph-isolation tests: primal outputs keep
`requires_grad` and `grad_fn`, while tangent outputs have
`requires_grad == False` and `grad_fn is None`, even when the input tangent was
created with `requires_grad=True`.

## Nanochat-Shaped GPT Compatibility

Milestone 4 aims to use the same public interface inside nanochat-style GPT
training:

```python
with modular_hvp(model, v):
    loss = model(idx, targets)
    loss.backward()
```

The compatibility work is being kept architecture-agnostic. The runtime changes
are generic graph/source handling; the backend additions are primitive tensor
coverage. We do not add GPT-block, residual-path, or attention-block Hessian
rules.

Local tests do not execute the production nanochat training stack. The local
environment currently has torch 2.2.2, while nanochat pins a newer torch release
and uses `torch.nn.functional.rms_norm`. Instead, the regression suite includes
a nanochat-shaped token model that exercises:

- integer token IDs and `nn.Embedding`;
- learned scalar parameters used directly in arithmetic;
- indexing/slicing such as `x[:, 1:]` and `x[..., :d]`;
- `torch.cat`, residual-style addition, multiplication/division, dtype casts;
- `torch.sigmoid`, `torch.relu(...).square()`, and tanh logit soft-capping;
- in-forward `F.cross_entropy(..., ignore_index=-1)`.

The suite also includes a nanoGPT-shaped model copied at the structural level
from nanoGPT's `model.py`: custom `LayerNorm` implemented with
`F.layer_norm`, tied token embedding / LM-head weight, learned position
embedding broadcast over the batch dimension, `ModuleDict`/`ModuleList`
composition, causal self-attention, `Tensor.split`, residual adds, GELU MLPs,
tuple model outputs `(logits, loss)`, and in-forward cross entropy. Two local
paths are tested:

- SDPA/flash-style causal attention with `dropout=0`, checked against an
  equivalent slow-attention reverse-over-reverse reference because this local
  PyTorch build does not provide the second derivative of the fused flash
  attention backward.
- Slow causal attention with `masked_fill`, softmax, and training-mode dropout,
  checked directly against reverse-over-reverse on the same realized dropout
  masks.

Both nanoGPT-shaped ModularHVP runs monkeypatch `torch.autograd.grad` to fail
inside `modular_hvp`, guarding the one-forward/one-backward graph runtime
against fallback. The changes needed for this were generic primitive/runtime
coverage: functional/native LayerNorm records, dropout records using PyTorch's
saved multiplier, `masked_fill`, `Tensor.split`, broadcast-aware add records,
and top-level tuple-output handling.

RMSNorm graph support is feature-gated behind the ATen operator exposed by the
installed torch build. It is not locally executable under torch 2.2.2, so it
must be validated in the nanochat production environment.

Fused SDPA with nonzero dropout remains unsupported in the public runtime
because the fused primitive does not expose the realized dropout mask needed to
mirror the exact primal pass. Use `dropout=0` on the fused path or a decomposed
slow attention path when dropout must be active.

Performance audit note: after adding the token/GPT-shaped graph coverage, the
runtime was checked for accidental repetitive computation. The implementation
still performs one forward and one backward. No per-parameter replay was found.
Two low-risk caches were added: composite-module raw parameter sources are
collected once per context, and the direct ATen graph-dispatch set is built once
at import time.

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
| `modular_hvp` | 0.000e+00 | 0.000e+00 | 49.852 ms | 1.00x | 202.26 MiB | 1.00x |
| `backpack_hmp` | 3.725e-09 | 4.425e-07 | 77.740 ms | 1.56x | 368.62 MiB | 1.82x |
| `backpack_autodiff` | 3.725e-09 | 3.035e-07 | 82.542 ms | 1.66x | 257.10 MiB | 1.27x |
| `torch_backward` | n/a | n/a | 9.381 ms | 0.19x | 183.14 MiB | 0.91x |

### Linear-Module Block Comparison

The module-wise MLP benchmark groups each `Linear` module's `weight` and `bias`
into one block, while different `Linear` modules remain separate blocks:

```bash
uv run python benchmarks/compare_mlp_linear_blocks.py \
  --depths 4 50 \
  --batch-size 256 --d-in 784 --d-hidden 256 --d-out 10 \
  --dtype float32 --warmup 1 --repeats 3
```

BackPACK HMP is not included in this table because the tested HMP interface
returns per-parameter Hessian-matrix products and does not directly expose the
within-module `weight`/`bias` cross terms. BackPACK's reverse-over-reverse
autodiff utility can do this grouping by passing each module's grouped
parameters to one `hessian_vector_product(...)` call, so it is the grouped
baseline below.

For both depths, `modular_linear_blocks` matches grouped reverse-over-reverse,
while `modular_per_parameter` differs from the grouped reference because it
intentionally drops the within-Linear `weight`/`bias` cross terms.
The sequential module-block path applies the downstream curvature action once
per local block and reuses it for the grouped parameter-gradient formulas, so a
`weight`/`bias` block does not evaluate the same activation-side curvature
twice. On the sequential path, grouped parameters also share the same combined
local tangent reference instead of cloning one identical activation tangent per
parameter; graph packets still keep independent tensors where values may be
mutated independently during traversal.

| Hidden Linear/ReLU blocks | Method | Max abs error vs grouped RoR | Max rel error vs grouped RoR | Mean time | Time vs `modular_linear_blocks` | Median peak RSS | Peak RSS vs `modular_linear_blocks` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `modular_per_parameter` | 2.230e-01 | 1.361e+00 | 17.089 ms | 1.05x | 177.04 MiB | 1.00x |
| 4 | `modular_linear_blocks` | 1.863e-09 | 4.091e-07 | 16.284 ms | 1.00x | 176.47 MiB | 1.00x |
| 4 | `backpack_autodiff_linear_blocks` | 0.000e+00 | 0.000e+00 | 17.082 ms | 1.05x | 229.12 MiB | 1.30x |
| 4 | `torch_backward` | n/a | n/a | 2.417 ms | 0.15x | 171.74 MiB | 0.97x |
| 50 | `modular_per_parameter` | 3.155e-01 | 1.143e+00 | 2127.611 ms | 1.85x | 257.64 MiB | 1.00x |
| 50 | `modular_linear_blocks` | 9.686e-08 | 1.387e-06 | 1149.908 ms | 1.00x | 257.39 MiB | 1.00x |
| 50 | `backpack_autodiff_linear_blocks` | 0.000e+00 | 0.000e+00 | 2251.654 ms | 1.96x | 324.59 MiB | 1.26x |
| 50 | `torch_backward` | n/a | n/a | 27.991 ms | 0.02x | 206.99 MiB | 0.80x |

### Full-MLP HVP Comparison

The same block interface can group all sequential MLP parameters into one
block:

```python
blocks = {"full": tuple(name for name, _ in model.named_parameters())}
```

This is mathematically the full Hessian-vector product for the MLP parameters,
so it is compared against one reverse-over-reverse call over all parameters and
against PyTorch's `torch.func.jvp(torch.func.grad(...))` baseline:

```bash
uv run python benchmarks/compare_mlp_full_hvp.py \
  --depths 4 50 \
  --batch-size 256 --d-in 784 --d-hidden 256 --d-out 10 \
  --dtype float32 --warmup 1 --repeats 3
```

BackPACK RoR is called once with the complete parameter list, not once per
parameter. The `torch.func` baseline uses detached functional parameters and an
explicit MSE expression equivalent to `nn.MSELoss(reduction="mean")`; this
avoids a local torch 2.2 `ZeroTensor` immutability failure in
`jvp(grad(nn.MSELoss(...)))`.

Full-block results are correct, but they are not the setting where ModularHVP's
block-scoped advantage is strongest. With one giant block, the problem becomes
ordinary full-HVP computation, and optimized PyTorch full-HVP baselines are
competitive or faster on this CPU run.

| Hidden Linear/ReLU blocks | Method | Max abs error vs full RoR | Max rel error vs full RoR | Mean time | Time vs `modular_full` | Median peak RSS | Peak RSS vs `modular_full` |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `modular_full` | 1.490e-08 | 2.676e-07 | 21.178 ms | 1.00x | 192.04 MiB | 1.00x |
| 4 | `backpack_autodiff_full` | 0.000e+00 | 0.000e+00 | 16.130 ms | 0.76x | 232.92 MiB | 1.21x |
| 4 | `torch_func_jvp_full` | 3.725e-08 | 3.568e-07 | 24.044 ms | 1.14x | 249.98 MiB | 1.30x |
| 4 | `torch_backward` | n/a | n/a | 4.079 ms | 0.19x | 171.84 MiB | 0.89x |
| 50 | `modular_full` | 1.118e-08 | 1.090e-06 | 271.466 ms | 1.00x | 390.67 MiB | 1.00x |
| 50 | `backpack_autodiff_full` | 0.000e+00 | 0.000e+00 | 228.550 ms | 0.84x | 375.58 MiB | 0.96x |
| 50 | `torch_func_jvp_full` | 2.235e-08 | 1.048e-06 | 165.997 ms | 0.61x | 346.02 MiB | 0.89x |
| 50 | `torch_backward` | n/a | n/a | 37.370 ms | 0.14x | 206.95 MiB | 0.53x |

## Toy CNN Comparison

The CNN comparison uses the same public `modular_hvp(...)` interface and the
same BackPACK baselines on a single-chain Conv/ReLU/AvgPool/Conv/ReLU/Flatten
toy CNN:

```bash
uv run python benchmarks/compare_toy_cnn.py \
  --batch-size 64 --image-size 16 --width 16 --d-out 10 \
  --dtype float32 --warmup 1 --repeats 3
```

Ratio columns compare each method against `modular_hvp`; values above `1.0x`
mean the method is slower or uses more peak RSS than `modular_hvp`.

| Method | Max abs error | Max rel error | Mean time | Time vs `modular_hvp` | Median peak RSS | Peak RSS vs `modular_hvp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modular_hvp` | 0.000e+00 | 0.000e+00 | 11.802 ms | 1.00x | 176.95 MiB | 1.00x |
| `backpack_hmp` | 3.353e-08 | 9.943e-07 | 31.235 ms | 2.65x | 274.18 MiB | 1.55x |
| `backpack_autodiff` | 2.980e-08 | 1.967e-07 | 18.869 ms | 1.60x | 228.97 MiB | 1.29x |
| `torch_backward` | n/a | n/a | 2.384 ms | 0.20x | 171.78 MiB | 0.97x |

BackPACK baseline limitations observed while setting up CNN comparisons:

- BackPACK HMP works on the float32 toy CNN above, but the same Conv/ReLU/
  AvgPool/Flatten/Linear HMP path failed locally in float64 with an internal
  dtype mismatch.
- BackPACK HMP does not currently provide an HMP extension for
  `nn.BatchNorm2d`; the local `modular_hvp` runtime supports eval-mode
  `BatchNorm2d` in sequential CNNs and ResNet-style DAGs, and tests it against
  per-parameter autodiff HVPs.
- BackPACK's reverse-over-reverse autodiff HVP utility remained usable for
  these toy CNN checks, but it keeps the expected reverse-over-reverse cost
  profile.

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

## ResNet-Style DAG Support

The default runtime now tests residual graphs against per-parameter autodiff,
including a small ResNet20-shaped eval-mode network with three two-convolution
basic blocks per stage. The runtime records residual joins as primitive tensor
addition edges and preserves the public invariant of exactly one `p.hvp` per
parameter. It does not expose branch-specific objects such as `hvp_main` or
`hvp_skip`.

The ResNet20 comparison script checks the public `modular_hvp(...)` interface
against BackPACK's reverse-over-reverse HVP utility and attempts BackPACK HMP:

```bash
uv run python benchmarks/compare_resnet20.py \
  --batch-size 2 --image-size 8 --width 2 --d-out 3 \
  --dtype float32 --warmup 1 --repeats 3
```

Observed result:

| Method | Max abs error | Max rel error | Mean time | Time vs `modular_hvp` | Median peak RSS | Peak RSS vs `modular_hvp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modular_hvp` | 0.000e+00 | 0.000e+00 | 402.629 ms | 1.00x | 176.46 MiB | 1.00x |
| `backpack_hmp` | unsupported | unsupported | unsupported | n/a | n/a | n/a |
| `backpack_autodiff` | 1.490e-08 | 8.160e-07 | 753.163 ms | 1.87x | 237.53 MiB | 1.35x |
| `torch_backward` | n/a | n/a | 11.208 ms | 0.03x | 172.73 MiB | 0.98x |

BackPACK HMP is unsupported on this residual graph in the installed BackPACK
version: after `extend(..., use_converter=True)`, HMP fails on BackPACK's
internal residual `SumModule`.

Performance note: the DAG path avoids the earlier correctness bug at residual
joins by propagating graph-indexed tangent packets in one backward pass. It is
still slower than the single-chain path because graph joins and nonlinear
backward JVPs need packet bookkeeping, but it no longer replays a downstream
graph action per parameter.

## Transformer / Multihead Attention Support

Transformer support is implemented by extending primitive ATen coverage and the
generic graph traversal, not by adding a transformer-level Hessian rule. The
runtime supports small eval-mode Transformer blocks with `LayerNorm`, packed
self-attention `nn.MultiheadAttention(batch_first=True)`, residual additions,
GELU MLPs, and MSE loss.

Current `MultiheadAttention` runtime constraints:

- self-attention only (`query is key is value`);
- `batch_first=True`;
- packed `in_proj_weight`/`in_proj_bias`;
- no `attn_mask` or `key_padding_mask`;
- dropout disabled by eval mode or `dropout=0.0`;
- output attention weights may be requested (`need_weights=True`) or skipped
  (`need_weights=False`, fused SDPA path).

The unfused comparison uses `need_weights=True`, which lowers attention into
matmul/div/softmax/matmul primitives and allows BackPACK autodiff to run:

```bash
uv run python benchmarks/compare_toy_transformer.py \
  --attention unfused --dtype float32 --warmup 1 --repeats 3
```

| Method | Max abs error | Max rel error | Mean time | Time vs `modular_hvp` | Median peak RSS | Peak RSS vs `modular_hvp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modular_hvp` | 0.000e+00 | 0.000e+00 | 93.576 ms | 1.00x | 174.84 MiB | 1.00x |
| `backpack_hmp` | unsupported | unsupported | unsupported | n/a | n/a | n/a |
| `backpack_autodiff` | 2.608e-08 | 1.239e-06 | 120.301 ms | 1.29x | 239.34 MiB | 1.37x |
| `torch_backward` | n/a | n/a | 3.752 ms | 0.04x | 167.34 MiB | 0.96x |

The fused comparison uses `need_weights=False`, which dispatches through
scaled-dot-product attention. ModularHVP handles the primitive fused attention
path in the same one-forward/one-backward runtime. BackPACK autodiff is
unsupported in this installed PyTorch/BackPACK stack because the second
derivative for `aten::_scaled_dot_product_flash_attention_backward` is not
implemented.

```bash
uv run python benchmarks/compare_toy_transformer.py \
  --attention fused --dtype float32 --warmup 1 --repeats 3
```

| Method | Max abs error | Max rel error | Mean time | Time vs `modular_hvp` | Median peak RSS | Peak RSS vs `modular_hvp` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `modular_hvp` | 0.000e+00 | 0.000e+00 | 103.341 ms | 1.00x | 175.91 MiB | 1.00x |
| `backpack_hmp` | unsupported | unsupported | unsupported | n/a | n/a | n/a |
| `backpack_autodiff` | unsupported | unsupported | unsupported | n/a | n/a | n/a |
| `torch_backward` | n/a | n/a | 3.740 ms | 0.04x | 167.54 MiB | 0.95x |

Larger unfused scaling checks after packet-retention memory optimization:

```bash
uv run python benchmarks/compare_toy_transformer.py \
  --attention unfused --batch-size 8 --seq-len 32 --d-model 64 \
  --n-heads 4 --layers 4 --dtype float32 --warmup 1 --repeats 2
```

| Config | ModularHVP mean time | BackPACK autodiff mean time | Autodiff time vs ModularHVP | ModularHVP median peak RSS | Autodiff peak RSS vs ModularHVP |
| --- | ---: | ---: | ---: | ---: | ---: |
| B8 T32 D64 L4 | 985.949 ms | 1680.610 ms | 1.70x | 277.99 MiB | 0.98x |
| B8 T16 D128 L4 | 1223.163 ms | 1962.607 ms | 1.60x | 305.95 MiB | 0.92x |
| B8 T16 D64 L8 | 3065.688 ms | 3869.132 ms | 1.26x | 401.43 MiB | 0.83x |

The same B8 T32 D64 L4 run before this memory pass used about 370.46 MiB
median peak RSS and 1321.786 ms for ModularHVP, so retained-packet pruning
reduced peak RSS by about 25% while improving time on that stress setting.

BackPACK baseline limitations observed on this toy Transformer:

- BackPACK HMP fails during extension with no HMP extension for the top-level
  toy Transformer module.
- BackPACK autodiff matches ModularHVP on the unfused path but uses higher
  process RSS in this run.
- BackPACK autodiff cannot run the fused flash-attention path because the
  required second derivative is unavailable.

Current limitations:

- BatchNorm2d is supported in eval mode only in the public HVP runtime.
- Supported transformer coverage targets eval-mode blocks built from `Linear`,
  `LayerNorm`, `GELU`, eval-mode `Dropout`, residual addition, split/chunk,
  view/reshape, transpose/contiguous, matmul, scalar division, softmax,
  scaled-dot-product attention, and the constrained `MultiheadAttention` path
  listed above.
- Training-mode dropout, masks in attention, cross-attention, and unpacked
  `MultiheadAttention` projection weights remain outside the optimized public
  runtime path.
- Shared and tied parameters are supported when each use site is covered by the
  existing primitive/local graph records. The regression suite covers a tied
  `nn.Embedding`/linear LM head while forcing `torch.autograd.grad` to fail
  inside `modular_hvp`, which guards against reintroducing the old fallback.

## Primitive Coverage Roadmap

Architecture support should be added by extending ATen-level `DualTensor`
rules, not by special-casing architecture motifs. For example, ResNet residual
connections are just ordinary tensor addition (`x + f(x)`), so `aten.add` is
the primitive rule; there should be no residual-specific rule.

Implemented CNN/ResNet forward primitives:

- `aten.convolution`: JVP uses
  `conv(x_dot, W) + conv(x, W_dot) + b_dot`.
- `aten.native_batch_norm`: training-mode JVP includes the tangent of batch
  mean, variance, and inverse standard deviation; eval-mode treats running
  statistics as constants.
- `aten.avg_pool2d` and `aten._adaptive_avg_pool2d`: linear same-op tangent
  rules.
- `aten.max_pool2d_with_indices`: tangent gathers from `x_dot` at the primal
  argmax indices.
- ResNet residual paths continue to use existing `aten.add`; there is no
  residual-specific rule.

Implemented transformer unfused primitives:

- Tuple/view primitives: `aten.split`, `aten.split_with_sizes`, Python
  `Tensor.split`/`chunk`, `aten.slice`, `aten.select`, view/reshape,
  transpose, contiguous, and chunk-style graph edges.
- Attention primitives: matmul/bmm, scalar division, `aten._softmax`, and
  softmax backward JVPs. Slow causal masking uses primitive `aten.masked_fill`.
- Normalization/MLP primitives: `aten.native_layer_norm`, LayerNorm backward
  JVPs, functional `F.layer_norm` lowered to native LayerNorm records,
  `aten.gelu`, exact GELU backward JVPs, and training/eval dropout as a
  primitive linear edge.
- Embedding forward JVP is implemented in the `DualTensor` backend for
  unfused forward composition tests.

Implemented fused attention primitives:

- `aten.scaled_dot_product_attention`: forward JVP treats SDPA as an atomic
  primitive when it appears at dispatch.
- `aten._scaled_dot_product_flash_attention`: forward JVP support for the
  output tensor when PyTorch dispatches through the flash-attention ATen
  primitive.
- The public runtime records scaled-dot-product attention as a primitive DAG
  edge and propagates its local backward JVP without exporting an attention- or
  branch-specific public quantity.

Remaining primitive families:

- Broader Transformer fastpath kernels should be added only when they actually
  appear at ATen dispatch. Do not replace them with module-level transformer
  rules.

Each added primitive should get a focused finite-difference test and then a
small composition test for the target architecture.
