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
