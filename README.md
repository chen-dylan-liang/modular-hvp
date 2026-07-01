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

The current backend is intentionally fake: it preserves the runtime shape and
writes zero HVP tensors. Primitive dual tensor rules are the next milestone.
