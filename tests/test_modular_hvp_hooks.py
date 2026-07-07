from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

import modular_hvp.eager as eager_runtime
from modular_hvp import is_dual, modular_hvp
from modular_hvp.backend import FakeDualBackend, LocalDualActivations


class ToyBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module | None = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None
        self.relu2 = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu2(out)


class ToyResNet20(nn.Module):
    def __init__(self, width: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
        )
        self.stage1 = self._make_stage(width, width, blocks=3, stride=1)
        self.stage2 = self._make_stage(width, 2 * width, blocks=3, stride=2)
        self.stage3 = self._make_stage(2 * width, 4 * width, blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4 * width, num_classes)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers: list[nn.Module] = [
            ToyBasicBlock(in_channels, out_channels, stride=stride)
        ]
        for _ in range(1, blocks):
            layers.append(ToyBasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.pool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 8, n_heads: int = 2) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.1)
        self.ff2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=-1)
        batch, tokens, channels3 = qkv.shape
        channels = channels3 // 3
        q = q.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, tokens, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head**0.5)
        attention = torch.softmax(scores, dim=-1)
        y = (attention @ v).transpose(1, 2).contiguous().view(
            batch,
            tokens,
            channels,
        )
        x = x + self.proj(y)
        return x + self.ff2(self.drop(self.gelu(self.ff1(self.ln2(x)))))


class TinyMultiheadAttentionBlock(nn.Module):
    def __init__(self, *, need_weights: bool) -> None:
        super().__init__()
        self.need_weights = need_weights
        self.ln = nn.LayerNorm(8)
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.ff = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(x)
        y, _ = self.attn(h, h, h, need_weights=self.need_weights)
        return x + self.ff(y)


class TinyNanoChatPattern(nn.Module):
    def __init__(self, vocab_size: int = 11, d_model: int = 6) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.scale = nn.Parameter(torch.tensor([1.0, 0.25], dtype=torch.float64))
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        x = self.emb(idx)
        x = self.scale[0] * x + self.scale[1] * x
        gate = torch.sigmoid(x[:, 1:, :1])
        x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        x = torch.relu(x).square()
        logits = self.proj(x).float()
        logits = torch.tanh(logits / 5.0) * 5.0
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )


class TinyNanoChatQKVAttention(nn.Module):
    def __init__(
        self,
        vocab_size: int = 13,
        block_size: int = 5,
        d_model: int = 8,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(block_size, d_model)
        self.ln = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)).view(
                1,
                1,
                block_size,
                block_size,
            ),
        )

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch, tokens = idx.shape
        pos = torch.arange(tokens, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        h = self.ln(x)
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
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
        x = x + self.out_proj(y)
        logits = self.head(x)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1,
        )


class TinyTiedEmbeddingLM(nn.Module):
    def __init__(self, vocab_size: int = 13, d_model: int = 5) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.wte(idx))
        logits = self.lm_head(hidden)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))


class TinyNanoGPTConfig:
    def __init__(
        self,
        *,
        block_size: int = 6,
        vocab_size: int = 17,
        n_layer: int = 1,
        n_head: int = 2,
        n_embd: int = 8,
        dropout: float = 0.0,
        bias: bool = True,
        flash: bool = True,
    ) -> None:
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.flash = flash


class TinyNanoGPTLayerNorm(nn.Module):
    def __init__(self, ndim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim, dtype=torch.float64))
        self.bias = (
            nn.Parameter(torch.zeros(ndim, dtype=torch.float64)) if bias else None
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class TinyNanoGPTCausalSelfAttention(nn.Module):
    def __init__(self, config: TinyNanoGPTConfig) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = config.flash
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1,
                    1,
                    config.block_size,
                    config.block_size,
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, channels = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        k = k.view(batch, tokens, self.n_head, head_dim).transpose(1, 2)
        q = q.view(batch, tokens, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, tokens, self.n_head, head_dim).transpose(1, 2)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / (head_dim**0.5))
            att = att.masked_fill(self.bias[:, :, :tokens, :tokens] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.resid_dropout(self.c_proj(y))


class TinyNanoGPTMLP(nn.Module):
    def __init__(self, config: TinyNanoGPTConfig) -> None:
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class TinyNanoGPTBlock(nn.Module):
    def __init__(self, config: TinyNanoGPTConfig) -> None:
        super().__init__()
        self.ln_1 = TinyNanoGPTLayerNorm(config.n_embd, bias=config.bias)
        self.attn = TinyNanoGPTCausalSelfAttention(config)
        self.ln_2 = TinyNanoGPTLayerNorm(config.n_embd, bias=config.bias)
        self.mlp = TinyNanoGPTMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TinyNanoGPT(nn.Module):
    def __init__(self, config: TinyNanoGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(config.vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "drop": nn.Dropout(config.dropout),
                "h": nn.ModuleList(
                    [TinyNanoGPTBlock(config) for _ in range(config.n_layer)]
                ),
                "ln_f": TinyNanoGPTLayerNorm(config.n_embd, bias=config.bias),
            }
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, tokens = idx.size()
        pos = torch.arange(0, tokens, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss


def _tangents_by_name(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: torch.ones_like(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _block_autodiff_hvps(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    tangents: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    loss = loss_fn(model(x), target)
    hvps: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        gradient = torch.autograd.grad(
            loss,
            [parameter],
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )[0]
        directional_gradient = (gradient * tangents[name]).sum()
        hvps[name] = torch.autograd.grad(
            directional_gradient,
            [parameter],
            retain_graph=True,
            materialize_grads=True,
        )[0].detach()
    return hvps


def _block_autodiff_hvps_from_loss(
    loss: torch.Tensor,
    model: nn.Module,
    tangents: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    hvps: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        gradient = torch.autograd.grad(
            loss,
            [parameter],
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )[0]
        directional_gradient = (gradient * tangents[name]).sum()
        hvps[name] = torch.autograd.grad(
            directional_gradient,
            [parameter],
            retain_graph=True,
            materialize_grads=True,
        )[0].detach()
    return hvps


def _custom_block_autodiff_hvps(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    tangents: Mapping[str, torch.Tensor],
    blocks: Mapping[Any, Sequence[str]],
) -> dict[str, torch.Tensor]:
    loss = loss_fn(model(x), target)
    named_parameters = dict(model.named_parameters())
    hvps: dict[str, torch.Tensor] = {}
    for block_names in blocks.values():
        block_parameters = [named_parameters[name] for name in block_names]
        block_tangents = [tangents[name] for name in block_names]
        gradients = torch.autograd.grad(
            loss,
            block_parameters,
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )
        directional_gradient = sum(
            (gradient * tangent_value).sum()
            for gradient, tangent_value in zip(gradients, block_tangents, strict=True)
        )
        block_hvps = torch.autograd.grad(
            directional_gradient,
            block_parameters,
            retain_graph=True,
            materialize_grads=True,
        )
        for name, hvp in zip(block_names, block_hvps, strict=True):
            hvps[name] = hvp.detach()
    return hvps


def _custom_block_autodiff_hvps_from_loss(
    loss: torch.Tensor,
    model: nn.Module,
    tangents: Mapping[str, torch.Tensor],
    blocks: Mapping[Any, Sequence[str]],
) -> dict[str, torch.Tensor]:
    named_parameters = dict(model.named_parameters())
    hvps: dict[str, torch.Tensor] = {}
    for block_names in blocks.values():
        block_parameters = [named_parameters[name] for name in block_names]
        block_tangents = [tangents[name] for name in block_names]
        gradients = torch.autograd.grad(
            loss,
            block_parameters,
            create_graph=True,
            retain_graph=True,
            materialize_grads=True,
        )
        directional_gradient = sum(
            (gradient * tangent_value).sum()
            for gradient, tangent_value in zip(gradients, block_tangents, strict=True)
        )
        block_hvps = torch.autograd.grad(
            directional_gradient,
            block_parameters,
            retain_graph=True,
            materialize_grads=True,
        )
        for name, hvp in zip(block_names, block_hvps, strict=True):
            hvps[name] = hvp.detach()
    return hvps


def _finite_difference_block_hvps(
    model: nn.Module,
    loss_fn: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    tangents: Mapping[str, torch.Tensor],
    *,
    h: float = 1e-5,
) -> dict[str, torch.Tensor]:
    hvps: dict[str, torch.Tensor] = {}
    parameters = dict(model.named_parameters())
    base_state = {name: parameter.detach().clone() for name, parameter in parameters.items()}
    for name, parameter in parameters.items():
        with torch.no_grad():
            parameter.copy_(base_state[name] + h * tangents[name])
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), target).backward()
        plus_grad = parameter.grad.detach().clone()

        with torch.no_grad():
            parameter.copy_(base_state[name] - h * tangents[name])
        model.zero_grad(set_to_none=True)
        loss_fn(model(x), target).backward()
        minus_grad = parameter.grad.detach().clone()

        with torch.no_grad():
            parameter.copy_(base_state[name])
        hvps[name] = (plus_grad - minus_grad) / (2 * h)
    return hvps


class RecordingBackend(FakeDualBackend):
    def __init__(self) -> None:
        self.local_forward_modules: list[str] = []
        self.backward_modules: list[str] = []

    def local_forward(
        self,
        *,
        module: nn.Module,
        original_forward: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
    ) -> tuple[Any, LocalDualActivations]:
        self.local_forward_modules.append(module.__class__.__name__)
        return super().local_forward(
            module=module,
            original_forward=original_forward,
            args=args,
            kwargs=kwargs,
            active_param_tangents=active_param_tangents,
        )

    def dual_backward(
        self,
        *,
        module: nn.Module,
        local_dual_activations: LocalDualActivations,
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
        grad_input: Sequence[torch.Tensor | None],
        grad_output: Sequence[torch.Tensor | None],
    ) -> Mapping[nn.Parameter, torch.Tensor]:
        self.backward_modules.append(module.__class__.__name__)
        return super().dual_backward(
            module=module,
            local_dual_activations=local_dual_activations,
            active_param_tangents=active_param_tangents,
            grad_input=grad_input,
            grad_output=grad_output,
        )


class CountingBackend(FakeDualBackend):
    def __init__(self) -> None:
        self.calls_by_parameter_id: dict[int, int] = {}

    def dual_backward(
        self,
        *,
        module: nn.Module,
        local_dual_activations: LocalDualActivations,
        active_param_tangents: Mapping[nn.Parameter, torch.Tensor],
        grad_input: Sequence[torch.Tensor | None],
        grad_output: Sequence[torch.Tensor | None],
    ) -> Mapping[nn.Parameter, torch.Tensor]:
        for parameter in active_param_tangents:
            self.calls_by_parameter_id[id(parameter)] = (
                self.calls_by_parameter_id.get(id(parameter), 0) + 1
            )
        return super().dual_backward(
            module=module,
            local_dual_activations=local_dual_activations,
            active_param_tangents=active_param_tangents,
            grad_input=grad_input,
            grad_output=grad_output,
        )


def test_default_modular_hvp_matches_block_autodiff_on_mlp() -> None:
    torch.manual_seed(0)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    baseline_loss = criterion(baseline(x), target)
    baseline_loss.backward()
    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    assert torch.allclose(loss.detach(), baseline_loss.detach())
    for (_, expected), (_, actual) in zip(
        baseline.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.allclose(actual.grad, expected.grad)
        assert actual.hvp is not None
    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_modular_hvp_supports_linear_module_blocks_on_sequential_mlp() -> None:
    torch.manual_seed(42)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    blocks = {
        model[0]: ("0.weight", "0.bias"),
        model[2]: ("2.weight", "2.bias"),
    }
    baseline_blocks = {
        baseline[0]: ("0.weight", "0.bias"),
        baseline[2]: ("2.weight", "2.bias"),
    }

    reference_hvps = _custom_block_autodiff_hvps(
        baseline,
        criterion,
        x,
        target,
        tangents,
        baseline_blocks,
    )
    with modular_hvp(model, tangents, blocks=blocks):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_custom_blocks_may_leave_parameters_as_singletons() -> None:
    torch.manual_seed(44)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    blocks = {model[0]: ("0.weight", "0.bias")}
    baseline_blocks = {
        baseline[0]: ("0.weight", "0.bias"),
        "2.weight": ("2.weight",),
        "2.bias": ("2.bias",),
    }

    reference_hvps = _custom_block_autodiff_hvps(
        baseline,
        criterion,
        x,
        target,
        tangents,
        baseline_blocks,
    )
    with modular_hvp(model, tangents, blocks=blocks):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_linear_module_blocks_keep_within_module_cross_terms() -> None:
    torch.manual_seed(43)
    baseline = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(4, 2, dtype=torch.float64)
    target = torch.randn(4, 1, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    module_blocks = {
        model[0]: ("0.weight", "0.bias"),
        model[2]: ("2.weight", "2.bias"),
    }
    baseline_module_blocks = {
        baseline[0]: ("0.weight", "0.bias"),
        baseline[2]: ("2.weight", "2.bias"),
    }

    module_hvps = _custom_block_autodiff_hvps(
        baseline,
        criterion,
        x,
        target,
        tangents,
        baseline_module_blocks,
    )
    parameter_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    with modular_hvp(model, tangents, blocks=module_blocks):
        loss = criterion(model(x), target)
        loss.backward()

    assert not torch.allclose(
        module_hvps["0.weight"],
        parameter_hvps["0.weight"],
        rtol=1e-10,
        atol=1e-10,
    )
    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, module_hvps[name], rtol=1e-10, atol=1e-10)


def test_modular_hvp_supports_full_block_on_sequential_mlp() -> None:
    torch.manual_seed(44)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    block_names = tuple(name for name, _ in model.named_parameters())
    blocks = {"full": block_names}

    reference_hvps = _custom_block_autodiff_hvps(
        baseline,
        criterion,
        x,
        target,
        tangents,
        blocks,
    )
    with modular_hvp(model, tangents, blocks=blocks):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_custom_block_rejects_disconnected_parameter_dependency_graph() -> None:
    torch.manual_seed(45)
    model = nn.Sequential(
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 4),
        nn.ReLU(),
        nn.Linear(4, 2),
    ).double()

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    blocks = {
        "disconnected": ("0.weight", "4.weight"),
        "0.bias": ("0.bias",),
        "2.weight": ("2.weight",),
        "2.bias": ("2.bias",),
        "4.bias": ("4.bias",),
    }

    with pytest.raises(NotImplementedError, match="disconnected"):
        with modular_hvp(model, tangents, blocks=blocks):
            loss = criterion(model(x), target)
            loss.backward()


def test_custom_block_rejects_unsupported_multi_leaf_parameterized_op() -> None:
    torch.manual_seed(46)
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(2, 2, kernel_size=3, padding=1),
        nn.Flatten(),
        nn.Linear(2 * 4 * 4, 3),
    ).double()

    x = torch.randn(2, 1, 4, 4, dtype=torch.float64)
    target = torch.randn(2, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    blocks = {
        "conv_pair": ("0.weight", "0.bias", "2.weight", "2.bias"),
        "linear_weight": ("4.weight",),
        "linear_bias": ("4.bias",),
    }

    with pytest.raises(NotImplementedError, match="backward-input JVP"):
        with modular_hvp(model, tangents, blocks=blocks):
            loss = criterion(model(x), target)
            loss.backward()


def test_default_modular_hvp_matches_block_autodiff_on_sequential_cnn() -> None:
    torch.manual_seed(20)
    baseline = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 4 * 4, 3),
    ).double()
    model = nn.Sequential(
        nn.Conv2d(1, 2, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(2 * 4 * 4, 3),
    ).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(4, 1, 8, 8, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_matches_block_autodiff_on_eval_cnn_with_batchnorm_and_pooling() -> None:
    torch.manual_seed(21)
    baseline = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(12, 2),
    ).double().eval()
    model = nn.Sequential(
        nn.Conv2d(1, 3, kernel_size=3, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.AdaptiveAvgPool2d((2, 2)),
        nn.Flatten(),
        nn.Linear(12, 2),
    ).double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(3, 1, 8, 8, dtype=torch.float64)
    target = torch.randn(3, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_matches_block_autodiff_on_residual_block() -> None:
    torch.manual_seed(22)
    baseline = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        ToyBasicBlock(4, 4),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    ).double().eval()
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        ToyBasicBlock(4, 4),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    ).double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 3, 6, 6, dtype=torch.float64)
    target = torch.randn(2, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_matches_block_autodiff_on_resnet20_shape() -> None:
    torch.manual_seed(23)
    baseline = ToyResNet20(width=2, num_classes=3).double().eval()
    model = ToyResNet20(width=2, num_classes=3).double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 3, 8, 8, dtype=torch.float64)
    target = torch.randn(2, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-9, atol=1e-10)


def test_default_modular_hvp_matches_block_autodiff_on_transformer_block() -> None:
    torch.manual_seed(24)
    baseline = TinyTransformerBlock().double().eval()
    model = TinyTransformerBlock().double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 4, 8, dtype=torch.float64)
    target = torch.randn_like(x)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-6,
            atol=1e-8,
        )


def test_default_modular_hvp_matches_block_autodiff_on_multihead_attention_unfused() -> None:
    torch.manual_seed(25)
    baseline = TinyMultiheadAttentionBlock(need_weights=True).double().eval()
    model = TinyMultiheadAttentionBlock(need_weights=True).double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 4, 8, dtype=torch.float64)
    target = torch.randn_like(x)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-6,
            atol=1e-8,
        )


def test_default_modular_hvp_matches_block_autodiff_on_token_gpt_pattern() -> None:
    torch.manual_seed(28)
    baseline = TinyNanoChatPattern().double()
    model = TinyNanoChatPattern().double()
    model.load_state_dict(baseline.state_dict())

    idx = torch.randint(0, 11, (2, 4), dtype=torch.long)
    targets = torch.randint(0, 11, (2, 4), dtype=torch.long)
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _block_autodiff_hvps_from_loss(
        baseline(idx, targets),
        baseline,
        tangents,
    )
    with modular_hvp(model, tangents):
        loss = model(idx, targets)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_custom_qkv_block_on_nanochat_shaped_attention_matches_grouped_autodiff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(47)
    baseline = TinyNanoChatQKVAttention().double()
    model = TinyNanoChatQKVAttention().double()
    model.load_state_dict(baseline.state_dict())

    idx = torch.randint(0, 13, (2, 5), dtype=torch.long)
    targets = torch.randint(0, 13, (2, 5), dtype=torch.long)
    targets[0, -1] = -1
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    qkv_names = (
        "q_proj.weight",
        "q_proj.bias",
        "k_proj.weight",
        "k_proj.bias",
        "v_proj.weight",
        "v_proj.bias",
    )
    blocks: dict[str, tuple[str, ...]] = {"qkv": qkv_names}
    for name in tangents:
        if name not in qkv_names:
            blocks[name] = (name,)

    reference_hvps = _custom_block_autodiff_hvps_from_loss(
        baseline(idx, targets),
        baseline,
        tangents,
        blocks,
    )

    def fail_autograd_grad(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("modular_hvp must not use reverse-over-reverse fallback")

    monkeypatch.setattr(torch.autograd, "grad", fail_autograd_grad)
    with modular_hvp(model, tangents, blocks=blocks):
        loss = model(idx, targets)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-8,
            atol=1e-9,
        )


def test_default_modular_hvp_matches_finite_difference_on_multihead_attention_fused() -> None:
    torch.manual_seed(26)
    baseline = TinyMultiheadAttentionBlock(need_weights=False).double().eval()
    model = TinyMultiheadAttentionBlock(need_weights=False).double().eval()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(2, 4, 8, dtype=torch.float64)
    target = torch.randn_like(x)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    reference_hvps = _finite_difference_block_hvps(
        baseline,
        criterion,
        x,
        target,
        tangents,
    )
    with modular_hvp(model, tangents):
        loss = criterion(model(x), target)
        loss.backward()

    for name, parameter in model.named_parameters():
        assert parameter.hvp is not None
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-5,
            atol=1e-7,
        )


def test_default_modular_hvp_uses_original_module_forward() -> None:
    class OffsetLinear(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input) + 1.25

    torch.manual_seed(0)
    baseline = nn.Sequential(OffsetLinear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model = nn.Sequential(OffsetLinear(3, 4), nn.ReLU(), nn.Linear(4, 2)).double()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    baseline_output = baseline(x)
    baseline_loss = criterion(baseline_output, target)
    baseline_loss.backward()
    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    with modular_hvp(model, tangents):
        output = model(x)
        loss = criterion(output, target)
        loss.backward()

    assert torch.allclose(output.detach(), baseline_output.detach())
    assert torch.allclose(loss.detach(), baseline_loss.detach())
    for (_, expected), (_, actual) in zip(
        baseline.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.allclose(actual.grad, expected.grad)
        assert actual.hvp is not None
    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_runtime_uses_autograd_saved_activation_refs() -> None:
    class OffsetLinear(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return super().forward(input) + 1.25

    torch.manual_seed(0)
    layer = OffsetLinear(3, 4).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    linear_output = layer(x)
    linear_ref = eager_runtime._make_linear_input_activation_ref(linear_output, x)

    assert linear_ref.fallback is None
    assert torch.equal(linear_ref.resolve_and_release(), x)
    assert linear_ref.grad_fn is None

    relu_output = torch.relu(linear_output)
    relu_ref = eager_runtime._make_relu_output_activation_ref(relu_output)

    assert torch.equal(relu_ref.resolve_and_release(), relu_output)
    assert relu_ref.grad_fn is None


def test_default_modular_hvp_consumes_local_dual_parameter_in_forward() -> None:
    class InspectLinear(nn.Linear):
        calls: int
        dual_weight_calls: int
        dual_bias_calls: int
        primal_calls: int

        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__(in_features, out_features)
            self.calls = 0
            self.dual_weight_calls = 0
            self.dual_bias_calls = 0
            self.primal_calls = 0
            self.dual_input_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            weight_is_dual = is_dual(self.weight)
            bias_is_dual = is_dual(self.bias)
            self.dual_input_calls += int(is_dual(input))
            self.dual_weight_calls += int(weight_is_dual)
            self.dual_bias_calls += int(bias_is_dual)
            self.primal_calls += int(not weight_is_dual and not bias_is_dual)
            return super().forward(input)

    torch.manual_seed(0)
    model = nn.Sequential(InspectLinear(3, 2)).double()
    x = torch.randn(5, 3, dtype=torch.float64)
    target = torch.randn(5, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    with modular_hvp(model, tangents):
        criterion(model(x), target).backward()

    layer = model[0]
    assert layer.calls == 1
    assert layer.dual_weight_calls == 1
    assert layer.dual_bias_calls == 0
    assert layer.dual_input_calls == 0
    assert layer.primal_calls == 0
    for parameter in model.parameters():
        assert parameter.hvp is not None


def test_default_modular_hvp_does_not_export_dual_activations() -> None:
    class InspectReLU(nn.ReLU):
        def __init__(self) -> None:
            super().__init__()
            self.dual_input_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.dual_input_calls += int(is_dual(input))
            return super().forward(input)

    class InspectLinear(nn.Linear):
        def __init__(self, in_features: int, out_features: int) -> None:
            super().__init__(in_features, out_features)
            self.dual_input_calls = 0

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            self.dual_input_calls += int(is_dual(input))
            return super().forward(input)

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(3, 5),
        InspectReLU(),
        InspectLinear(5, 4),
        InspectReLU(),
        InspectLinear(4, 2),
    ).double()
    x = torch.randn(6, 3, dtype=torch.float64)
    target = torch.randn(6, 2, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    with modular_hvp(model, tangents):
        criterion(model(x), target).backward()

    for module in model:
        if isinstance(module, (InspectReLU, InspectLinear)):
            assert module.dual_input_calls == 0
    for parameter in model.parameters():
        assert parameter.hvp is not None


def test_default_modular_hvp_accumulates_reused_parameter_hvps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(1)

    def make_model() -> nn.Sequential:
        shared = nn.Linear(3, 3).double()
        return nn.Sequential(shared, nn.ReLU(), shared)

    baseline = make_model()
    model = make_model()
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(4, 3, dtype=torch.float64)
    target = torch.randn(4, 3, dtype=torch.float64)
    criterion = nn.MSELoss()
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    reference_hvps = _block_autodiff_hvps(baseline, criterion, x, target, tangents)

    def fail_autograd_grad(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("modular_hvp must not use reverse-over-reverse fallback")

    monkeypatch.setattr(torch.autograd, "grad", fail_autograd_grad)
    with modular_hvp(model, tangents):
        criterion(model(x), target).backward()

    for name, parameter in model.named_parameters():
        assert torch.allclose(parameter.hvp, reference_hvps[name], rtol=1e-10, atol=1e-10)


def test_default_modular_hvp_supports_tied_embedding_lm_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(29)
    baseline = TinyTiedEmbeddingLM().double()
    model = TinyTiedEmbeddingLM().double()
    model.load_state_dict(baseline.state_dict())

    idx = torch.randint(0, 13, (3, 4), dtype=torch.long)
    targets = torch.randint(0, 13, (3, 4), dtype=torch.long)
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }
    assert set(tangents) == {"wte.weight"}

    reference_hvps = _block_autodiff_hvps_from_loss(
        baseline(idx, targets),
        baseline,
        tangents,
    )

    def fail_autograd_grad(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("modular_hvp must not use reverse-over-reverse fallback")

    monkeypatch.setattr(torch.autograd, "grad", fail_autograd_grad)
    with modular_hvp(model, tangents):
        model(idx, targets).backward()

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-10,
            atol=1e-10,
        )


@pytest.mark.parametrize(
    ("flash", "dropout"),
    [
        (True, 0.0),
        (False, 0.2),
    ],
)
def test_default_modular_hvp_supports_nanogpt_shaped_training(
    monkeypatch: pytest.MonkeyPatch,
    flash: bool,
    dropout: float,
) -> None:
    torch.manual_seed(41)
    config = TinyNanoGPTConfig(flash=flash, dropout=dropout)
    model = TinyNanoGPT(config).double()
    reference_config = TinyNanoGPTConfig(flash=False, dropout=dropout)
    baseline = TinyNanoGPT(reference_config).double()
    baseline.load_state_dict(model.state_dict(), strict=False)
    baseline.train()
    model.train()

    idx = torch.randint(0, config.vocab_size, (2, config.block_size), dtype=torch.long)
    targets = torch.randint(
        0,
        config.vocab_size,
        (2, config.block_size),
        dtype=torch.long,
    )
    targets[0, -1] = -1
    tangents = {
        name: torch.randn_like(parameter)
        for name, parameter in model.named_parameters()
    }

    torch.manual_seed(123)
    _, baseline_loss = baseline(idx, targets)
    assert baseline_loss is not None
    reference_hvps = _block_autodiff_hvps_from_loss(
        baseline_loss,
        baseline,
        tangents,
    )

    def fail_autograd_grad(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("modular_hvp must not use reverse-over-reverse fallback")

    monkeypatch.setattr(torch.autograd, "grad", fail_autograd_grad)
    torch.manual_seed(123)
    with modular_hvp(model, tangents):
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(
            parameter.hvp,
            reference_hvps[name],
            rtol=1e-9 if parameter.dtype == torch.float64 else 1e-4,
            atol=1e-9 if parameter.dtype == torch.float64 else 1e-5,
        )


def test_explicit_hook_backend_preserves_primal_forward_and_gradients() -> None:
    torch.manual_seed(0)
    baseline = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    model.load_state_dict(baseline.state_dict())

    x = torch.randn(5, 3)
    target = torch.randn(5, 2)
    criterion = nn.MSELoss()

    baseline_loss = criterion(baseline(x), target)
    baseline_loss.backward()

    backend = RecordingBackend()
    with modular_hvp(model, _tangents_by_name(model), backend=backend):
        loss = criterion(model(x), target)
        loss.backward()

    assert torch.allclose(loss.detach(), baseline_loss.detach())
    for (_, expected), (_, actual) in zip(
        baseline.named_parameters(), model.named_parameters(), strict=True
    ):
        assert torch.allclose(actual.grad, expected.grad)
        assert actual.hvp is not None
        assert torch.equal(actual.hvp, torch.zeros_like(actual))
    assert backend.local_forward_modules == ["Linear", "Linear"]
    assert backend.backward_modules == ["Linear", "Linear"]


def test_default_context_restores_forward_and_removes_active_flags() -> None:
    model = nn.Linear(3, 2)
    original_forward = model.forward

    with modular_hvp(model, _tangents_by_name(model)):
        assert model.forward is not original_forward

    assert "forward" not in model.__dict__
    assert not hasattr(model, "_modular_hvp_eager_active")
    assert model.forward.__func__ is original_forward.__func__


def test_explicit_hook_context_restores_forward_and_removes_active_flags() -> None:
    model = nn.Linear(3, 2)
    original_forward = model.forward

    with modular_hvp(model, _tangents_by_name(model), backend=FakeDualBackend()):
        assert model.forward is not original_forward

    assert "forward" not in model.__dict__
    assert not hasattr(model, "_modular_hvp_runtime_active")
    assert model.forward.__func__ is original_forward.__func__


def test_parameter_object_tangent_keys_are_supported() -> None:
    model = nn.Linear(3, 2)
    tangents = {
        parameter: torch.ones_like(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }

    with modular_hvp(model, tangents):
        output = model(torch.randn(4, 3))
        assert type(output) is torch.Tensor
        loss = nn.MSELoss()(output, torch.zeros_like(output))
        loss.backward()

    for parameter in model.parameters():
        assert parameter.hvp is not None
        assert parameter.hvp.shape == parameter.shape


def test_missing_tangent_is_rejected() -> None:
    model = nn.Linear(3, 2)

    with pytest.raises(ValueError, match="missing tangents"):
        modular_hvp(model, {"weight": torch.ones_like(model.weight)})


def test_shape_mismatch_is_rejected() -> None:
    model = nn.Linear(3, 2)
    tangents = _tangents_by_name(model)
    tangents["weight"] = torch.ones(1)

    with pytest.raises(ValueError, match="shape"):
        modular_hvp(model, tangents)


def test_duplicate_name_and_object_tangent_is_rejected() -> None:
    model = nn.Linear(3, 2)
    tangents: dict[str | nn.Parameter, torch.Tensor] = {
        "weight": torch.ones_like(model.weight),
        model.weight: torch.ones_like(model.weight),
        "bias": torch.ones_like(model.bias),
    }

    with pytest.raises(ValueError, match="duplicate"):
        modular_hvp(model, tangents)


class ReusedLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.linear(x)


def test_reused_module_consumes_forward_records_lifo_and_accumulates_hvp() -> None:
    model = ReusedLinear()
    backend = CountingBackend()

    with modular_hvp(model, _tangents_by_name(model), backend=backend):
        model(torch.randn(2, 3)).sum().backward()

    for parameter in model.parameters():
        assert backend.calls_by_parameter_id[id(parameter)] == 2
        assert parameter.hvp is not None
        assert torch.equal(parameter.hvp, torch.zeros_like(parameter))


def test_existing_stale_hvp_is_cleared_on_entry() -> None:
    model = nn.Linear(3, 2)
    model.weight.hvp = torch.full_like(model.weight, 17.0)

    with modular_hvp(model, _tangents_by_name(model)):
        output = model(torch.randn(4, 3))
        assert type(output) is torch.Tensor
        nn.MSELoss()(output, torch.zeros_like(output)).backward()

    assert model.weight.hvp is not None
    assert not torch.equal(model.weight.hvp, torch.full_like(model.weight, 17.0))
