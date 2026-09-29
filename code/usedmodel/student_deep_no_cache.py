"""Causal RoPE Transformer with RMSNorm and gated feed-forward layers.

Only training-set gradients determine its parameters. No state survives a call.
Ablation: set USE_ROPE=False and USE_SWIGLU=False, then retrain from scratch.
"""
import torch
from torch import nn
from torch.nn import functional as F

USE_ROPE = True
USE_SWIGLU = True


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight)


def rotate_half(x):
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width, heads, context):
        super().__init__()
        self.heads = heads
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        hidden = 512 if USE_SWIGLU else 768
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False) if USE_SWIGLU else None
        self.down = nn.Linear(hidden, width, bias=False)
        dim = width // heads
        inv = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        phase = torch.outer(torch.arange(context, dtype=torch.float32), inv)
        self.register_buffer('cos', torch.repeat_interleave(phase.cos(), 2, dim=-1), persistent=False)
        self.register_buffer('sin', torch.repeat_interleave(phase.sin(), 2, dim=-1), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        if USE_ROPE:
            cos, sin = self.cos[:length].to(q.dtype), self.sin[:length].to(q.dtype)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin
        attended = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + F.dropout(self.proj(attended.transpose(1, 2).reshape(batch, length, width)), p=0.10, training=self.training)
        h = self.norm2(x)
        if USE_SWIGLU:
            h = F.silu(self.gate(h)) * self.up(h)
        else:
            h = F.gelu(self.gate(h))
        return x + F.dropout(self.down(h), p=0.10, training=self.training)


class StudentModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        width, heads, depth = 192, 6, 8
        self.token = nn.Embedding(config['vocab'], width)
        self.blocks = nn.ModuleList([Block(width, heads, self.context) for _ in range(depth)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, config['vocab'], bias=False)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=.02 / (2 * depth) ** .5)
            nn.init.normal_(block.down.weight, std=.02 / (2 * depth) ** .5)
        self.head.weight = self.token.weight

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=.02)

    def forward(self, ids):
        x = self.token(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

    def predict_log_probs(self, ids):
        return F.log_softmax(self(ids).float(), dim=-1)


def build_model(config):
    return StudentModel(config)
