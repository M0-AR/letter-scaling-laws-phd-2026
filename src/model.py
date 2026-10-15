"""Tiny decoder-only Transformer for 27-letter (a-z + space) next-char prediction.

From-scratch implementation, no copy-paste from nanoGPT/minGPT.
Architecture: token+position embedding -> N x pre-LN block (causal MHA + MLP) -> LN -> head.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalMHA(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # causal scaled dot-product
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_head))
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).view(1, 1, T, T)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalMHA(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class LetterGPT(nn.Module):
    def __init__(self, vocab: int = 27, context: int = 64, d_model: int = 32,
                 n_layer: int = 2, n_head: int = 4, dropout: float = 0.0):
        super().__init__()
        self.context = context
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(context, d_model)
        self.blocks = nn.Sequential(*[Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.context, f"context {T} > max {self.context}"
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None, :, :]
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, n: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(n):
            c = idx[:, -self.context:]
            logits, _ = self(c)
            probs = F.softmax(logits[:, -1, :] / max(temperature, 1e-6), dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


def count_params(model: nn.Module, exclude_embedding: bool = False) -> int:
    total = 0
    for name, p in model.named_parameters():
        if exclude_embedding and ("tok" in name or "pos" in name):
            continue
        total += p.numel()
    return total


# 8 sizes spanning ~0.5k .. ~3.5M is aspirational on laptop; we use
# depth/width ladder whose *non-embedding* counts grow ~4x per step.
LADDER = [
    {"d_model": 8, "n_layer": 1, "n_head": 2},
    {"d_model": 12, "n_layer": 2, "n_head": 2},
    {"d_model": 24, "n_layer": 2, "n_head": 4},
    {"d_model": 32, "n_layer": 2, "n_head": 4},
    {"d_model": 64, "n_layer": 2, "n_head": 4},
    {"d_model": 128, "n_layer": 2, "n_head": 4},
    {"d_model": 256, "n_layer": 2, "n_head": 8},
    {"d_model": 384, "n_layer": 2, "n_head": 8},
]
