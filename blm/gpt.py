#!/usr/bin/env python3
"""PyTorch implementation of a lightweight GPT-style autoregressive byte model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass
class BytesGPTConfig:
    """Configuration container for :class:`BytesGPT`."""

    block_size: int = 512
    vocab_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


class CausalSelfAttention(nn.Module):
    """Multi-head masked self-attention."""

    def __init__(self, config: BytesGPTConfig) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("Embedding dimension must be divisible by number of heads")

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        bias = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("bias", bias.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, embed = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=2)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * self.scale
        mask = self.bias[:, :, :seqlen, :seqlen]
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, embed)
        y = self.resid_drop(self.out_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network used inside the transformer block."""

    def __init__(self, config: BytesGPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc(x)
        x = self.act(x)
        x = self.proj(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block with pre-layer normalization."""

    def __init__(self, config: BytesGPTConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BytesGPT(nn.Module):
    """A compact GPT-style transformer for byte sequences."""

    def __init__(self, config: BytesGPTConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def forward(self, idx: Tensor, targets: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        bsz, seqlen = idx.shape
        if seqlen > self.config.block_size:
            raise ValueError(
                f"Sequence length {seqlen} exceeds block size {self.config.block_size}"
            )

        device = idx.device
        positions = torch.arange(0, seqlen, dtype=torch.long, device=device)

        x = self.embedding(idx) + self.position(positions)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        idx: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None and top_k < logits.size(-1):
                top_values, _ = torch.topk(logits, top_k)
                min_top_values = top_values[:, [-1]]
                logits = torch.where(
                    logits < min_top_values,
                    torch.tensor(float("-inf"), device=logits.device),
                    logits,
                )

            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

    def save_weights(self, path: str | bytes) -> None:
        torch.save({"model": self.state_dict(), "config": self.config.__dict__}, path)

    def load_weights(self, path: str | bytes, strict: bool = True) -> None:
        payload = torch.load(path, map_location=self.device)
        state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        self.load_state_dict(state, strict=strict)

    @staticmethod
    def encode(data: Sequence[int] | bytes | bytearray | str) -> list[int]:
        return encode(data)

    @staticmethod
    def decode(tokens: Iterable[int]) -> str:
        return decode(tokens)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def encode(data: Sequence[int] | bytes | bytearray | str) -> list[int]:
    """Convert input to a list of byte token IDs."""

    if isinstance(data, str):
        data = data.encode("utf-8")
    if isinstance(data, (bytes, bytearray, memoryview)):
        return [int(b) for b in bytes(data)]
    if isinstance(data, Iterable):
        return [int(b) & 0xFF for b in data]
    raise TypeError(f"Unsupported type for encode: {type(data)!r}")


def decode(tokens: Iterable[int]) -> str:
    """Decode byte token IDs back into a UTF-8 string."""

    return bytes(int(t) & 0xFF for t in tokens).decode("utf-8", "replace")


__all__ = ["BytesGPTConfig", "BytesGPT", "encode", "decode"]