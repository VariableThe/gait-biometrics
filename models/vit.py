"""SEGait-ConViT: Vision Transformer for efficient gait recognition (paper B7).

Efficient Gait Recognition with SEGait-ConViT: A Vision Transformer Model
(Procedia CS, 2025). This module provides 1D patch embedding, a pre-LN
ConViT block, and `GaitViT` — a ViT classifier over sensor windows.

With T=128 and patch_len=16 the input is split into 8 patches. A learned
CLS token and learned positional embeddings are prepended; pre-LN
transformer blocks process the sequence; a mean-pooled head produces the
class logits.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed1d(nn.Module):
    """1D patch embedding: (N, C, T) -> (N, n_patches, embed_dim).

    A Conv1d with kernel_size=stride=patch_len performs non-overlapping
    patch extraction and projection in one step.

    Args:
        n_channels: number of sensor channels.
        patch_len: length of each patch (also the stride).
        embed_dim: output embedding dimension.
        stride: patch stride; defaults to `patch_len` (non-overlapping).
    """

    def __init__(self, n_channels: int, patch_len: int, embed_dim: int,
                 stride: int | None = None) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride if stride is not None else patch_len
        self.proj = nn.Conv1d(
            n_channels, embed_dim, kernel_size=patch_len, stride=self.stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C, T) -> (N, embed_dim, n_patches) -> (N, n_patches, embed_dim)
        h = self.proj(x)
        return h.transpose(1, 2)


class ConViTBlock(nn.Module):
    """Pre-LN transformer block: LayerNorm -> MHA -> residual, LN -> MLP -> residual.

    Pre-LN (LayerNorm before each sub-layer) trains stably at lr 0.001
    without warm-up.

    Args:
        embed_dim: embedding dimension.
        n_heads: number of attention heads.
        mlp_ratio: hidden-dim multiplier for the MLP.
        dropout: dropout probability.
    """

    def __init__(self, embed_dim: int, n_heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN: normalize before each sub-layer.
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.norm2(x)
        x = x + self.mlp(h)
        return x


class GaitViT(nn.Module):
    """Vision Transformer classifier over 1D sensor patches.

    PatchEmbed -> prepend CLS token -> add learned positional embeddings ->
    pre-LN ConViT blocks -> final LN -> mean-pool patch tokens -> classifier.

    Args:
        n_channels: number of sensor channels in the input (N, C, T).
        n_classes: number of activity classes.
        patch_len: length of each patch (T=128, patch_len=16 -> 8 patches).
        embed_dim: transformer embedding dimension.
        depth: number of ConViT blocks.
        n_heads: number of attention heads.
    """

    def __init__(self, n_channels: int, n_classes: int, patch_len: int = 16,
                 embed_dim: int = 128, depth: int = 4, n_heads: int = 4) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed1d(n_channels, patch_len, embed_dim)
        # n_patches = T // patch_len (for T=128, patch_len=16 -> 8)
        n_patches = 128 // patch_len

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            ConViTBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        # Initialize positional embeddings and CLS token.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding: (N, C, T) -> (N, n_patches, embed_dim)
        h = self.patch_embed(x)
        n_patches = h.size(1)

        # Prepend CLS token and add positional embeddings.
        cls = self.cls_token.expand(h.size(0), -1, -1)  # (N, 1, embed_dim)
        h = torch.cat([cls, h], dim=1)  # (N, n_patches+1, embed_dim)
        h = h + self.pos_embed[:, : n_patches + 1, :]

        # Transformer blocks.
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Mean-pool over patch tokens (exclude CLS at index 0).
        pooled = h[:, 1:, :].mean(dim=1)  # (N, embed_dim)
        return self.head(pooled)
