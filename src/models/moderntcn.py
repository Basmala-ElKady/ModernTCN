"""A compact, causal ModernTCN-style forecasting model.

This is an independent educational implementation, not a copy of the authors'
code.  It preserves the important ideas: patch embedding, large-kernel
depthwise temporal mixing, a small-kernel re-parameterisation branch, and
pointwise feed-forward layers.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .revin import RevIN


class PatchEmbedding(nn.Module):
    """Turn each univariate series into overlapping learned temporal patches."""

    def __init__(self, patch_size: int, patch_stride: int, embed_dim: int):
        super().__init__()
        if patch_size < 1 or patch_stride < 1:
            raise ValueError("patch_size and patch_stride must be positive")
        self.patch_size, self.patch_stride = patch_size, patch_stride
        self.proj = nn.Conv1d(1, embed_dim, patch_size, stride=patch_stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L] -> [B*C, D, P].  Right padding repeats only the last
        # observed value, so the final patch always contains the last sample.
        b, c, length = x.shape
        right = max(0, self.patch_size - length)
        if length >= self.patch_size:
            right += (self.patch_stride - (length - self.patch_size) % self.patch_stride) % self.patch_stride
        if right:
            x = F.pad(x, (0, right), mode="replicate")
        return self.proj(x.reshape(b * c, 1, x.shape[-1]))


class ChannelLayerNorm(nn.Module):
    """Layer norm over channels independently at every patch position."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class LargeKernelDepthwiseConv(nn.Module):
    """Causal large depthwise convolution plus an active small-kernel branch."""

    def __init__(self, channels: int, kernel_size: int, small_kernel_size: int):
        super().__init__()
        if kernel_size < 1 or small_kernel_size < 1:
            raise ValueError("kernel sizes must be positive")
        if small_kernel_size > kernel_size:
            raise ValueError("small_kernel_size cannot exceed kernel_size")
        self.kernel_size, self.small_kernel_size = kernel_size, small_kernel_size
        self.large = nn.Conv1d(channels, channels, kernel_size, groups=channels)
        # This is deliberately retained at training/evaluation time (D4), not
        # fused away: gradients flow through both branches.
        self.small = nn.Conv1d(channels, channels, small_kernel_size, groups=channels)

    @staticmethod
    def _causal(conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        return conv(F.pad(x, (conv.kernel_size[0] - 1, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._causal(self.large, x) + self._causal(self.small, x)


class ModernTCNBlock(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, small_kernel_size: int,
                 ffn_ratio: float, dropout: float):
        super().__init__()
        hidden = max(1, int(round(embed_dim * ffn_ratio)))
        self.norm1 = ChannelLayerNorm(embed_dim)
        self.token_mixer = LargeKernelDepthwiseConv(embed_dim, kernel_size, small_kernel_size)
        self.norm2 = ChannelLayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(embed_dim, hidden, 1), nn.GELU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, embed_dim, 1), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class ModernTCN(nn.Module):
    """Forecast independently embedded variables with a shared ModernTCN trunk.

    Input and output use the project-wide convention ``[B, time, variables]``.
    """

    def __init__(self, num_variables: int, input_length: int, forecast_horizon: int,
                 embed_dim: int = 64, depth: int = 3, kernel_size: int = 31,
                 small_kernel_size: int = 5, patch_size: int = 16,
                 patch_stride: int = 8, ffn_ratio: float = 2.0, dropout: float = 0.1,
                 use_revin: bool = True, revin_affine: bool = True):
        super().__init__()
        if min(num_variables, input_length, forecast_horizon, embed_dim, depth) < 1:
            raise ValueError("model dimensions must be positive")
        self.C, self.L, self.H = num_variables, input_length, forecast_horizon
        self.use_revin = use_revin
        self.revin = RevIN(num_variables, affine=revin_affine) if use_revin else None
        self.patch_embed = PatchEmbedding(patch_size, patch_stride, embed_dim)
        self.blocks = nn.Sequential(*[
            ModernTCNBlock(embed_dim, kernel_size, small_kernel_size, ffn_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = ChannelLayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, forecast_horizon)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return causal patch features shaped ``[B, C, D, P]`` for ERF tests."""
        if x.ndim != 3 or x.shape[1:] != (self.L, self.C):
            raise ValueError(f"Expected [B, {self.L}, {self.C}], got {tuple(x.shape)}")
        if self.revin is not None:
            x = self.revin.normalize(x)
        b = x.shape[0]
        h = self.blocks(self.patch_embed(x.transpose(1, 2)))
        h = self.norm(h)
        return h.reshape(b, self.C, h.shape[1], h.shape[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)[:, :, :, -1]       # [B, C, D]
        y = self.head(h).transpose(1, 2)                 # [B, H, C]
        return self.revin.denormalize(y) if self.revin is not None else y

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: Mapping) -> "ModernTCN":
        m = config.get("moderntcn", config.get("modern_tcn", {}))
        return cls(
            num_variables=config["num_variables"], input_length=config["input_length"],
            forecast_horizon=config["forecast_horizon"], embed_dim=m.get("embed_dim", 64),
            depth=m.get("depth", 3), kernel_size=m.get("kernel_size", 31),
            small_kernel_size=m.get("small_kernel_size", 5), patch_size=m.get("patch_size", 16),
            patch_stride=m.get("patch_stride", 8), ffn_ratio=m.get("ffn_ratio", 2.0),
            dropout=m.get("dropout", 0.1), use_revin=m.get("use_revin", True),
            revin_affine=m.get("revin_affine", True),
        )


def build_moderntcn(config: Mapping) -> ModernTCN:
    return ModernTCN.from_config(config)
