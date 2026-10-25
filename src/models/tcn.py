"""TCN baseline (Bai, Kolter, Koltun 2018): stacked residual blocks of causal,
dilated 1D convolutions. Written from scratch.

External interface (shared by all models):
    x [B, L, C]  ->  predictions [B, H, C]
Forecast head: features at the LAST time step (whose causal receptive field
covers the whole window) -> Linear -> H*C values.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Module):
    """Conv1d with left-only padding: output[t] depends on input[<= t]."""

    def __init__(self, c_in: int, c_out: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(c_in, c_out, kernel_size, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T]
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(c_in, c_out, kernel_size, dilation)
        self.conv2 = CausalConv1d(c_out, c_out, kernel_size, dilation)
        self.drop = nn.Dropout(dropout)
        self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        out = self.drop(F.relu(self.conv1(x)))
        out = self.drop(F.relu(self.conv2(out)))
        return F.relu(out + self.down(x))


def receptive_field(kernel_size: int, num_levels: int) -> int:
    """Two convs per block, dilation 2^i: RF = 1 + 2(k-1)(2^n - 1)."""
    return 1 + 2 * (kernel_size - 1) * (2**num_levels - 1)


def levels_to_cover(kernel_size: int, input_length: int) -> int:
    n = 1
    while receptive_field(kernel_size, n) < input_length:
        n += 1
    return n


class TCN(nn.Module):
    def __init__(self, num_variables, input_length, forecast_horizon,
                 hidden_channels=64, kernel_size=3, num_levels=None, dropout=0.1):
        super().__init__()
        self.H, self.C = forecast_horizon, num_variables
        if num_levels is None:  # smallest depth whose RF covers the input window
            num_levels = levels_to_cover(kernel_size, input_length)
        self.num_levels = num_levels
        self.receptive_field = receptive_field(kernel_size, num_levels)
        blocks, c_in = [], num_variables
        for i in range(num_levels):
            blocks.append(TemporalBlock(c_in, hidden_channels, kernel_size, 2**i, dropout))
            c_in = hidden_channels
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_channels, forecast_horizon * num_variables)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """x [B, L, C] -> per-step features [B, hidden, L] (used by causality test / ERF)."""
        return self.network(x.transpose(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.forward_features(x)[:, :, -1]            # [B, hidden]
        return self.head(h).view(-1, self.H, self.C)      # [B, H, C]


def build_tcn(config: dict) -> TCN:
    """Reads config['tcn'] (TCN hyperparameters) plus shared shape keys."""
    m = config.get("tcn", {})
    return TCN(
        num_variables=config["num_variables"],
        input_length=config["input_length"],
        forecast_horizon=config["forecast_horizon"],
        hidden_channels=m.get("hidden_channels", 64),
        kernel_size=m.get("kernel_size", 3),
        num_levels=m.get("num_levels"),
        dropout=m.get("dropout", 0.1),
    )
