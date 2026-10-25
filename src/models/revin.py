"""Reversible instance normalization for multivariate forecasting."""
from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Normalize each sample/variable over time and later restore its scale.

    The statistics belong to the current input window, so they are deliberately
    kept outside ``state_dict`` and are refreshed on every ``norm`` call.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self._mean: torch.Tensor | None = None
        self._stdev: torch.Tensor | None = None

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [B, L, C], got {tuple(x.shape)}")
        self._mean = x.mean(dim=1, keepdim=True).detach()
        # unbiased=False is intentional: a length-one window remains defined.
        self._stdev = x.var(dim=1, keepdim=True, unbiased=False).add(self.eps).sqrt().detach()
        x = (x - self._mean) / self._stdev
        return x * self.weight + self.bias if self.affine else x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self._mean is None or self._stdev is None:
            raise RuntimeError("Call normalize() before denormalize().")
        if self.affine:
            x = (x - self.bias) / self.weight
        return x * self._stdev + self._mean

    def forward(self, x: torch.Tensor, mode: str = "norm") -> torch.Tensor:
        if mode == "norm":
            return self.normalize(x)
        if mode == "denorm":
            return self.denormalize(x)
        raise ValueError("mode must be 'norm' or 'denorm'")
