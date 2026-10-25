from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["MovingAverageDecomposition", "DLinear", "count_parameters"]


# Return the number of (trainable) parameters of "model".
def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


# Split a series into trend (moving average) and seasonal (remainder).
class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}.")
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2

    # Decompose x of shape [B, L, C].
    # Returns: (seasonal, trend), each of shape [B, L, C].
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape [B, L, C], got {tuple(x.shape)}.")
        xt = x.transpose(1, 2)  # [B, C, L]: pooling runs over the last axis
        if self.pad > 0:
            xt = F.pad(xt, (self.pad, self.pad), mode="replicate")
        trend = F.avg_pool1d(xt, kernel_size=self.kernel_size, stride=1).transpose(1, 2)
        return x - trend, trend

# DLinear: decomposition + one linear layer per component.
class DLinear(nn.Module):
    def __init__(
        self,
        input_length: int = 336,
        forecast_horizon: int = 96,
        moving_avg_kernel: int = 25,
        individual: bool = False,
        num_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        if input_length < 1 or forecast_horizon < 1:
            raise ValueError("input_length and forecast_horizon must be positive.")
        if individual and (num_features is None or num_features < 1):
            raise ValueError("individual=True requires a positive num_features.")

        self.input_length = input_length
        self.forecast_horizon = forecast_horizon
        self.individual = individual
        self.num_features = num_features
        self.decomposition = MovingAverageDecomposition(moving_avg_kernel)

        # Weights are stored as [C?, H, L]; the leading dim exists only if individual.
        lead: Tuple[int, ...] = (num_features,) if individual else ()
        self.seasonal_weight = nn.Parameter(torch.empty(*lead, forecast_horizon, input_length))
        self.seasonal_bias = nn.Parameter(torch.empty(*lead, forecast_horizon))
        self.trend_weight = nn.Parameter(torch.empty(*lead, forecast_horizon, input_length))
        self.trend_bias = nn.Parameter(torch.empty(*lead, forecast_horizon))
        self.reset_parameters()

    # init
    # Uniform(-1/sqrt(L), 1/sqrt(L)) init, the same scheme as "nn.Linear".
    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.input_length)
        for p in (self.seasonal_weight, self.seasonal_bias, self.trend_weight, self.trend_bias):
            nn.init.uniform_(p, -bound, bound)

    # helpers
    # Linear map along time. z: [B, C, L] -> [B, C, H].
    def _project(self, z: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        if self.individual:
            return torch.einsum("bcl,chl->bch", z, weight) + bias
        return F.linear(z, weight, bias)  # broadcast over B and C

    # forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input of shape [B, L, C], got {tuple(x.shape)}.")
        if x.shape[1] != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, got {x.shape[1]}.")
        if self.individual and x.shape[2] != self.num_features:
            raise ValueError(f"Expected {self.num_features} features, got {x.shape[2]}.")

        seasonal, trend = self.decomposition(x) # [B, L, C] each
        seasonal = seasonal.transpose(1, 2) # [B, C, L]
        trend = trend.transpose(1, 2) # [B, C, L]
        out = self._project(seasonal, self.seasonal_weight, self.seasonal_bias) + self._project(trend, self.trend_weight, self.trend_bias) # [B, C, H]
        return out.transpose(1, 2) # [B, H, C]

    # utilities
    # Number of parameters of this model.
    def num_parameters(self, trainable_only: bool = True) -> int:
        return count_parameters(self, trainable_only)

    # Closed-form parameter count: 2 * (L*H + H) (times C if individual).
    @staticmethod
    def expected_num_parameters(
        input_length: int,
        forecast_horizon: int,
        individual: bool = False,
        num_features: int = 1,
    ) -> int:
        per_branch = input_length * forecast_horizon + forecast_horizon
        return 2 * per_branch * (num_features if individual else 1)

    # Configuration and parameter count, for logging / the experiment record.
    def config_summary(self) -> dict:
        return {
            "model": "dlinear",
            "input_length": self.input_length,
            "forecast_horizon": self.forecast_horizon,
            "moving_avg_kernel": self.decomposition.kernel_size,
            "individual": self.individual,
            "num_features": self.num_features,
            "num_parameters": self.num_parameters(),
        }

    #  Build from the shared config dict.
    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "DLinear":
        return cls(
            input_length = config.get("input_length", 336),
            forecast_horizon = config["forecast_horizon"],
            moving_avg_kernel = config.get("moving_avg_kernel", 25),
            individual = config.get("individual", False),
            num_features = config.get("num_features", config.get("num_variables")),
        )