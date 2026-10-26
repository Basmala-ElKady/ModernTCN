"""Failure case analysis: TCN degradation on long-horizon ETTm1 forecasting.

Assignment Requirement 6:
"Find a setting in which your model clearly fails and explain why.
The missing P wave in Lab 4, Part D, is the kind of failure meant here."

Demonstrates and quantifies where the Lab 4 TCN baseline fails drastically
relative to ModernTCN on ETTm1 with a 336-step forecast horizon.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.data.dataset import build_dataloaders
from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed


def compute_per_window_mse(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    """Compute per-window mean squared error across all test samples.

    Returns:
        1D numpy array of shape [num_test_windows].
    """
    model = model.to(device)
    model.eval()
    errors: list[float] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            pred = model(x)
            per_window = (pred - y).square().mean(dim=(1, 2))
            errors.extend(per_window.cpu().numpy().tolist())

    return np.asarray(errors, dtype=np.float64)


def compare_window_errors(
    tcn_errors: np.ndarray,
    moderntcn_errors: np.ndarray,
    dataset: str = "ETTm1",
    horizon: int = 336,
    seed: int = 42,
    top_k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Compare window errors between TCN and ModernTCN.

    Returns:
        tuple containing:
            - df: Full DataFrame of per-window errors
            - worst_windows: Top K worst windows for TCN (largest error gap)
            - summary: Aggregate failure metrics dictionary
    """
    if len(tcn_errors) != len(moderntcn_errors):
        raise ValueError(
            f"Mismatched test windows count: TCN {len(tcn_errors)} vs ModernTCN {len(moderntcn_errors)}"
        )

    tcn_minus_modern = tcn_errors - moderntcn_errors
    ratio = tcn_errors / np.clip(moderntcn_errors, 1e-12, None)

    df = pd.DataFrame({
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "test_window_index": np.arange(len(tcn_errors)),
        "tcn_mse": tcn_errors,
        "moderntcn_mse": moderntcn_errors,
        "tcn_minus_moderntcn_mse": tcn_minus_modern,
        "tcn_error_ratio": ratio,
    })

    worst_windows = (
        df.sort_values("tcn_minus_moderntcn_mse", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )

    total_windows = len(df)
    windows_tcn_worse = int((tcn_minus_modern > 0).sum())

    summary = {
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "total_test_windows": total_windows,
        "windows_tcn_worse": windows_tcn_worse,
        "percent_tcn_worse": 100.0 * windows_tcn_worse / max(total_windows, 1),
        "tcn_mean_mse": float(np.mean(tcn_errors)),
        "moderntcn_mean_mse": float(np.mean(moderntcn_errors)),
        "mean_mse_gap": float(np.mean(tcn_minus_modern)),
        "max_error_ratio": float(np.max(ratio)),
    }

    return df, worst_windows, summary


def run_failure_analysis(
    data_dir: str | Path,
    tcn_checkpoint: str | Path | None,
    moderntcn_checkpoint: str | Path | None,
    output_dir: str | Path = "results/failure_analysis",
    dataset: str = "ETTm1",
    horizon: int = 336,
    seed: int = 42,
    device_name: str = "cpu",
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Execute failure analysis comparing TCN and ModernTCN test errors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    common_cfg = {
        "data_dir": str(data_dir),
        "dataset": dataset,
        "input_length": 336,
        "forecast_horizon": horizon,
        "batch_size": 32,
        "num_variables": 7,
        "seed": seed,
    }

    tcn_cfg = {
        **common_cfg,
        "model_name": "tcn",
        "tcn": {
            "hidden_channels": 64,
            "kernel_size": 3,
            "dropout": 0.1,
            "use_revin": False,
            "revin_affine": True,
        },
    }

    modern_cfg = {
        **common_cfg,
        "model_name": "moderntcn",
        "moderntcn": {
            "embed_dim": 64,
            "depth": 3,
            "kernel_size": 31,
            "small_kernel_size": 5,
            "patch_size": 16,
            "patch_stride": 8,
            "ffn_ratio": 2.0,
            "dropout": 0.1,
            "use_revin": True,
            "revin_affine": True,
        },
    }

    set_seed(seed)
    loaders = build_dataloaders(tcn_cfg)

    # Build and load TCN
    tcn_model = build_model(tcn_cfg)
    if tcn_checkpoint and Path(tcn_checkpoint).exists():
        load_checkpoint(tcn_checkpoint, tcn_model, device=str(device))
    tcn_errors = compute_per_window_mse(tcn_model, loaders["test"], device)

    # Build and load ModernTCN
    modern_model = build_model(modern_cfg)
    if moderntcn_checkpoint and Path(moderntcn_checkpoint).exists():
        load_checkpoint(moderntcn_checkpoint, modern_model, device=str(device))
    modern_errors = compute_per_window_mse(modern_model, loaders["test"], device)

    df, worst, summary = compare_window_errors(
        tcn_errors,
        modern_errors,
        dataset=dataset,
        horizon=horizon,
        seed=seed,
    )

    df.to_csv(output_dir / "ettm1_h336_window_errors.csv", index=False)
    worst.to_csv(output_dir / "worst_tcn_windows.csv", index=False)

    return df, worst, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Failure case analysis: TCN vs ModernTCN")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--tcn-ckpt", required=True)
    parser.add_argument("--modern-ckpt", required=True)
    parser.add_argument("--output-dir", default="results/failure_analysis")
    parser.add_argument("--dataset", default="ETTm1")
    parser.add_argument("--horizon", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    _, _, summary = run_failure_analysis(
        data_dir=args.data_dir,
        tcn_checkpoint=args.tcn_ckpt,
        moderntcn_checkpoint=args.modern_ckpt,
        output_dir=args.output_dir,
        dataset=args.dataset,
        horizon=args.horizon,
        seed=args.seed,
        device_name=args.device,
    )
    print("Failure analysis completed. Summary:")
    print(summary)


if __name__ == "__main__":
    main()
