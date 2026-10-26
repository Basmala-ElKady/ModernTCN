"""Effective Receptive Field (ERF) analysis via input gradient autograd.

Topic 6 requirement (d):
"Measure the effective receptive field of the TCN and of your ModernTCN-style
model with the autograd method of Lab 4, and plot input sensitivity against lag."
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models.factory import build_model
from src.utils.checkpoint import load_checkpoint


def compute_effective_receptive_field(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_positions: Sequence[int] | None = None,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Compute ERF sensitivity across input timesteps using PyTorch autograd.

    Args:
        model: Trained or initialized forecasting model.
        x: Input tensor of shape [1, seq_len, num_variables].
        target_positions: Forecast horizon indices to evaluate. If None, evaluates
            [0, H//4, H//2, 3H//4, H-1].
        device: Device to run computation on.

    Returns:
        tuple containing:
            - erf: Mean absolute gradient at each input timestep [seq_len]
            - erf_normalized: Normalized ERF (scaled to max 1.0)
            - summary: Dictionary with steps needed for 50% and 90% gradient mass
    """
    model = model.to(device)
    model.eval()

    if x.dim() == 2:
        x = x.unsqueeze(0)
    x = x.to(device=device, dtype=torch.float32)
    x = x.detach().clone()
    x.requires_grad_(True)

    prediction = model(x)
    seq_len = x.shape[1]
    pred_len = prediction.shape[1]
    num_variables = prediction.shape[2]

    if target_positions is None:
        target_positions = sorted(set([
            0,
            pred_len // 4,
            pred_len // 2,
            (3 * pred_len) // 4,
            pred_len - 1,
        ]))

    gradient_profiles: list[np.ndarray] = []

    for pos in target_positions:
        for var_idx in range(num_variables):
            grad = torch.autograd.grad(
                outputs=prediction[0, pos, var_idx],
                inputs=x,
                retain_graph=True,
                create_graph=False,
            )[0]
            # Mean absolute gradient across input variables
            profile = grad[0].abs().mean(dim=-1).detach().cpu().numpy()
            gradient_profiles.append(profile)

    erf = np.mean(np.stack(gradient_profiles, axis=0), axis=0)
    max_val = float(erf.max())
    erf_normalized = erf / max_val if max_val > 0 else erf

    # Calculate steps for 50% and 90% cumulative gradient mass
    total_mass = float(erf.sum())
    summary: dict[str, int] = {}
    if total_mass > 0:
        # Lag ordering: from latest input timestep (lag 0) back to earliest (lag seq_len-1)
        ordered_by_lag = erf[::-1]
        cumsum_pct = np.cumsum(ordered_by_lag) / total_mass
        for pct in (0.50, 0.90):
            idx = int(np.searchsorted(cumsum_pct, pct))
            idx = min(idx, seq_len - 1)
            summary[f"steps_for_{int(pct*100)}pct_gradient_mass"] = idx + 1
    else:
        summary["steps_for_50pct_gradient_mass"] = 0
        summary["steps_for_90pct_gradient_mass"] = 0

    return erf, erf_normalized, summary


def plot_receptive_field(
    df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Effective Receptive Field (Autograd Input Sensitivity)",
) -> None:
    """Plot normalized gradient profile against steps before forecast."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 4))
    plt.plot(df["steps_before_forecast"], df["normalized_gradient"], color="#1f77b4", lw=2)
    plt.xlabel("Steps before forecast (lag)")
    plt.ylabel("Normalized mean absolute gradient")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()


def run_erf_analysis(
    model: torch.nn.Module | None = None,
    checkpoint_path: str | Path | None = None,
    config: dict | None = None,
    output_dir: str | Path = "results/erf_analysis",
    file_prefix: str = "moderntcn_etth1_h96",
    device_name: str = "cpu",
    seq_len: int = 336,
    num_variables: int = 7,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Execute complete ERF analysis and export CSV and PNG artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name)

    if model is None:
        if config is None:
            config = {
                "model_name": "moderntcn",
                "input_length": seq_len,
                "forecast_horizon": 96,
                "num_variables": num_variables,
                "moderntcn": {
                    "embed_dim": 64,
                    "depth": 3,
                    "kernel_size": 15,
                    "small_kernel_size": 5,
                    "patch_size": 16,
                    "patch_stride": 8,
                    "ffn_ratio": 2.0,
                    "dropout": 0.1,
                    "use_revin": False,
                    "revin_affine": True,
                },
            }
        model = build_model(config)
        if checkpoint_path and Path(checkpoint_path).exists():
            load_checkpoint(checkpoint_path, model, device=str(device))

    # Standard input sample
    dummy_x = torch.randn(1, seq_len, num_variables, device=device)

    erf, erf_normalized, summary = compute_effective_receptive_field(
        model=model,
        x=dummy_x,
        device=device,
    )

    df = pd.DataFrame({
        "input_timestep": np.arange(seq_len),
        "steps_before_forecast": np.arange(seq_len - 1, -1, -1),
        "mean_absolute_gradient": erf,
        "normalized_gradient": erf_normalized,
    })

    csv_path = output_dir / f"{file_prefix}_erf.csv"
    png_path = output_dir / f"{file_prefix}_erf.png"
    df.to_csv(csv_path, index=False)
    plot_receptive_field(df, png_path, title=f"ERF Profile: {file_prefix}")

    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Effective Receptive Field via Autograd")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint path")
    parser.add_argument("--model", default="moderntcn", choices=["moderntcn", "tcn", "dlinear"])
    parser.add_argument("--output-dir", default="results/erf_analysis")
    parser.add_argument("--prefix", default="erf_analysis")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = {
        "model_name": args.model,
        "input_length": 336,
        "forecast_horizon": 96,
        "num_variables": 7,
    }
    if args.model == "moderntcn":
        config["moderntcn"] = {
            "embed_dim": 64,
            "depth": 3,
            "kernel_size": 15,
            "small_kernel_size": 5,
            "patch_size": 16,
            "patch_stride": 8,
            "ffn_ratio": 2.0,
            "dropout": 0.1,
            "use_revin": False,
            "revin_affine": True,
        }
    elif args.model == "tcn":
        config["tcn"] = {
            "hidden_channels": 64,
            "kernel_size": 3,
            "dropout": 0.1,
            "use_revin": False,
            "revin_affine": True,
        }

    run_erf_analysis(
        checkpoint_path=args.checkpoint,
        config=config,
        output_dir=args.output_dir,
        file_prefix=args.prefix,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
