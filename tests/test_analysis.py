"""Unit tests for analysis modules: kernel ablation, ERF, and failure analysis."""
from pathlib import Path
import numpy as np
import pytest
import torch

from src.analysis.ablation import aggregate_kernel_results, count_parameters
from src.analysis.failure_analysis import compare_window_errors, compute_per_window_mse
from src.analysis.receptive_field import compute_effective_receptive_field, plot_receptive_field
from src.models.moderntcn import ModernTCN
from src.models.tcn import TCN


def test_kernel_parameter_count_comparable():
    """Verify kernel sizes 15, 31, and 51 have parameter counts within 10%."""
    base_kwargs = {
        "num_variables": 7,
        "input_length": 336,
        "forecast_horizon": 96,
        "patch_size": 16,
        "patch_stride": 8,
        "embed_dim": 64,
        "depth": 3,
        "small_kernel_size": 5,
        "ffn_ratio": 2.0,
        "dropout": 0.1,
        "use_revin": True,
        "revin_affine": True,
    }

    m15 = ModernTCN(**{**base_kwargs, "kernel_size": 15})
    m31 = ModernTCN(**{**base_kwargs, "kernel_size": 31})
    m51 = ModernTCN(**{**base_kwargs, "kernel_size": 51})

    p15 = count_parameters(m15)
    p31 = count_parameters(m31)
    p51 = count_parameters(m51)

    # Base model has ~65,000 parameters
    # Difference should be under 10% of p31
    assert abs(p15 - p31) / p31 < 0.10
    assert abs(p51 - p31) / p31 < 0.10


def test_aggregate_kernel_results():
    """Verify aggregation handles multiple seeds and calculates sample std (ddof=1)."""
    raw_rows = [
        {"dataset": "ETTh1", "horizon": 96, "kernel_size": 15, "test_mse": 0.380, "test_mae": 0.400, "status": "completed"},
        {"dataset": "ETTh1", "horizon": 96, "kernel_size": 15, "test_mse": 0.378, "test_mae": 0.402, "status": "completed"},
        {"dataset": "ETTh1", "horizon": 96, "kernel_size": 15, "test_mse": 0.379, "test_mae": 0.401, "status": "completed"},
        {"dataset": "ETTh1", "horizon": 96, "kernel_size": 31, "test_mse": 0.382, "test_mae": 0.405, "status": "completed"},
        {"dataset": "ETTh1", "horizon": 96, "kernel_size": 31, "test_mse": 0.380, "test_mae": 0.403, "status": "completed"},
    ]

    summary = aggregate_kernel_results(raw_rows)
    assert len(summary) == 2

    k15 = next(s for s in summary if s["kernel_size"] == 15)
    assert k15["runs"] == 3
    assert np.isclose(k15["mse_mean"], 0.379, atol=1e-5)
    assert k15["mse_std"] > 0.0

    k31 = next(s for s in summary if s["kernel_size"] == 31)
    assert k31["runs"] == 2
    assert np.isclose(k31["mse_mean"], 0.381, atol=1e-5)


def test_compute_effective_receptive_field_moderntcn():
    """Verify ERF autograd calculation produces normalized sensitivity profile."""
    model = ModernTCN(
        num_variables=3,
        input_length=64,
        forecast_horizon=16,
        patch_size=8,
        patch_stride=4,
        embed_dim=16,
        depth=1,
        kernel_size=7,
        small_kernel_size=3,
        use_revin=False,
    )
    x = torch.randn(1, 64, 3)

    erf, erf_norm, summary = compute_effective_receptive_field(
        model=model,
        x=x,
        target_positions=[0, 8, 15],
        device="cpu",
    )

    assert len(erf) == 64
    assert len(erf_norm) == 64
    assert np.all(erf >= 0)
    assert np.isclose(erf_norm.max(), 1.0)
    assert "steps_for_50pct_gradient_mass" in summary
    assert "steps_for_90pct_gradient_mass" in summary
    assert 1 <= summary["steps_for_50pct_gradient_mass"] <= 64
    assert summary["steps_for_50pct_gradient_mass"] <= summary["steps_for_90pct_gradient_mass"]


def test_compare_window_errors():
    """Verify window error comparison and identification of worst failure cases."""
    # 5 test windows
    tcn_errs = np.array([1.5, 0.4, 2.8, 0.5, 3.2])
    modern_errs = np.array([0.3, 0.35, 0.4, 0.45, 0.5])

    df, worst, summary = compare_window_errors(
        tcn_errors=tcn_errs,
        moderntcn_errors=modern_errs,
        dataset="ETTm1",
        horizon=336,
        seed=42,
        top_k=2,
    )

    assert len(df) == 5
    assert len(worst) == 2
    # The largest gap is index 4: 3.2 - 0.5 = 2.7
    assert worst.iloc[0]["test_window_index"] == 4
    # The second largest gap is index 2: 2.8 - 0.4 = 2.4
    assert worst.iloc[1]["test_window_index"] == 2

    assert summary["total_test_windows"] == 5
    assert summary["windows_tcn_worse"] == 5
    assert summary["percent_tcn_worse"] == 100.0


def test_plot_receptive_field(tmp_path: Path):
    """Verify plot generation writes a valid image file."""
    import pandas as pd
    df = pd.DataFrame({
        "input_timestep": np.arange(10),
        "steps_before_forecast": np.arange(9, -1, -1),
        "mean_absolute_gradient": np.linspace(0.01, 0.1, 10),
        "normalized_gradient": np.linspace(0.1, 1.0, 10),
    })
    out_png = tmp_path / "test_erf.png"
    plot_receptive_field(df, out_png, title="Test Plot")
    assert out_png.exists()
    assert out_png.stat().st_size > 0
