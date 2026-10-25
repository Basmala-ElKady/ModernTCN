import numpy as np
import pytest

from src.analysis.revin_ablation import aggregate_results, percent_improvement, sample_std


def _rows():
    return [
        {"dataset": "ETTh1", "horizon": 96, "use_revin": False, "seed": i, "mse": v, "mae": v / 2}
        for i, v in enumerate((3.0, 5.0, 7.0))
    ] + [
        {"dataset": "ETTh1", "horizon": 96, "use_revin": True, "seed": i, "mse": v, "mae": v / 2}
        for i, v in enumerate((2.0, 4.0, 6.0))
    ]


def test_sample_std_uses_ddof_one():
    values = [3.0, 5.0, 7.0]
    assert sample_std(values) == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(ValueError):
        sample_std([1.0])


def test_summary_and_relative_improvement():
    summary = aggregate_results(_rows())
    disabled = next(row for row in summary if not row["use_revin"])
    enabled = next(row for row in summary if row["use_revin"])
    assert disabled["mse_sample_std"] == pytest.approx(2.0)
    assert enabled["mse_improvement_percent"] == pytest.approx(20.0)
    assert enabled["mae_improvement_percent"] == pytest.approx(20.0)
    assert percent_improvement(0.0, 1.0) is None
