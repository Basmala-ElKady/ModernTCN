"""ETT preprocessing: loading, chronological split borders, train-only scaler.

Protocol (standard Informer / ModernTCN split, feature mode 'M' = all 7 variables):
  ETTh1: 12 / 4 / 4 months -> 8640 / 2880 / 2880 rows   (first 14400 of 17420 rows)
  ETTm1: 12 / 4 / 4 months -> 34560 / 11520 / 11520 rows (first 57600 of 69680 rows)
Months are 30 days (standard ETT convention). Rows after the test segment are unused.

Validation and test segments are extended backwards by `seq_len` rows so the first
target window of each split begins exactly at the split boundary. Those extra rows
are INPUT context only; no target value ever lies in an earlier split.
The scaler is fitted on the training rows only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MONTH_H = 30 * 24
SPLIT_ROWS = {  # (train, val, test)
    "ETTh1": (12 * MONTH_H, 4 * MONTH_H, 4 * MONTH_H),
    "ETTm1": (12 * MONTH_H * 4, 4 * MONTH_H * 4, 4 * MONTH_H * 4),
}
NUM_VARIABLES = 7  # HUFL, HULL, MUFL, MULL, LUFL, LULL, OT


class StandardScaler:
    """Per-variable z-score. fit() must only ever see training rows."""

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, data: np.ndarray) -> "StandardScaler":
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + self.eps
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        return ((data - self.mean) / self.std).astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


def load_ett(csv_path: str) -> tuple[np.ndarray, list[str]]:
    """Return (values [T, 7] float32, column names). Drops the 'date' column."""
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "date"]
    return df[cols].to_numpy(dtype=np.float32), cols


def get_split_borders(dataset: str, seq_len: int = 336) -> dict[str, tuple[int, int]]:
    """Row ranges [start, end) of each split, including the seq_len lookback."""
    if dataset not in SPLIT_ROWS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {list(SPLIT_ROWS)}")
    n_tr, n_va, n_te = SPLIT_ROWS[dataset]
    return {
        "train": (0, n_tr),
        "val": (n_tr - seq_len, n_tr + n_va),
        "test": (n_tr + n_va - seq_len, n_tr + n_va + n_te),
    }


def split_and_normalize(
    data: np.ndarray, dataset: str, seq_len: int = 336
) -> tuple[dict[str, np.ndarray], StandardScaler]:
    """Chronological split; scaler fitted on the training rows ONLY."""
    expected = sum(SPLIT_ROWS[dataset])
    if len(data) < expected:
        raise ValueError(f"{dataset}: need >= {expected} rows, got {len(data)}")
    borders = get_split_borders(dataset, seq_len)
    tr0, tr1 = borders["train"]
    scaler = StandardScaler().fit(data[tr0:tr1])
    splits = {k: scaler.transform(data[a:b]) for k, (a, b) in borders.items()}
    return splits, scaler
