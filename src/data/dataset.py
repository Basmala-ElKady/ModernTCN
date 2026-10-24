"""Sliding-window datasets and dataloaders for ETTh1 / ETTm1.

Each item: x [seq_len, C], y [pred_len, C]  (normalised, float32).
Batches are therefore x [B, L, C] and y [B, H, C], matching the shared model API.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .preprocessing import StandardScaler, load_ett, split_and_normalize


class WindowDataset(Dataset):
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data = torch.from_numpy(np.ascontiguousarray(data))
        self.seq_len, self.pred_len = seq_len, pred_len
        self.n = len(data) - seq_len - pred_len + 1
        if self.n <= 0:
            raise ValueError("Split too short for the requested seq_len + pred_len")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        s = i + self.seq_len
        return self.data[i:s], self.data[s : s + self.pred_len]


def build_datasets(
    csv_path: str, dataset: str, seq_len: int = 336, pred_len: int = 96
) -> tuple[dict[str, WindowDataset], StandardScaler]:
    data, _ = load_ett(csv_path)
    splits, scaler = split_and_normalize(data, dataset, seq_len)
    return {k: WindowDataset(v, seq_len, pred_len) for k, v in splits.items()}, scaler


def build_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Config keys used: dataset, data_dir, input_length, forecast_horizon,
    batch_size, (optional) num_workers, seed."""
    csv_path = os.path.join(config["data_dir"], f"{config['dataset']}.csv")
    sets, _ = build_datasets(
        csv_path, config["dataset"], config["input_length"], config["forecast_horizon"]
    )
    g = torch.Generator().manual_seed(config.get("seed", 0))
    nw = config.get("num_workers", 0)
    bs = config["batch_size"]
    return {
        "train": DataLoader(sets["train"], bs, shuffle=True, generator=g, num_workers=nw),
        "val": DataLoader(sets["val"], bs, shuffle=False, num_workers=nw),
        "test": DataLoader(sets["test"], bs, shuffle=False, num_workers=nw),
    }
