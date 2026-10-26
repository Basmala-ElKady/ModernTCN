"""Kernel size ablation study for ModernTCN depthwise convolution.

Topic 6 requirement (e):
"Ablation: at least three kernel sizes for the large depthwise convolution,
at comparable parameter count. Does accuracy follow the effective receptive field?"

Evaluates ModernTCN with depthwise kernel sizes [15, 31, 51] on ETTh1 (horizon 96)
across multiple seeds (e.g., 42, 123, 2026).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

from src.data.dataset import build_dataloaders
from src.models.factory import build_model
from src.training.evaluate import evaluate_model
from src.training.train import train_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed

RAW_COLUMNS = (
    "model",
    "dataset",
    "horizon",
    "kernel_size",
    "seed",
    "status",
    "best_epoch",
    "best_val_mse",
    "test_mse",
    "test_mae",
    "training_seconds",
    "checkpoint",
)

SUMMARY_COLUMNS = (
    "dataset",
    "horizon",
    "kernel_size",
    "runs",
    "mse_mean",
    "mse_std",
    "mae_mean",
    "mae_std",
    "best_val_mse_mean",
)


def sample_std(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1) across independent runs."""
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def aggregate_kernel_results(rows: Iterable[Mapping]) -> list[dict]:
    """Aggregate raw seed rows by kernel size with mean and sample std."""
    grouped: dict[tuple[str, int, int], list[Mapping]] = defaultdict(list)
    for row in rows:
        if row.get("status", "completed") != "completed":
            continue
        key = (str(row["dataset"]), int(row["horizon"]), int(row["kernel_size"]))
        grouped[key].append(row)

    summary: list[dict] = []
    for (dataset, horizon, kernel_size), group in sorted(grouped.items()):
        mse = [float(r["test_mse"]) for r in group]
        mae = [float(r["test_mae"]) for r in group]
        val_mse = [float(r["best_val_mse"]) for r in group if "best_val_mse" in r]
        summary.append({
            "dataset": dataset,
            "horizon": horizon,
            "kernel_size": kernel_size,
            "runs": len(group),
            "mse_mean": float(np.mean(mse)),
            "mse_std": sample_std(mse),
            "mae_mean": float(np.mean(mae)),
            "mae_std": sample_std(mae),
            "best_val_mse_mean": float(np.mean(val_mse)) if val_mse else float("nan"),
        })
    return summary


def run_kernel_ablation(
    data_dir: str | Path,
    output_dir: str | Path,
    kernel_sizes: Sequence[int] = (15, 31, 51),
    seeds: Sequence[int] = (42, 123, 2026),
    dataset: str = "ETTh1",
    horizon: int = 96,
    input_length: int = 336,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    epochs: int = 10,
    patience: int = 3,
    device_name: str = "auto",
) -> tuple[list[dict], list[dict]]:
    """Execute the depthwise kernel size ablation experiment."""
    output_dir = Path(output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    base_config = {
        "data_dir": str(data_dir),
        "dataset": dataset,
        "input_length": input_length,
        "forecast_horizon": horizon,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "device": str(device),
        "num_variables": 7,
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

    raw_rows: list[dict] = []
    total_runs = len(kernel_sizes) * len(seeds)
    run_idx = 0

    for kernel_size in kernel_sizes:
        for seed in seeds:
            run_idx += 1
            ckpt_path = ckpt_dir / f"moderntcn_k{kernel_size}_{dataset}_h{horizon}_seed{seed}.pt"

            run_config = deepcopy(base_config)
            run_config["seed"] = seed
            run_config["moderntcn"]["kernel_size"] = kernel_size

            set_seed(seed)
            loaders = build_dataloaders(run_config)
            model = build_model(run_config)

            param_count = count_parameters(model)
            started = time.time()

            train_info = train_model(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                config=run_config,
                checkpoint_path=ckpt_path,
            )

            load_checkpoint(ckpt_path, model, device=str(device))
            test_metrics = evaluate_model(model, loaders["test"], device)
            elapsed = time.time() - started

            row = {
                "model": "moderntcn",
                "dataset": dataset,
                "horizon": horizon,
                "kernel_size": kernel_size,
                "seed": seed,
                "status": "completed",
                "best_epoch": train_info.get("best_epoch", 0),
                "best_val_mse": train_info.get("best_val_mse", 0.0),
                "test_mse": float(test_metrics["mse"]),
                "test_mae": float(test_metrics["mae"]),
                "training_seconds": round(elapsed, 2),
                "checkpoint": str(ckpt_path),
                "param_count": param_count,
            }
            raw_rows.append(row)

    summary_rows = aggregate_kernel_results(raw_rows)

    # Save CSVs
    raw_csv = output_dir / "kernel_ablation_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[c for c in RAW_COLUMNS if c in raw_rows[0]])
        writer.writeheader()
        writer.writerows([{k: r.get(k, "") for k in writer.fieldnames} for r in raw_rows])

    summary_csv = output_dir / "kernel_ablation_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    return raw_rows, summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="ModernTCN kernel size ablation runner")
    parser.add_argument("--data-dir", default="data", help="Directory containing ETT datasets")
    parser.add_argument("--output-dir", default="results/kernel_ablation", help="Directory for results")
    parser.add_argument("--kernels", nargs="+", type=int, default=[15, 31, 51], help="Kernel sizes to test")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2026], help="Random seeds")
    parser.add_argument("--dataset", default="ETTh1", help="Dataset name")
    parser.add_argument("--horizon", type=int, default=96, help="Forecast horizon")
    parser.add_argument("--device", default="auto", help="Compute device (cuda/cpu/auto)")
    args = parser.parse_args()

    run_kernel_ablation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        kernel_sizes=args.kernels,
        seeds=args.seeds,
        dataset=args.dataset,
        horizon=args.horizon,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
