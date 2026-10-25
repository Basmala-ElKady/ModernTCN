"""Fair ModernTCN RevIN ablation runner and cross-seed result aggregation.

Example (requires ``data/ETTh1.csv``)::

    python -m src.analysis.revin_ablation --data-dir data

The runner trains RevIN-on and RevIN-off independently for every same seed;
it selects checkpoints only by validation MSE and evaluates those checkpoints
once on the test split.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch
import yaml

from src.data.dataset import build_dataloaders
from src.models.moderntcn import build_moderntcn
from src.training.evaluate import evaluate_model
from src.training.train import train_model
from src.utils.checkpoint import load_checkpoint
from src.utils.seed import set_seed


RAW_COLUMNS = ("dataset", "horizon", "use_revin", "seed", "mse", "mae", "best_epoch", "best_val_mse", "checkpoint")
SUMMARY_COLUMNS = (
    "dataset", "horizon", "use_revin", "runs", "mse_mean", "mse_sample_std",
    "mae_mean", "mae_sample_std", "mse_improvement_percent", "mae_improvement_percent",
)


def sample_std(values: Iterable[float]) -> float:
    """Sample standard deviation for reporting across independent seeds."""
    values = list(values)
    if len(values) < 2:
        raise ValueError("Sample standard deviation requires at least two seed results")
    return float(np.std(values, ddof=1))


def percent_improvement(without_revin: float, with_revin: float) -> float | None:
    """Positive means RevIN improved (lowered) an error metric."""
    if without_revin == 0:
        return None
    return 100.0 * (without_revin - with_revin) / without_revin


def aggregate_results(rows: Iterable[Mapping]) -> list[dict]:
    """Aggregate raw seed rows; intentionally uses ddof=1, never scaler stats."""
    grouped: dict[tuple[str, int, bool], list[Mapping]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["horizon"]), bool(row["use_revin"]))].append(row)

    summary: list[dict] = []
    for (dataset, horizon, use_revin), group in sorted(grouped.items()):
        mse = [float(r["mse"]) for r in group]
        mae = [float(r["mae"]) for r in group]
        summary.append({
            "dataset": dataset, "horizon": horizon, "use_revin": use_revin,
            "runs": len(group), "mse_mean": float(np.mean(mse)),
            "mse_sample_std": sample_std(mse), "mae_mean": float(np.mean(mae)),
            "mae_sample_std": sample_std(mae),
            "mse_improvement_percent": None, "mae_improvement_percent": None,
        })

    paired = {(r["dataset"], r["horizon"]): r for r in summary if not r["use_revin"]}
    for row in summary:
        if row["use_revin"] and (base := paired.get((row["dataset"], row["horizon"]))) is not None:
            row["mse_improvement_percent"] = percent_improvement(base["mse_mean"], row["mse_mean"])
            row["mae_improvement_percent"] = percent_improvement(base["mae_mean"], row["mae_mean"])
    return summary


def _write_csv(path: Path, rows: list[Mapping], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _flat_base_config(config_path: Path, data_dir: str) -> dict:
    with config_path.open(encoding="utf-8") as f:
        source = yaml.safe_load(f)
    data, training = source["data"], source["training"]
    return {
        "data_dir": data_dir, "input_length": data["input_length"],
        "batch_size": training["batch_size"], "learning_rate": training["learning_rate"],
        "epochs": training["epochs"], "patience": training["patience"],
        "device": source["experiment"].get("device", "auto"), "num_variables": 7,
        # Record every architecture value rather than relying on source-code
        # defaults, so a future code change cannot silently alter this ablation.
        "moderntcn": {
            "embed_dim": 64, "depth": 3, "kernel_size": 31,
            "small_kernel_size": 5, "patch_size": 16, "patch_stride": 8,
            "ffn_ratio": 2.0, "dropout": 0.1, "revin_affine": True,
        },
    }


def run_ablation(base_config: Mapping, datasets: Iterable[str], horizons: Iterable[int],
                 seeds: Iterable[int], output_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Run matched RevIN-off/on experiments and save raw plus summary artifacts."""
    output_dir = Path(output_dir)
    checkpoints = output_dir / "checkpoints"
    raw: list[dict] = []
    seeds, datasets, horizons = list(seeds), list(datasets), list(horizons)
    if len(seeds) < 2:
        raise ValueError("Use at least two seeds; the required protocol uses 0, 1, 2.")

    for dataset in datasets:
        for horizon in horizons:
            for use_revin in (False, True):
                for seed in seeds:
                    set_seed(seed)
                    config = deepcopy(dict(base_config))
                    config.update({"dataset": dataset, "forecast_horizon": horizon, "seed": seed})
                    modern = dict(config.get("moderntcn", {}))
                    modern["use_revin"] = use_revin
                    config["moderntcn"] = modern
                    label = f"{dataset}_h{horizon}_revin-{str(use_revin).lower()}_seed-{seed}"
                    checkpoint = checkpoints / f"{label}.pt"
                    loaders = build_dataloaders(config)
                    model = build_moderntcn(config)
                    train_result = train_model(model, loaders["train"], loaders["val"], config, checkpoint)
                    device = next(model.parameters()).device
                    load_checkpoint(checkpoint, model, device=str(device))
                    metrics = evaluate_model(model, loaders["test"], device)
                    raw.append({
                        "dataset": dataset, "horizon": horizon, "use_revin": use_revin, "seed": seed,
                        "mse": metrics["mse"], "mae": metrics["mae"],
                        "best_epoch": train_result["best_epoch"], "best_val_mse": train_result["best_val_mse"],
                        "checkpoint": str(checkpoint),
                    })
                    _write_csv(output_dir / "revin_ablation_raw.csv", raw, RAW_COLUMNS)

    summary = aggregate_results(raw)
    _write_csv(output_dir / "revin_ablation_summary.csv", summary, SUMMARY_COLUMNS)
    manifest = {"seeds": seeds, "datasets": datasets, "horizons": horizons,
                "base_config": dict(base_config), "reporting_std_ddof": 1,
                "standard_scaler_ddof": 0}
    (output_dir / "revin_ablation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return raw, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched ModernTCN RevIN ablation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="results/metrics/revin_ablation")
    parser.add_argument("--datasets", nargs="+", default=["ETTh1"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[96, 336])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    missing = [str(Path(args.data_dir) / f"{dataset}.csv") for dataset in args.datasets
               if not (Path(args.data_dir) / f"{dataset}.csv").is_file()]
    if missing:
        parser.error(
            "Missing dataset file(s): " + ", ".join(missing) + ". "
            "Download/copy the ETT CSV file(s) there, or pass their parent folder with --data-dir."
        )
    base = _flat_base_config(Path(args.config), args.data_dir)
    raw, summary = run_ablation(base, args.datasets, args.horizons, args.seeds, args.output_dir)
    print(f"Completed {len(raw)} runs. Results: {Path(args.output_dir).resolve()}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
