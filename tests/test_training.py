import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.training.evaluate import evaluate_model
from src.training.metrics import calculate_metrics
from src.training.train import train_model


class DummyForecastModel(torch.nn.Module):
    def __init__(self, horizon: int):
        super().__init__()
        self.projection = torch.nn.Linear(1, 1)
        self.horizon = horizon

    def forward(self, x):
        # x: [B, L, C]
        last_step = x[:, -1:, :]
        last_step = last_step.transpose(1, 2)  # [B, C, 1]

        projected = self.projection(last_step)
        projected = projected.transpose(1, 2)  # [B, 1, C]

        return projected.expand(-1, self.horizon, -1)


class TestTrainingInfrastructure(unittest.TestCase):
    def test_metrics(self):
        predictions = torch.tensor([1.0, 3.0])
        targets = torch.tensor([2.0, 1.0])

        metrics = calculate_metrics(predictions, targets)

        self.assertAlmostEqual(metrics["mse"], 2.5)
        self.assertAlmostEqual(metrics["mae"], 1.5)

    def test_train_and_checkpoint(self):
        torch.manual_seed(42)

        x = torch.randn(8, 336, 3)
        y = torch.randn(8, 96, 3)

        loader = DataLoader(
            TensorDataset(x, y),
            batch_size=4,
            shuffle=False,
        )

        model = DummyForecastModel(horizon=96)

        config = {
            "learning_rate": 0.001,
            "epochs": 2,
            "patience": 2,
            "device": "cpu",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "best.pt"

            result = train_model(
                model=model,
                train_loader=loader,
                val_loader=loader,
                config=config,
                checkpoint_path=checkpoint_path,
            )

            self.assertTrue(checkpoint_path.exists())
            self.assertGreaterEqual(result["best_epoch"], 1)
            self.assertTrue(torch.isfinite(
                torch.tensor(result["best_val_mse"])
            ).item())


if __name__ == "__main__":
    unittest.main()