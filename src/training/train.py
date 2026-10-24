from pathlib import Path

import torch

from src.training.evaluate import evaluate_model
from src.utils.checkpoint import save_checkpoint


def train_model(
    model: torch.nn.Module,
    train_loader,
    val_loader,
    config: dict,
    checkpoint_path: str | Path,
) -> dict:
    """Train a model and keep the best checkpoint by validation MSE."""

    device_name = config.get("device", "auto")

    if device_name == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = torch.device(device_name)

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
    )

    best_val_mse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    epochs = int(config["epochs"])
    patience = int(config.get("patience", epochs))

    history = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_squared_error = 0.0
        total_elements = 0

        for x, y in train_loader:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)

            predictions = model(x)

            if predictions.shape != y.shape:
                raise ValueError(
                    f"Prediction shape {predictions.shape} "
                    f"does not match target shape {y.shape}"
                )

            loss = torch.mean((predictions - y) ** 2)
            loss.backward()
            optimizer.step()

            total_squared_error += (
                (predictions.detach() - y).square().sum().item()
            )
            total_elements += y.numel()

        if total_elements == 0:
            raise ValueError("The training DataLoader is empty.")

        train_mse = total_squared_error / total_elements
        val_metrics = evaluate_model(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_mse": train_mse,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train_mse={train_mse:.6f} | "
            f"val_mse={val_metrics['mse']:.6f} | "
            f"val_mae={val_metrics['mae']:.6f}"
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mse=best_val_mse,
                config=config,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    return {
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }