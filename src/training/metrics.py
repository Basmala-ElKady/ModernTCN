import torch


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """Calculate MSE and MAE in the supplied value space."""

    if predictions.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: predictions={predictions.shape}, "
            f"targets={targets.shape}"
        )

    predictions = predictions.detach().float()
    targets = targets.detach().float()

    mse = torch.mean((predictions - targets) ** 2)
    mae = torch.mean(torch.abs(predictions - targets))

    return {
        "mse": mse.item(),
        "mae": mae.item(),
    }