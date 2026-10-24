import torch

from src.training.metrics import calculate_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model on a validation or test loader."""

    model.eval()

    total_squared_error = 0.0
    total_absolute_error = 0.0
    total_elements = 0

    for x, y in data_loader:
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        predictions = model(x)

        if predictions.shape != y.shape:
            raise ValueError(
                f"Prediction shape {predictions.shape} "
                f"does not match target shape {y.shape}"
            )

        squared_error = (predictions - y).float().square()
        absolute_error = (predictions - y).float().abs()

        total_squared_error += squared_error.sum().item()
        total_absolute_error += absolute_error.sum().item()
        total_elements += y.numel()

    if total_elements == 0:
        raise ValueError("The evaluation DataLoader is empty.")

    return {
        "mse": total_squared_error / total_elements,
        "mae": total_absolute_error / total_elements,
    }