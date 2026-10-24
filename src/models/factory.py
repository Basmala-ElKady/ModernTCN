from src.models.tcn import TCN
from src.models.moderntcn import ModernTCN
from src.models.dlinear import DLinear


MODEL_REGISTRY = {
    "tcn": TCN,
    "moderntcn": ModernTCN,
    "dlinear": DLinear,
}


def build_model(config: dict):
    """Build a model from the shared experiment configuration."""

    model_name = config["model_name"].lower()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY)}"
        )

    return MODEL_REGISTRY[model_name](config)