from src.models.tcn import TCN, build_tcn
from src.models.moderntcn import ModernTCN, build_moderntcn
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

    if model_name == "tcn":
        return build_tcn(config)
    if model_name == "moderntcn":
        return build_moderntcn(config)
    return DLinear.from_config(config)
