import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python, NumPy and PyTorch RNGs for a reproducible experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN benchmarking can choose a different algorithm between runs.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
