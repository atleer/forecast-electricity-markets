# %% Set seed and turn of non-deterministic behavior for reproducibility

import torch

def set_seed(seed: int = 2026) -> None:
    """Set seed and disable non-deterministic behavior for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # when set to true, cuda tests multiple convolution algorithm to find fastest for shape, set to true the default is used
    torch.backends.cudnn.benchmark = False
    # only use deterministic algorithms
    torch.backends.cudnn.deterministic = True
