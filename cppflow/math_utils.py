from typing import Tuple

# import numpy as np
import torch

def tile_tensor(x: torch.Tensor, k: int) -> torch.Tensor:
    """Tiles a tensor along the first dimension k times.

    Example input, output:
        k: 3, x: [
                  [1, 2, 3],
                  [4, 5, 6],
                ]

        output:
            [
                  [1, 2, 3],
                  [4, 5, 6],
                  [1, 2, 3],
                  [4, 5, 6],
            ]
    """
    return torch.tile(x, (k, 1))
