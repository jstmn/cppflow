import unittest

import torch

from jrl.robots import Panda

from cppflow.utils import set_seed
from cppflow.config import device
from cppflow.self_collision_ml.model import build_model

torch.set_default_dtype(torch.float32)
torch.set_default_device(device)

# set_seed()

torch.set_printoptions(linewidth=200)


class SelfCollisionMLTest(unittest.TestCase):
    def test_build_model(self):
        ndofs = 7
        n_distances = 11
        mlp_n_layers = 3
        mlp_width = 120
        mlp_activation = "leaky-relu"
        model = build_model(ndofs, n_distances, mlp_n_layers, mlp_width, mlp_activation)

        # Note: weird result here - the minimum value is almost always at the same column accross different elements in
        # the batch.
        configs = 3 * torch.rand((5, 7))
        distance_predictions = model(configs)
        assert distance_predictions.shape == (5, 11)
        mins = distance_predictions.min(dim=1).values
        assert mins.numel() == 5


if __name__ == "__main__":
    unittest.main()
