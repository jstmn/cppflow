import unittest

import torch
from cppflow.evaluation_utils import angular_changes, calculate_per_timestep_mjac_deg, calculate_mjac_deg

_2PI = 2 * torch.pi
torch.set_default_dtype(torch.float32)
torch.set_default_device("cpu")


class EvaluationTest(unittest.TestCase):
    def test_mjac_consistency(self):
        qpath = torch.randn((10, 3))
        self.assertAlmostEqual(calculate_mjac_deg(qpath), calculate_per_timestep_mjac_deg(qpath).max())
        self.assertAlmostEqual(calculate_mjac_deg(qpath), torch.rad2deg(angular_changes(qpath).abs().max()))

    def test_angular_changes(self):
        """Test that the angular_changes() function returns the correct values."""
        qpath = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(expected, angular_changes(qpath))

        #
        qpath = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0.1],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0.1],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(expected, angular_changes(qpath))

        #
        qpath = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0.1],
                [0, 0, -0.1],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [0, 0, 0.1],
                [0, 0, -0.2],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(expected, angular_changes(qpath))

        #
        qpath = torch.tensor(
            [
                [0, -0.05, 0],
                [0, 0, 0.1],
                [0, 0, -0.1],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [0, 0.05, 0.1],
                [0, 0, -0.2],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(expected, angular_changes(qpath))

        #
        qpath = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, _2PI - 0.1],
                [0, 0, 0],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [0, 0, -0.1],
                [0, 0, 0.1],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(expected, angular_changes(qpath))

        #
        qpath = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, _2PI - 0.1],
                [-0.5, 0, 0.2],
            ],
            dtype=torch.float32,
        )
        expected = torch.tensor(
            [
                [0, 0, -0.1],
                [-0.5, 0, 0.3],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(expected, angular_changes(qpath))


if __name__ == "__main__":
    unittest.main()
