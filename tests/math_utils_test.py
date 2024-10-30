import unittest

import torch
from cppflow.math_utils import tile_tensor


class MathUtilsTest(unittest.TestCase):
    def test_tile_tensor(self):
        x_input = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        expected_output = torch.tensor(
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ]
        )
        returned_output = tile_tensor(x_input, 3)
        self.assertEqual(returned_output.shape[0], 9)
        self.assertEqual(returned_output.shape[1], 3)
        torch.testing.assert_close(expected_output, returned_output)


if __name__ == "__main__":
    unittest.main()
