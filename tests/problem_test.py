import unittest

from cppflow.problem import problem_from_filename, Problem
from cppflow.utils import to_torch

import numpy as np


class ProblemTest(unittest.TestCase):
    def _get_problem(self, target_path):
        return Problem(
            target_path=target_path,
            robot=None,
            name=None,
            full_name=None,
            obstacles=[],
        )

    # Tests
    # TODO
    def test_offset_target_path(self):
        pass

    def test_problem(self):
        problem = problem_from_filename("panda__square")
        self.assertIsInstance(problem, Problem)

    def test_path_length_calculation(self):
        target_path = np.array(
            [
                [0, 0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 1.0, 0, 0, 0],
                [0, 0, 0, 1.0, 0, 0, 0],
            ]
        )
        problem = self._get_problem(target_path)

        # Position
        expected_pos = 0.0
        expected_rot = 0.0
        self.assertAlmostEqual(expected_pos, problem.path_length_cumultive_positional_change_cm)
        self.assertAlmostEqual(expected_rot, problem.path_length_cumulative_rotational_change_deg)

        # ------
        # Test 2
        # quaternions from https://www.andre-gaschler.com/rotationconverter/
        target_path = np.array(
            [
                [0, 0, 0, 1.0, 0, 0, 0],
                [0.1, 0, 0, 0.9990482, 0, 0, 0.0436194],  # [ 0.9990482, 0, 0, 0.0436194 ] - 5  deg about [0, 0, 1]
                [0.1, 0, 0, 0.9914449, 0, 0, 0.1305262],  # [ 0.9914449, 0, 0, 0.1305262 ] - 15 deg about [0, 0, 1]
                [0.2, 0, 0, 0.9063078, 0, 0, 0.4226183],  # [ 0.9063078, 0, 0, 0.4226183 ] - 50 deg about [0, 0, 1]
            ]
        )
        problem = self._get_problem(target_path)
        #
        expected_pos = 20.0
        expected_rot = 5.0 + 10.0 + 35.0
        self.assertAlmostEqual(expected_pos, problem.path_length_cumultive_positional_change_cm)
        self.assertAlmostEqual(expected_rot, problem.path_length_cumulative_rotational_change_deg, places=4)

        # ------
        # Test 3
        # triangle values from https://www.calculator.net/right-triangle-calculator.html?
        target_path = np.array(
            [
                [0, 0, 0, 1.0, 0, 0, 0],
                [1, 2, 0, 0.9990482, 0.0, 0.0, 0.0436194],  # [0, 0, 5]
                [0, 0, 0, 0.9952465, 0.0870728, -0.0038017, 0.0434534],  # [10, 0, 5]
                [0, 2, 7, 0.9941335, 0.0888853, 0.039614, 0.0472101],  # [10, 5, 5 ]
            ]
        )
        problem = self._get_problem(target_path)
        #
        expected_pos = (2.23607 + 2.23607 + 7.28011) * 100
        expected_rot = 5.0 + 10.0 + 5.0
        self.assertAlmostEqual(expected_pos, problem.path_length_cumultive_positional_change_cm, places=3)
        self.assertAlmostEqual(expected_rot, problem.path_length_cumulative_rotational_change_deg, places=3)


if __name__ == "__main__":
    unittest.main()
