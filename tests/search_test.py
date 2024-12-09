import unittest

import torch
import numpy as np
from jrl.robots import Fetch

from cppflow.search import joint_limit_almost_violations_3d
from cppflow.problem import problem_from_filename
from cppflow.data_types import PlannerSearcher
from cppflow.utils import set_seed

set_seed()

torch.set_default_dtype(torch.float32)
torch.set_default_device("cuda:0")
torch.set_printoptions(linewidth=200)
np.set_printoptions(linewidth=500)


class SearchTest(unittest.TestCase):
    # python tests/search_test.py SearchTest.test_joint_limit_almost_violations
    def test_joint_limit_almost_violations(self):
        robot = Fetch()
        # qs = torch.zeros((1, 3, 8)) # [k x ntimesteps x n_dofs]
        qs = torch.zeros((2, 3, 8))  # [k x ntimesteps x n_dofs]
        qs[0, 0] = torch.tensor([0.051, 0, 0, 0, 0, 0, 0, 0])  # all good
        qs[0, 1] = torch.tensor([0.38615 - 0.001, 0, 0, 0, 0, 0, 0, 0])  # prismatic is out of bounds
        qs[0, 2] = torch.tensor([0.38615 - 0.051, 0, 0, 0, 0, 0, 0, 0])  # all good
        qs[1, 0] = torch.tensor([0.38615 - 0.051, 0, 0, -np.pi, 0, 0, 0, 0])  # out of bounds
        qs[1, 1] = torch.tensor([0.38615 - 0.051, 0, 0, -np.pi + 0.11, 0, 0, 0, 0])  # all good
        qs[1, 2] = torch.tensor([0.38615 - 0.051, 0, 0, -np.pi + 0.11, 0, 0, 0, np.pi - 0.25])  # all good
        eps_revolute = 0.1
        eps_prismatic = 0.05

        # (0, 0.38615),
        # (-1.6056, 1.6056),
        # (-1.221, 1.518),  # shoulder_lift_joint
        # (-np.pi, np.pi),  # upperarm_roll_joint
        # (-2.251, 2.251),  # elbow_flex_joint
        # (-np.pi, np.pi),  # forearm_roll_joint
        # (-2.16, 2.16),  # wrist_flex_joint
        # (-np.pi, np.pi),  # wrist_roll_joint
        # expected = torch.zeros((1, 3)) # [k x ntimesteps]
        expected = torch.zeros((2, 3))  # [k x ntimesteps]
        expected[0, 0] = 0
        expected[0, 1] = 1
        expected[0, 2] = 0

        expected[1, 0] = 1
        expected[1, 1] = 0
        expected[1, 2] = 0

        returned = joint_limit_almost_violations_3d(robot, qs, eps_revolute=eps_revolute, eps_prismatic=eps_prismatic)
        print("\nexpected:\n", expected)
        print("returned:\n", returned)
        print("\ndiff:\n", expected - returned)
        torch.testing.assert_close(returned, expected)

    def test_joint_limit_avoidance(self):
        """Test that the optimal path is returned by dp_search"""
        problem_name = "tests/fetch__s__truncated.yaml"
        problem = problem_from_filename("", filepath_override=problem_name)
        planner = PlannerSearcher(problem.robot)
        k = 20
        result = planner.generate_plan(problem, run_batch_ik=False, k=k)
        plan = result.plan
        other_plans = result.other_plans
        eps = np.deg2rad(1)
        for joint_idx, (l, u) in enumerate(problem.robot.actuated_joints_limits):
            violating_l = plan.q_path[:, joint_idx] < l + eps
            violating_u = plan.q_path[:, joint_idx] > u - eps
            self.assertFalse(torch.any(violating_l))
            self.assertFalse(torch.any(violating_u))
        for op in other_plans:
            self.assertLessEqual(plan.mjac, op.mjac, msg=f"  BUG FOUND - {plan.mjac} > {op.mjac}")


if __name__ == "__main__":
    unittest.main()
