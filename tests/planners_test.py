import unittest

import torch
import numpy as np
from jrl.robot import Robot
from jrl.robots import Panda

from cppflow.evaluation_utils import angular_changes, prismatic_changes
from cppflow.planners import Planner
from cppflow.utils import set_seed
from cppflow import config
from cppflow.planners import PlannerSearcher
from cppflow.problem import problem_from_filename

set_seed()

DEVICE = config.DEVICE
torch.set_printoptions(threshold=10000, precision=6, sci_mode=False, linewidth=200)


def _value_in_tensor(t: torch.Tensor, v: float) -> bool:
    """Check if a value is in a tensor"""
    return ((t - v).abs() < 1e-8).any()


torch.set_default_dtype(torch.float32)
torch.set_default_device(config.DEVICE)


class PlannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.panda = Panda()
        self.mocked_planner = Planner(self.panda, is_mock=True)

        # We assume the network width is 9 in this testsuite (see `IkflowModelParameters` in ikflow/model.py). Here is
        # where we verify that assumption
        self.assertEqual(self.mocked_planner._network_width, 9)  # pylint: disable=protected-access

    def assert_tensor_is_unique(self, x: torch.Tensor):
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # there should be one match: itself
                matching = (x - x[i, j]).abs() < 1e-8
                matching_count = matching.sum().item() - 1
                if matching_count > 0:
                    tupes = matching.nonzero(as_tuple=True)
                    print(f"\nError: value at x[{i}, {j}]={x[i,j]} found at {matching_count} other locations:")
                    # tupes[0]: row idxs
                    # tupes[1]: column idxs
                    for c, (ii, jj) in enumerate(zip(tupes[0], tupes[1])):
                        print(f"  {c}: x[{ii}, {jj}]=\t{x[ii,jj]}")
                self.assertEqual(matching_count, 0)

    def assert_nearly_uniform(self, x: torch.Tensor, lower_bound: float, upper_bound: float, max_delta: float = 0.03):
        self.assertGreater(x.min().item(), lower_bound)
        self.assertLess(x.max().item(), upper_bound)
        self.assertAlmostEqual(x.mean().item(), 0, delta=max_delta)

    def assert_latent_is_per_k(self, latent: torch.Tensor, target_path_n_waypoints: int, k: int):
        """Test that a batched latent tensor 'latent' has 'k' unique latent vectors, which are copied
        'target_path_n_waypoints' times each.
        """
        for i in range(k):
            offset = i * target_path_n_waypoints
            for j in range(target_path_n_waypoints):
                torch.testing.assert_close(latent[offset], latent[offset + j])

            for ii in range(k):
                if i == ii:
                    continue
                offset2 = ii * target_path_n_waypoints
                l1 = latent[offset : offset + target_path_n_waypoints]
                l2 = latent[offset2 : offset2 + target_path_n_waypoints]
                self.assertFalse((l1 == l2).any())

    # ==================================================================================================================
    # Tests

    def test_Plan(self):
        problem = problem_from_filename("", filepath_override="tests/fetch__s__truncated.yaml")
        assert isinstance(problem.robot, Robot), f"problem.robot is {type(problem.robot)}, should be Robot"
        planner = Planner(problem.robot, is_mock=True)
        problem.target_path = problem.target_path[0:5, :]

        qpath = torch.randn((5, 8))
        plan = planner._plan_from_qpath(qpath, problem)
        qpath_rot, qpath_pri = problem.robot.split_configs_to_revolute_and_prismatic(qpath)
        assert qpath_rot.shape == (5, 7)
        assert qpath_pri.shape == (5, 1), f"qpath_pri.shape should be (5, 1), is {qpath_pri.shape} (equals {qpath_pri})"

        # revolute
        self.assertAlmostEqual(plan.mjac_per_timestep_deg.max().item(), plan.mjac_deg)
        self.assertAlmostEqual(torch.rad2deg(angular_changes(qpath_rot).abs().max()), plan.mjac_deg)
        # prismatic
        self.assertAlmostEqual(plan.mjac_per_timestep_cm.max().item(), plan.mjac_cm)
        self.assertAlmostEqual(100 * prismatic_changes(qpath_pri).abs().max(), plan.mjac_cm)

    def test_batch_ik_order(self):
        """Check that the output is the same for the same input"""

        # Test #1: All zeros
        problem_name = "fetch_arm__s"
        problem = problem_from_filename(problem_name)
        planner = PlannerSearcher(problem.robot)
        verbosity = 0

        set_seed()  # set seed to ensure that ikflow returns the same tensor back
        plan_pre_search, _, _ = planner.generate_plan(
            problem, k=15, verbosity=verbosity, batch_ik_order="pre_search", run_batch_ik=True
        )
        set_seed()
        plan_post_search, _, _ = planner.generate_plan(
            problem, k=15, verbosity=verbosity, batch_ik_order="post_search", run_batch_ik=True
        )
        set_seed()
        plan_no_batch_ik, _, _ = planner.generate_plan(problem, k=15, verbosity=verbosity, run_batch_ik=False)

        self.assertTrue(plan_pre_search.q_path.shape == plan_post_search.q_path.shape)
        self.assertGreater(np.absolute((plan_pre_search.q_path - plan_post_search.q_path)).max(), 1e-6)
        self.assertTrue(plan_pre_search.q_path.shape == plan_no_batch_ik.q_path.shape)
        self.assertGreater(np.absolute((plan_pre_search.q_path - plan_no_batch_ik.q_path)).max(), 1e-6)

    def test__get_k_ikflow_qpaths_same_input_same_output(self):
        """Check that the output is the same for the same input"""
        k = 2

        # Test #1: All zeros
        ee_path = torch.zeros((5, 7), device=DEVICE)
        batched_latents = torch.zeros((10, 9), device=DEVICE)
        ikflow_qpaths, _ = self.mocked_planner._get_k_ikflow_qpaths(  # pylint: disable=protected-access
            k, ee_path, batched_latents, verbosity=0
        )
        torch.testing.assert_close(ikflow_qpaths[0], ikflow_qpaths[1])

        # Test #2: Changing end effector path
        ee_path = torch.zeros((5, 7), device=DEVICE)
        ee_path[:, 0] = torch.arange(5)

        batched_latents = torch.zeros((10, 9), device=DEVICE)
        ikflow_qpaths, _ = self.mocked_planner._get_k_ikflow_qpaths(  # pylint: disable=protected-access
            k, ee_path, batched_latents, verbosity=0
        )
        torch.testing.assert_close(ikflow_qpaths[0], ikflow_qpaths[1])

    def test__get_k_ikflow_qpaths_no_repeats_fixed_ee_path(self):
        """Check that no elements from one qpath tensor are found in another. This should be the case when the two
        qpaths have different latent vectors
        """
        k = 2
        ee_path = torch.zeros((5, 7), device=DEVICE)
        batched_latents = torch.zeros((10, 9), device=DEVICE)  # latents for qpath1 are all 0
        batched_latents[5:, :] = 1.0  # latents for qpath2 are all 1

        # qpath1 and qpath2 should be completely different, because they have different latent vectors and the same
        # ee_path. Elements
        ikflow_qpaths, _ = self.mocked_planner._get_k_ikflow_qpaths(  # pylint: disable=protected-access
            k, ee_path, batched_latents, verbosity=0, clamp_to_joint_limits=False
        )
        qpath_0 = ikflow_qpaths[0]
        qpath_1 = ikflow_qpaths[1]
        self.assertEqual(qpath_0.shape, (5, 7))
        self.assertEqual(qpath_1.shape, (5, 7))

        # Check that no value from qpath[0] is in qpath[1]. This should happen with exceedingly low probability
        for i in range(5):
            for j in range(7):
                value_k0 = qpath_0[i, j]
                in_k2 = _value_in_tensor(qpath_1, value_k0).item()
                self.assertFalse(
                    in_k2,
                    f"Error: found value qpath_0[{i}, {j}]={value_k0} found in qpaths_1.\nqpaths_1={qpath_1.data}",
                )

    def test__get_k_ikflow_qpaths_no_repeats_changing_ee_path(self):
        """Check that no elements match one another in returned qpath tensors. with a changing target pose value this
        should be the case

        Note: This test will fail if clamp_to_joint_limits is enabled.
        """
        # Setup inputs
        k = 2
        ee_path = torch.zeros((5, 7), device=DEVICE)
        ee_path[:, 0] = torch.arange(5)
        batched_latents = torch.zeros((10, 9), device=DEVICE)  # latents for qpath1 are all 0
        batched_latents[5:, :] = 1.0  # latents for qpath2 are all 1

        # Call ikflow
        ikflow_qpaths, _ = self.mocked_planner._get_k_ikflow_qpaths(  # pylint: disable=protected-access
            k,
            ee_path,
            batched_latents,
            clamp_to_joint_limits=False,
        )
        qpath_0 = ikflow_qpaths[0]
        qpath_1 = ikflow_qpaths[1]

        # Validate output
        self.assertEqual(qpath_0.shape, (5, 7))
        self.assertEqual(qpath_1.shape, (5, 7))
        self.assert_tensor_is_unique(qpath_0)
        self.assert_tensor_is_unique(qpath_1)

    def test__get_fixed_random_latent(self):
        """Sanity check _get_fixed_random_latent()"""
        # --------------------
        # Test 1: Per-k
        k = 15
        target_path_n_waypoints = 300
        distribution = "uniform"
        latent_vector_scale = 1.5
        per_k_or_timestep = "per_k"
        latent = self.mocked_planner._get_fixed_random_latent(
            k, target_path_n_waypoints, distribution, latent_vector_scale, per_k_or_timestep
        )
        # Test that the latents are nearly uniform, and also that there are k unique ones
        self.assertEqual(latent.shape, (4500, 9))
        self.assert_nearly_uniform(latent, -0.75, 0.75)
        self.assert_latent_is_per_k(latent, target_path_n_waypoints, k)

        # gaussian
        distribution = "gaussian"
        latent = self.mocked_planner._get_fixed_random_latent(
            k, target_path_n_waypoints, distribution, latent_vector_scale, per_k_or_timestep
        )
        self.assertEqual(latent.shape, (4500, 9))
        self.assert_latent_is_per_k(latent, target_path_n_waypoints, k)

        # --------------------
        # Test 2: Per timestep
        k = 5
        target_path_n_waypoints = 25
        distribution = "uniform"
        latent_vector_scale = 3.0
        per_k_or_timestep = "per_timestep"
        latent = self.mocked_planner._get_fixed_random_latent(
            k, target_path_n_waypoints, distribution, latent_vector_scale, per_k_or_timestep
        )
        # Test that the latents are nearly uniform, and also that they are all unique
        self.assertEqual(latent.shape, (125, 9))
        self.assert_nearly_uniform(latent, -1.5, 1.5)
        self.assert_tensor_is_unique(latent)

        distribution = "gaussian"
        latent = self.mocked_planner._get_fixed_random_latent(
            k, target_path_n_waypoints, distribution, latent_vector_scale, per_k_or_timestep
        )
        self.assertEqual(latent.shape, (125, 9))
        self.assert_tensor_is_unique(latent)


if __name__ == "__main__":
    unittest.main()
