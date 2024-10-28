import unittest

import torch
from jrl.robots import Panda, Fetch

from cppflow.problem import problem_from_filename
from cppflow.utils import set_seed, to_torch, make_text_green_or_red
from cppflow.optimization_utils import (
    LmResidualFns,
    OptimizationParameters,
    filter_rows_from_r_J_differencing,
    _get_prismatic_and_revolute_row_mask,
    _get_rotation_and_position_row_mask,
)
from cppflow.lm_residuals_naive import LmResidualFnsNaive


pi = torch.pi
torch.set_printoptions(linewidth=300, sci_mode=False, precision=6, edgeitems=17)
# np.set_printoptions(suppress=True, linewidth=120)
set_seed()


def _get_pose_and_differencing_optimization_parameters(
    alpha_pos: float, alpha_rot: float, alpha_diff: float
) -> OptimizationParameters:
    return OptimizationParameters(
        # General
        seed_w_only_pose=False,
        do_use_penalty_method=False,
        # Alphas
        alpha_position=alpha_pos,
        alpha_rotation=alpha_rot,
        alpha_differencing=alpha_diff,
        alpha_virtual_configs=0.0,
        alpha_self_collision=0.0,
        alpha_env_collision=0.0,
        # Pose
        pose_do_scale_down_satisfied=False,
        pose_ignore_satisfied_threshold_scale=None,
        pose_ignore_satisfied_scale_down=None,
        # Differencing
        use_differencing=True,
        differencing_do_ignore_satisfied=False,
        differencing_ignore_satisfied_margin_deg=None,
        differencing_ignore_satisfied_margin_cm=None,
        differencing_do_scale_satisfied=False,
        differencing_scale_down_satisfied_scale=None,
        differencing_scale_down_satisfied_shift_invalid_to_threshold=None,
        # Virtual Configs
        use_virtual_configs=False,
        virtual_configs=None,
        n_virtual_configs=None,
        # Collisions
        use_self_collisions=False,
        use_env_collisions=False,
    )


class OptimizationUtilsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panda = Panda()
        cls.fetch = Fetch()

    # ------------------------------------------------------------------------------------------------------------------
    #   TESTS
    #

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__get_prismatic_and_revolute_row_mask
    def test__get_prismatic_and_revolute_row_mask(self):
        """Test that _get_prismatic_and_revolute_row_mask() is returning correct masks"""
        expected_revolute = torch.tensor(
            [False, True, True, True, True, True, True, True, False, True, True, True, True, True, True, True],
            dtype=torch.bool,
        )
        expected_prismatic = torch.tensor(
            [
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            dtype=torch.bool,
        )
        revolute_rows, prismatic_rows = _get_prismatic_and_revolute_row_mask(self.fetch, 16)
        torch.testing.assert_close(expected_revolute, revolute_rows)
        torch.testing.assert_close(expected_prismatic, prismatic_rows)

        expected_revolute = torch.tensor(
            [True, True, True, True, True, True, True, True, True, True, True, True, True, True], dtype=torch.bool
        )
        expected_prismatic = torch.tensor(
            [False, False, False, False, False, False, False, False, False, False, False, False, False, False],
            dtype=torch.bool,
        )
        revolute_rows, prismatic_rows = _get_prismatic_and_revolute_row_mask(self.panda, 14)
        torch.testing.assert_close(expected_revolute, revolute_rows)
        torch.testing.assert_close(expected_prismatic, prismatic_rows)

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test___get_rotation_and_position_row_mask
    def test___get_rotation_and_position_row_mask(self):
        expected_rotation = torch.tensor(
            [True, True, True, False, False, False, True, True, True, False, False, False], dtype=torch.bool
        )
        expected_position = torch.tensor(
            [False, False, False, True, True, True, False, False, False, True, True, True], dtype=torch.bool
        )
        rotation_rows, position_rows = _get_rotation_and_position_row_mask(2)
        torch.testing.assert_close(expected_rotation, rotation_rows)
        torch.testing.assert_close(expected_position, position_rows)

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__scale_down_rows_from_r_J_differencing_below_error
    def test__scale_down_rows_from_r_J_differencing_below_error(self):
        set_seed()

        # input
        r_in = torch.tensor(
            [
                [0.5],  # prismatic
                [0.1],
                [1.6],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],  # end of config 1
                [-0.4],  # prismatic
                [1.7],
                [-1.7],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],  # end of config 1
                [0.2],  # prismatic
                [0.01],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],  # end of config 3
            ]
        )
        J_in = torch.ones((24, 32))
        mjac_threshold_m = 0.25
        mjac_threshold_rad = 1.5
        scale = 0.5

        # expected
        J_expected = 0.5 * torch.ones((24, 32))
        J_expected[0, :] = 1.0
        J_expected[2, :] = 1.0
        J_expected[8, :] = 1.0
        J_expected[9, :] = 1.0
        J_expected[10, :] = 1.0

        r_expected = torch.tensor(
            [
                [0.5],  # prismatic
                [0.1 / 2],
                [1.6],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],  # end of config 1
                [-0.4],  # prismatic
                [1.7],
                [-1.7],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],  # end of config 1
                [0.2 / 2],  # prismatic
                [0.01 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],  # end of config 3
            ]
        )
        invalid_row_idxs_expected = torch.tensor(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool  # q1  # q2
        )

        # returned
        J_returned, r_returned, invalid_row_idxs_returned = (
            LmResidualFns._scale_down_rows_from_r_J_differencing_below_error(
                robot=self.fetch,
                r=r_in,
                J=J_in,
                mjac_threshold_m=mjac_threshold_m,
                mjac_threshold_rad=mjac_threshold_rad,
                scale=scale,
            )
        )
        torch.testing.assert_close(invalid_row_idxs_expected, invalid_row_idxs_returned)
        torch.testing.assert_close(J_expected, J_returned)
        torch.testing.assert_close(r_expected, r_returned)

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__scale_down_rows_from_r_J_differencing_below_error_with_shift
    def test__scale_down_rows_from_r_J_differencing_below_error_with_shift(self):
        """Test that _scale_down_rows_from_r_J_differencing_below_error() works correctly with shift_invalid_to_threshold=True"""

        # input
        r_in = torch.tensor(
            [
                [0.5],  # prismatic
                [0.1],
                [1.6],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],  # end of config 1
                [-0.4],  # prismatic
                [1.7],
                [-1.7],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],  # end of config 1
                [0.2],  # prismatic
                [0.01],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
                [0.1],  # end of config 3
            ]
        )
        J_in = torch.ones((24, 32))
        mjac_threshold_m = 0.25
        mjac_threshold_rad = 1.5
        scale = 0.5

        # expected
        J_expected = 0.5 * torch.ones((24, 32))
        J_expected[0, :] = 1.0
        J_expected[2, :] = 1.0
        J_expected[8, :] = 1.0
        J_expected[9, :] = 1.0
        J_expected[10, :] = 1.0

        r_expected = torch.tensor(
            [
                [0.5 - 0.25],  # prismatic
                [0.1 / 2],
                [1.6 - 1.5],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],  # end of config 1
                [-0.4 + 0.25],  # prismatic
                [1.7 - 1.5],
                [-1.7 + 1.5],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],  # end of config 1
                [0.2 / 2],  # prismatic
                [0.01 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],
                [0.1 / 2],  # end of config 3
            ]
        )
        invalid_row_idxs_expected = torch.tensor(
            [1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.bool  # q1  # q2
        )
        # returned
        J_returned, r_returned, invalid_row_idxs_returned = (
            LmResidualFns._scale_down_rows_from_r_J_differencing_below_error(
                robot=self.fetch,
                r=r_in,
                J=J_in,
                mjac_threshold_m=mjac_threshold_m,
                mjac_threshold_rad=mjac_threshold_rad,
                scale=scale,
                shift_invalid_to_threshold=True,
            )
        )
        torch.testing.assert_close(invalid_row_idxs_expected, invalid_row_idxs_returned)
        torch.testing.assert_close(J_expected, J_returned)
        torch.testing.assert_close(r_expected, r_returned)

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__scale_down_rows_from_r_J_differencing_below_error_all_revolute
    def test__scale_down_rows_from_r_J_differencing_below_error_all_revolute(self):
        set_seed()

        # input
        r_in = torch.tensor(
            [
                [0.5],
                [0.1],
                [1.6],
                [0.1],
                [0.1],
                [0.1],
                [0.1],
            ]
        )
        J_in = torch.ones((7, 14))
        mjac_threshold_m = 0.25
        mjac_threshold_rad = 1.5
        scale = 0.5

        # returned
        LmResidualFns._scale_down_rows_from_r_J_differencing_below_error(
            robot=self.panda,
            r=r_in,
            J=J_in,
            mjac_threshold_m=mjac_threshold_m,
            mjac_threshold_rad=mjac_threshold_rad,
            scale=scale,
        )
        print(
            make_text_green_or_red(
                "test__scale_down_rows_from_r_J_differencing_below_error_all_revolute() passed", True
            )
        )

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__get_residual_differencing
    def test__get_residual_differencing(self):
        set_seed()
        qs = torch.randn((5, 8))
        returned = LmResidualFns._get_residual_differencing(self.fetch, qs)
        naive = LmResidualFnsNaive._get_residual_differencing(self.fetch, qs)
        torch.testing.assert_close(naive, returned)
        print(make_text_green_or_red("test__get_residual_differencing() passed", True))

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__get_jacobian_differencing
    def test__get_jacobian_differencing(self):
        set_seed()
        qs = torch.randn((5, 8))
        returned = LmResidualFns._get_jacobian_differencing(self.fetch, qs)
        naive = LmResidualFnsNaive._get_jacobian_differencing(self.fetch, qs)
        torch.testing.assert_close(naive, returned)

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__get_r_and_J__pose
    def test__get_r_and_J__pose(self):
        opt_params = OptimizationParameters(
            # General
            do_use_penalty_method=False,
            seed_w_only_pose=False,
            # Alphas
            alpha_position=0.25,
            alpha_rotation=1.5,
            alpha_differencing=0.0,
            alpha_virtual_configs=0.0,
            alpha_self_collision=0.0,
            alpha_env_collision=0.0,
            # Pose
            pose_do_scale_down_satisfied=False,
            pose_ignore_satisfied_threshold_scale=None,
            pose_ignore_satisfied_scale_down=None,
            # Differencing
            use_differencing=False,
            differencing_do_ignore_satisfied=None,
            differencing_ignore_satisfied_margin_deg=None,
            differencing_ignore_satisfied_margin_cm=None,
            differencing_do_scale_satisfied=False,
            differencing_scale_down_satisfied_scale=None,
            differencing_scale_down_satisfied_shift_invalid_to_threshold=None,
            # Virtual Configs
            use_virtual_configs=False,
            virtual_configs=None,
            n_virtual_configs=None,
            # Collisions
            use_self_collisions=False,
            use_env_collisions=False,
        )
        set_seed()

        qs = torch.tensor(
            [
                [-0.05, 0, 0, 0, 0, 0, 0, 0],
                [0.25, 0, 0, 0, 0, 0, 0, 0],
                [0.1, 0, 0, 0, 0, 0, 0, 0],
            ]
        )
        target_path = self.fetch.forward_kinematics(
            torch.tensor(
                [
                    [0.05, 0, 0, 0, 0, 0, 0, 0],
                    [0.2, 0, 0, 0, 0, 0, 0, 0],
                    [0.1, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )

        expected = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0.1 * 0.25],
                [0, 0, 0, 0, 0, -0.05 * 0.25],
                [0, 0, 0, 0, 0, 0 * 0.25],
            ]
        ).reshape((18, 1))
        _, r_naive = LmResidualFnsNaive.get_r_and_J(opt_params, self.fetch, qs, target_path)
        _, r = LmResidualFns.get_r_and_J(opt_params, self.fetch, qs, target_path)
        torch.testing.assert_close(expected, r_naive.pose)
        torch.testing.assert_close(expected, r.pose)

    # python tests/optimization_utils_test.py OptimizationUtilsTest.test__scale_down_rows_from_r_J_pose_below_error
    def test__scale_down_rows_from_r_J_pose_below_error(self):
        """Test that _scale_down_rows_from_r_J_pose_below_error() is working as expected"""
        set_seed()
        poses = self.fetch.forward_kinematics(
            torch.tensor(
                [
                    [0.05, 0, 0, 0, 0, 0, 0, 0],
                    [0.2, 0, 0, 0, 0, 0, 0, 0],
                    [0.1, 0, 0, 0, 0, 0, 0, 0],
                    [0.0, 0, 0, 0, 0, 0, 0, 0],
                ]
            )
        )
        qs = torch.tensor(
            [
                [0.15, 0, 0, 0, 0, 0, 0, 0],
                [0.05, 0, 0, 0, 0, 0, 0, 0],
                [0.1, 0, 0, 0, 0, 0, 0, 0],
                0.001 * torch.randn(8),
            ]
        )

        # Sanity check that the the pose error is as expected
        r_gt = torch.tensor(
            [
                [0, 0, 0, 0, 0, -0.1],
                [0, 0, 0, 0, 0, 0.15],
                [0, 0, 0, 0, 0, 0],
                [0.000807, -0.002434, 0.000551, 0.000002, 0.000606, 0.001627],
            ]
        ).reshape((24, 1))
        torch.testing.assert_close(r_gt, LmResidualFns._get_residual_pose(self.fetch, qs, poses)[0])
        torch.testing.assert_close(r_gt, LmResidualFnsNaive._get_residual_pose(self.fetch, qs, poses)[0])
        J = torch.zeros((24, 32))

        # Check that the appropriate rows are filtered
        expected = torch.tensor(
            [
                [0, 0, 0, 0, 0, -0.03],
                [0, 0, 0, 0, 0, 0.15],
                [0, 0, 0, 0, 0, 0],
                [0.000807, -0.002434, 0.000551, 0.000002 * 0.3, 0.000606 * 0.3, 0.001627 * 0.3],
            ]
        ).reshape(
            (24, 1)
        )  # scale r0_gt by some factor below the threshold
        threshold_m = 0.125
        threshold_rad = 1e-8
        scale = 0.3
        r_returned, _, _ = LmResidualFns._scale_down_rows_from_r_J_pose_below_error(
            r_gt.clone(), J, threshold_m, threshold_rad, scale
        )
        r_naive, _ = LmResidualFnsNaive._scale_down_rows_from_r_J_pose_below_error(
            r_gt.clone(), J, threshold_m, threshold_rad, scale
        )
        torch.testing.assert_close(r_returned, expected)
        torch.testing.assert_close(r_naive, expected)

    def test_get_r_and_J(self):
        """Test that the updated residual and jacobian from LmResidualFns match the naive implementation."""
        opt_params = OptimizationParameters(
            # General
            do_use_penalty_method=False,
            seed_w_only_pose=False,
            # Alphas
            alpha_position=0.5,
            alpha_rotation=2.0,
            alpha_differencing=3.0,
            alpha_virtual_configs=4.0,
            alpha_self_collision=0.0,
            alpha_env_collision=0.0,
            # Pose
            pose_do_scale_down_satisfied=True,
            pose_ignore_satisfied_threshold_scale=0.5,
            pose_ignore_satisfied_scale_down=0.5,
            # Differencing
            use_differencing=False,
            differencing_do_ignore_satisfied=True,
            differencing_ignore_satisfied_margin_deg=0.5,
            differencing_ignore_satisfied_margin_cm=0.5,
            differencing_do_scale_satisfied=False,
            differencing_scale_down_satisfied_scale=None,
            differencing_scale_down_satisfied_shift_invalid_to_threshold=None,
            # Virtual Configs
            use_virtual_configs=False,
            virtual_configs=None,
            n_virtual_configs=None,
            # Collisions
            use_self_collisions=False,
            use_env_collisions=False,
        )
        set_seed()
        problem = problem_from_filename("panda__1cube", robot=self.panda)
        x = to_torch(problem.robot.sample_joint_angles(problem.target_path.shape[0]))
        J_naive, r_naive = LmResidualFnsNaive.get_r_and_J(opt_params, problem.robot, x, problem.target_path)
        J, r = LmResidualFns.get_r_and_J(opt_params, problem.robot, x, problem.target_path)

        torch.testing.assert_close(r_naive.pose, r.pose)
        torch.testing.assert_close(r_naive.differencing, r.differencing)
        torch.testing.assert_close(r_naive.virtual_configs, r.virtual_configs)
        torch.testing.assert_close(J_naive.pose, J.pose)
        torch.testing.assert_close(J_naive.differencing, J.differencing)
        torch.testing.assert_close(J_naive.virtual_configs, J.virtual_configs)

    def test_filter_rows_from_r_J_differencing(self):
        """Test that filter_rows_from_r_J_differencing() is working for robots with and without prismatic joints"""
        # Test 1: all rows are filtered, all revolute
        r = torch.tensor(
            [
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # end of config 1
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # end of config 2
            ]
        )
        J = torch.zeros((14, 14))
        threshold_rad = 0.1
        threshold_m = 0.5
        r_filtered, _ = filter_rows_from_r_J_differencing(
            self.panda, r, J, threshold_rad, threshold_m, shift_to_threshold=True
        )
        r_expected = torch.empty((0, 1))
        torch.testing.assert_close(r_filtered, r_expected)

        # Test 2: some rows are filtered, all revolute
        threshold_rad = 0.1
        threshold_m = 0.5
        r = torch.tensor(
            [
                [0.15],
                [-0.15],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.5],  # end of config 1
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [1.5],  # end of config 2
            ]
        )
        J = torch.zeros((14, 14))
        r_filtered, _ = filter_rows_from_r_J_differencing(
            self.panda, r, J, threshold_rad, threshold_m, shift_to_threshold=True
        )
        r_expected = torch.tensor(
            [
                [0.05],
                [-0.05],
                [0.4],  # end of config 1
                [1.4],  # end of config 2
            ]
        )
        torch.testing.assert_close(r_filtered, r_expected)

        # Test 3: all rows are filtered, some revolute
        threshold_m = 0.5
        threshold_rad = 0.1
        r = torch.tensor(
            [
                [0.4],  # prismatic
                [0.05],
                [-0.05],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # end of config 1
                [-0.1],  # prismatic
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # end of config 1
            ]
        )
        J = torch.zeros((16, 16))
        r_filtered, _ = filter_rows_from_r_J_differencing(
            self.fetch, r, J, threshold_rad, threshold_m, shift_to_threshold=True
        )
        r_expected = torch.empty((0, 1))
        torch.testing.assert_close(r_filtered, r_expected)

        # Test 4: some rows are filtered, some revolute
        threshold_m = 0.5
        threshold_rad = 0.1
        r = torch.tensor(
            [
                [0.6],  # prismatic
                [0.25],
                [-0.25],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # end of config 1
                [-0.7],  # prismatic
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],  # end of config 1
            ]
        )
        J = torch.zeros((16, 16))
        r_filtered, _ = filter_rows_from_r_J_differencing(
            self.fetch, r, J, threshold_rad, threshold_m, shift_to_threshold=True
        )
        r_expected = torch.tensor(
            [
                [0.1],  # prismatic
                [0.15],
                [-0.15],  # end of config 1
                [-0.2],  # prismatic, end of config 1
            ]
        )
        torch.testing.assert_close(r_filtered, r_expected)

    def test_differencing_no_ignore_satisfied(self):
        """Test that get_residual() returns the expected residual"""

        ndof = 7  # for panda
        n = 4
        robot = self.panda
        target_path = torch.zeros((n, 7))
        x = torch.tensor(
            [
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
                [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
            ]
        )
        expected = torch.tensor(
            [
                0.09,
                0.18,
                0.27,
                0.36,
                0.45,
                0.54,
                0.63,
                -0.09,
                -0.18,
                -0.27,
                -0.36,
                -0.45,
                -0.54,
                -0.63,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )[:, None]
        assert expected.shape == ((n - 1) * ndof, 1)
        assert x.shape == (n, ndof)
        opt_params = _get_pose_and_differencing_optimization_parameters(1.0, 1.0, 1.0)

        _, r = LmResidualFns.get_r_and_J(opt_params, robot, x, target_path)
        r_diff = r.differencing
        assert r_diff.shape == ((n - 1) * ndof, 1)
        torch.testing.assert_close(r_diff, expected)


if __name__ == "__main__":
    unittest.main()
