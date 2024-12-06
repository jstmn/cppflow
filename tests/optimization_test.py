import unittest
from time import time

import numpy as np
import torch

from cppflow.problem import problem_from_filename
from cppflow.utils import set_seed, to_torch, make_text_green_or_red
from cppflow.optimization import (
    OptimizationParameters,
    OptimizationProblem,
    OptimizationState,
    levenberg_marquardt_full,
    levenberg_marquardt_only_pose,
)
from cppflow.planners import PlannerSearcher
from cppflow.optimization_utils import LmResidualFns

PI = torch.pi
torch.set_printoptions(linewidth=5000, precision=8, sci_mode=False)
np.set_printoptions(suppress=True, linewidth=120)
set_seed()


def levenberg_marquardt_full_naive(
    opt_problem: OptimizationProblem,
    opt_state: OptimizationState,
    opt_params: OptimizationParameters,
    lambd: float,
    return_residual: bool = False,
):
    """Calculate an update to x using the levenberg marquardt optimization procedure. Includes residual terms for
    minimizing the difference between sequential configs.
    """
    assert opt_problem.parallel_count == 1
    assert opt_problem.robot.name != "fetch", "fetch unsupported. need to deal with prismatic joint first"
    n = opt_state.x.shape[0]
    ndof = opt_problem.robot.ndof
    # ---
    jacobian, residual = LmResidualFns.get_r_and_J(
        opt_params,
        opt_problem.robot,
        opt_state.x,
        opt_problem.target_path,
        Tcuboids=opt_problem.problem.obstacles_Tcuboids,
        cuboids=opt_problem.problem.obstacles_cuboids,
    )
    J = jacobian.get_J()
    r = residual.get_r()
    assert r.shape[0] == J.shape[0], f"Shape error. r: {r.shape}, J: {J.shape}"
    # Solve (J^T*J + lambd*I)*delta_X = J^T*r
    # Naive solution - create giant, sparse matrix. Cholesky decomposition should reduce solve time
    J_T = torch.transpose(J, 0, 1)  #
    eye = torch.eye(n * ndof, dtype=opt_state.x.dtype, device=opt_state.x.device)
    lhs = torch.matmul(J_T, J) + lambd * eye  # [n*ndof x n*ndof]
    rhs = torch.matmul(J_T, r)  # [n ndof 1]
    delta_x = torch.linalg.solve(lhs, rhs)  # [n ndof 1]
    delta_x = delta_x.reshape((n, ndof))
    if return_residual:
        return opt_state.x + delta_x, J, r
    return opt_state.x + delta_x


class OptimizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fetch_arm__square = problem_from_filename("fetch_arm__square")
        cls.fetch_arm = cls.fetch_arm__square.robot

    # ------------------------------------------------------------------------------------------------------------------
    #   TESTS
    #

    def test_batch_J_matches_full_only_pose(self):
        """Test that levenberg_marquardt_full() and levenberg_marquardt_only_pose() return the same result when
        differencing, virtual configs, and obstacle avoidance is disabled.
        """
        set_seed()
        problem = self.fetch_arm__square
        plan = PlannerSearcher(problem.robot).generate_plan(problem, k=20, verbosity=0).plan
        n = 50
        x_seed = problem.robot.clamp_to_joint_limits((plan.q_path + 0.1 * torch.randn_like(plan.q_path))[0:n])

        opt_params = get_pose_only_optimization_parameters(1.1, 1.0)
        opt_problem = OptimizationProblem(problem, x_seed, plan.target_path[0:n], 0, False, 1, None)
        opt_state = OptimizationState(x_seed.clone(), 0, time())

        #
        x_batched, J_batched, r_batched = levenberg_marquardt_only_pose(
            opt_problem, opt_state, opt_params, return_residual=True
        )
        x_full, J_full, r_full = levenberg_marquardt_full(opt_problem, opt_state, opt_params, return_residual=True)

        for i in range(n):
            torch.testing.assert_close(
                J_batched[i], J_full[i * 6 : (i + 1) * 6, i * 7 : (i + 1) * 7], atol=1e-5, rtol=0.0
            )  # 7 = ndof
            torch.testing.assert_close(r_batched[i], r_full[i * 6 : (i + 1) * 6], atol=1e-5, rtol=0.0)
        torch.testing.assert_close(x_batched, x_full, atol=0.005, rtol=0.0)
        print(make_text_green_or_red("test passed", True))

    def test_cholesky(self):
        """Check that cholesky decomposition returns the same value as torch.linalg.solve"""

        opt_params = OptimizationParameters(
            # General
            seed_w_only_pose=False,
            # Alphas
            alpha_position=0.75,
            alpha_rotation=0.5,
            alpha_differencing=0.01,
            alpha_virtual_configs=None,
            alpha_self_collision=0.01,
            alpha_env_collision=0.025,
            # Pose
            pose_do_scale_down_satisfied=True,
            pose_ignore_satisfied_threshold_scale=0.5,
            pose_ignore_satisfied_scale_down=0.5,
            # Differencing
            use_differencing=True,
            differencing_do_ignore_satisfied=True,
            differencing_ignore_satisfied_margin_deg=1.0,
            differencing_ignore_satisfied_margin_cm=1.0,
            # Virtual Configs
            use_virtual_configs=False,
            virtual_configs=None,
            n_virtual_configs=None,
            # Collisions
            use_self_collisions=True,
            use_env_collisions=True,
        )
        set_seed()
        problem = self.fetch_arm__square
        robot = problem.robot
        n = 50
        qs = to_torch(robot.sample_joint_angles(n))
        target_path = robot.forward_kinematics(qs + 0.1 * torch.randn_like(qs))

        opt_problem = OptimizationProblem(problem, qs.clone(), target_path, 0, False, 1, None)
        opt_state = OptimizationState(qs.clone(), 0, time())
        lambd = 0.00001

        x_new_cholesky, J_chl, r_chl = levenberg_marquardt_full(
            opt_problem, opt_state, opt_params, lambd=lambd, return_residual=True
        )
        x_new_naive, J_naive, r_naive = levenberg_marquardt_full_naive(
            opt_problem, opt_state, opt_params, lambd=lambd, return_residual=True
        )

        torch.testing.assert_close(J_naive, J_chl, atol=5e-4, rtol=0.0)
        torch.testing.assert_close(r_naive, r_chl, atol=5e-4, rtol=0.0)
        torch.testing.assert_close(x_new_cholesky, x_new_naive, atol=5e-4, rtol=0.0)

    # TODO: update to use new LM code
    # def test_ignore_satisfied_mse(self):
    #     """Test that pose error is ignored when below the success threshold"""

    #     # Test 1: pose error is zero, mjac nonzero
    #     qs = torch.tensor(
    #         [
    #             [0.0] * 7,
    #             [PI / 4] * 7,
    #             [PI / 2] * 7,
    #             [PI / 4] * 7,
    #             [3 * PI / 4] * 7,
    #             [PI] * 7,
    #         ],
    #         dtype=torch.float32,
    #         device="cuda",
    #     )
    #     poses = problem.robot.forward_kinematics(qs)
    #     loss = optimizer.loss_fn(qs, poses, -1, 0)[0].item()
    #     qdeltas = torch.rad2deg(
    #         torch.tensor(
    #             [
    #                 [PI / 4] * 7,
    #                 [PI / 4] * 7,
    #                 [-PI / 4] * 7,
    #                 [PI / 2] * 7,
    #                 [PI / 4] * 7,
    #             ],
    #             dtype=torch.float32,
    #             device="cuda",
    #         ).abs()
    #     )
    #     error = torch.maximum(qdeltas - SUCCESS_THRESHOLD_mjac_DEG, torch.zeros_like(qdeltas))
    #     expected = torch.pow(error, 2).mean().item()
    #     self.assertAlmostEqual(loss, expected, msg=f"test 1 failed", delta=0.001)

    #     #
    #     # Test 2: mjac below threshold, pose error high
    #     qs = torch.deg2rad(
    #         torch.tensor(
    #             [
    #                 [0.0] * 7,
    #                 [1.0] * 7,
    #                 [SUCCESS_THRESHOLD_mjac_DEG - 0.1] * 7,
    #                 [1.0] * 7,
    #                 [0.0] * 7,
    #                 [-1.0] * 7,
    #             ],
    #             dtype=torch.float32,
    #             device="cuda",
    #         )
    #     )
    #     _, poses = problem.robot.sample_joint_angles_and_poses(6)
    #     poses = to_torch(poses)
    #     loss = optimizer.loss_fn(qs, poses, -1, 0)[0].item()

    #     t_error_cm, R_error_deg = calculate_pose_error_cm_deg(problem.robot, qs, poses)
    #     t_error_cm -= SUCCESS_THRESHOLD_translation_ERR_MAX_CM
    #     R_error_deg -= SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG
    #     expected = torch.pow(t_error_cm, 2).mean().item() + torch.pow(R_error_deg, 2).mean().item()
    #     self.assertAlmostEqual(loss, expected)


if __name__ == "__main__":
    unittest.main()
