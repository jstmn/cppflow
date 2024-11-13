from dataclasses import dataclass
from typing import Optional, Dict
from time import time
import warnings

import torch
from jrl.utils import safe_mkdir

from cppflow.visualization import plot_optimized_trajectory
from cppflow.utils import make_text_green_or_red
from cppflow.problem import Problem
from cppflow.plan import write_qpath_to_results_df
from cppflow.evaluation_utils import angular_changes
from cppflow.config import SELF_COLLISIONS_IGNORED, ENV_COLLISIONS_IGNORED

from cppflow.optimization_utils import (
    x_is_valid,
    clamp_to_joint_limits,
    get_6d_pose_errors,
    OptimizationParameters,
    LmResidualFns,
)
from cppflow.lm_hyper_parameters import ALT_LOSS_V2_1_POSE, ALT_LOSS_V2_1_DIFF



@dataclass
class OptimizationProblem:
    problem: Problem
    seed: torch.Tensor
    target_path: torch.Tensor
    verbosity: int
    parallel_count: int
    results_df: Optional[Dict]

    @property
    def robot(self):
        return self.problem.robot

    @property
    def n_timesteps(self):
        return self.problem.n_timesteps


@dataclass
class OptimizationState:
    x: torch.Tensor
    n_steps: int
    t0: float


@dataclass
class OptimizationResult:
    x_opt: torch.Tensor
    n_steps_taken: int
    is_valid: bool
    parallel_seed_idx: int


# batched
def levenberg_marquardt_only_pose(
    opt_problem: OptimizationProblem,
    opt_state: OptimizationState,
    opt_params: OptimizationParameters,
    return_residual: bool = False,
):
    """Run levenberg marquardt optimization using pose error only in the residual. This lets updates be calculated
    using batched matrix operations which is much faster than solving the full, sparse linear system.
    """
    n = opt_state.x.shape[0]
    ndof = opt_problem.robot.ndof

    error, _ = get_6d_pose_errors(opt_problem.robot, opt_state.x, opt_problem.target_path)  # [n 6 1]
    J_batch = opt_problem.robot.jacobian(opt_state.x)  # [n 6 ndof]
    assert J_batch.shape == (n, 6, ndof)

    error[:, 3:, 0] *= opt_params.alpha_position
    error[:, :3, 0] *= opt_params.alpha_rotation
    J_batch[:, 3:] *= opt_params.alpha_position
    J_batch[:, :3] *= opt_params.alpha_rotation

    J_batch_T = torch.transpose(J_batch, 1, 2)  # [n ndof 6]
    assert J_batch_T.shape == (n, ndof, 6), f"error, J_batch_T: {J_batch_T.shape}, should be {(n, ndof, 6)}"

    eye = torch.eye(ndof, device=opt_state.x.device)[None, :, :].repeat(n, 1, 1)
    lhs_A = torch.bmm(J_batch_T, J_batch) + opt_params.lm_lambda * eye  # [n ndof ndof]
    rhs_B = torch.bmm(J_batch_T, error)  # [n ndof 1]
    delta_x = torch.linalg.solve(lhs_A, rhs_B)  # [n ndof 1]

    if return_residual:
        return opt_state.x + torch.squeeze(delta_x), J_batch, error
    return opt_state.x + torch.squeeze(delta_x)


def _lm_full_step(J: torch.Tensor, r: torch.Tensor, x: torch.Tensor, lambd: float):
    n, ndof = x.shape

    # Solve (J^T*J + lambd*I)*delta_X = J^T*r
    # From wikipedia (https://en.wikipedia.org/wiki/Cholesky_decomposition)
    #  Problem: solve Ax=b
    #  Solution:
    #    1. find L s.t. A = L*L^T
    #    2. solve L*y = b for y by forward substitution
    #    3. solve L^T*x = y for y by backward substitution
    # eye = torch.eye(n * ndof, dtype=opt_state.x.dtype, device=opt_state.x.device)
    eye = torch.eye(n * ndof)
    J_T = torch.transpose(J, 0, 1)
    A = torch.matmul(J_T, J) + lambd * eye  # [n*ndof x n*ndof]
    b = torch.matmul(J_T, r)
    L = torch.linalg.cholesky(A, upper=False)
    y = torch.linalg.solve_triangular(L, b, upper=False)
    delta_x = torch.linalg.solve_triangular(L.T, y, upper=True).reshape((n, ndof))
    return x + delta_x


def levenberg_marquardt_full(
    opt_problem: OptimizationProblem,
    opt_state: OptimizationState,
    opt_params: OptimizationParameters,
    return_residual: bool = False,
):
    """Update x using the levenberg marquardt optimization procedure. Using cholesky decomposition to solve the least
    squares equation is ~4x faster than the naive approach (torch.linalg.solve).

        0.010395956039428712 s per LM update - naive
        0.0028159618377685547 s per LM update - cholesky
    """
    assert opt_problem.parallel_count == 1

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
    x_new = _lm_full_step(J, r, opt_state.x, opt_params.lm_lambda)
    if return_residual:
        return x_new, jacobian, residual
    return x_new


def run_lm_alternating_loss(
    opt_problem: OptimizationProblem,
    opt_state: OptimizationState,
    params_diff: OptimizationParameters,
    params_pose: OptimizationParameters,
    return_residuals: bool = False,
    tmax_return_if_no_valid_after: int = 25,
    tmax: Optional[float] = TMAX,
    max_n_steps: int = ALTERNATING_LOSS_MAX_N_STEPS,
    return_if_valid_after_n_steps: int = ALTERNATING_LOSS_RETURN_IF_SOL_FOUND_AFTER,
    convergence_threshold: float = ALTERNATING_LOSS_CONVERGENCE_THRESHOLD,
    verbosity: int = 0,
    save_images: bool = False,
    results_df: Optional[Dict] = None,
):
    """_summary_

    Timing data for only-pose step:
        levenberg_marquardt_only_pose(): 0.017729759216308594 sec for pose-only step
        levenberg_marquardt_full():      0.03213906288146973 sec for pose-only step
    """
    assert (max_n_steps is None) == (return_if_valid_after_n_steps is None)
    if tmax is None:
        assert (max_n_steps is not None) and (return_if_valid_after_n_steps is not None)
        assert return_if_valid_after_n_steps <= max_n_steps
    if max_n_steps is None:
        assert tmax is not None

    def calc_TL(qpath):
        qpath_rev, _ = opt_problem.problem.robot.split_configs_to_revolute_and_prismatic(qpath)
        return angular_changes(qpath_rev).abs().sum().item()

    def printc(*args, **_kwargs):
        if verbosity > 0:
            print(*args, **_kwargs)

    params_diff = OptimizationParameters(
        **params_diff.__dict__
    )  # copy to avoid modifying in-place - virtual_configs does this
    params_pose = OptimizationParameters(**params_pose.__dict__)

    # Save image of seed trajectory
    if save_images:
        t0_str = str(time()).split(".")[0][-4:]
        safe_mkdir(f"images/{opt_problem.problem.full_name}")
        with open(f"images/{opt_problem.problem.full_name}/(LmAlternating, {t0_str})", "w") as f:
            f.write(f"<empty>")
        plot_optimized_trajectory(
            opt_problem.problem,
            opt_problem.seed,
            opt_state.x,
            show=False,
            figsize=(13, 14),
            save_filepath=f"images/{opt_problem.problem.full_name}/(LmAlternating, {t0_str})   step={100 + 0}.png",
            suptitle=f"Initial seed (TL={round(calc_TL(opt_state.x), 3)})",
        )

    # Run optimization
    if params_diff.virtual_configs.numel() == 0:
        params_diff.virtual_configs = opt_problem.seed.clone()

    tl = calc_TL(opt_state.x)
    printc("TL_0:", tl)
    Js = []
    rs = []
    tls = [tl]
    tls_post_differencing = []
    tls_post_differencing_delta_idxs = []
    tls_post_differencing_deltas = []
    is_valids = []
    last_valid = None
    last_valid_idx = -1
    pose_pos_valid = True  # arbitrarily set to false to start - this leads with pose updates
    pose_rot_valid = False
    mjac_rev_valid = False
    mjac_pris_valid = False
    is_a_self_collision = False
    is_a_env_collision = False
    converged = False

    # TODO: verify that no code is running unnecessarily during execution

    t0 = time()
    i = 0
    while True:
        printc("i:", i)

        # printout metrics for current x
        if verbosity > 0:
            s = ""
            for txt, val in zip(
                ("pose.pos", "pose.rot", "mjac.rev", "mjac.pri"),
                (pose_pos_valid, pose_rot_valid, mjac_rev_valid, mjac_pris_valid),
            ):
                if opt_problem.problem.robot.ndof == 7:
                    if ".pri" in txt:
                        continue
                s += f"{make_text_green_or_red(txt, val)}, "
            if is_a_self_collision is not None and is_a_self_collision:
                s += f"{make_text_green_or_red('self-collision', False)}, "
            if is_a_env_collision is not None and is_a_env_collision:
                s += f"{make_text_green_or_red('env-collision', False)}, "
            printc("  ", s[0:-2])

        # Take update step
        if pose_pos_valid and pose_rot_valid:
            # update the virtual_configs to the current solution
            params_diff.virtual_configs = opt_state.x.clone()
            printc("  ----> differencing")
            x_new, J, r = levenberg_marquardt_full(opt_problem, opt_state, params_diff, return_residual=True)
        else:
            printc("  --> only pose")
            x_new, J, r = levenberg_marquardt_only_pose(opt_problem, opt_state, params_pose, return_residual=True)
        opt_state.x = clamp_to_joint_limits(opt_problem.robot, x_new)

        # update results_df
        if results_df is not None:
            t0i = time()
            write_qpath_to_results_df(results_df, opt_state.x.clone(), opt_problem.problem)  # ~0.07943 sec
            t0 += time() - t0i

        # Analyze new x
        tl_new = calc_TL(opt_state.x)
        printc(f"  tl:", tl_new)
        if return_residuals:
            tl = tl_new
            tls.append(tl_new)
            Js.append(J)
            rs.append(r)
        if pose_pos_valid and pose_rot_valid:
            if not converged and len(tls_post_differencing) > 0:
                diff = abs(tl_new - tls_post_differencing[-1])
                tls_post_differencing_delta_idxs.append(i)
                tls_post_differencing_deltas.append(diff)
                printc("  diff:", diff)
                if diff < convergence_threshold:
                    printc(f"  converged after {i} steps - diff={diff} < {convergence_threshold}")
                    converged = True
                    # break now if the previous step was valid. in this scenario we are saying that the benefit of the
                    # reduced TL is not worth the time required to take the current differencing step in addition to
                    # however many additional steps are required to reduce the pose error back to below the threshold.
                    # Some rough calculations:
                    #   Assume the time per step is 0.075s. If the TL (rad) reduction from this step is 0.3, then we are
                    #   getting a reduction of 0.3 TL for 0.15s of computation. This is a reduction of 2 TL/s which is
                    #   too low to be worth it. The TL can be as high as 80. A scaling of this threshold may be
                    #   appropriate, but that's more work for a minor benefit.
                    if last_valid_idx == i - 1:
                        printc("  last valid was last step, exiting")
                        break
            tls_post_differencing.append(tl_new)

        # save trajectory plot
        if save_images:
            plot_optimized_trajectory(
                opt_problem.problem,
                opt_problem.seed,
                opt_state.x,
                show=False,
                figsize=(13, 14),
                save_filepath=(
                    f"images/{opt_problem.problem.full_name}/(LmAlternating, {t0_str})   step={100 + i+1}.png"
                ),
                suptitle=f"step {i+1} (TL={round(calc_TL(opt_state.x), 3)})",
            )

        # Check for validity
        (
            x_sol,
            _,
            (pose_pos_valid, pose_rot_valid, mjac_rev_valid, mjac_pris_valid, is_a_self_collision, is_a_env_collision),
        ) = x_is_valid(opt_problem.problem, opt_problem.target_path, opt_state.x, 1)
        if x_sol is not None:
            last_valid_idx = i
            last_valid = opt_state.x.clone()
            # update the virtual_configs to this solution
            params_diff.virtual_configs = x_sol.clone()
            is_valids.append(i + 1)
            if converged:
                printc(make_text_green_or_red(f"  x is valid and TL has converged, exiting", True))
                break
            printc(make_text_green_or_red(f"  x is valid, continuing", True))

        # return if found a valid solution and we have taken enough steps ('return_if_valid_after_n_steps')
        if tmax is None:
            if i >= return_if_valid_after_n_steps and last_valid is not None:
                printc(make_text_green_or_red(f"  returning last valid x", True))
                opt_state.x = last_valid.clone()
                break
            if i >= max_n_steps:
                printc(make_text_green_or_red(f"  max_n_steps={max_n_steps} reached, no valid x found", True))
                break
        else:
            if time() - t0 > tmax:
                printc(make_text_green_or_red(f"  tmax={tmax} reached, returning last valid x", True))
                opt_state.x = last_valid.clone()
                break
            if i > tmax_return_if_no_valid_after and last_valid is None:
                printc(
                    make_text_green_or_red(
                        f"  no valid solution found after {tmax_return_if_no_valid_after} steps, breaking", False
                    )
                )
                break

        i += 1

    printc(f"{time() - t0} sec for {i+1} optimization steps  -->  {(time() - t0)/(i+1)} s/step")
    x_return = last_valid if last_valid is not None else opt_state.x
    if return_residuals:
        return (
            x_return,
            last_valid is not None,
            Js,
            rs,
            tls,
            is_valids,
            tls_post_differencing_delta_idxs,
            tls_post_differencing_deltas,
        )
    return OptimizationResult(x_opt=x_return, n_steps_taken=i, is_valid=last_valid is not None, parallel_seed_idx=0)


def run_lm_optimization(
    problem: Problem,
    x_seed: torch.Tensor,
    tmax: float,
    max_n_steps: int,
    return_if_valid_after_n_steps: int,
    convergence_threshold: float,
    parallel_count: int = 1,
    results_df: Optional[Dict] = None,
    verbosity: int = 1,
) -> OptimizationResult:
    """Optimizer a joint angle vector (or multiple simultaneously).

    Args:
        x0 (torch.Tensor): Initial seed(s) for the optimization
        max_n_steps (int): Number of steps to optimize for
        parallel_count (int): Number of seeds. This optimizer is agnostic to the number of initial qpath seeds used.
    """
    if SELF_COLLISIONS_IGNORED:
        warnings.warn("robot-robot are collisions will be ignored during LM optimization")
    if ENV_COLLISIONS_IGNORED:
        warnings.warn("environment-robot collisions will be ignored during LM optimization")


    TODO: use the following:
        tmax
        max_n_steps
        return_if_valid_after_n_steps
        convergence_threshold

    stacked_target_path = (
        problem.target_path if parallel_count == 1 else torch.vstack([problem.target_path] * parallel_count)
    )
    assert stacked_target_path.shape == (problem.n_timesteps * parallel_count, 7)
    assert stacked_target_path.shape[0] == x_seed.shape[0]
    assert x_seed.shape[1] == problem.robot.ndof
    assert isinstance(max_n_steps, int), f"error: max_n_steps must be int, is {type(max_n_steps)}"

    opt_problem = OptimizationProblem(problem, x_seed, stacked_target_path, verbosity, parallel_count, results_df)
    opt_state = OptimizationState(x_seed.clone(), 0, time())

    # lm alternating loss
    return run_lm_alternating_loss(
        opt_problem,
        opt_state,
        ALT_LOSS_V2_1_DIFF,
        ALT_LOSS_V2_1_POSE,
        verbosity=verbosity,
        results_df=results_df,
    )
