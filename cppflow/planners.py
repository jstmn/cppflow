from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
from time import time
import warnings

from ikflow.ikflow_solver import IKFlowSolver
from ikflow.model_loading import get_ik_solver
from ikflow.model import IkflowModelParameters, TINY_MODEL_PARAMS
from jrl.robot import Robot
from jrl.robots import Panda, Fetch, FetchArm
import matplotlib.pyplot as plt
import torch
import numpy as np

from cppflow.utils import TimerContext
from cppflow.problem import Problem
from cppflow.config import DEBUG_MODE_ENABLED, DEVICE
from cppflow.math_utils import tile_tensor
from cppflow.search import dp_search
from cppflow.evaluation_utils import calculate_mjac_deg, calculate_per_timestep_mjac_cm
from cppflow.optimization import run_lm_optimization
from cppflow.plan import Plan, plan_from_qpath, write_qpath_to_results_df
from cppflow.collision_detection import qpaths_batched_self_collisions, qpaths_batched_env_collisions


ROBOT_TO_IKFLOW_MODEL = {
    # --- Panda
    Panda.name: "panda__full__lp191_5.25m",
    # --- Fetch
    Fetch.name: "fetch_full_temp_nsc_tpm",
    # --- FetchArm
    FetchArm.name: "fetch_arm__large__mh186_9.25m",
}

MOCK_IKFLOW_PARAMS = IkflowModelParameters()

DEFAULT_RERUN_NEW_K = 125  # the existing k configs will be added to this so no need to go overboard
DEFAULT_RERUN_MJAC_THRESHOLD_DEG = 13.0
DEFAULT_RERUN_MJAC_THRESHOLD_CM = 3.42


def printc(s, kwargs):
    if kwargs["verbosity"] > 0:
        print(s)


def _print_kwargs(kwargs):
    for key, v in kwargs.items():
        if key == "results_df" or key[0] == "_":
            continue
        if isinstance(v, tuple):
            print(f"  {key}: (", end=" ")
            for vv in v:
                if isinstance(vv, torch.Tensor):
                    print(vv.shape, end=", ")
                else:
                    print(vv, end=", ")
            print(")")
            continue
        print(f"  {key}: {v}")
    print()


def _plot_self_collisions(self_collision_violations: torch.Tensor):
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.imshow(self_collision_violations.cpu().numpy(), vmin=0, vmax=1)
    plt.title("self collision violations")
    plt.xlabel("timestep")
    plt.ylabel("k")
    # plt.colorbar()
    plt.grid(True, which="both", axis="both")
    plt.savefig("debug__self_collision_violations.png", bbox_inches="tight")
    plt.close()


def _plot_env_collisions(env_collision_violations: torch.Tensor):
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.imshow(env_collision_violations.cpu().numpy(), vmin=0, vmax=1)
    plt.title("environment collision violations")
    plt.xlabel("timestep")
    plt.ylabel("k")
    # plt.colorbar()
    plt.grid(True, which="both", axis="both")
    plt.savefig("debug__env_collision_violations.png", bbox_inches="tight")
    plt.close()


def _get_mjacs(robot: Robot, qpath: torch.Tensor):
    qps_revolute, qps_prismatic = robot.split_configs_to_revolute_and_prismatic(qpath)
    if qps_prismatic.numel() > 0:
        return calculate_mjac_deg(qps_revolute), calculate_per_timestep_mjac_cm(qps_prismatic).abs().max().item()
    return calculate_mjac_deg(qps_revolute), 0.0


@dataclass
class TimingData:
    total: float
    ikflow: float
    coll_checking: float
    batch_opt: float
    dp_search: float
    optimizer: float


@dataclass
class PlannerResult:
    plan: Plan
    timing: TimingData
    other_plans: List[Plan]
    other_plans_names: List[str]
    debug_info: dict


class Planner:
    def __init__(self, robot: Robot, is_mock: bool = False):
        if not is_mock:
            self._ikflow_model_name = ROBOT_TO_IKFLOW_MODEL[robot.name]
            self._ikflow_solver, _ = get_ik_solver(self._ikflow_model_name, robot=robot)
        else:
            print("Warning: Using a mocked IKFlow solver - this model has random weights")
            self._ikflow_model_name = "none - mocked"
            self._ikflow_solver = IKFlowSolver(TINY_MODEL_PARAMS, robot)

        self._network_width = self._ikflow_solver.network_width

    # Public methods
    @property
    def ikflow_model_name(self) -> str:
        return self._ikflow_model_name

    @property
    def robot(self) -> Robot:
        return self._ikflow_solver.robot

    @property
    def ikflow_solver(self) -> IKFlowSolver:
        return self._ikflow_solver

    @property
    def network_width(self) -> int:
        return self._network_width

    @property
    def name(self) -> str:
        return str(self.__class__.__name__)

    @abstractmethod
    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        raise NotImplementedError()

    # Private methods
    def _get_fixed_random_latent(
        self,
        k: int,
        target_path_n_waypoints: int,
        distribution: str,
        latent_scale_parameter: float,
        per_k_or_timestep: str,
    ) -> torch.Tensor:
        """Returns the latent vector for the IKFlow call.

        Notes:
            1. For 'uniform', latent_scale_parameter is the width of the sampling area for each dimension. The value 2.0
                is recommended. This was found in a hyperparameter search in the 'search_scatchpad.ipynb' notebook on
                March 13.

        Args:
            k: Number of paths
            target_path_n_waypoints: Number of waypoints in the target pose path
        """
        assert per_k_or_timestep in {"per_k", "per_timestep"}
        shape = (
            (k, self._network_width)
            if per_k_or_timestep == "per_k"
            else (k * target_path_n_waypoints, self._network_width)
        )
        if distribution == "gaussian":
            latents = torch.randn(shape, device=DEVICE) * latent_scale_parameter  # [k x network_width]
        elif distribution == "uniform":
            width = latent_scale_parameter
            latents = torch.rand(shape, device=DEVICE) * width - (width / 2)
        if per_k_or_timestep == "per_k":
            return torch.repeat_interleave(latents, target_path_n_waypoints, dim=0)
        return latents

    def _get_k_ikflow_qpaths(
        self,
        k: int,
        ee_path: torch.Tensor,
        batched_latent: torch.Tensor,
        verbosity: int = 1,
        clamp_to_joint_limits: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns k different config space paths for the given ee_path."""
        n = ee_path.shape[0]

        with torch.inference_mode(), TimerContext("running IKFlow", enabled=verbosity > 0):
            ee_path_tiled = tile_tensor(ee_path, k)
            # TODO: Query model directly (remove 'solve_n_poses()' call)
            ikf_sols = self._ikflow_solver.generate_ik_solutions(
                ee_path_tiled,
                latent=batched_latent,
                # refine_solutions=False,
                clamp_to_joint_limits=clamp_to_joint_limits,
            )
        paths = [ikf_sols[i * n : (i * n) + n, :] for i in range(k)]
        return paths, ee_path_tiled


    def _run_pipeline(self, problem: Problem, **kwargs) -> Tuple[torch.Tensor, bool, TimingData]:
        k = kwargs["k"] if "k" in kwargs else 175
        if "verbosity" not in kwargs:
            kwargs["verbosity"] = 1
        verbosity = kwargs["verbosity"] if "verbosity" in kwargs else 1
        latent_distribution = kwargs["latent_distribution"] if "latent_distribution" in kwargs else "uniform"
        latent_scale_parameter = kwargs["latent_scale_parameter"] if "latent_scale_parameter" in kwargs else 2.0
        latent_drawing = kwargs["latent_drawing"] if "latent_drawing" in kwargs else "per_k"
        run_batch_opt = kwargs["run_batch_opt"] if "run_batch_opt" in kwargs else False
        run_dp_search = kwargs["run_dp_search"] if "run_dp_search" in kwargs else True
        existing_q_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None
        assert latent_drawing in {"per_timestep", "per_k"}
        assert latent_distribution in {"uniform", "gaussian"}
        assert latent_scale_parameter > 0.0
        if not run_dp_search:
            assert run_batch_opt

        if verbosity > 1:
            print(f"_run_pipeline() | kwargs:")
            _print_kwargs(kwargs)

        # ikflow
        t0_ikflow = time()
        batched_latents = self._get_fixed_random_latent(
            k, problem.n_timesteps, latent_distribution, latent_scale_parameter, latent_drawing
        )
        ikflow_qpaths, _ = self._get_k_ikflow_qpaths(k, problem.target_path, batched_latents, verbosity)
        time_ikflow = time() - t0_ikflow

        # save initial solution
        if "only_1st" in kwargs and kwargs["only_1st"]:
            return (
                ikflow_qpaths[0],
                False,
                TimingData(-1, time_ikflow, 0.0, 0.0, 0.0, 0.0),
                {},
                (ikflow_qpaths[0], None, None),
            )

        # Collision checking
        t0_col_check = time()
        qs = torch.stack(ikflow_qpaths)

        with TimerContext("calculating self-colliding configs", enabled=verbosity > 0):
            self_collision_violations = qpaths_batched_self_collisions(problem, qs)
            pct_colliding = (torch.sum(self_collision_violations) / (k * problem.n_timesteps)).item() * 100
            assert pct_colliding < 95.0, f"too many env collisions: {pct_colliding} %"
            if verbosity > 0:
                print(f"  self_collision violations: {pct_colliding} %")

            if verbosity > 2:
                warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
                _plot_self_collisions(self_collision_violations)

        with TimerContext("calculating env-colliding configs", enabled=verbosity > 0):
            env_collision_violations = qpaths_batched_env_collisions(problem, qs)
            pct_colliding = (torch.sum(env_collision_violations) / (k * problem.n_timesteps)).item() * 100
            assert pct_colliding < 95.0, f"too many env collisions: {pct_colliding} %"
            if verbosity > 0:
                print(f"  env_collision violations: {pct_colliding} %")

            if verbosity > 2:
                warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
                _plot_env_collisions(env_collision_violations)

        if existing_q_data is not None:
            qs_prev, self_collision_violations_prev, env_collision_violations_prev = existing_q_data
            qs = torch.cat([qs_prev, qs], dim=0)
            self_collision_violations = torch.cat([self_collision_violations_prev, self_collision_violations], dim=0)
            env_collision_violations = torch.cat([env_collision_violations_prev, env_collision_violations], dim=0)

        time_coll_check = time() - t0_col_check
        debug_info = {}
        q_data = (qs, self_collision_violations, env_collision_violations)


        time_batch_opt = 0.0

        # dp_search
        t0_dp_search = time()
        with TimerContext(f"running dynamic programming search with qs: {qs.shape}", enabled=verbosity > 0):
            qpath_search = dp_search(
                self.robot, qs, self_collision_violations, env_collision_violations, verbosity=verbosity
            ).to(DEVICE)
        time_dp_search = time() - t0_dp_search
        if "results_df" in kwargs:
            write_qpath_to_results_df(kwargs["results_df"], qpath_search, problem)

        if DEBUG_MODE_ENABLED and "do_return_search_path_mjac" in kwargs and kwargs["do_return_search_path_mjac"]:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, qpath_search)
            debug_info["search_path_mjac-cm"] = mjac_cm
            debug_info["search_path_mjac-deg"] = mjac_deg
            _, qps_prismatic = problem.robot.split_configs_to_revolute_and_prismatic(qpath_search)

            search_path_min_dist_to_jlim_cm = -1
            search_path_min_dist_to_jlim_deg = 10000

            for i, (l, u) in enumerate(problem.robot.actuated_joints_limits):
                if i == 0 and problem.robot.has_prismatic_joints:
                    search_path_min_dist_to_jlim_cm = 100 * min(
                        torch.min(torch.abs(qps_prismatic - l)).item(), torch.min(torch.abs(qps_prismatic - u)).item()
                    )
                    continue
                search_path_min_dist_to_jlim_deg = min(
                    torch.rad2deg(torch.min(torch.abs(qpath_search[:, i] - l))).item(),
                    torch.rad2deg(torch.min(torch.abs(qpath_search[:, i] - u))).item(),
                    search_path_min_dist_to_jlim_deg,
                )
            debug_info["search_path_min_dist_to_jlim_cm"] = search_path_min_dist_to_jlim_cm
            debug_info["search_path_min_dist_to_jlim_deg"] = search_path_min_dist_to_jlim_deg

        return (
            qpath_search,
            False,
            TimingData(-1, time_ikflow, time_coll_check, time_batch_opt, time_dp_search, 0.0),
            debug_info,
            q_data,
        )


class PlannerSearcher(Planner):
    """PlannerSearcher creates a finds a solution by performing a search through a graph constructed by connecting k
    ikflow generated cspace plans
    """

    def __init__(self, robot: Robot):
        super().__init__(robot)

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        """Runs dp_search and returns"""
        assert problem.robot.name == self.robot.name
        kwargs["run_batch_opt"] = False
        kwargs["run_dp_search"] = True
        do_rerun_if_large_dp_search_mjac = (
            kwargs["do_rerun_if_large_dp_search_mjac"] if "do_rerun_if_large_dp_search_mjac" in kwargs else False
        )
        rerun_mjac_threshold_deg = (
            kwargs["rerun_mjac_threshold_deg"]
            if "rerun_mjac_threshold_deg" in kwargs
            else DEFAULT_RERUN_MJAC_THRESHOLD_DEG
        )
        rerun_mjac_threshold_cm = (
            kwargs["rerun_mjac_threshold_cm"]
            if "rerun_mjac_threshold_cm" in kwargs
            else DEFAULT_RERUN_MJAC_THRESHOLD_CM
        )

        t0 = time()
        qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)

        # rerun dp_search with larger k if mjac is too high
        if do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, qpath_search)
            if mjac_deg > rerun_mjac_threshold_deg or mjac_cm > rerun_mjac_threshold_cm:
                printc(
                    f"rerunning dp_search with larger k because mjac is too high: {mjac_deg} deg, {mjac_cm} cm", kwargs
                )
                kwargs["k"] = 350  # DEFAULT_RERUN_NEW_K
                qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = _get_mjacs(problem.robot, qpath_search)
                printc(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", kwargs)

        time_total = time() - t0
        return PlannerResult(
            plan_from_qpath(qpath_search.detach(), problem),
            TimingData(time_total, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, 0.0),
            [],
            [],
            debug_info,
        )


# ----------------------------------------------------------------------------------------------------------------------
# ---
# --- Optimizers
# ---


class _PlannerSearcherOptimizer(Planner):
    # Generalized planner that runs dp_search before LM optimization. Must specify the optimization version to use

    DEFAULT_MAX_N_OPTIMIZATION_STEPS = 20

    def __init__(self, robot: Robot):
        super().__init__(robot)

    def _generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        t0 = time()
        if "verbosity" not in kwargs:
            kwargs["verbosity"] = 1
        do_rerun_if_optimization_fails = (
            kwargs["do_rerun_if_optimization_fails"] if "do_rerun_if_optimization_fails" in kwargs else False
        )
        do_rerun_if_large_dp_search_mjac = (
            kwargs["do_rerun_if_large_dp_search_mjac"] if "do_rerun_if_large_dp_search_mjac" in kwargs else False
        )
        rerun_mjac_threshold_deg = (
            kwargs["rerun_mjac_threshold_deg"]
            if "rerun_mjac_threshold_deg" in kwargs
            else DEFAULT_RERUN_MJAC_THRESHOLD_DEG
        )
        rerun_mjac_threshold_cm = (
            kwargs["rerun_mjac_threshold_cm"]
            if "rerun_mjac_threshold_cm" in kwargs
            else DEFAULT_RERUN_MJAC_THRESHOLD_CM
        )
        rerun_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None
        if rerun_data is not None:
            printc("WAHOO!! I AM IN AN INNER LOOP", kwargs)

        results_df = kwargs["results_df"] if "results_df" in kwargs else None
        # q_data = (qs, self_collision_violations, env_collision_violations)
        search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)

        # return only the IST plan
        if "only_1st" in kwargs and kwargs["only_1st"]:
            return PlannerResult(
                plan_from_qpath(search_qpath, problem), TimingData(time() - t0, 0, 0, 0, 0, 0), [], [], {}
            )

        # rerun dp_search with larger k if mjac is too high
        if do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, search_qpath)
            if mjac_deg > rerun_mjac_threshold_deg or mjac_cm > rerun_mjac_threshold_cm:
                printc(
                    f"rerunning dp_search with larger k because mjac is too high: {mjac_deg} deg, {mjac_cm} cm", kwargs
                )
                kwargs["k"] = DEFAULT_RERUN_NEW_K
                kwargs["rerun_data"] = q_data
                search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = _get_mjacs(problem.robot, search_qpath)
                printc(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", kwargs)

        if kwargs["verbosity"] > 1:
            print("\ndp_search path:")
            print(plan_from_qpath(search_qpath, problem))

        # batch_opt was successful
        if is_valid:
            time_total = time() - t0
            return PlannerResult(
                plan_from_qpath(search_qpath, problem),
                TimingData(time_total, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, 0.0),
                [],
                [],
                debug_info,
            )

        # batch_opt was not successful - continue to LM optimization
        with TimerContext(f"running run_lm_optimization()", enabled=kwargs["verbosity"] > 0):
            t0_opt = time()
            optimization_result = run_lm_optimization(
                problem,
                search_qpath,
                max_n_steps=_PlannerSearcherOptimizer.DEFAULT_MAX_N_OPTIMIZATION_STEPS,
                results_df=results_df,
                verbosity=kwargs["verbosity"],
            )
            time_optimizer = time() - t0_opt
            time_total = time() - t0
            debug_info["n_optimization_steps"] = optimization_result.n_steps_taken

        # Optionally rerun if optimization failed
        if not optimization_result.is_valid and do_rerun_if_optimization_fails and rerun_data is None:
            printc(f"rerunning dp_search because optimization failed", kwargs)
            kwargs["rerun_data"] = q_data
            return self._generate_plan(problem, **kwargs)

        if "results_df" in kwargs:
            write_qpath_to_results_df(kwargs["results_df"], optimization_result.x_opt, problem)

        return PlannerResult(
            plan_from_qpath(optimization_result.x_opt.detach(), problem),
            TimingData(time_total, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, time_optimizer),
            [plan_from_qpath(search_qpath.detach(), problem)],
            ["search-plan"],
            debug_info,
        )


class CppFlowPlanner(Planner):
    def __init__(self, robot: Robot):
        super().__init__(robot)
        self._searcher_optimizer = _PlannerSearcherOptimizer(robot)

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        return self._searcher_optimizer._generate_plan(problem, **kwargs)  # pylint: disable=protected-access
