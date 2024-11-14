from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Dict
from time import time
import warnings

from ikflow.ikflow_solver import IKFlowSolver
from ikflow.model_loading import get_ik_solver
from ikflow.model import IkflowModelParameters, TINY_MODEL_PARAMS
from jrl.robot import Robot
from jrl.robots import Panda, Fetch, FetchArm
import torch

from cppflow.utils import (
    TimerContext,
    print_v1,
    print_v2,
    print_v3,
    _plot_self_collisions,
    _plot_env_collisions,
    _get_mjacs,
)
from cppflow.problem import Problem
from cppflow.config import DEBUG_MODE_ENABLED, DEVICE
from cppflow.search import dp_search
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

OPTIMIZATION_CONVERGENCE_THRESHOLD = 0.005


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


@dataclass
class PlannerSettings:
    k: int
    tmax_sec: float
    anytime_mode_enabled: bool
    latent_distribution: str = "uniform"
    latent_vector_scale: float = 2.0
    latent_drawing: str = "per_k"
    run_dp_search: bool = True
    do_rerun_if_optimization_fails: bool = False
    do_rerun_if_large_dp_search_mjac: bool = False
    rerun_mjac_threshold_deg: bool = DEFAULT_RERUN_MJAC_THRESHOLD_DEG
    rerun_mjac_threshold_cm: bool = DEFAULT_RERUN_MJAC_THRESHOLD_CM
    do_return_search_path_mjac: bool = False
    return_only_1st_plan: bool = False
    verbosity: int = 1

    def __post_init__(self):
        assert self.latent_drawing in {"per_timestep", "per_k"}
        assert self.latent_distribution in {"uniform", "gaussian"}
        assert self.latent_vector_scale > 0.0


def add_search_path_mjac(debug_info: Dict, problem: Problem, qpath_search: torch.Tensor):
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


class Planner:
    def __init__(self, settings: PlannerSettings, robot: Robot, is_mock: bool = False):
        if not is_mock:
            self._ikflow_model_name = ROBOT_TO_IKFLOW_MODEL[robot.name]
            self._ikflow_solver, _ = get_ik_solver(self._ikflow_model_name, robot=robot)
        else:
            print("Warning: Using a mocked IKFlow solver - this model has random weights")
            self._ikflow_model_name = "none - mocked"
            self._ikflow_solver = IKFlowSolver(TINY_MODEL_PARAMS, robot)

        self._network_width = self._ikflow_solver.network_width
        self._cfg = settings

    def set_settings(self, settings: PlannerSettings):
        self._cfg = settings

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
    def _get_fixed_random_latent(self, target_path_n_waypoints: int) -> torch.Tensor:
        """Returns the latent vector for the IKFlow call.

        Notes:
            1. For 'uniform', latent_vector_scale is the width of the sampling area for each dimension. The value 2.0
                is recommended. This was found in a hyperparameter search in the 'search_scatchpad.ipynb' notebook on
                March 13.

        Args:
            k: Number of paths
            target_path_n_waypoints: Number of waypoints in the target pose path
        """
        shape = (
            (self._cfg.k, self._network_width)
            if self._cfg.latent_drawing == "per_k"
            else (self._cfg.k * target_path_n_waypoints, self._network_width)
        )
        if self._cfg.latent_distribution == "gaussian":
            latents = torch.randn(shape, device=DEVICE) * self._cfg.latent_vector_scale  # [k x network_width]
        elif self._cfg.latent_distribution == "uniform":
            width = self._cfg.latent_vector_scale
            latents = torch.rand(shape, device=DEVICE) * width - (width / 2)
        if self._cfg.latent_drawing == "per_k":
            return torch.repeat_interleave(latents, target_path_n_waypoints, dim=0)
        return latents

    def _get_k_ikflow_qpaths(
        self,
        ee_path: torch.Tensor,
        batched_latent: torch.Tensor,
        use_rerun_k: bool,
        clamp_to_joint_limits: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns k different config space paths for the given ee_path."""
        n = ee_path.shape[0]
        k = self._cfg.k if not use_rerun_k else DEFAULT_RERUN_NEW_K
        with torch.inference_mode(), TimerContext("running IKFlow", enabled=self._cfg.verbosity > 0):
            ee_path_tiled = ee_path.repeat((k, 1))
            # TODO: Query model directly (remove 'generate_ik_solutions()' call)
            ikf_sols = self._ikflow_solver.generate_ik_solutions(
                ee_path_tiled,
                latent=batched_latent,
                clamp_to_joint_limits=clamp_to_joint_limits,
            )
        paths = [ikf_sols[i * n : (i * n) + n, :] for i in range(k)]
        return paths, ee_path_tiled

    def _run_pipeline(self, problem: Problem, **kwargs) -> Tuple[torch.Tensor, bool, TimingData]:
        existing_q_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None

        # ikflow
        t0_ikflow = time()
        batched_latents = self._get_fixed_random_latent(problem.n_timesteps)
        ikflow_qpaths, _ = self._get_k_ikflow_qpaths(problem.target_path, batched_latents, use_rerun_k=existing_q_data is not None)
        time_ikflow = time() - t0_ikflow

        # save initial solution
        if self._cfg.return_only_1st_plan:
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

        with TimerContext("calculating self-colliding configs", enabled=self._cfg.verbosity > 0):
            self_collision_violations = qpaths_batched_self_collisions(problem, qs)
            pct_colliding = (torch.sum(self_collision_violations) / (self._cfg.k * problem.n_timesteps)).item() * 100
            assert pct_colliding < 95.0, f"too many env collisions: {pct_colliding} %"
            print_v2(f"  self_collision violations: {pct_colliding} %")

            if self._cfg.verbosity > 2:
                warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
                _plot_self_collisions(self_collision_violations)

        with TimerContext("calculating env-colliding configs", enabled=self._cfg.verbosity > 0):
            env_collision_violations = qpaths_batched_env_collisions(problem, qs)
            pct_colliding = (torch.sum(env_collision_violations) / (self._cfg.k * problem.n_timesteps)).item() * 100
            assert pct_colliding < 95.0, f"too many env collisions: {pct_colliding} %"
            print_v2(f"  env_collision violations: {pct_colliding} %")

            if self._cfg.verbosity > 2:
                warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
                _plot_env_collisions(env_collision_violations)

        if existing_q_data is not None:
            qs_prev, self_collision_violations_prev, env_collision_violations_prev = existing_q_data
            qs = torch.cat([qs_prev, qs], dim=0)
            self_collision_violations = torch.cat([self_collision_violations_prev, self_collision_violations], dim=0)
            env_collision_violations = torch.cat([env_collision_violations_prev, env_collision_violations], dim=0)

        # qs is [ k x n_timesteps x ndof ]
        if problem.initial_configuration is not None:
            qs = torch.cat([problem.initial_configuration.expand((self.cfg.k, 1, self.robot.ndof)), qs], dim=0)
        # print(qs.shape)
        # exit()

        time_coll_check = time() - t0_col_check
        debug_info = {}
        q_data = (qs, self_collision_violations, env_collision_violations)

        # dp_search
        t0_dp_search = time()
        with TimerContext(f"running dynamic programming search with qs: {qs.shape}", enabled=self._cfg.verbosity > 0):
            qpath_search = dp_search(self.robot, qs, self_collision_violations, env_collision_violations).to(DEVICE)
        time_dp_search = time() - t0_dp_search
        if "results_df" in kwargs:
            write_qpath_to_results_df(kwargs["results_df"], qpath_search, problem)

        if DEBUG_MODE_ENABLED and self._cfg.do_return_search_path_mjac:
            add_search_path_mjac(debug_info, problem, qpath_search)

        return (
            qpath_search,
            False,
            TimingData(-1, time_ikflow, time_coll_check, 0.0, time_dp_search, 0.0),
            debug_info,
            q_data,
        )


class PlannerSearcher(Planner):
    """PlannerSearcher creates a finds a solution by performing a search through a graph constructed by connecting k
    ikflow generated cspace plans
    """

    def __init__(self, settings: PlannerSettings, robot: Robot):
        super().__init__(settings, robot)

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        """Runs dp_search and returns"""
        assert problem.robot.name == self.robot.name
        kwargs["run_dp_search"] = True

        t0 = time()
        qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)

        # rerun dp_search with larger k if mjac is too high
        if self._cfg.do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, qpath_search)
            if mjac_deg > self._cfg.rerun_mjac_threshold_deg or mjac_cm > self._cfg.rerun_mjac_threshold_cm:
                print_v1(
                    f"rerunning dp_search with larger k b/c mjac is too high: {mjac_deg} deg, {mjac_cm} cm",
                    verbosity=self._cfg.verbosity,
                )
                qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = _get_mjacs(problem.robot, qpath_search)
                print_v1(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", verbosity=self._cfg.verbosity)

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


class CppFlowPlanner(Planner):
    # Generalized planner that runs dp_search before LM optimization. Must specify the optimization version to use

    def __init__(self, settings: PlannerSettings, robot: Robot):
        super().__init__(settings, robot)

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        t0 = time()
        initial_k = self._cfg.k

        rerun_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None
        results_df = kwargs["results_df"] if "results_df" in kwargs else None
        search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)

        # Optionally return only the 1ST plan
        if self._cfg.return_only_1st_plan:
            return PlannerResult(
                plan_from_qpath(search_qpath, problem), TimingData(time() - t0, 0, 0, 0, 0, 0), [], [], {}
            )

        # rerun dp_search with larger k if mjac is too high
        if self._cfg.do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, search_qpath)
            if mjac_deg > self._cfg.rerun_mjac_threshold_deg or mjac_cm > self._cfg.rerun_mjac_threshold_cm:
                print_v1(
                    f"rerunning dp_search with larger k b/c mjac is too high: {mjac_deg} deg, {mjac_cm} cm",
                    verbosity=self._cfg.verbosity,
                )
                kwargs["rerun_data"] = q_data
                search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = _get_mjacs(problem.robot, search_qpath)
                print_v1(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", verbosity=self._cfg.verbosity)

        print_v1("\ndp_search path:", verbosity=self._cfg.verbosity)
        print_v1(str(plan_from_qpath(search_qpath, problem)), verbosity=self._cfg.verbosity)

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
        with TimerContext(f"running run_lm_optimization()", enabled=self._cfg.verbosity > 0):
            t0_opt = time()
            if self._cfg.anytime_mode_enabled:
                optimization_result = run_lm_optimization(
                    problem,
                    search_qpath,
                    max_n_steps=50,
                    tmax_sec=50,
                    return_if_valid_after_n_steps=int(1e8),
                    convergence_threshold=OPTIMIZATION_CONVERGENCE_THRESHOLD,
                    results_df=results_df,
                    verbosity=self._cfg.verbosity,
                )
            else:
                optimization_result = run_lm_optimization(
                    problem,
                    search_qpath,
                    tmax_sec=self._cfg.tmax_sec - (time() - t0),
                    max_n_steps=20,
                    return_if_valid_after_n_steps=15,
                    convergence_threshold=0.1,
                    results_df=results_df,
                    verbosity=self._cfg.verbosity,
                )

            time_optimizer = time() - t0_opt
            time_total = time() - t0
            debug_info["n_optimization_steps"] = optimization_result.n_steps_taken

        # Optionally rerun if optimization failed
        if not optimization_result.is_valid and self._cfg.do_rerun_if_optimization_fails and rerun_data is None:
            print_v1(f"rerunning dp_search because optimization failed", verbosity=self._cfg.verbosity)
            kwargs["rerun_data"] = q_data
            return self.generate_plan(problem, **kwargs)

        if "results_df" in kwargs:
            write_qpath_to_results_df(kwargs["results_df"], optimization_result.x_opt, problem)

        return PlannerResult(
            plan_from_qpath(optimization_result.x_opt.detach(), problem),
            TimingData(time_total, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, time_optimizer),
            [plan_from_qpath(search_qpath.detach(), problem)],
            ["search-plan"],
            debug_info,
        )
