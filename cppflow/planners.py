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
    make_text_green_or_red,
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
SINGLE_PT_ZERO = torch.zeros(1)

DEFAULT_RERUN_NEW_K = 125  # the existing k configs will be added to this so no need to go overboard
# DEFAULT_RERUN_NEW_K = 10  # the existing k configs will be added to this so no need to go overboard
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
    run_dp_search: bool = True
    do_rerun_if_optimization_fails: bool = False
    do_rerun_if_large_dp_search_mjac: bool = False
    rerun_mjac_threshold_deg: bool = DEFAULT_RERUN_MJAC_THRESHOLD_DEG
    rerun_mjac_threshold_cm: bool = DEFAULT_RERUN_MJAC_THRESHOLD_CM
    do_return_search_path_mjac: bool = False
    return_only_1st_plan: bool = False
    verbosity: int = 1

    def __post_init__(self):
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
    def _sample_latents(self, k: int, n_timesteps: int) -> torch.Tensor:
        """Returns the latent vector for the IKFlow call.

        Notes:
            1. For 'uniform', latent_vector_scale is the width of the sampling area for each dimension. The value 2.0
                is recommended. This was found in a hyperparameter search in the 'search_scatchpad.ipynb' notebook on
                March 13.

        Args:
            k: Number of paths
            n_timesteps: Number of waypoints in the target pose path
        """
        shape = (k, self._network_width)
        if self._cfg.latent_distribution == "gaussian":
            latents = torch.randn(shape, device=DEVICE) * self._cfg.latent_vector_scale  # [k x network_width]
        elif self._cfg.latent_distribution == "uniform":
            width = self._cfg.latent_vector_scale
            latents = torch.rand(shape, device=DEVICE) * width - (width / 2)
        return torch.repeat_interleave(latents, n_timesteps, dim=0)

    def _sample_latents_near(self, k: int, n_timesteps: int, center_latent: torch.Tensor) -> torch.Tensor:
        """Returns the latent vector for the IKFlow call.

        Notes:
            1. For 'uniform', latent_vector_scale is the width of the sampling area for each dimension. The value 2.0
                is recommended. This was found in a hyperparameter search in the 'search_scatchpad.ipynb' notebook on
                March 13.

        Args:
            k: Number of paths
            n_timesteps: Number of waypoints in the target pose path
        """
        assert center_latent.numel() == self._network_width, "given latent should be same dim as network width"
        shape = (k, self._network_width)
        width = self._cfg.latent_vector_scale
        latents = torch.rand(shape, device=DEVICE) * width - (width / 2) + center_latent  # [k x network_width]
        latents[0] = center_latent
        return torch.repeat_interleave(latents, n_timesteps, dim=0)

    def _get_k_ikflow_qpaths(
        self,
        ee_path: torch.Tensor,
        batched_latent: torch.Tensor,
        k: int,
        clamp_to_joint_limits: bool = True,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Returns k different config space paths for the given ee_path."""
        n = ee_path.shape[0]
        with torch.inference_mode(), TimerContext("running IKFlow", enabled=self._cfg.verbosity > 0):
            ee_path_tiled = ee_path.repeat((k, 1))
            ikf_sols = self._ikflow_solver.generate_ik_solutions(
                ee_path_tiled,
                latent=batched_latent,
                clamp_to_joint_limits=clamp_to_joint_limits,
            )
        paths = [ikf_sols[i * n : (i * n) + n, :] for i in range(k)]
        return paths

    def _get_configuration_corresponding_latent(self, qs: torch.Tensor, ee_pose: torch.Tensor) -> torch.Tensor:
        """Get the latent vectors that corresponds to the given configurations"""
        with torch.inference_mode(), TimerContext(
            "running IKFlow in reverse to get latent for initial_configuration", enabled=self._cfg.verbosity > 0
        ):
            if self.robot.ndof != self._ikflow_solver.network_width:
                model_input = torch.cat(
                    [qs.view(1, self.robot.ndof), torch.zeros(1, self._ikflow_solver.network_width - self.robot.ndof)],
                    dim=1,
                )
            else:
                model_input = qs
            conditional = torch.cat([ee_pose.view(1, 7), SINGLE_PT_ZERO.view(1, 1)], dim=1)
            output_rev, _ = self._ikflow_solver.nn_model(model_input, c=conditional, rev=False)
            return output_rev

    def _run_pipeline(self, problem: Problem, **kwargs) -> Tuple[torch.Tensor, bool, TimingData]:
        """Runs IKFlow, collision checking, and search."""
        existing_q_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None
        if "initial_q_latent" not in kwargs:
            kwargs["initial_q_latent"] = None

        # ikflow
        t0_ikflow = time()
        k = self._cfg.k if (existing_q_data is None) else DEFAULT_RERUN_NEW_K

        # Get latent matching 'problem.initial_configuration'
        if (problem.initial_configuration is not None) and (kwargs["initial_q_latent"] is None):
            kwargs["initial_q_latent"] = self._get_configuration_corresponding_latent(
                problem.initial_configuration, problem.target_path[0]
            )

        # Sample latents
        if kwargs["initial_q_latent"] is not None:
            batched_latents = self._sample_latents_near(
                k, problem.n_timesteps, kwargs["initial_q_latent"]
            )  # [ (n_timesteps * k) x network_width ]=
        else:
            batched_latents = self._sample_latents(k, problem.n_timesteps)  # [ (n_timesteps * k) x network_width ]

        # Run IKFlow
        ikflow_qpaths = self._get_k_ikflow_qpaths(problem.target_path, batched_latents, k)
        time_ikflow = time() - t0_ikflow

        # Optionally save initial solution
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
        qs = torch.stack(ikflow_qpaths)  # [ k x n_timesteps x ndof ]
        k_current = qs.shape[0]  # may be different from self._cfg.k if existing_q_data is not None

        with TimerContext("calculating self-colliding configs", enabled=self._cfg.verbosity > 0):
            self_collision_violations = qpaths_batched_self_collisions(problem, qs)
            pct_colliding = (torch.sum(self_collision_violations) / (k_current * problem.n_timesteps)).item() * 100

            torch.save(plan_from_qpath(qs[0, :, :], problem), f"many_env_collisions[1].pt")
            torch.save(plan_from_qpath(qs[1, :, :], problem), f"many_env_collisions[0].pt")
            torch.save(plan_from_qpath(qs[2, :, :], problem), f"many_env_collisions[2].pt")

            assert pct_colliding < 95.0, f"too many env collisions: {pct_colliding} %"
            print_v2(f"  self_collision violations: {pct_colliding} %")

            if self._cfg.verbosity > 2:
                warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
                _plot_self_collisions(self_collision_violations)

        with TimerContext("calculating env-colliding configs", enabled=self._cfg.verbosity > 0):
            env_collision_violations = qpaths_batched_env_collisions(problem, qs)
            pct_colliding = (torch.sum(env_collision_violations) / (k_current * problem.n_timesteps)).item() * 100
            assert pct_colliding < 95.0, f"too many env collisions: {pct_colliding} %"
            print_v2(f"  env_collision violations: {pct_colliding} %")
            if self._cfg.verbosity > 2:
                warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
                _plot_env_collisions(env_collision_violations)

        # Append previous qs to the new qs if it exists
        if existing_q_data is not None:
            qs_prev, self_collision_violations_prev, env_collision_violations_prev = existing_q_data
            qs = torch.cat([qs_prev, qs], dim=0)
            self_collision_violations = torch.cat([self_collision_violations_prev, self_collision_violations], dim=0)
            env_collision_violations = torch.cat([env_collision_violations_prev, env_collision_violations], dim=0)

        # qs is [ k x n_timesteps x ndof ]
        if problem.initial_configuration is not None:
            k_current = qs.shape[0]  # may be different from self._cfg.k if existing_q_data is not None
            q0 = problem.initial_configuration.expand((k_current, 1, self.robot.ndof))
            qs = torch.cat([q0, qs], dim=1)
            zeros = SINGLE_PT_ZERO.expand((k_current, 1))
            self_collision_violations = torch.cat([zeros, self_collision_violations], dim=1)
            env_collision_violations = torch.cat([zeros, env_collision_violations], dim=1)

        time_coll_check = time() - t0_col_check
        debug_info = {}

        # dp_search
        t0_dp_search = time()
        with TimerContext(f"running dynamic programming search with qs: {qs.shape}", enabled=self._cfg.verbosity > 0):

            # [ ntimesteps, ndof ]
            qpath_search = dp_search(self.robot, qs, self_collision_violations, env_collision_violations).to(DEVICE)

        # Remove initial configuration if it was added
        if problem.initial_configuration is not None:
            qpath_search = qpath_search[1:, :]  # remove initial configuration
            qs = qs[:, 1:, :]
            self_collision_violations = self_collision_violations[:, 1:]
            env_collision_violations = env_collision_violations[:, 1:]

        # Creating q_data needs to come after dp_search, so that in the case that there is an initial configuration
        # provided, it (the initial configuration) is removed from qs.
        q_data = (qs, self_collision_violations, env_collision_violations)
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


# ----------------------------------------------------------------------------------------------------------------------
# ---
# --- Planners
# ---


class PlannerSearcher(Planner):
    """PlannerSearcher creates a finds a solution by performing a search through a graph constructed by connecting k
    ikflow generated cspace plans
    """

    def __init__(self, settings: PlannerSettings, robot: Robot):
        super().__init__(settings, robot)
        assert self._cfg.run_dp_search

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        """Runs dp_search and returns"""
        assert problem.robot.name == self.robot.name

        t0 = time()
        qpath_search, _, td, debug_info, _ = self._run_pipeline(problem, **kwargs)

        # rerun dp_search with larger k if mjac is too high
        if self._cfg.do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, qpath_search)
            if mjac_deg > self._cfg.rerun_mjac_threshold_deg or mjac_cm > self._cfg.rerun_mjac_threshold_cm:
                print_v1(
                    f"\nRerunning dp_search with larger k b/c mjac is too high: {mjac_deg} deg, {mjac_cm} cm",
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


class CppFlowPlanner(Planner):
    # Generalized planner that runs dp_search before LM optimization. Must specify the optimization version to use

    def __init__(self, settings: PlannerSettings, robot: Robot):
        super().__init__(settings, robot)

    def generate_plan(self, problem: Problem, **kwargs) -> PlannerResult:
        t0 = time()
        rerun_data = kwargs["rerun_data"] if "rerun_data" in kwargs else None
        results_df = kwargs["results_df"] if "results_df" in kwargs else None
        search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)

        def time_is_exceeded():
            return time() - t0 > self._cfg.tmax_sec

        def return_(qpath):
            return PlannerResult(
                plan_from_qpath(qpath, problem),
                TimingData(time() - t0, td.ikflow, td.coll_checking, td.batch_opt, td.dp_search, 0.0),
                [],
                [],
                debug_info,
            )

        # Optionally return only the 1st plan
        if self._cfg.return_only_1st_plan:
            return PlannerResult(
                plan_from_qpath(search_qpath, problem), TimingData(time() - t0, 0, 0, 0, 0, 0), [], [], {}
            )

        # rerun dp_search with larger k if mjac is too high
        if self._cfg.do_rerun_if_large_dp_search_mjac:
            mjac_deg, mjac_cm = _get_mjacs(problem.robot, search_qpath)
            if mjac_deg > self._cfg.rerun_mjac_threshold_deg or mjac_cm > self._cfg.rerun_mjac_threshold_cm:
                print_v1(
                    f"{ make_text_green_or_red('Rerunning', False)} dp_search with larger k b/c mjac is too high:"
                    f" {mjac_deg} deg, {mjac_cm} cm",
                    verbosity=self._cfg.verbosity,
                )
                kwargs["rerun_data"] = q_data
                search_qpath, is_valid, td, debug_info, q_data = self._run_pipeline(problem, **kwargs)
                mjac_deg, mjac_cm = _get_mjacs(problem.robot, search_qpath)
                print_v1(f"new mjac after dp_search with larger k: {mjac_deg} deg,  cm", verbosity=self._cfg.verbosity)

        print_v2("\ndp_search path:", verbosity=self._cfg.verbosity)
        print_v2(str(plan_from_qpath(search_qpath, problem)), verbosity=self._cfg.verbosity)
        # print_v2(str(plan_from_qpath(search_qpath, problem)), verbosity=self._cfg.verbosity)

        # return if not anytime mode and search path is valid, or out of time
        if ((not self._cfg.anytime_mode_enabled) and is_valid) or time_is_exceeded():
            return return_(search_qpath)

        # Run optimization
        with TimerContext(f"running run_lm_optimization()", enabled=self._cfg.verbosity > 0):
            t0_opt = time()
            if self._cfg.anytime_mode_enabled:
                optimization_result = run_lm_optimization(
                    problem,
                    search_qpath,
                    max_n_steps=50,
                    tmax_sec=self._cfg.tmax_sec - (time() - t0),
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
                    max_n_steps=100,
                    return_if_valid_after_n_steps=0,
                    convergence_threshold=0.1,
                    results_df=results_df,
                    verbosity=self._cfg.verbosity,
                )
            td.optimizer = time() - t0_opt
            debug_info["n_optimization_steps"] = optimization_result.n_steps_taken

        # update convergence result
        if "results_df" in kwargs:
            write_qpath_to_results_df(kwargs["results_df"], optimization_result.x_opt, problem)

        # Check if past tmax_sec
        if time_is_exceeded():
            return return_(optimization_result.x_opt.detach())

        # Optionally rerun if optimization failed
        if not optimization_result.is_valid and self._cfg.do_rerun_if_optimization_fails and rerun_data is None:
            print_v1(f"\nRerunning dp_search because optimization failed", verbosity=self._cfg.verbosity)
            kwargs["rerun_data"] = q_data
            return self.generate_plan(problem, **kwargs)

        return return_(optimization_result.x_opt.detach())
