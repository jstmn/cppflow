from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from time import time

from jrl.robot import Robot
from jrl.math_utils import quaternion_norm, geodesic_distance_between_quaternions
import torch
import numpy as np
from klampt import RigidObjectModel

from cppflow.config import (
    DEFAULT_RERUN_MJAC_THRESHOLD_DEG,
    DEFAULT_RERUN_MJAC_THRESHOLD_CM,
    SUCCESS_THRESHOLD_initial_q_norm_dist,
)
from cppflow.utils import make_text_green_or_red, cm_to_mm, m_to_mm
from cppflow.evaluation_utils import (
    angular_changes,
    prismatic_changes,
    joint_limits_exceeded,
    errors_are_below_threshold,
    calculate_per_timestep_mjac_deg,
    calculate_per_timestep_mjac_cm,
)


@dataclass
class TimingData:
    total: float
    ikflow: float
    coll_checking: float
    batch_opt: float
    dp_search: float
    optimizer: float

    def __str__(self):
        def cond(t):
            if t < 1e-6:
                return t
            return f"{t:.5f}"

        s = "TimingData {\n"
        s += f"  total:         {cond(self.total)}\n"
        s += f"  ikflow:        {cond(self.ikflow)}\n"
        s += f"  coll_checking: {cond(self.coll_checking)}\n"
        s += f"  batch_opt:     {cond(self.batch_opt)}\n"
        s += f"  dp_search:     {cond(self.dp_search)}\n"
        s += f"  optimizer:     {cond(self.optimizer)}\n"
        s += "}"
        return s


@dataclass
class Constraints:
    max_allowed_position_error_cm: float
    max_allowed_rotation_error_deg: float
    max_allowed_mjac_deg: float
    max_allowed_mjac_cm: float

    @property
    def max_allowed_position_error_m(self):
        return self.max_allowed_position_error_cm / 100


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


@dataclass
class Plan:
    # Plan data
    q_path: torch.Tensor
    q_path_revolute: torch.Tensor
    q_path_prismatic: torch.Tensor
    pose_path: torch.Tensor
    target_path: torch.Tensor
    robot_joint_limits: List[Tuple[float, float]]

    # Errors
    self_colliding_per_ts: torch.Tensor
    env_colliding_per_ts: torch.Tensor
    positional_errors: torch.Tensor
    rotational_errors: torch.Tensor
    provided_initial_configuration: Optional[torch.Tensor]

    constraints: Constraints

    def __post_init__(self):
        assert isinstance(self.q_path, torch.Tensor)
        assert self.q_path.shape == (self.target_path.shape[0], len(self.robot_joint_limits)), (
            f"Error: qpath.shape = {self.q_path.shape}, should be"
            f" {(self.target_path.shape[0], len(self.robot_joint_limits))}"
        )
        assert isinstance(self.q_path_revolute, torch.Tensor)
        assert isinstance(self.q_path_prismatic, torch.Tensor)
        assert isinstance(self.pose_path, torch.Tensor)
        assert isinstance(self.target_path, torch.Tensor)
        assert self.positional_errors.numel() == self.q_path.shape[0]
        assert self.rotational_errors.numel() == self.q_path.shape[0]
        assert self.target_path.shape[0] == self.q_path.shape[0]

    def append_to_results_df(self, df_wrapped: Dict):
        tnow = time()
        new_row = [
            # "Time Elapsed (s)", "is valid", "Mean Pos Error (mm)", "Max Pos Error (mm)", "Mean Rot Error (deg)", "Max Rot Error (deg)", "Mjac (deg)", "Mjac (cm)", "pct_self-colliding", "pct_env-colliding", "path_length_rad", "path_length_m"
            0,
            self.is_valid,
            self.mean_positional_error_mm,
            self.max_positional_error_mm,
            self.mean_rotational_error_deg,
            self.max_rotational_error_deg,
            self.mjac_deg,
            self.mjac_cm,
            (self.self_colliding_per_ts.sum() / self.self_colliding_per_ts.numel()).item(),
            (self.env_colliding_per_ts.sum() / self.env_colliding_per_ts.numel()).item(),
            self.path_length_rad,
            self.path_length_m,
        ]
        df_wrapped["t0"] += time() - tnow
        new_row[0] = time() - df_wrapped["t0"]
        df_wrapped["df"].loc[len(df_wrapped["df"])] = new_row

    # Path length
    @property
    def path_length_rad(self) -> float:
        return angular_changes(self.q_path_revolute).abs().sum().item()

    @property
    def path_length_m(self) -> float:
        if self.q_path_prismatic.numel() > 0:
            return prismatic_changes(self.q_path_prismatic).abs().sum().item()
        return 0.0

    @property
    def is_a_prismatic_joint(self) -> bool:
        return self.q_path_prismatic.numel() > 0

    # Rotational error
    @property
    def rotational_errors_deg(self):
        return torch.rad2deg(self.rotational_errors)

    @property
    def max_rotational_error_deg(self) -> float:
        return float(self.rotational_errors_deg.max())

    @property
    def mean_rotational_error_deg(self) -> float:
        return float(self.rotational_errors_deg.mean())

    # Positional error
    @property
    def positional_errors_cm(self):
        return 100 * self.positional_errors

    @property
    def positional_errors_mm(self):
        return 1000 * self.positional_errors

    @property
    def max_positional_error_cm(self) -> float:
        return float(self.positional_errors_cm.max())

    @property
    def max_positional_error_mm(self) -> float:
        return self.max_positional_error_cm * 10.0

    @property
    def mean_positional_error_cm(self) -> float:
        return float(self.positional_errors_cm.mean())

    @property
    def mean_positional_error_mm(self) -> float:
        return self.mean_positional_error_cm * 10.0

    # Mjac - revolute
    @property
    def mjac_per_timestep_deg(self):
        return calculate_per_timestep_mjac_deg(self.q_path_revolute)

    @property
    def mjac_deg(self) -> float:
        """Per-timestep mjac for the robots revolute joints"""
        return self.mjac_per_timestep_deg.max().item()

    # Mjac - prismatic
    @property
    def mjac_per_timestep_cm(self):
        """Per-timestep mjac for the robots prismatic joints"""
        if self.q_path_prismatic.numel() == 0:
            return torch.zeros(self.target_path.shape[0] - 1, device=self.q_path.device, dtype=self.q_path.dtype)
        return calculate_per_timestep_mjac_cm(self.q_path_prismatic)

    @property
    def mjac_cm(self) -> float:
        if self.q_path_prismatic.numel() == 0:
            return 0.0
        return float(self.mjac_per_timestep_cm.max())

    # is valid
    @property
    def joint_limits_violated(self) -> bool:
        return joint_limits_exceeded(self.robot_joint_limits, self.q_path)[0]

    @property
    def initial_q_norm_dist(self) -> float:
        if self.provided_initial_configuration is None:
            return 0.0
        return torch.norm(self.provided_initial_configuration - self.q_path[0]).item()

    @property
    def is_valid(self) -> bool:
        joint_limits_violated, _ = joint_limits_exceeded(self.robot_joint_limits, self.q_path)
        iv = (
            not joint_limits_violated
            and errors_are_below_threshold(
                self.constraints.max_allowed_position_error_cm,
                self.constraints.max_allowed_rotation_error_deg,
                self.constraints.max_allowed_mjac_deg,
                self.constraints.max_allowed_mjac_cm,
                self.positional_errors_cm,
                self.rotational_errors_deg,
                self.mjac_per_timestep_deg,
                self.mjac_per_timestep_cm,
            )[0]
            and self.self_colliding_per_ts.sum() == 0
            and self.env_colliding_per_ts.sum() == 0
            and self.initial_q_norm_dist < SUCCESS_THRESHOLD_initial_q_norm_dist
        )
        if isinstance(iv, torch.Tensor):
            return iv.item()
        return iv

    def __str__(self) -> str:
        s = "Plan {\n"

        # is valid
        mjac_deg_valid = self.mjac_deg < self.constraints.max_allowed_mjac_deg
        mjac_cm_valid = self.mjac_cm < self.constraints.max_allowed_mjac_cm
        max_t_error_valid = self.max_positional_error_cm < self.constraints.max_allowed_position_error_cm
        max_R_error_valid = self.max_rotational_error_deg < self.constraints.max_allowed_rotation_error_deg
        joint_limits_violated, joint_limit_violation_pct = joint_limits_exceeded(self.robot_joint_limits, self.q_path)
        self_coll_valid = self.self_colliding_per_ts.sum() == 0
        env_coll_valid = self.env_colliding_per_ts.sum() == 0
        is_valid = self.is_valid
        initial_q_valid = self.initial_q_norm_dist < SUCCESS_THRESHOLD_initial_q_norm_dist
        assert is_valid == (
            mjac_deg_valid
            and mjac_cm_valid
            and max_t_error_valid
            and max_R_error_valid
            and (not joint_limits_violated)
            and self_coll_valid
            and env_coll_valid
            and initial_q_valid
        ), (
            f"self.isvalid disagrees with manual calculation.\n  self.is_valid: {is_valid}\n  mjac_deg_valid:"
            f" {mjac_deg_valid}\n  mjac_cm_valid: {mjac_cm_valid}\n  "
            f" max_t_error_valid: {max_t_error_valid}\n  max_R_error_valid:"
            f" {max_R_error_valid}\n  joint_limits_valid: {not joint_limits_violated}"
        )
        round_amt = 5
        s += f"  is_valid:         \t   {make_text_green_or_red(is_valid, is_valid)}\n"
        s += (
            f"  mjac < {self.constraints.max_allowed_mjac_deg} deg:                 "
            f" {make_text_green_or_red(mjac_deg_valid, mjac_deg_valid)}\n"
        )
        s += (
            f"  mjac < {self.constraints.max_allowed_mjac_cm} cm:                  "
            f" {make_text_green_or_red(mjac_cm_valid, mjac_cm_valid)}\n"
        )
        s += (
            f"  max positional error < {10*self.constraints.max_allowed_position_error_cm} mm:  "
            f" {make_text_green_or_red(max_t_error_valid, max_t_error_valid)}\n"
        )
        s += (
            f"  max rotational error < {self.constraints.max_allowed_rotation_error_deg} deg: "
            f" {make_text_green_or_red(max_R_error_valid, max_R_error_valid)}\n"
        )
        s += (
            "  joint limits in bounds:        "
            f"  {make_text_green_or_red(not joint_limits_violated, not joint_limits_violated)}\n"
        )
        s += f"  close-to-initial-configuration:  {make_text_green_or_red(initial_q_valid, initial_q_valid)}\n"
        if joint_limits_violated:
            for i, violation_pct in enumerate(joint_limit_violation_pct):
                s += (
                    f"    % of q_path[:, {i}] violating JL:"
                    f"  {make_text_green_or_red(violation_pct.item(), abs(violation_pct) < 1e-8)}\n"
                )
        s += (
            "  # self collisions:              "
            f" {make_text_green_or_red(self.self_colliding_per_ts.sum().item(), self.self_colliding_per_ts.sum() == 0)}\n"
        )
        s += (
            "  # env. collisions:              "
            f" {make_text_green_or_red(self.env_colliding_per_ts.sum().item(), self.env_colliding_per_ts.sum() == 0)}\n"
        )
        s += "  .\n"  # sep
        s += f"  mjac:                  {round(self.mjac_deg, round_amt)} deg\n"
        s += f"  mjac:                  {round(self.mjac_cm, round_amt)} cm\n"
        s += f"  ave positional error:  {round(self.mean_positional_error_mm, round_amt)} mm\n"
        s += f"  max positional error:  {round(self.max_positional_error_mm, round_amt)} mm\n"
        s += f"  ave rotational error:  {round(self.mean_rotational_error_deg, round_amt)} deg\n"
        s += f"  max rotational error:  {round(self.max_rotational_error_deg, round_amt)} deg\n"
        s += f"  q_initial norm dist:   {round(self.initial_q_norm_dist, round_amt)}\n"
        s += "  .\n"  # sep
        s += f"  trajectory length:     {round(self.path_length_rad, round_amt)} rad\n"
        s += f"  trajectory length:     {round(self.path_length_m, round_amt)} m\n"
        s += "}"
        return s


@dataclass
class PlanNp:
    """Convenience class for converting everything in a Plan object to numpy arrays"""

    plan: Plan

    def __getattribute__(self, attr):
        if attr == "plan":
            return super().__getattribute__("__dict__")["plan"]
        assert attr in dir(self.plan), f"Error: '{attr}' not found in Plan class (all items: {dir(self.plan)})"

        item = self.plan.__getattribute__(attr)
        if isinstance(item, torch.Tensor):
            return item.cpu().numpy()
        return item


@dataclass
class PlannerResult:
    plan: Plan
    timing: TimingData
    other_plans: List[Plan]
    other_plans_names: List[str]
    debug_info: dict


@dataclass
class Problem:
    constraints: Constraints
    target_path: torch.Tensor
    initial_configuration: Optional[torch.Tensor]
    robot: Robot
    name: str
    full_name: str
    obstacles: Optional[List]
    obstacles_Tcuboids: Optional[List]
    obstacles_cuboids: Optional[List]
    obstacles_klampt: List[RigidObjectModel]

    @property
    def n_timesteps(self) -> int:
        return self.target_path.shape[0]

    @property
    def fancy_name(self) -> str:
        return f"{self.robot.formal_robot_name} - {self.name}"

    @property
    def path_length_cumultive_positional_change_cm(self) -> float:
        """Sum of the positional difference between consecutive poses of the target path"""
        return float(torch.norm(self.target_path[1:, 0:3] - self.target_path[:-1, 0:3], dim=1).sum()) * 100.0

    @property
    def path_length_cumulative_rotational_change_deg(self) -> float:
        """Sum of the geodesic distance between consecutive poses of the target path"""
        q0 = self.target_path[0:-1, 3:7]
        q1 = self.target_path[1:, 3:7]
        # Copied from geodesic_distance_between_quaternions():
        #   dot = torch.clip(torch.sum(q1 * q2, dim=1), -1, 1)
        #   distance = 2 * torch.acos(torch.clamp(dot, -1 + acos_clamp_epsilon, 1 - acos_clamp_epsilon))
        acos_clamp_epsilon = 1e-7
        dot = torch.sum(q0 * q1, dim=1)
        dot_is_1 = torch.logical_or(dot > 1 - acos_clamp_epsilon, dot < -1 + acos_clamp_epsilon)
        imagined_error_per_elem = 2 * torch.acos(torch.tensor([1 - acos_clamp_epsilon], device=self.target_path.device))
        total_imagined_error = imagined_error_per_elem * torch.sum(dot_is_1)
        #
        rotational_changes = geodesic_distance_between_quaternions(q0, q1)
        return torch.rad2deg(rotational_changes.abs().sum() - total_imagined_error).item()

    def write_qpath_to_results_df(self, results_df: Dict, qpath: torch.Tensor):
        raise NotImplementedError("need to refactor to avoid circular dependencies from plan_from_qpath()")
        tnow = time()
        plan = plan_from_qpath(qpath, problem)
        results_df["t0"] += time() - tnow
        plan.append_to_results_df(results_df)

    def __str__(self, prefix: str = "") -> str:
        n = self.target_path.shape[0]
        s = f"{prefix}<Problem>"
        s += f"\n{prefix}  name:                        " + self.name
        s += f"\n{prefix}  full_name:                   " + self.full_name
        s += f"\n{prefix}  robot:                       " + str(self.robot)
        s += f"\n{prefix}  target_path:                 " + str(self.target_path.shape)
        s += (
            f"\n{prefix}  path length (m):            "
            f" {round(self.path_length_cumultive_positional_change_cm/100.0, 4)}"
        )
        s += f"\n{prefix}  path length (deg)            {round(self.path_length_cumulative_rotational_change_deg, 4)}"
        s += f"\n{prefix}  number of waypoints:         {n}"
        return s + "\n"

    def __post_init__(self):
        assert isinstance(self.robot, Robot), f"Error - self.robot is {type(self.robot)}, should be Robot type"
        # Sanity check target poses
        norms = quaternion_norm(self.target_path[:, 3:7])
        if max(norms) > 1.01 or min(norms) < 0.99:
            raise ValueError("quaternion(s) are not unit quaternion(s)")
        if self.initial_configuration is not None:
            np.set_printoptions(precision=5, suppress=True)
            torch.set_printoptions(precision=5, sci_mode=False)
            max_translation_mm = cm_to_mm(self.constraints.max_allowed_mjac_cm)
            assert (
                len(self.initial_configuration.shape) == 2
            ), f"'initial_configuration' should be [1, ndof], is {self.initial_configuration.shape}"
            fk_jrl = self.robot.forward_kinematics(self.initial_configuration)[0]
            waypoint_0_np = self.target_path[0].cpu().numpy()
            fk_error_jrl: torch.Tensor = fk_jrl - self.target_path[0]
            fk_error_klampt: np.ndarray = (
                self.robot.forward_kinematics_klampt(self.initial_configuration.cpu().numpy()) - waypoint_0_np[None, :]
            )[0]
            assert (
                fk_error_jrl[0:3].numel() == 3
            ), f"Fk position error term should be shaped [3], is {fk_error_jrl.shape}"
            assert (
                fk_error_klampt[0:3].size == 3
            ), f"Fk position error term should be shaped [3], is {fk_error_klampt.shape}"

            fk_error_pos_mm = m_to_mm(fk_error_jrl[0:3].norm().item())
            fk_error_klampt_pos_mm = m_to_mm(np.linalg.norm(fk_error_klampt[0:3]))

            if fk_error_pos_mm < max_translation_mm:
                raise ValueError(
                    f"Position error for `initial_configuration` is too large ({fk_error_pos_mm:.5f} >"
                    f" {max_translation_mm} mm)\n    FK_jrl(q_initial) = t({fk_jrl[0:3].cpu().numpy()}),"
                    f" quat({fk_jrl[3:7].cpu().numpy()})\n    waypoint[0] = t({waypoint_0_np[0:3]}),"
                    f" quat({waypoint_0_np[3:7]})\n    delta(FK(q_initial) - waypoint[0]) ="
                    f" t({fk_error_jrl.cpu().numpy()[0:3]}), quat({fk_error_jrl.cpu().numpy()[3:7]})"
                )
            if fk_error_klampt_pos_mm < max_translation_mm:
                raise ValueError(
                    f"Position error for `initial_configuration` is too large ({fk_error_klampt_pos_mm:.5f} >"
                    f" {max_translation_mm} mm)\n    FK_klampt(q_initial) = t({fk_error_klampt[0:3]}),"
                    f" quat({fk_error_klampt[3:7]})\n    waypoint[0] = t({waypoint_0_np[0:3]}),"
                    f" quat({waypoint_0_np[3:7]})\n    delta(FK(q_initial) - waypoint[0]) ="
                    f" t({fk_error_klampt.cpu().numpy()[0:3]}), quat({fk_error_klampt.cpu().numpy()[3:7]})"
                )
