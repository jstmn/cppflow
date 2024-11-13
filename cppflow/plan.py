from dataclasses import dataclass
from typing import List, Tuple, Dict
from time import time

import torch

from cppflow.problem import Problem, problem_from_filename
from cppflow.config import (
    SUCCESS_THRESHOLD_translation_ERR_MAX_CM,
    SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG,
    SUCCESS_THRESHOLD_mjac_DEG,
    SUCCESS_THRESHOLD_mjac_CM,
    env_collisions_ignored,
    self_collisions_ignored,
)

from cppflow.utils import make_text_green_or_red, to_torch
from cppflow.evaluation_utils import (
    positional_errors,
    rotational_errors,
    angular_changes,
    prismatic_changes,
    joint_limits_exceeded,
    errors_are_below_threshold,
    calculate_per_timestep_mjac_deg,
    calculate_per_timestep_mjac_cm,
)
from cppflow.collision_detection import self_colliding_configs_klampt, env_colliding_configs_klampt


def write_qpath_to_results_df(results_df: Dict, qpath: torch.Tensor, problem: Problem):
    tnow = time()
    plan = plan_from_qpath(qpath, problem)
    results_df["t0"] += time() - tnow
    plan.append_to_results_df(results_df)


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
    def is_valid(self) -> bool:
        joint_limits_violated, _ = joint_limits_exceeded(self.robot_joint_limits, self.q_path)
        iv = (
            not joint_limits_violated
            and errors_are_below_threshold(
                self.positional_errors_cm,
                self.rotational_errors_deg,
                self.mjac_per_timestep_deg,
                self.mjac_per_timestep_cm,
            )[0]
            and self.self_colliding_per_ts.sum() == 0
            and self.env_colliding_per_ts.sum() == 0
        )
        if isinstance(iv, torch.Tensor):
            return iv.item()
        return iv

    def __str__(self) -> str:
        s = "<Plan>" + "\n"

        # is valid
        mjac_deg_valid = self.mjac_deg < SUCCESS_THRESHOLD_mjac_DEG
        mjac_cm_valid = self.mjac_cm < SUCCESS_THRESHOLD_mjac_CM
        max_t_error_valid = self.max_positional_error_cm < SUCCESS_THRESHOLD_translation_ERR_MAX_CM
        max_R_error_valid = self.max_rotational_error_deg < SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG
        joint_limits_violated, joint_limit_violation_pct = joint_limits_exceeded(self.robot_joint_limits, self.q_path)
        self_coll_valid = self.self_colliding_per_ts.sum() == 0
        env_coll_valid = self.env_colliding_per_ts.sum() == 0
        is_valid = self.is_valid
        assert is_valid == (
            mjac_deg_valid
            and mjac_cm_valid
            and max_t_error_valid
            and max_R_error_valid
            and (not joint_limits_violated)
            and self_coll_valid
            and env_coll_valid
        ), (
            f"self.isvalid disagrees with manual calculation.\n  self.is_valid: {is_valid}\n  mjac_deg_valid:"
            f" {mjac_deg_valid}\n  mjac_cm_valid: {mjac_cm_valid}\n  "
            f" max_t_error_valid: {max_t_error_valid}\n  max_R_error_valid:"
            f" {max_R_error_valid}\n  joint_limits_valid: {not joint_limits_violated}"
        )
        round_amt = 5
        s += f"  is_valid:         {make_text_green_or_red(is_valid, is_valid)}\n"
        s += (
            f"    mjac < {SUCCESS_THRESHOLD_mjac_DEG} deg:                   "
            f"{make_text_green_or_red(mjac_deg_valid, mjac_deg_valid)}\n"
        )
        s += (
            f"    mjac < {SUCCESS_THRESHOLD_mjac_CM} cm:                    "
            f"{make_text_green_or_red(mjac_cm_valid, mjac_cm_valid)}\n"
        )
        s += (
            f"    max positional error < {10*SUCCESS_THRESHOLD_translation_ERR_MAX_CM} mm:  "
            f"{make_text_green_or_red(max_t_error_valid, max_t_error_valid)}\n"
        )
        s += (
            f"    max rotational error < {SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG} deg: "
            f"{make_text_green_or_red(max_R_error_valid, max_R_error_valid)}\n"
        )
        s += (
            "    joint limits in bounds:        "
            f" {make_text_green_or_red(not joint_limits_violated, not joint_limits_violated)}\n"
        )
        if joint_limits_violated:
            for i, violation_pct in enumerate(joint_limit_violation_pct):
                s += (
                    f"      % of q_path[:, {i}] violating JL:"
                    f" {make_text_green_or_red(violation_pct.item(), abs(violation_pct) < 1e-8)}\n"
                )
        s += (
            "    # self collisions:              "
            f"{make_text_green_or_red(self.self_colliding_per_ts.sum().item(), self.self_colliding_per_ts.sum() == 0)}\n"
        )
        s += (
            "    # env. collisions:              "
            f"{make_text_green_or_red(self.env_colliding_per_ts.sum().item(), self.env_colliding_per_ts.sum() == 0)}\n"
        )

        # ---
        s += "  ---\n"
        s += f"                  mjac: {round(self.mjac_deg, round_amt)} deg\n"
        s += f"                  mjac: {round(self.mjac_cm, round_amt)} cm\n"
        s += f"  ave positional error: {round(self.mean_positional_error_mm, round_amt)} mm\n"
        s += f"  max positional error: {round(self.max_positional_error_mm, round_amt)} mm\n"
        s += f"  ave rotational error: {round(self.mean_rotational_error_deg, round_amt)} deg\n"
        s += f"  max rotational error: {round(self.max_rotational_error_deg, round_amt)} deg\n"
        s += "  ---\n"
        s += f"       path length rad: {round(self.path_length_rad, round_amt)}\n"
        s += f"         path length m: {round(self.path_length_m, round_amt)}\n"
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


def plan_from_qpath(qpath: torch.Tensor, problem: Problem) -> Plan:
    """Converts a qpath to a plan.

    Note: mean runtime is 0.08366363048553467. Don't call this when timed
    """
    assert isinstance(qpath, torch.Tensor), f"qpath must be a torch.Tensor, got {type(qpath)}"
    traced_path = problem.robot.forward_kinematics(qpath, out_device=qpath.device)
    qpath_revolute, qpath_prismatic = problem.robot.split_configs_to_revolute_and_prismatic(qpath)

    # Use klampts collision checker here instead of jrl's capsule-capsule checking. klampt uses the collision
    # geometry of the robot which is a tighter bound.
    self_colliding = self_colliding_configs_klampt(problem, qpath)
    if self_collisions_ignored:
        self_colliding = torch.zeros_like(self_colliding)

    env_colliding = env_colliding_configs_klampt(problem, qpath)
    if env_collisions_ignored:
        env_colliding = torch.zeros_like(env_colliding)

    return Plan(
        q_path=qpath,
        q_path_revolute=qpath_revolute,
        q_path_prismatic=qpath_prismatic,
        pose_path=traced_path,
        target_path=problem.target_path,
        robot_joint_limits=problem.robot.actuated_joints_limits,
        self_colliding_per_ts=self_colliding,
        env_colliding_per_ts=env_colliding,
        positional_errors=positional_errors(traced_path, problem.target_path),
        rotational_errors=rotational_errors(traced_path, problem.target_path),
    )


""" python cppflow/plan.py
"""

if __name__ == "__main__":
    problem = problem_from_filename("fetch__square")
    qpath = to_torch(problem.robot.sample_joint_angles(problem.n_timesteps))
    ts = []
    for _ in range(10):
        t0 = time()
        plan_from_qpath(qpath, problem)
        ts.append(time() - t0)
    print("total time for 10 runs:", sum(ts))
    print("          mean runtime:", sum(ts) / len(ts))
