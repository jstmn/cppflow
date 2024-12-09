from typing import List, Tuple

from jrl.robot import Robot
from jrl.math_utils import geodesic_distance_between_quaternions

import numpy as np
import torch

from cppflow.config import DEVICE


# =================================
#   ==  Trajectory validation  ==
#


def joint_limits_exceeded(robot_joint_limits: List[Tuple[float, float]], qs: np.ndarray) -> Tuple[bool, List[float]]:
    """Return the percent of configurations for each joint that are violating the given joint limits."""
    assert len(robot_joint_limits) == qs.shape[1]
    n = qs.shape[0]
    violation_pcts = []
    for i, (l, u) in enumerate(robot_joint_limits):
        assert l < u
        qs_i = qs[:, i]
        n_violating = (qs_i < l).sum() + (u < qs_i).sum()
        violation_pcts.append(100 * n_violating / n)
    return any([vp > 0 for vp in violation_pcts]), violation_pcts


def errors_are_below_threshold(
    max_allowed_position_error_cm: float,
    max_allowed_rotation_error_deg: float,
    max_allowed_mjac_deg: float,
    max_allowed_mjac_cm: float,
    error_t_cm: torch.Tensor,
    error_R_deg: torch.Tensor,
    qdeltas_revolute_deg: torch.Tensor,
    qdeltas_prismatic_cm: torch.Tensor,
    verbosity: int = 0,
) -> bool:
    """Check whether the given errors are below the set success thresholds."""
    pose_pos_valid = (error_t_cm.max() < max_allowed_position_error_cm).item()
    pose_rot_valid = (error_R_deg.max() < max_allowed_rotation_error_deg).item()
    if verbosity > 0 and not pose_pos_valid:
        print(
            "errors_are_below_threshold() | pose-position is invalid (error_t_cm.max() <"
            f" max_allowed_position_error_cm): {error_t_cm.max()} <"
            f" {max_allowed_position_error_cm}"
        )
    if verbosity > 0 and not pose_rot_valid:
        print(
            "errors_are_below_threshold() | pose-rotation is invalid (error_R_deg.max() <"
            f" max_allowed_rotation_error_deg): {error_R_deg.max()} < {max_allowed_rotation_error_deg}"
        )

    mjac_rev_valid = qdeltas_revolute_deg.abs().max() < max_allowed_mjac_deg
    mjac_pris_valid = (
        qdeltas_prismatic_cm.abs().max() < max_allowed_mjac_deg if qdeltas_prismatic_cm.numel() > 0 else True
    )
    if verbosity > 0:
        if not mjac_rev_valid:
            print(
                f"errors_are_below_threshold() | mjac_rev is invalid: {qdeltas_revolute_deg.abs().max():.3f} >"
                f" {max_allowed_mjac_deg:.3f}"
            )
        if not mjac_pris_valid:
            print(
                f"errors_are_below_threshold() | mjac_pris is invalid: {qdeltas_prismatic_cm.abs().max():.3f} >"
                f" {max_allowed_mjac_deg:.3f}"
            )
    return pose_pos_valid and pose_rot_valid and mjac_rev_valid and mjac_pris_valid, (
        pose_pos_valid,
        pose_rot_valid,
        mjac_rev_valid,
        mjac_pris_valid,
    )


# ===============================
#   ==  Joint angle changes  ==
#


def calculate_mjac_deg(x: torch.Tensor) -> float:
    """Calculate the maximum change in configuration space over a path in configuration space."""
    return torch.rad2deg(angular_changes(x).abs().max()).item()


def calculate_per_timestep_mjac_deg(x: torch.Tensor) -> torch.Tensor:
    """Calculate the maximum change in configuration space over a path in configuration space."""
    return torch.max(torch.rad2deg(angular_changes(x).abs()), dim=1).values


def calculate_per_timestep_mjac_cm(x: torch.Tensor) -> torch.Tensor:
    return 100 * torch.max(prismatic_changes(x).abs(), dim=1).values


def prismatic_changes(x: torch.Tensor) -> torch.Tensor:
    return x[1:] - x[0:-1]


def get_mjacs(robot: Robot, qpath: torch.Tensor):
    qps_revolute, qps_prismatic = robot.split_configs_to_revolute_and_prismatic(qpath)
    if qps_prismatic.numel() > 0:
        return calculate_mjac_deg(qps_revolute), calculate_per_timestep_mjac_cm(qps_prismatic).abs().max().item()
    return calculate_mjac_deg(qps_revolute), 0.0


# ======================
#   ==  Pose error  ==
#


def calculate_pose_error_cm_deg(robot: Robot, x: torch.Tensor, target_path: torch.Tensor):
    """Calculate the positional and rotational errors in cm and rad of a config path"""
    traced_path = robot.forward_kinematics(x, out_device=DEVICE)
    return 100 * positional_errors(target_path, traced_path), torch.rad2deg(rotational_errors(target_path, traced_path))


def calculate_pose_error_mm_deg_and_mjac_cm_deg(robot: Robot, x: torch.Tensor, target_path: torch.Tensor):
    """Calculate the positional and rotational errors in mm of a config path"""
    traced_path = robot.forward_kinematics(x, out_device=DEVICE)
    x_revolute, x_prismatic = robot.split_configs_to_revolute_and_prismatic(x)
    mjac_cm = 0.0
    if x_prismatic.numel() > 0:
        mjac_cm = float(100 * x_prismatic.abs().max())
    return (
        1000 * positional_errors(target_path, traced_path),
        torch.rad2deg(rotational_errors(target_path, traced_path)),
        float(torch.rad2deg(angular_changes(x_revolute).abs().max())),
        mjac_cm,
    )


def positional_errors(path_1: torch.Tensor, path_2: torch.Tensor) -> torch.Tensor:
    """Return the positional errors between two pose paths"""
    return torch.norm(path_1[:, :3] - path_2[:, :3], dim=1)


def rotational_errors(path_1: torch.Tensor, path_2: torch.Tensor) -> torch.Tensor:
    """Computes the summed rotational error between two cartesian space paths."""
    return geodesic_distance_between_quaternions(path_1[:, 3:], path_2[:, 3:])


def angular_changes(qpath: torch.Tensor) -> torch.Tensor:
    """Computes the change in the configuration space path. Respects jumps from 0 <-> 2pi

    WARNING: Results may be negative. Be sure to call .abs() if calculating the maximum absolute joint angle change

    Returns: a [n x ndof] array of the change in each joint angle over the n timesteps.
    """
    dqs = qpath[1:] - qpath[0:-1]
    if isinstance(qpath, torch.Tensor):
        return torch.remainder(dqs + torch.pi, 2 * torch.pi) - torch.pi
    return np.remainder(dqs + np.pi, 2 * np.pi) - np.pi
