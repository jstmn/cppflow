from dataclasses import dataclass
from typing import Optional, Tuple, List
from time import time
import warnings

import pandas as pd
import torch
import numpy as np
from jrl.math_utils import quaternion_inverse, quaternion_product, quaternion_to_rpy, angular_subtraction
from jrl.robot import Robot

from cppflow.utils import cm_to_m, make_text_green_or_red
from cppflow.problem import Problem
from cppflow.config import (
    SUCCESS_THRESHOLD_translation_ERR_MAX_CM,
    SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG,
    SUCCESS_THRESHOLD_mjac_DEG,
    SUCCESS_THRESHOLD_mjac_CM,
    env_collisions_ignored,
    self_collisions_ignored,
)
from cppflow.plan import write_qpath_to_results_df
from cppflow.lm_hyper_parameters import OptimizationParameters
from cppflow.evaluation_utils import (
    angular_changes,
    prismatic_changes,
    errors_are_below_threshold,
    calculate_pose_error_cm_deg,
)
from cppflow.collision_detection import (
    self_colliding_configs_klampt,
    env_colliding_configs_klampt,
    env_colliding_configs_capsule,
)

_SUCCESS_THRESHOLD_translation_ERR_MAX_M = SUCCESS_THRESHOLD_translation_ERR_MAX_CM / 100
_SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG = np.deg2rad(SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG)




@dataclass
class LmResidual:
    # pose
    pose: Optional[torch.Tensor]
    pose_invalid_row_idxs: Optional[torch.Tensor]
    time_pose: float
    time_pose_post_processing: float

    # differencing
    differencing: Optional[torch.Tensor]
    differencing_invalid_row_idxs: Optional[torch.Tensor]
    time_differencing: float
    time_differencing_post_processing: float

    # virtual configs
    virtual_configs: Optional[torch.Tensor]
    time_vqs: float
    time_vqs_post_processing: float

    # self collisions
    self_collisions: Optional[torch.Tensor]
    time_scs: float
    time_scs_post_processing: float

    # env collisions
    env_collisions: Optional[torch.Tensor]
    time_ecs: float
    time_ecs_post_processing: float

    def __post_init__(self):
        for v in [
            self.time_pose,
            self.time_pose_post_processing,
            self.time_differencing,
            self.time_differencing_post_processing,
            self.time_vqs,
            self.time_vqs_post_processing,
            self.time_scs,
            self.time_scs_post_processing,
            self.time_ecs,
            self.time_ecs_post_processing,
        ]:
            assert isinstance(v, (float, int))

    def total_time(self) -> float:
        return (
            self.time_pose
            + self.time_pose_post_processing
            + self.time_differencing
            + self.time_differencing_post_processing
            + self.time_vqs
            + self.time_vqs_post_processing
            + self.time_scs
            + self.time_scs_post_processing
            + self.time_ecs
            + self.time_ecs_post_processing
        )

    def get_r(self) -> torch.Tensor:
        """Get the residual vector"""
        r_list = [
            r_i
            for r_i in [
                self.pose,
                self.differencing,
                self.virtual_configs,
                self.self_collisions,
                self.env_collisions,
            ]
            if r_i is not None
        ]
        return torch.cat(r_list, dim=0)

    def verify_J(self, J: "LmJacobian"):
        if self.pose_invalid_row_idxs is not None:
            assert (
                J.pose_invalid_row_idxs is not None
                and J.pose_invalid_row_idxs.shape == self.pose_invalid_row_idxs.shape
            )
        if self.differencing_invalid_row_idxs is not None:
            assert (
                J.differencing_invalid_row_idxs is not None
                and J.differencing_invalid_row_idxs.shape == self.differencing_invalid_row_idxs.shape
            )
        if self.pose is not None:
            assert J.pose is not None and J.pose.shape[0] == self.pose.shape[0]
        if self.differencing is not None:
            assert J.differencing is not None and J.differencing.shape[0] == self.differencing.shape[0]
        if self.virtual_configs is not None:
            assert J.virtual_configs is not None and J.virtual_configs.shape[0] == self.virtual_configs.shape[0]
        if (self.self_collisions is not None) and (self.self_collisions.numel() > 0):
            assert (J.self_collisions is not None) and (J.self_collisions.shape[0] == self.self_collisions.shape[0])
        if (self.env_collisions is not None) and self.env_collisions.numel() > 0:
            assert J.env_collisions is not None and J.env_collisions.shape[0] == self.env_collisions.shape[0]


@dataclass
class LmJacobian:
    # pose
    pose: Optional[torch.Tensor]
    pose_invalid_row_idxs: Optional[torch.Tensor]
    time_pose: float
    time_pose_post_processing: float

    # differencing
    differencing: Optional[torch.Tensor]
    differencing_invalid_row_idxs: Optional[torch.Tensor]
    time_differencing: float
    time_differencing_post_processing: float

    # virtual configs
    virtual_configs: Optional[torch.Tensor]
    time_vqs: float
    time_vqs_post_processing: float

    # self collisions
    self_collisions: Optional[torch.Tensor]
    time_scs: float
    time_scs_post_processing: float

    # env collisions
    env_collisions: Optional[torch.Tensor]
    time_ecs: float
    time_ecs_post_processing: float

    def __post_init__(self):
        for v in [
            self.time_pose,
            self.time_pose_post_processing,
            self.time_differencing,
            self.time_differencing_post_processing,
            self.time_vqs,
            self.time_vqs_post_processing,
            self.time_scs,
            self.time_scs_post_processing,
            self.time_ecs,
            self.time_ecs_post_processing,
        ]:
            assert isinstance(v, (float, int))

    def total_time(self) -> float:
        return (
            self.time_pose
            + self.time_pose_post_processing
            + self.time_differencing
            + self.time_differencing_post_processing
            + self.time_vqs
            + self.time_vqs_post_processing
            + self.time_scs
            + self.time_scs_post_processing
            + self.time_ecs
            + self.time_ecs_post_processing
        )

    def get_J(self) -> torch.Tensor:
        """Get the jacobian tensor"""
        J_list = [
            J_i
            for J_i in [
                self.pose,
                self.differencing,
                self.virtual_configs,
                self.self_collisions,
                self.env_collisions,
            ]
            if J_i is not None
        ]
        return torch.cat(J_list, dim=0)

    def verify_r(self, r: "LmResidual"):
        if self.pose is not None:
            assert r.pose is not None and r.pose.shape[0] == self.pose.shape[0]
        if self.pose_invalid_row_idxs is not None:
            assert (
                r.pose_invalid_row_idxs is not None
                and r.pose_invalid_row_idxs.shape == self.pose_invalid_row_idxs.shape
            )
        if self.differencing is not None:
            assert r.differencing is not None and r.differencing.shape[0] == self.differencing.shape[0]
        if self.differencing_invalid_row_idxs is not None:
            assert (
                r.differencing_invalid_row_idxs is not None
                and r.differencing_invalid_row_idxs.shape == self.differencing_invalid_row_idxs.shape
            )
        if self.virtual_configs is not None:
            assert r.virtual_configs is not None and r.virtual_configs.shape[0] == self.virtual_configs.shape[0]

        if (self.self_collisions is not None) and self.self_collisions.numel() > 0:
            assert r.self_collisions is not None and r.self_collisions.shape[0] == self.self_collisions.shape[0]
        if (self.env_collisions is not None) and self.env_collisions.numel() > 0:
            assert r.env_collisions is not None and r.env_collisions.shape[0] == self.env_collisions.shape[0]


def _get_prismatic_and_revolute_row_mask(robot: Robot, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a mask for the revolute and prismatic indexes of a [n x __] tensor

    Timing data:
        ~5 ms / call using torch.tensor([i for i in range(n) if i % robot.ndof in robot.revolute_joint_idxs]] = 1
        0.08778095245361328 ms / call with 'fetch_hello' using tile()
    """
    assert n % robot.ndof == 0, f"error - n {n} is not divisible by ndof {robot.ndof}"
    n_qs = n // robot.ndof
    revolute_rows = torch.zeros(robot.ndof, dtype=torch.bool)
    revolute_rows[robot.revolute_joint_idxs] = 1
    revolute_rows = revolute_rows.tile(n_qs)
    return revolute_rows, torch.logical_not(revolute_rows)


def _get_rotation_and_position_row_mask(n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a mask for the revolute and prismatic indexes of a [n x __] tensor

    Timing data:
        0.795588493347168 ms / run using 'rotation_rows[[i for i in range(6 * n) if i % 6 in (0, 1, 2)]] = 1'
        0.455532073974609 ms / run using same as above but with 'torch.logical_not(rotation_rows)' for position
        0.1395559310913086 ms / run using 'rotation_rows[list(range(3))] = 1; rotation_rows.tile(n)'
    """
    rotation_rows = torch.zeros(6, dtype=torch.bool)
    rotation_rows[list(range(3))] = 1
    rotation_rows = rotation_rows.tile(n)
    return rotation_rows, torch.logical_not(rotation_rows)


class LmResidualFns:
    """The residual is constructed by pose, differencing, virtual config, and collision sub-residuals. This class
    generates all but the collision sub-residuals, which are available as methods in the Robot class. The idea of having
    these all be static methods is to keep them organized and make it clear that they should be used in the same
    way. This is as close to a c++ namespace as you can get afaik.
    """

    # =======================================================================
    # Pose

    @staticmethod
    def _get_residual_pose(robot: Robot, x: torch.Tensor, target_path: torch.Tensor) -> torch.Tensor:
        """Pose residuals"""
        n = x.shape[0]
        r = torch.zeros((6 * n, 1), dtype=x.dtype, device=x.device)
        pose_errors, current_poses = get_6d_pose_errors(robot, x, target_path)  # [n 6 1]
        r[0 : 6 * n, 0] = pose_errors.flatten()
        return r, current_poses

    @staticmethod
    def _get_jacobian_pose(robot: Robot, x: torch.Tensor):
        """Get the pose components of the jacobian of the residual"""
        n = x.shape[0]
        ndof = robot.ndof

        J = torch.zeros((6 * n, ndof * n))

        # Add robot FK jacobians to J (J is shorthand for J_residual(X))
        J_fk_batch = robot.jacobian(x)  # [n 6 ndof]
        assert J_fk_batch.shape == (n, 6, ndof)
        for i in range(n):
            J[6 * i : (6 * i) + 6, i * ndof : (i * ndof) + ndof] = J_fk_batch[i]
        return J

    @staticmethod
    def _scale_down_rows_from_r_J_pose_below_error(
        r: torch.Tensor,
        J: torch.Tensor,
        error_threshold_m: float,
        error_threshold_rad: float,
        scale: float,
        shift_invalid_to_threshold: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scales the values in r_pose, J_pose for which the positional or rotational error in r is below the provided
        thresholds
        """
        assert r.shape[0] == J.shape[0]
        assert r.shape[0] % 6 == 0
        assert r.shape[0] == r.numel()
        assert 0.0 <= scale < 1.0, "scale should be in [0, 1)"
        r_numel = r.numel()

        rotation_rows, position_rows = _get_rotation_and_position_row_mask(r_numel // 6)
        rot_below_threshold_rows = r[:, 0].abs() < error_threshold_rad
        pos_below_threshold_rows = r[:, 0].abs() < error_threshold_m
        #
        do_scale_rotation = torch.logical_and(rot_below_threshold_rows, rotation_rows)
        do_scale_position = torch.logical_and(pos_below_threshold_rows, position_rows)

        # Finally, scale down the rows that are below the threshold and return
        r[do_scale_position, :] *= scale
        r[do_scale_rotation, :] *= scale
        J[do_scale_position, :] *= scale
        J[do_scale_rotation, :] *= scale

        # TODO: Test shift_invalid_to_threshold
        if shift_invalid_to_threshold:
            invalid_rot = torch.logical_and(torch.logical_not(rot_below_threshold_rows), rotation_rows)
            invalid_pos = torch.logical_and(torch.logical_not(pos_below_threshold_rows), position_rows)
            print(invalid_rot.sum().item(), invalid_pos.sum().item(), "below threshold")
            r[torch.logical_and((r < -error_threshold_rad)[:, 0], invalid_rot)] += error_threshold_rad
            r[torch.logical_and((r > error_threshold_rad)[:, 0], invalid_rot)] -= error_threshold_rad
            r[torch.logical_and((r < -error_threshold_m)[:, 0], invalid_pos)] += error_threshold_m
            r[torch.logical_and((r > error_threshold_m)[:, 0], invalid_pos)] -= error_threshold_m

        invalid_row_idxs = torch.logical_not(torch.logical_or(do_scale_rotation, do_scale_position))
        return r, J, invalid_row_idxs

    # =======================================================================
    # Differencing

    @staticmethod
    def _get_residual_differencing(robot: Robot, x: torch.Tensor) -> torch.Tensor:
        """Differencing residuals"""
        n = x.shape[0]
        ndof = robot.ndof
        return angular_changes(x).reshape(((n - 1) * ndof, 1))

    @staticmethod
    def _get_jacobian_differencing(robot: Robot, x: torch.Tensor):
        """Get the differencing components of the jacobian of the residual"""
        n = x.shape[0]
        ndof = robot.ndof
        zeros = torch.zeros((ndof * (n - 1), ndof * n))
        J = torch.diagonal_scatter(zeros, torch.ones(ndof * (n - 1)), 0)
        J = torch.diagonal_scatter(J, -torch.ones(ndof * (n - 1)), offset=ndof)
        return J

    @staticmethod
    def _scale_down_rows_from_r_J_differencing_below_error(
        robot: Robot,
        r: torch.Tensor,
        J: torch.Tensor,
        mjac_threshold_m: float,
        mjac_threshold_rad: float,
        scale: float,
        shift_invalid_to_threshold: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scales the values in r_pose, J_pose for which the positional or rotational error in r is below the provided
        thresholds
        """
        ndof = robot.ndof
        n = (r.shape[0] // ndof) + 1
        r_numel = r.numel()
        assert 0.0 <= scale < 1.0, "values should be scaled down, not up"
        assert r.shape[0] % ndof == 0
        assert J.shape == (
            (n - 1) * ndof,
            n * ndof,
        ), f"J is {J.shape}, should be ((n-1)*ndof, n*ndof) {(n-1)*ndof, n*ndof}"
        assert r.shape == ((n - 1) * ndof, 1)
        #
        revolute_rows, prismatic_rows = _get_prismatic_and_revolute_row_mask(robot, r_numel)
        #
        prismatic_below_threshold_rows = r[:, 0].abs() < mjac_threshold_m
        revolute_below_threshold_rows = r[:, 0].abs() < mjac_threshold_rad
        #
        valid_prismatic = torch.logical_and(prismatic_below_threshold_rows, prismatic_rows)
        invalid_prismatic = torch.logical_and(torch.logical_not(prismatic_below_threshold_rows), prismatic_rows)
        valid_revolute = torch.logical_and(revolute_below_threshold_rows, revolute_rows)
        invalid_revolute = torch.logical_and(torch.logical_not(revolute_below_threshold_rows), revolute_rows)

        # Finally, scale down the rows that are below the threshold and return
        r[valid_prismatic, :] *= scale
        r[valid_revolute, :] *= scale
        J[valid_prismatic, :] *= scale
        J[valid_revolute, :] *= scale

        if shift_invalid_to_threshold:
            r[torch.logical_and((r < -mjac_threshold_rad)[:, 0], invalid_revolute)] += mjac_threshold_rad
            r[torch.logical_and((r > mjac_threshold_rad)[:, 0], invalid_revolute)] -= mjac_threshold_rad
            r[torch.logical_and((r < -mjac_threshold_m)[:, 0], invalid_prismatic)] += mjac_threshold_m
            r[torch.logical_and((r > mjac_threshold_m)[:, 0], invalid_prismatic)] -= mjac_threshold_m

        return J, r, torch.logical_not(torch.logical_or(valid_prismatic, valid_revolute))

    @staticmethod
    def _scale_down_differencing_near_pose_error_spikes(
        invalid_pose_idxs: torch.Tensor,
        J_diff: torch.Tensor,
        r_diff: torch.Tensor,
        scale: float,
        n_timesteps_around_spike: int,
    ):
        assert False, "this procedure doesn't help at all"
        assert J_diff.shape[0] == r_diff.shape[0]
        if invalid_pose_idxs.sum() == 0:
            return J_diff, r_diff

        n = J_diff.shape[0]
        scale_mask = torch.zeros(n, dtype=torch.bool)
        scale_idxs = torch.where(invalid_pose_idxs)[0]
        scale_mask[scale_idxs] = 1

        for i in range(1, n_timesteps_around_spike + 1):
            offset = i * torch.ones_like(scale_idxs)
            scale_mask[torch.clamp(scale_idxs + offset, min=0, max=n - 1)] = 1
            scale_mask[torch.clamp(scale_idxs - offset, min=0, max=n - 1)] = 1

        J_diff[scale_mask, :] *= scale
        r_diff[scale_mask, :] *= scale
        return J_diff, r_diff

    # =======================================================================
    # Virtual configs

    @staticmethod
    def _get_jacobian_virtual_configs(pms: OptimizationParameters, robot: Robot, x: torch.Tensor):
        """Get the virtual config differencing components of the jacobian of the residual"""
        assert pms.virtual_configs.shape == x.shape
        n = x.shape[0]
        ndof = robot.ndof
        n_vqs = pms.n_virtual_configs  # number of virtual configs
        J = torch.zeros((ndof * 2 * n_vqs, ndof * n))

        for i in range(n_vqs):
            row = i * ndof
            col = i * ndof
            # should be positive according to calculations, this way works empirically though
            J[row : row + ndof, col : col + ndof] = -torch.eye(ndof)

        #
        row0_right = ndof * n_vqs
        col0_right = n * ndof - ndof * n_vqs
        for i in range(n_vqs):
            row = row0_right + i * ndof
            col = col0_right + i * ndof
            J[row : row + ndof, col : col + ndof] = -torch.eye(ndof)  # same as above
        return J

    @staticmethod
    def _get_residual_virtual_joints(pms: OptimizationParameters, robot: Robot, x: torch.Tensor) -> torch.Tensor:
        """Virtual joint residuals"""
        assert x.shape == pms.virtual_configs.shape
        assert (
            2 * pms.n_virtual_configs < x.shape[0]
        ), f"Error, {2 * pms.n_virtual_configs} virtual configs provided, but only {x.shape[0]} configs provided"
        n_vqs = pms.n_virtual_configs  # number of virtual configs
        x_virtual = pms.virtual_configs
        n = x.shape[0]
        ndof = robot.ndof
        r = torch.zeros((2 * n_vqs * ndof, 1))

        for i in range(n_vqs):
            row = ndof * i
            r[row : row + ndof, 0] = angular_subtraction(x[i, :], x_virtual[i, :])

        for i in range(n_vqs):
            row = n_vqs * ndof + ndof * i
            config_idx = n - n_vqs + i
            r[row : row + ndof, 0] = angular_subtraction(x[config_idx, :], x_virtual[config_idx, :])

        # sanity check
        # if (x - x_virtual).abs().max() > 1e-8:
        #     for i in range(2 * n_vqs):
        #         row = ndof * i
        #         assert (
        #             r[row : row + ndof, 0].abs().max() > 1e-8
        #         ), f"r[row: row + ndof, 0].abs().max() is suspiciously small (only {r[row: row + ndof, 0].abs().max()})"

        return r

    @staticmethod
    def get_r_and_J(
        pms: OptimizationParameters,
        robot: Robot,
        x: torch.Tensor,
        target_path: torch.Tensor,
        Tcuboids: Optional[List] = None,
        cuboids: Optional[List] = None,
    ) -> Tuple[LmJacobian, LmResidual]:
        """Get the residual of the current x being optimized. Additionally fill out the jacobian of this residual by
        following the derived expression
        """
        n = x.shape[0]
        residual = LmResidual(None, None, 0, 0, None, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0)
        jacobian = LmJacobian(None, None, 0, 0, None, None, 0, 0, None, 0, 0, None, 0, 0, None, 0, 0)

        # =======================================================================
        # Pose error
        #
        if pms.use_pose:
            t0 = time()
            J_pose = LmResidualFns._get_jacobian_pose(robot, x)
            jacobian.time_pose = time() - t0

            t0 = time()
            r_pose, _ = LmResidualFns._get_residual_pose(robot, x, target_path)
            residual.time_pose = time() - t0

            t0 = time()
            if pms.pose_do_scale_down_satisfied:
                pose_error_threshold_pos_m = (
                    pms.pose_ignore_satisfied_threshold_scale * _SUCCESS_THRESHOLD_translation_ERR_MAX_M
                )
                pose_error_threshold_rot_rad = (
                    pms.pose_ignore_satisfied_threshold_scale * _SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG
                )

                r_pose, J_pose, pose_invalid_row_idxs = LmResidualFns._scale_down_rows_from_r_J_pose_below_error(
                    r_pose,
                    J_pose,
                    error_threshold_m=pose_error_threshold_pos_m,
                    error_threshold_rad=pose_error_threshold_rot_rad,
                    scale=pms.pose_ignore_satisfied_scale_down,
                )
                residual.pose_invalid_row_idxs = pose_invalid_row_idxs
                jacobian.pose_invalid_row_idxs = pose_invalid_row_idxs

            # --- Multiply by alpha_rotation, alpha_position
            rotation_rows, position_rows = _get_rotation_and_position_row_mask(n)
            r_pose[rotation_rows, :] *= pms.alpha_rotation
            r_pose[position_rows, :] *= pms.alpha_position
            J_pose[rotation_rows, :] *= pms.alpha_rotation
            J_pose[position_rows, :] *= pms.alpha_position
            # ---
            residual.time_pose_post_processing = time() - t0
            jacobian.time_pose_post_processing = time() - t0
            residual.pose = r_pose
            jacobian.pose = J_pose

        # =======================================================================
        # Differencing
        #
        assert not (
            pms.differencing_do_scale_satisfied and pms.differencing_do_ignore_satisfied
        ), f"use one or the other, not both"
        if pms.use_differencing:
            t0 = time()
            r_differencing = LmResidualFns._get_residual_differencing(robot, x)
            residual.time_differencing = time() - t0

            t0 = time()
            J_differencing = LmResidualFns._get_jacobian_differencing(robot, x)
            jacobian.time_differencing = time() - t0

            # post processing
            t0 = time()
            if pms.differencing_do_ignore_satisfied or pms.differencing_do_scale_satisfied:
                mjac_threshold_rad = np.deg2rad(
                    SUCCESS_THRESHOLD_mjac_DEG - pms.differencing_ignore_satisfied_margin_deg
                )
                mjac_threshold_m = cm_to_m(SUCCESS_THRESHOLD_mjac_CM - pms.differencing_ignore_satisfied_margin_cm)

            # Option 1: ignore satisfied diffs
            if pms.differencing_do_ignore_satisfied:
                r_differencing, J_differencing = filter_rows_from_r_J_differencing(
                    robot,
                    r_differencing,
                    J_differencing,
                    threshold_rad=mjac_threshold_rad,
                    threshold_m=mjac_threshold_m,
                    shift_to_threshold=True,
                )

            # Option 2: scale satisfied diffs
            if pms.differencing_do_scale_satisfied:
                J_differencing, r_differencing, differencing_invalid_row_idxs = (
                    LmResidualFns._scale_down_rows_from_r_J_differencing_below_error(
                        robot,
                        r_differencing,
                        J_differencing,
                        mjac_threshold_m=mjac_threshold_m,
                        mjac_threshold_rad=mjac_threshold_rad,
                        scale=pms.differencing_scale_down_satisfied_scale,
                        shift_invalid_to_threshold=pms.differencing_scale_down_satisfied_shift_invalid_to_threshold,
                    )
                )
                residual.differencing_invalid_row_idxs = differencing_invalid_row_idxs
                jacobian.differencing_invalid_row_idxs = differencing_invalid_row_idxs

                # This doesn't work well - optimization diverges
                # if pms.differencing_do_scale_down_near_pose_error_spikes:
                #     J_differencing, r_differencing = LmResidualFns._scale_down_differencing_near_pose_error_spikes(
                #         residual.pose_invalid_row_idxs,
                #         J_differencing,
                #         r_differencing,
                #         pms.differencing_scale_down_near_pose_error_spikes_scale,
                #         pms.differencing_scale_down_near_pose_error_spikes_n_timesteps
                #     )

            if robot.has_prismatic_joints and not pms.differencing_do_ignore_satisfied:
                _, prismatic_rows = _get_prismatic_and_revolute_row_mask(robot, r_differencing.shape[0])
                r_differencing[prismatic_rows] *= pms.alpha_differencing_prismatic_scaling
                J_differencing[prismatic_rows] *= pms.alpha_differencing_prismatic_scaling

            residual.differencing = pms.alpha_differencing * r_differencing
            jacobian.differencing = pms.alpha_differencing * J_differencing

            residual.time_differencing_post_processing = time() - t0
            jacobian.time_differencing_post_processing = time() - t0

        # =======================================================================
        # Virtual configs
        #
        if pms.use_virtual_configs:
            t0 = time()
            residual.virtual_configs = (
                pms.alpha_virtual_configs
                * pms.alpha_differencing
                * LmResidualFns._get_residual_virtual_joints(pms, robot, x)
            )
            residual.time_vqs = time() - t0
            t0 = time()
            jacobian.virtual_configs = (
                pms.alpha_virtual_configs
                * pms.alpha_differencing
                * LmResidualFns._get_jacobian_virtual_configs(pms, robot, x)
            )
            jacobian.time_vqs = time() - t0
            t0 = time()
            residual.time_differencing_post_processing = time() - t0
            jacobian.time_differencing_post_processing = time() - t0

        # =======================================================================
        # Self collisions
        #

        if pms.use_self_collisions:
            # Everything below this is considered a collision and will be penalized. Note that these distances are
            # between the minimum bounding capsules for each joint, so a negative distance does not necessarily mean a
            # collision is occuring. Final collision checking uses the collision meshes provided in the urdf so is more
            # precise.
            safety_margin_m = 0.0  # 0.0075

            # residual
            t0 = time()
            distances = robot.self_collision_distances(x) - safety_margin_m
            r_self_collision = -pms.alpha_self_collision * distances
            residual.time_scs = time() - t0

            # residual - post processing
            t0 = time()
            n_selfcoll_pairs = r_self_collision.shape[1]
            r_self_collision = r_self_collision.reshape(-1, 1)
            mask = (r_self_collision > 0).reshape(-1)
            residual.self_collisions = r_self_collision[mask, :]
            batchmask = mask.reshape(x.shape[0], -1).any(dim=1)
            x_selfcoll = x[batchmask, :]
            residual.time_scs_post_processing = time() - t0

            # jacobian
            t0 = time()
            if torch.any(batchmask):
                J_self_collision = torch.zeros(x.shape[0], n_selfcoll_pairs, x.shape[1])
                J_self_collision[batchmask, :, :] = pms.alpha_self_collision * robot.self_collision_distances_jacobian(
                    x_selfcoll
                )
                t0 = time()
                J_self_collision = torch.block_diag(*torch.unbind(J_self_collision))
                jacobian.self_collisions = J_self_collision[mask, :]
                jacobian.time_scs_post_processing = time() - t0
            jacobian.time_scs = time() - t0

        # =======================================================================
        # Env collisions
        #
        if pms.use_env_collisions and len(Tcuboids) > 0:
            # Obstacle error
            r_env_collisions = []
            J_env_collisions = []
            for Tcuboid, cuboid in zip(Tcuboids, cuboids):
                # residual
                t0 = time()
                safety_margin_m = 0.0  # 0.0025
                distances = robot.env_collision_distances(x, cuboid, Tcuboid) - safety_margin_m
                env_residuals = -pms.alpha_env_collision * distances
                residual.time_ecs += time() - t0

                # residual - post processing
                t0 = time()
                n_envcoll_pairs = env_residuals.shape[1]
                env_residuals = env_residuals.reshape(-1, 1)
                mask = (env_residuals > 0).reshape(-1)
                env_residuals = env_residuals[mask, :]
                if mask.sum() > 0:
                    r_env_collisions.append(env_residuals)
                batchmask = mask.reshape(x.shape[0], -1).any(dim=1)
                x_envcoll = x[batchmask, :]
                residual.time_ecs_post_processing += time() - t0

                # jacobian
                if torch.any(batchmask):
                    t0 = time()
                    J_env_collision = torch.zeros(x.shape[0], n_envcoll_pairs, x.shape[1])
                    J_env_collision[batchmask] = pms.alpha_env_collision * robot.env_collision_distances_jacobian(
                        x_envcoll, cuboid, Tcuboid
                    )
                    jacobian.time_ecs += time() - t0

                    # jacobian - post processing
                    t0 = time()
                    J_env_collision = torch.block_diag(*torch.unbind(J_env_collision))
                    J_env_collision = J_env_collision[mask, :]
                    J_env_collisions.append(J_env_collision)
                    jacobian.time_ecs_post_processing = time() - t0

            assert len(r_env_collisions) == len(J_env_collisions)
            if len(r_env_collisions) > 0:
                residual.env_collisions = torch.cat(r_env_collisions, dim=0)
                jacobian.env_collisions = torch.cat(J_env_collisions, dim=0)

            # Save timing. meaningless for now, saved in case any post processing is added

        jacobian.verify_r(residual)
        residual.verify_J(jacobian)
        return jacobian, residual


# TODO: i'm pretty sure we shouldn't be doing this, instead scaling down errors below the threshold instead like we do
# for pose
def filter_rows_from_r_J_differencing(
    robot: Robot,
    r: torch.Tensor,
    J: torch.Tensor,
    threshold_rad: float,
    threshold_m: float,
    shift_to_threshold: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert r.shape[0] == J.shape[0]
    assert r.shape[0] % robot.ndof == 0, f"r.shape[0]: {r.shape[0]}, robot.ndof: {robot.ndof} for {robot}"
    n_configs = r.shape[0] / robot.ndof
    assert abs(int(n_configs) - n_configs) < 1e-8
    n_configs = int(n_configs)

    #
    revolute_idxs = torch.zeros(robot.ndof, dtype=torch.bool)
    revolute_idxs[robot.revolute_joint_idxs] = 1
    revolute_idxs = revolute_idxs.repeat((n_configs, 1)).reshape(n_configs * robot.ndof)
    prismatic_idxs = torch.zeros(robot.ndof, dtype=torch.bool)
    prismatic_idxs[robot.prismatic_joint_idxs] = 1
    prismatic_idxs = prismatic_idxs.repeat((n_configs, 1)).reshape(n_configs * robot.ndof)

    invalid_idxs_rev = torch.logical_and(r.abs()[:, 0] > threshold_rad, revolute_idxs)
    invalid_idxs_pris = torch.logical_and(r.abs()[:, 0] > threshold_m, prismatic_idxs)
    keep_idxs = torch.logical_or(invalid_idxs_rev, invalid_idxs_pris)
    if not shift_to_threshold:
        return r[keep_idxs, :], J[keep_idxs, :]

    r[torch.logical_and((r < -threshold_rad)[:, 0], revolute_idxs)] += threshold_rad
    r[torch.logical_and((r > threshold_rad)[:, 0], revolute_idxs)] -= threshold_rad
    r[torch.logical_and((r < -threshold_m)[:, 0], prismatic_idxs)] += threshold_m
    r[torch.logical_and((r > threshold_m)[:, 0], prismatic_idxs)] -= threshold_m
    return r[keep_idxs, :], J[keep_idxs, :]


def get_jacobian_finite_differencing(
    robot: Robot, opt_params: OptimizationParameters, x: torch.Tensor, target_path: torch.Tensor
):
    """Compute the jacobian of the residual vector using finite differencing"""
    n = x.shape[0]
    ndof = robot.ndof
    r_numel = 6 * n + ndof * (n - 1)

    if opt_params.use_virtual_configs is not None:
        assert opt_params.n_virtual_configs > 0
        r_numel += 2 * opt_params.n_virtual_configs * ndof

    J = torch.zeros((r_numel, ndof * n), device=x.device, dtype=x.dtype)
    _, r = LmResidualFns.get_r_and_J(opt_params, robot, x, target_path)
    r = r.get_r()
    assert r.numel() == r_numel
    eps = 0.01

    print(f"_/{r_numel}")
    for i in range(r_numel):
        print(f"{i}", end=" ")
        for j in range(n * ndof):
            x_diff = x.clone()
            timestep, rem = divmod(j, ndof)
            x_diff[timestep, rem] += eps
            _, r_new = LmResidualFns.get_r_and_J(opt_params, robot, x_diff, target_path)
            r_new = r_new.get_r()
            J[i, j] = (r_new[i, 0] - r[i, 0]) / eps
    return J


def get_6d_pose_errors(robot: Robot, x: torch.Tensor, target_poses: torch.Tensor):
    """Returns [n x 6 x 1] tensor of pose errors.

    Notes:
        - [6 x 1] sub tensors are structured as [roll-error, pitch-error, yaw-error, x-error, y-error, z-error]^T
        - rotational and positional error is measured in radians and meters. This matches the units used by the FK
            jacobian returned by jrl.
    """
    n = x.shape[0]
    current_poses = robot.forward_kinematics(x, out_device=x.device, dtype=x.dtype)
    pose_errors = torch.zeros((n, 6, 1), device=x.device, dtype=x.dtype)
    for i in range(3):
        pose_errors[:, i + 3, 0] = target_poses[:, i] - current_poses[:, i]

    current_pose_quat_inv = quaternion_inverse(current_poses[:, 3:7])
    rotation_error_quat = quaternion_product(target_poses[:, 3:], current_pose_quat_inv)
    rotation_error_rpy = quaternion_to_rpy(rotation_error_quat)
    pose_errors[:, 0:3, 0] = rotation_error_rpy  #
    return pose_errors, current_poses


def clamp_to_joint_limits(robot: Robot, x: torch.Tensor, verbosity: int = 0) -> torch.Tensor:
    if verbosity > 0:
        for i, (l, u) in enumerate(robot.actuated_joints_limits):
            if x[:, i].min() < l:
                print(f"clamp_to_joint_limits() | joint {i} is below lower limit {l}")
            if x[:, i].max() > u:
                print(f"clamp_to_joint_limits() | joint {i} is above upper limit {u}")

    for i, (l, u) in enumerate(robot.actuated_joints_limits):
        x[:, i] = torch.clamp(x[:, i], l, u)
    return x


def meta_objective_function(error_pos_max_cm: float, error_rot_max_deg: float, mjac: float) -> float:
    raise NotImplementedError("add robot-robot, robot-env collision checks")
    return (
        max(error_pos_max_cm - SUCCESS_THRESHOLD_translation_ERR_MAX_CM, 0)
        + max(error_rot_max_deg - SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG, 0)
        + max(mjac - SUCCESS_THRESHOLD_mjac_DEG, 0)
    )


def x_is_valid(
    problem: Problem,
    target_path_stacked: torch.Tensor,
    x: torch.Tensor,
    parallel_count: int,
    results_df: Optional[pd.DataFrame] = None,
    verbosity: int = 0,
):
    n = problem.n_timesteps
    revolute_diffs_deg, prismatic_diffs_cm = None, None
    error_t_cm, error_R_deg = calculate_pose_error_cm_deg(problem.robot, x, target_path_stacked)

    if results_df is not None:
        max_pos_errors = []

    is_a_self_collision = None
    is_a_env_collision = None

    for i in range(parallel_count):
        x_i = x[i * n : (i + 1) * n, :]

        # TODO: check joint limits

        # Pose error
        x_i_revolute, x_i_prismatic = problem.robot.split_configs_to_revolute_and_prismatic(x_i)
        revolute_diffs_deg = torch.rad2deg(angular_changes(x_i_revolute))
        prismatic_diffs_cm = 100 * prismatic_changes(x_i_prismatic)
        pos_error = error_t_cm[i * n : (i + 1) * n]
        rot_error = error_R_deg[i * n : (i + 1) * n]
        if results_df is not None:
            max_pos_errors.append(pos_error.max().item())
        assert pos_error.shape == rot_error.shape
        assert pos_error.numel() == x_i.shape[0]
        assert revolute_diffs_deg.shape[0] + 1 == x_i.shape[0], (
            f"Error: revolute_diffs_deg is {revolute_diffs_deg.shape} but should be"
            f" {(x_i.shape[0]-1, problem.robot.ndof)}"
        )
        all_valid, (pose_pos_valid, pose_rot_valid, mjac_rev_valid, mjac_pris_valid) = errors_are_below_threshold(
            pos_error,
            rot_error,
            revolute_diffs_deg,
            prismatic_diffs_cm,
            verbosity=0,
        )
        if not all_valid:
            continue

        # TODO: It could be faster to do this in parallel with self_colliding_configs_capsule()
        if not self_collisions_ignored:
            self_colliding = self_colliding_configs_klampt(problem, x_i)
            # assert (
            #     not self_colliding.any()
            # ), f"Found a self-colliding config. Use this as an example for tuning self-collision avoidance"
            is_a_self_collision = self_colliding.any()
            if is_a_self_collision:
                continue

        # TODO: It could be faster to do this in parallel with configs_are_env_colliding_capsule()
        if not env_collisions_ignored:
            env_colliding = env_colliding_configs_klampt(problem, x_i)
            is_a_env_collision = env_colliding.any()
            # assert (
            #     not env_colliding.any()
            # ), f"Found a env-colliding config. Use this as an example for tuning env-collision avoidance"
            # if verbosity > 0:
            #     env_colliding_caps = env_colliding_configs_capsule(problem, x_i)
            #     print(f"{env_colliding.sum()} env collisions, {env_colliding_caps.sum()} env collisions (capsule)")
            if is_a_env_collision.any():
                continue

        if verbosity > 0:
            print("x_is_valid() |", make_text_green_or_red("x is valid", True))

        return (
            x_i,
            i,
            (pose_pos_valid, pose_rot_valid, mjac_rev_valid, mjac_pris_valid, is_a_self_collision, is_a_env_collision),
        )

    if results_df is not None:
        best_idx = max_pos_errors.index(min(max_pos_errors))
        x_i = x[best_idx * n : (best_idx + 1) * n, :]
        write_qpath_to_results_df(results_df, x_i, problem)

    if verbosity > 0:
        print("x_is_valid() |", make_text_green_or_red("x is invalid", False))

    return (
        None,
        None,
        (pose_pos_valid, pose_rot_valid, mjac_rev_valid, mjac_pris_valid, is_a_self_collision, is_a_env_collision),
    )
