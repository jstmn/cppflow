from typing import Optional
from dataclasses import dataclass
import warnings

import torch

from cppflow.utils import Hashable

ALTERNATING_LOSS_MAX_N_STEPS = 20
ALTERNATING_LOSS_RETURN_IF_SOL_FOUND_AFTER = 15
ALTERNATING_LOSS_CONVERGENCE_THRESHOLD = 0.3


@dataclass
class OptimizationParameters(Hashable):
    """Parameters for the optimizer"""

    seed_w_only_pose: bool
    lm_lambda: float

    # --- Alphas
    alpha_position: float
    alpha_rotation: float
    alpha_differencing: float
    alpha_differencing_prismatic_scaling: float
    # note: 'alpha_virtual_configs' multiplies 'alpha_differencing' to get the final scaling term
    alpha_virtual_configs: float
    alpha_self_collision: float
    alpha_env_collision: float

    # --- Pose error
    use_pose: bool
    pose_do_scale_down_satisfied: bool
    pose_ignore_satisfied_threshold_scale: float
    pose_ignore_satisfied_scale_down: float

    # --- Differencing error
    use_differencing: bool
    differencing_do_ignore_satisfied: bool
    differencing_ignore_satisfied_margin_deg: float
    differencing_ignore_satisfied_margin_cm: float
    # scale down satisfied differencing errors instead of filtering them out
    differencing_do_scale_satisfied: bool
    differencing_scale_down_satisfied_scale: float
    differencing_scale_down_satisfied_shift_invalid_to_threshold: (
        bool  # TODO: test out differencing_scale_down_satisfied_shift_invalid_to_threshold
    )

    # --- Virtual configs
    use_virtual_configs: bool
    virtual_configs: Optional[torch.Tensor]
    n_virtual_configs: int

    # --- Collisions
    use_self_collisions: bool
    use_env_collisions: bool

    def __post_init__(self):
        if self.differencing_do_scale_satisfied and not self.use_virtual_configs:
            warnings.warn(
                "differencing_do_scale_satisfied is True, but virtual_configs are disabled. Using virtual configs with"
                " do_scale_satisfied is recommended - otherwise the differencing residual for configs at start/end will"
                " be unbalanced."
            )
        if self.use_differencing:
            assert not (
                self.differencing_do_ignore_satisfied and self.differencing_do_scale_satisfied
            ), "use one or the other, not both"
        if self.differencing_do_ignore_satisfied or self.differencing_do_scale_satisfied:
            assert self.differencing_ignore_satisfied_margin_deg > 0
            assert self.differencing_ignore_satisfied_margin_cm > 0
        if self.use_virtual_configs:
            assert self.virtual_configs is not None
            assert isinstance(self.n_virtual_configs, int) and self.n_virtual_configs > 0
        if self.use_self_collisions:
            assert self.alpha_self_collision > 0
        if self.use_env_collisions:
            assert self.alpha_env_collision > 0
        if self.pose_do_scale_down_satisfied:
            assert isinstance(self.pose_ignore_satisfied_threshold_scale, float)
            assert self.pose_ignore_satisfied_threshold_scale > 0


# NOTE: parameters expect 1.5deg, 3cm padding in dp_search
# includes tweaks compared to V2
ALT_LOSS_V2_1_DIFF = OptimizationParameters(
    # General
    seed_w_only_pose=None,
    lm_lambda=1e-06,  # 1e-6 seems to be optimal. higher or lower has worse performance
    alpha_position=None,
    alpha_rotation=None,
    # alpha_differencing=0.005, # too high for fetch__hello
    alpha_differencing=0.00375,
    # alpha_differencing=0.0025,
    alpha_differencing_prismatic_scaling=1.0,
    alpha_virtual_configs=1.0,
    alpha_self_collision=0.01,
    alpha_env_collision=0.01,
    # --- Pose
    use_pose=False,
    pose_do_scale_down_satisfied=False,
    pose_ignore_satisfied_threshold_scale=None,
    pose_ignore_satisfied_scale_down=None,
    # --- Differencing
    use_differencing=True,
    differencing_do_ignore_satisfied=False,
    differencing_ignore_satisfied_margin_deg=None,
    differencing_ignore_satisfied_margin_cm=None,
    differencing_do_scale_satisfied=False,
    differencing_scale_down_satisfied_scale=None,
    differencing_scale_down_satisfied_shift_invalid_to_threshold=None,
    # --- Virtual Configs, collisions
    use_virtual_configs=True,  # need virtual configs when differencing_ignore_satisfied=False
    virtual_configs=torch.tensor([]),
    n_virtual_configs=4,
    use_self_collisions=True,
    use_env_collisions=True,
)
ALT_LOSS_V2_1_POSE = OptimizationParameters(
    # General
    seed_w_only_pose=None,
    # lm_lambda=1e-08, # this causes bad convergence with fetch__hello
    lm_lambda=1e-06,
    # --- Alphas
    alpha_position=3.5,
    alpha_rotation=0.35,
    alpha_differencing=None,
    alpha_differencing_prismatic_scaling=None,
    alpha_virtual_configs=None,
    alpha_self_collision=None,
    alpha_env_collision=None,
    # --- Pose
    use_pose=True,
    pose_do_scale_down_satisfied=False,
    pose_ignore_satisfied_threshold_scale=None,
    pose_ignore_satisfied_scale_down=None,
    # --- Differencing
    use_differencing=False,
    differencing_do_ignore_satisfied=False,
    differencing_ignore_satisfied_margin_deg=None,
    differencing_ignore_satisfied_margin_cm=None,
    differencing_do_scale_satisfied=False,
    differencing_scale_down_satisfied_scale=None,
    differencing_scale_down_satisfied_shift_invalid_to_threshold=True,
    # --- Virtual Configs, collisions
    use_virtual_configs=False,
    virtual_configs=None,
    n_virtual_configs=None,
    use_self_collisions=False,
    use_env_collisions=False,
)
