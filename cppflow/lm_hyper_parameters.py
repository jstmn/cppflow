import torch
from cppflow.optimization_utils import OptimizationParameters

ALTERNATING_LOSS_MAX_N_STEPS = 20
ALTERNATING_LOSS_RETURN_IF_SOL_FOUND_AFTER = 15
ALTERNATING_LOSS_CONVERGENCE_THRESHOLD = 0.3


# NOTE: parameters expect 1.5deg, 3cm padding in dp_search
# includes tweaks compared to V2
ALT_LOSS_V2_1_DIFF = OptimizationParameters(
    # General
    seed_w_only_pose=None,
    do_use_penalty_method=False,
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
    do_use_penalty_method=False,
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