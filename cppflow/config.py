import warnings

import torch
from jrl.config import DEVICE

print("cppflow/config.py | device:", DEVICE)

DEFAULT_TORCH_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)

VERBOSITY = 2

# Success criteria
SUCCESS_THRESHOLD_translation_ERR_MAX_CM = 0.01  # 0.1mm
SUCCESS_THRESHOLD_translation_ERR_MAX_MM = SUCCESS_THRESHOLD_translation_ERR_MAX_CM * 10
SUCCESS_THRESHOLD_rotation_ERR_MAX_DEG = 0.1  #
SUCCESS_THRESHOLD_mjac_CM = 2  # for prismatic joints
SUCCESS_THRESHOLD_mjac_DEG = 3  # for revolute joints

# LM optimization
SELF_COLLISIONS_IGNORED = False
ENV_COLLISIONS_IGNORED = False
DEBUG_MODE_ENABLED = False

if SELF_COLLISIONS_IGNORED:
    warnings.warn("robot-robot are collisions will be ignored during LM optimization")
if ENV_COLLISIONS_IGNORED:
    warnings.warn("environment-robot collisions will be ignored during LM optimization")
