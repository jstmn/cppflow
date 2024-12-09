import warnings

import torch
from jrl.config import DEVICE

print("cppflow/config.py | device:", DEVICE)

DEFAULT_TORCH_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)

VERBOSITY = 2

SUCCESS_THRESHOLD_initial_q_norm_dist = 0.01

DEFAULT_RERUN_MJAC_THRESHOLD_DEG = 13.0
DEFAULT_RERUN_MJAC_THRESHOLD_CM = 3.42
OPTIMIZATION_CONVERGENCE_THRESHOLD = 0.005


# LM optimization
SELF_COLLISIONS_IGNORED = False
ENV_COLLISIONS_IGNORED = False
DEBUG_MODE_ENABLED = False

if SELF_COLLISIONS_IGNORED:
    warnings.warn("robot-robot are collisions will be ignored during LM optimization")
if ENV_COLLISIONS_IGNORED:
    warnings.warn("environment-robot collisions will be ignored during LM optimization")
