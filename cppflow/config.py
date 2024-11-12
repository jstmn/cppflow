import warnings

import torch
from jrl.config import DEVICE

print("cppflow/config.py | device:", DEVICE)

DEFAULT_TORCH_DTYPE = torch.float32
torch.set_default_dtype(DEFAULT_TORCH_DTYPE)
torch.set_default_device(DEVICE)

VERBOSITY = 2

# Success criteria

# LM optimization
SELF_COLLISIONS_IGNORED = False
ENV_COLLISIONS_IGNORED = False
DEBUG_MODE_ENABLED = False
# DEBUG_MODE_ENABLED = True

if SELF_COLLISIONS_IGNORED:
    warnings.warn("robot-robot are collisions will be ignored during LM optimization")
if ENV_COLLISIONS_IGNORED:
    warnings.warn("environment-robot collisions will be ignored during LM optimization")


RESULTS_CSV_COLS = (
    "Time Elapsed (s)",
    "is valid",
    "Mean Pos Error (mm)",
    "Max Pos Error (mm)",
    "Mean Rot Error (deg)",
    "Max Rot Error (deg)",
    "Mjac (deg)",
    "Mjac (cm)",
    "pct_self-colliding",
    "pct_env-colliding",
    "path_length_rad",
    "path_length_m",
)
