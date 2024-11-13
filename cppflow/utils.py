from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import random
from time import time
import colorsys

from jrl.robot import Robot
import matplotlib.pyplot as plt
import pkg_resources
import torch
import numpy as np

from cppflow.evaluation_utils import calculate_mjac_deg, calculate_per_timestep_mjac_cm
from cppflow.config import DEFAULT_TORCH_DTYPE, DEVICE


def print_v1(s, verbosity=0, *args, **kwargs):
    """ Prints if verbsotity is 1 or greater
    """
    if verbosity >= 1:
        print(s, *args, **kwargs)

def print_v2(s, verbosity=0, *args, **kwargs):
    """ Prints if verbsotity is 2 or greater
    """
    if verbosity >= 2:
        print(s, *args, **kwargs)
        print(s)

def print_v3(s, verbosity=0, *args, **kwargs):
    """ Prints if verbsotity is 3 or greater
    """
    if verbosity >= 3:
        print(s, *args, **kwargs)

def _print_kwargs(kwargs, verbosity=0):
    if verbosity < 1:
        return
    for key, v in kwargs.items():
        if key == "results_df" or key[0] == "_":
            continue
        if isinstance(v, tuple):
            print(f"  {key}: (", end=" ")
            for vv in v:
                if isinstance(vv, torch.Tensor):
                    print(vv.shape, end=", ")
                else:
                    print(vv, end=", ")
            print(")")
            continue
        print(f"  {key}: {v}")
    print()


def _plot_self_collisions(self_collision_violations: torch.Tensor):
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.imshow(self_collision_violations.cpu().numpy(), vmin=0, vmax=1)
    plt.title("self collision violations")
    plt.xlabel("timestep")
    plt.ylabel("k")
    # plt.colorbar()
    plt.grid(True, which="both", axis="both")
    plt.savefig("debug__self_collision_violations.png", bbox_inches="tight")
    plt.close()


def _plot_env_collisions(env_collision_violations: torch.Tensor):
    plt.figure(figsize=(10, 10))
    plt.tight_layout()
    plt.imshow(env_collision_violations.cpu().numpy(), vmin=0, vmax=1)
    plt.title("environment collision violations")
    plt.xlabel("timestep")
    plt.ylabel("k")
    # plt.colorbar()
    plt.grid(True, which="both", axis="both")
    plt.savefig("debug__env_collision_violations.png", bbox_inches="tight")
    plt.close()

def _get_mjacs(robot: Robot, qpath: torch.Tensor):
    qps_revolute, qps_prismatic = robot.split_configs_to_revolute_and_prismatic(qpath)
    if qps_prismatic.numel() > 0:
        return calculate_mjac_deg(qps_revolute), calculate_per_timestep_mjac_cm(qps_prismatic).abs().max().item()
    return calculate_mjac_deg(qps_revolute), 0.0



TORM_TL_RESULTS = {
    "fetch_arm__circle": (11.105, None),
    "fetch_arm__hello": (61.029, None),
    "fetch_arm__rot_yz2": (29.913, None),
    "fetch_arm__s": (10.856, None),
    "fetch_arm__square": (14.841, None),
    "fetch__circle": (22.042, 0.498),
    "fetch__hello": (None, None),
    "fetch__rot_yz2": (30.419, 0.841),
    "fetch__s": (None, None),
    "fetch__square": (19.577, 0.56392),
    "panda__1cube": (8.493, None),
    "panda__2cubes": (12.628, None),
    "panda__flappy_bird": (None, None),
}


def get_TL_rad_torm_ratio(problem_name: str, tl_rad: Optional[float], tl_m: Optional[float]) -> Optional[float]:
    """Returns the ratio of trajectory length (TL) between the provided value and what torm reports"""
    tl_rad_torm = TORM_TL_RESULTS[problem_name][0]
    tl_m_torm = TORM_TL_RESULTS[problem_name][1]

    ratio_rev = None
    if tl_rad is not None and tl_rad_torm is not None:
        ratio_rev = tl_rad / tl_rad_torm

    ratio_pri = None
    if tl_m is not None and tl_m_torm is not None:
        ratio_pri = tl_m / tl_m_torm

    return ratio_rev, ratio_pri, tl_rad_torm, tl_m_torm


class Hashable:
    def get_hash(self, ignore: List[str] = []) -> str:
        hash_str = ""
        for k, v in self.__dict__.items():
            if k[0] == "_":
                continue
            if k in ignore:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() > 0:
                    hash_str += f"{k}={v.sum().item()}+{v.shape}+{v.min().item()}+{v.max().item()},"
                else:
                    hash_str += f"{k}=<empty_tensor>,"
                continue
            hash_str += f"{k}={v},"
        return calc_hash(hash_str)


class TimerContext:
    def __init__(self, name: str, enabled: bool = True, round_places: int = 4):
        self.name = name
        self.enabled = enabled
        self.round_places = round_places

    def __enter__(self):
        self.start = time()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        if exception_type is not None:
            raise RuntimeError(f"Error caught by '{self.name}': {exception_value}")
        if self.enabled:
            print(f" --> {self.name} took {round(time() - self.start, self.round_places)} seconds")


@dataclass
class TestSpecification(Hashable):
    planner: str
    problem: str
    k: int
    n_runs: int

    def __post_init__(self):
        assert self.planner == "CppFlowPlanner"


@dataclass
class TestResult:
    planner: str
    problem: str
    k: int
    # success info
    succeeded: bool
    is_self_collision: bool
    is_env_collision: bool
    mjac_rev_invalid: bool
    mjac_pri_invalid: bool
    pose_pos_invalid: bool
    pose_rot_invalid: bool
    # runtime
    n_optimization_steps: int
    time_total: float
    time_ikflow: float
    time_dp_search: float
    time_optimizer: float
    # other
    search_path_mjac_deg: float
    search_path_mjac_cm: float
    search_path_min_dist_to_jlim_cm: float
    search_path_min_dist_to_jlim_deg: float
    tl_rad: float
    tl_m: float

    def __post_init__(self):
        assert self.planner in {"CppFlowPlanner"}


def cm_to_m(x: float):
    return x / 100.0


def get_filepath(local_filepath: str):
    return pkg_resources.resource_filename(__name__, local_filepath)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(0)
    print("set_seed() - random int: ", torch.randint(0, 1000, (1, 1)).item())


def calc_hash(x: str, ignore_dict_keys: List = []) -> str:
    if isinstance(x, dict):
        hash_str = ""
        keys = sorted([str(k) for k in list(x.keys())])
        for k in keys:
            if k in ignore_dict_keys:
                continue
            hash_str += f"{k}:{x[k]}"
        return calc_hash(hash_str)
    if isinstance(x, (int, float)):
        x = str(x)
    res = 0
    for ch in x:
        res = (res * 281 ^ ord(ch) * 997) & 0xFFFFFFFF
    return str(hex(res)[2:].lower().zfill(8))[0:8]


def np_equal(a: np.ndarray, b: np.ndarray) -> bool:
    return (a.shape == b.shape) and (a == b).all()


def np_hash(arr: np.ndarray) -> int:
    return hash(str(arr))


def to_torch(x: np.ndarray, device: str = DEVICE, dtype: torch.dtype = DEFAULT_TORCH_DTYPE) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, device=device, dtype=dtype)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Return a tensor/np array as a numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def boolean_string(s):
    if isinstance(s, bool):
        return s
    if s.upper() not in {"FALSE", "TRUE"}:
        raise ValueError(f'input: "{s}" ("{type(s)}") is not a valid boolean string')
    return s.upper() == "TRUE"


def non_private_dict(d: Dict) -> Dict:
    r = {}
    for k, v in d.items():
        if k[0] == "_":
            continue
        r[k] = v
    return r


def add_prefix_to_dict(d: Dict, prefix: str) -> Dict:
    d_ret = {}
    for k, v in d.items():
        d_ret[f"{prefix}/{k}"] = v
    return d_ret


def dict_subtraction(d: Dict, keys_to_exclude: set) -> Dict:
    return {k: v for k, v in d.items() if k not in keys_to_exclude}


def assert_kwargs_is_valid(kwargs: Dict, allowed_keys: Dict):
    for key in kwargs:
        assert key in allowed_keys, f"Error, kwargs argument '{key}' not found in valid set {allowed_keys}"


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def make_text_green_or_red(text: str, print_green: bool) -> str:
    if print_green:
        s = bcolors.GREEN
    else:
        s = bcolors.FAIL
    return s + str(text) + bcolors.ENDC


def get_evenly_spaced_colors(n):
    base_color = 100  # Base color in HSL (green)
    colors = []
    for i in range(n):
        hue = (base_color + (i * (360 / n))) % 360  # Adjust the step size (30 degrees in this case)
        rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
        colors.append(rgb)
    return colors
