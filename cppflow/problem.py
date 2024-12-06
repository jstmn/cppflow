from typing import Optional, List, Dict
from dataclasses import dataclass
import os
import csv

import yaml
from jrl.robot import Robot
from jrl.robots import get_robot
from jrl.math_utils import quaternion_norm, geodesic_distance_between_quaternions, rpy_tuple_to_rotation_matrix
import numpy as np
from klampt.math import so3
from klampt.model import create
from klampt import RigidObjectModel
import torch

from cppflow import config
from cppflow.utils import get_filepath, to_torch

np.set_printoptions(suppress=True)

ALL_PROBLEM_FILENAMES = [
    "fetch_arm__hello",
    "fetch_arm__circle",
    # "fetch_arm__rot_yz",
    "fetch_arm__rot_yz2",
    "fetch_arm__s",
    "fetch_arm__square",
    "fetch__circle",
    "fetch__hello",
    # "fetch__rot_yz",
    "fetch__rot_yz2",
    "fetch__s",
    "fetch__square",
    "panda__1cube",
    "panda__2cubes",
    "panda__flappy_bird",
]

ALL_OBS_PROBLEM_FILENAMES = [
    "fetch_arm__circle",
    "fetch_arm__s",
    "fetch_arm__square",
    "fetch__circle",
    "fetch__s",
    "fetch__square",
    "panda__1cube",
    "panda__2cubes",
    "panda__flappy_bird",
]


@dataclass
class Problem:
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
            assert (
                len(self.initial_configuration.shape) == 2
            ), f"'initial_configuration' should be [1, ndof], is {self.initial_configuration.shape}"


def offset_target_path(
    robot: Robot,
    target_path: torch.Tensor,
    path_offset_frame: str,
    xyz_offset: List[float],
    R_offset: List[List[float]],
) -> torch.Tensor:
    """Offset the target path by a given xyz and rotation offset."""
    path = target_path.copy()  # copy to avoid any kind of weirdness

    if path_offset_frame == "world":
        world_T_path_offset = np.zeros(3)
    else:
        world_T_path_offset = robot.forward_kinematics_klampt(
            np.zeros(robot.ndof)[None, :], link_name=path_offset_frame
        )[0]
        # The path_offset reference frame can't be rotated w.r.t. the world frame, for now. This isn't a permenant
        # requirement, I just haven't implemented it yet
        np.testing.assert_allclose(world_T_path_offset[3:], np.array([1, 0, 0, 0]), atol=1e-8)

    for i in range(3):
        path[:, i] += xyz_offset[i] + world_T_path_offset[i]

    R_offset_klampt = so3.from_ndarray(np.array(R_offset))
    for i in range(path.shape[0]):
        R_i_klampt = so3.from_quaternion(path[i, 3:7])
        R_updated = so3.mul(R_i_klampt, R_offset_klampt)
        q_updated = np.array(so3.quaternion(R_updated))
        path[i, 3:7] = q_updated
    return to_torch(path)


def get_obstacles(robot: Robot, problem_dict: Dict, device=config.DEVICE):
    obstacles_unparsed = problem_dict["obstacles"] if "obstacles" in problem_dict else []
    obstacles = obstacles_unparsed
    obstacles_Tcuboids = []
    obstacles_cuboids = []
    if len(obstacles_unparsed) > 0:
        # assert "obstacle_xyz_offset" in problem_dict, f"'obstacle_xyz_offset' not found for {problem_filename}"
        obstacles_parsed = []
        for obs in obstacles:
            parsed = {}
            for d in obs:
                for k, v in d.items():
                    parsed[k] = v
            # Shift obstacles by 'obstacle_xyz_offset'
            parsed["x"] += problem_dict["obstacle_xyz_offset"][0]
            parsed["y"] += problem_dict["obstacle_xyz_offset"][1]
            parsed["z"] += problem_dict["obstacle_xyz_offset"][2]
            #
            obstacles_parsed.append(parsed)
        obstacles = obstacles_parsed
        for obs in obstacles:
            assert abs(obs["roll"]) < 1e-8 and abs(obs["pitch"]) < 1e-8 and abs(obs["yaw"]) < 1e-8
            cuboid = torch.tensor(
                [
                    -obs["size_x"] / 2,
                    -obs["size_y"] / 2,
                    -obs["size_z"] / 2,
                    obs["size_x"] / 2,
                    obs["size_y"] / 2,
                    obs["size_z"] / 2,
                ],
                device=device,
            )
            Tcuboid = torch.zeros((4, 4), device=device)
            Tcuboid[0, 3] = obs["x"]
            Tcuboid[1, 3] = obs["y"]
            Tcuboid[2, 3] = obs["z"]
            Tcuboid[:3, :3] = rpy_tuple_to_rotation_matrix((obs["roll"], obs["pitch"], obs["yaw"]))

            obstacles_Tcuboids.append(Tcuboid)
            obstacles_cuboids.append(cuboid)

    R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    obstacles_klampt = [
        create.box(
            width=obs["size_x"],
            height=obs["size_z"],
            depth=obs["size_y"],
            center=(0, 0, 0),
            t=(obs["x"], obs["y"], obs["z"]),
            R=R,
            mass=1,
            world=robot._klampt_world_model,  # passing robot._klampt_world_model adds the box as a RigidObjectModel to robot._klampt_world_model
        )
        for obs in obstacles
    ]
    robot.make_new_collision_checker()

    return obstacles, obstacles_Tcuboids, obstacles_cuboids, obstacles_klampt


def problem_from_filename(
    problem_filename: str, filepath_override: Optional[str] = None, robot: Optional[Robot] = None
) -> Problem:
    """Parse a yaml file and create a Problem object.

    A note: about 'obstacle_xyz_offset'. Some of the torm problems set the fetch torso_lift_link to 0.2. To account for
    this, the paths are optionally shifted so that the are in the correct location when the torso_lift_link is at 0,
    which it is for the Fetch.Arm problems in this directory.

    Args:
        problem_filename (str): _description_
        filepath_override (Optional[str], optional): Optionally provide the filepath directly. Used for testing.
                                                        Defaults to None.
    """

    # Get the filepath to the problem definition
    if filepath_override is None:
        assert (
            "yaml" not in problem_filename
        ), f"problem_filename should not include the .yaml file extension. '{problem_filename}'"
        filepath = get_filepath(os.path.join("problems/", problem_filename + ".yaml"))
    else:
        filepath = filepath_override

    # Parse the problem definition
    with open(filepath, "r") as f:
        problem_dict = yaml.load(f, Loader=yaml.FullLoader)

    if robot is None:
        robot = get_robot(problem_dict["robot"])
    else:
        assert "obstacles" not in problem_dict, f"Error - obstacles found for {problem_filename} but robot is provided"

    obstacles, obstacles_Tcuboids, obstacles_cuboids, obstacles_klampt = get_obstacles(robot, problem_dict)
    assert robot._klampt_world_model.numRigidObjects() == len(
        obstacles
    ), f"Error - {robot._klampt_world_model.numRigidObjects()} != {len(obstacles)}"

    # Load the end effector path
    path_name = problem_dict["path_name"]
    with open(get_filepath(os.path.join("paths/", path_name + ".csv")), "r") as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        rows = [[float(x) for x in row] for i, row in enumerate(csvreader) if i > 0]

    # Offset the end effector path according to 'path_xyz_offset', 'path_R_offset'. These are specified in the
    # 'path_offset_frame' frame
    # NOTE: 'path_R_offset' applies a rotation for each individual pose in the path, it does not rotate the path's
    # reference frame.
    original_path = np.array(rows)[:, 1:]
    target_path = offset_target_path(
        robot,
        original_path,
        problem_dict["path_offset_frame"],
        problem_dict["path_xyz_offset"],
        problem_dict["path_R_offset"],
    )
    name = path_name
    return Problem(
        target_path,
        None,
        robot,
        name,
        problem_filename,
        obstacles,
        obstacles_Tcuboids,
        obstacles_cuboids,
        obstacles_klampt,
    )


def get_problem_dict(problem_names: List[str]) -> Dict[str, Problem]:
    robot_map = {pr: None for pr in problem_names}
    if "fetch__hello" in problem_names and "fetch__rot_yz2" in problem_names:
        fetch = get_robot("fetch")
        robot_map["fetch__hello"] = fetch
        robot_map["fetch__rot_yz2"] = fetch
        print("FYI: using one Fetch robot for 'fetch__hello' and 'fetch__rot_yz2'")

    if "fetch_arm__hello" in problem_names and "fetch_arm__rot_yz2" in problem_names:
        fetch_arm = get_robot("fetch_arm")
        robot_map["fetch_arm__hello"] = fetch_arm
        robot_map["fetch_arm__rot_yz2"] = fetch_arm
        print("FYI: using one FetchArm robot for 'fetch_arm__hello' and 'fetch_arm__rot_yz2'")

    return {pname: problem_from_filename(pname, robot=robot_map[pname]) for pname in problem_names}


def get_all_problems() -> List[Problem]:
    problem_dict = get_problem_dict(ALL_PROBLEM_FILENAMES)
    return [problem_dict[pname] for pname in ALL_PROBLEM_FILENAMES]
