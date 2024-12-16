from typing import Optional, List, Dict
import os
import csv

import yaml
from jrl.robot import Robot
from jrl.robots import get_robot
from jrl.math_utils import rpy_tuple_to_rotation_matrix
import numpy as np
from klampt.math import so3
from klampt.model import create
import torch


from cppflow.config import ENV_COLLISIONS_IGNORED, SELF_COLLISIONS_IGNORED
from cppflow.evaluation_utils import positional_errors, rotational_errors
from cppflow.collision_detection import self_colliding_configs_klampt, env_colliding_configs_klampt
from cppflow.data_types import Problem, Plan, Constraints
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
    constraints: Constraints,
    problem_filename: str,
    filepath_override: Optional[str] = None,
    robot: Optional[Robot] = None,
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
        constraints,
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
    if SELF_COLLISIONS_IGNORED:
        self_colliding = torch.zeros_like(self_colliding)

    env_colliding = env_colliding_configs_klampt(problem, qpath)
    if ENV_COLLISIONS_IGNORED:
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
        provided_initial_configuration=problem.initial_configuration,
        constraints=problem.constraints,
    )
