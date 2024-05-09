from typing import List

import torch

from cppflow.problem import Problem
from cppflow.config import DEVICE


def get_only_non_colliding_qpaths(
    qpaths: List[torch.Tensor], self_colliding: torch.Tensor, env_colliding: torch.Tensor
):
    """Return only qpaths which are non colliding (robot-robot and robot-env).

    Args:
        qpaths (List[torch.Tensor]): List of [ntimesteps x ndof] tensors of configs
        self_colliding (torch.Tensor): [k x ntimesteps] tensor of bools indicating if the config results in a self
                                                        collision at each timestep
        env_colliding (torch.Tensor): [k x ntimesteps] tensor of bools indicating if the config is in collision with the
                                                        environment at each timestep
    """
    assert len(qpaths) == self_colliding.shape[0] == env_colliding.shape[0]
    colliding_idxs = torch.logical_or(self_colliding, env_colliding)
    to_keep = torch.sum(colliding_idxs, dim=1) == 0
    return [qpaths[i] for i in to_keep.nonzero()[:, 0]]


def qpaths_batched_env_collisions(problem: Problem, q: torch.Tensor) -> torch.Tensor:
    """Returns boolean tensor where 1 indicates the configuration is in collision with itself

    Args:
        q (torch.Tensor): [k x ntimesteps x n_dofs]

    Returns:
        torch.Tensor: [k x ntimesteps]
    """
    k, n, ndof = q.shape
    colliding = torch.zeros((k, n), dtype=torch.bool)
    q_2d = q.reshape((k * n, ndof))
    for obstacle_cuboid, obstacle_Tcuboid in zip(problem.obstacles_cuboids, problem.obstacles_Tcuboids):
        dists = problem.robot.env_collision_distances(q_2d, obstacle_cuboid, obstacle_Tcuboid)
        min_dists, _ = torch.min(dists, dim=1)
        assert min_dists.numel() == n * k
        colliding = torch.logical_or(colliding, (min_dists < 0).reshape((k, n)))
        # artificially set the collision threshold to get slightly in collision dp_search paths
        # import warnings
        # warnings.warn(f"FYI: setting {margin_m} m margin for environment collisions. REMOVE THIS AFTER DEBUGGING.")
        # margin_m = -0.01
        # colliding = torch.logical_or(colliding, (min_dists < margin_m).reshape((k, n)))
    return colliding


def qpaths_batched_self_collisions(problem: Problem, q: torch.Tensor) -> torch.Tensor:
    """Returns boolean tensor where 1 indicates the configuration is in collision with the environment

    Args:
        q (torch.Tensor): [k x ntimesteps x n_dofs]

    Returns:
        torch.Tensor: [k x ntimesteps]
    """
    assert str(q.device) == DEVICE, f"q.device != device ({q.device} == {DEVICE})"
    k, n, ndof = q.shape
    q_input = q.reshape((k * n, ndof))

    dists = problem.robot.self_collision_distances(q_input)
    min_dists, _ = torch.min(dists, dim=1)
    colliding = min_dists < 0
    colliding = colliding.reshape((k, n))
    return colliding


def self_colliding_configs_capsule(problem: Problem, qpath: torch.Tensor) -> torch.Tensor:
    dists = problem.robot.self_collision_distances(qpath)
    return torch.min(dists, dim=1)[0] < 0


def env_colliding_configs_capsule(problem: Problem, qpath: torch.Tensor) -> torch.Tensor:
    """Returns a boolean tensor of shape (n_timesteps,) where each value is True if the corresponding config in
    qpath is colliding with the environment and False otherwise.
    """
    colliding = torch.zeros(problem.n_timesteps, dtype=torch.bool)
    for obstacle_cuboid, obstacle_Tcuboid in zip(problem.obstacles_cuboids, problem.obstacles_Tcuboids):
        dists = problem.robot.env_collision_distances(qpath, obstacle_cuboid, obstacle_Tcuboid)
        min_dists = torch.min(dists, dim=1)[0]
        colliding = torch.logical_or(colliding, min_dists < 0)
    return colliding


def env_colliding_configs_klampt(problem: Problem, qpath: torch.Tensor) -> torch.Tensor:
    colliding = torch.zeros(problem.n_timesteps, dtype=torch.bool)
    for i, x in enumerate(qpath.cpu().numpy()):
        for j, _ in enumerate(problem.obstacles_klampt):
            try:
                if colliding[i]:
                    continue
            except IndexError as e:
                raise RuntimeError(
                    f"env_colliding_configs_klampt() | index error for 'colliding[i]'\nproblem: {problem}, qpath:"
                    f" {qpath.shape}, i: {i}, x: {x}"
                ) from e

            # Note: config_collides_with_env() accepts an int which refers to the index of the RigidBodyObject saved in
            # the WorldModel which is saved in robot. Obstacles are added to the WorldModel() in problem.py when calling
            # create.box(world=robot.world_model).
            colliding[i] = problem.robot.config_collides_with_env(x, j)
    return colliding


def self_colliding_configs_klampt(problem: Problem, qpath: torch.Tensor) -> torch.Tensor:
    colliding = torch.zeros(problem.n_timesteps, dtype=torch.bool)
    for i, x in enumerate(qpath.cpu().numpy()):
        try:
            colliding[i] = problem.robot.config_self_collides(x)
        except IndexError as e:
            raise RuntimeError(
                "self_colliding_configs_klampt() | index error for 'colliding[i] ="
                f" problem.robot.config_self_collides(x)'\nproblem: {problem}, qpath: {qpath.shape}, i: {i}, x: {x}"
            ) from e

    return colliding


def env_colliding_links_klampt(problem: Problem, q: torch.Tensor) -> List[str]:
    """Returns a list of links that are colliding with the environment at the given configuration."""
    links = []
    for j in range(len(problem.obstacles_klampt)):
        collisions = problem.robot.config_collides_with_env(q, j, return_detailed=True)
        # RobotModelLink, RigidObjectModel
        for robot_link, _ in collisions:  # robot_link, rigid_object
            links.append(robot_link.getName())
    return list(set(links))


def env_colliding_links_capsule(problem: Problem, q: torch.Tensor) -> List[str]:
    """Returns a list of links that are colliding with the environment at the given configuration."""
    links = []
    ordered_links = list(problem.robot._collision_capsules_by_link.keys())
    for obstacle_cuboid, obstacle_Tcuboid in zip(problem.obstacles_cuboids, problem.obstacles_Tcuboids):
        dists = problem.robot.env_collision_distances(q.unsqueeze(0), obstacle_cuboid, obstacle_Tcuboid)
        for i in range(dists.shape[1]):
            if dists[0, i] < 0:
                links.append(ordered_links[i])
    return list(set(links))
