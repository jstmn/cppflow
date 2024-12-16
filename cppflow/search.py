from typing import List
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from jrl.robot import Robot

from cppflow.data_types import Problem
from cppflow.config import DEVICE
from cppflow.utils import TimerContext, cm_to_m
from cppflow.collision_detection import qpaths_batched_self_collisions, qpaths_batched_env_collisions

K_JLIM_COST = 100
K_COLLISION_COST = 1000


# DEFAULT_JLIM_SAFETY_PADDING_REVOLUTE = np.deg2rad(1) # previous values
# DEFAULT_JLIM_SAFETY_PADDING_PRISMATIC = cm_to_m(0.25)
DEFAULT_JLIM_SAFETY_PADDING_REVOLUTE = np.deg2rad(1.5)
DEFAULT_JLIM_SAFETY_PADDING_PRISMATIC = cm_to_m(3)  # changed from 0.25 to 3 due to fetch__hello failures


# NOTE: Really bad LM convergence when increasing eps past 2deg for 'fetch_arm__hello', 'fetch_arm__rot_yz'
def joint_limit_almost_violations_3d(
    robot: Robot,
    qs: torch.Tensor,
    eps_revolute: float = DEFAULT_JLIM_SAFETY_PADDING_REVOLUTE,
    eps_prismatic: float = DEFAULT_JLIM_SAFETY_PADDING_PRISMATIC,
) -> torch.Tensor:
    """Returns a tensor with 1's where the joint angle is within eps of its limit and 0 otherwise

    Timing data:
        - fetch-circle: k=175 -> 0.001915, k=300 -> 0.002455 s

    Timing from the previous implementation:
        - fetch-circle: k=175 -> 0.0723s, k=300 -> 0.1263 s

    Args:
        qs (torch.Tensor): Should be [k x ntimesteps x n_dofs]

    Returns:
        torch.Tensor: [k x ntimesteps]
    """
    assert len(qs.shape) == 3
    l_lim = torch.tensor([l for l, _ in robot.actuated_joints_limits], dtype=qs.dtype, device=qs.device)
    u_lim = torch.tensor([u for _, u in robot.actuated_joints_limits], dtype=qs.dtype, device=qs.device)
    l_lim[robot.prismatic_joint_idxs] += eps_prismatic
    l_lim[robot.revolute_joint_idxs] += eps_revolute
    u_lim[robot.prismatic_joint_idxs] -= eps_prismatic
    u_lim[robot.revolute_joint_idxs] -= eps_revolute
    return torch.logical_or((qs < l_lim).any(dim=2), (qs > u_lim).any(dim=2)).type(torch.float32)


def dp_search_slow(problem: Problem, qpaths: List[torch.Tensor], use_cuda: bool = False, verbosity: int = 1):
    q = torch.stack(qpaths).detach().to(DEVICE)  # [k x ntimesteps x n_dofs]
    k, ntimesteps, ndof = q.shape
    q_device = "cpu" if not use_cuda else DEVICE
    memo = torch.zeros((k, ntimesteps), dtype=torch.int32)

    with TimerContext("calculating configs near joint limits in dp_search()", enabled=verbosity > 0):
        jlimit_violations = joint_limit_almost_violations_3d(problem.robot, q).to(q_device)

    with TimerContext("calculating self-colliding configs in dp_search()", enabled=verbosity > 0):
        self_collision_violations = qpaths_batched_self_collisions(problem, q).to(q_device)

    with TimerContext("calculating env-colliding configs in dp_search()", enabled=verbosity > 0):
        env_collision_violations = qpaths_batched_env_collisions(problem, q).to(q_device)

    costs = torch.zeros((k, ntimesteps), device=q_device, dtype=q.dtype)
    q_costs_external = (
        K_JLIM_COST * jlimit_violations
        + K_COLLISION_COST * env_collision_violations
        + K_COLLISION_COST * self_collision_violations
    )

    costs[:, 0] = q_costs_external[:, 0]
    q = q.to(q_device)

    for t in range(1, ntimesteps):
        for ki in range(k):
            q_k_t = q[ki, t, :]
            dqs = q_k_t - q[:, t - 1, :]  # [n_dofs] - [k x n_dofs]
            absdqs = torch.abs(torch.remainder(dqs + torch.pi, 2 * torch.pi) - torch.pi)
            maxdqs, _ = torch.max(absdqs, 1)
            t_next_cost = torch.maximum(maxdqs, costs[:, t - 1])
            t_next_cost += q_costs_external[ki, t]
            costs[ki, t], memo[ki, t] = torch.min(t_next_cost, 0)

    # Extract best path
    best_path = torch.zeros((ntimesteps, ndof), dtype=q.dtype).to(q.device)
    _, i = torch.min(costs[:, -1], 0)
    for t in range(ntimesteps - 1, -1, -1):
        best_path[t, :] = q[i, t, :]
        i = memo[i, t]

    return best_path


def _get_mjacs(q: torch.Tensor, robot: Robot, prismatic_joint_scaling: float = 5.0) -> torch.Tensor:
    """Returns a tensor of the maximum joint angle changes for each timestep from each of the k paths to one another.

    Args:
        q (torch.Tensor): [k, ntimesteps, ndof] tensor of joint angles
        prismatic_joint_scaling (float, optional): Scaling factor for change in value of the prismatic joints. Without
                                                    this scaling term, q-deltas between prismatic and revolute joints
                                                    are equally weighted, which means a change of 1 rad = 1 meter. This
                                                    would make 57.2958 deg = 100cm -> 5 deg = 8.726cm. A value of 5.0
                                                    was found to provide a good balance between decreasing the prismatic
                                                    mjac while not increasing the revolute mjac too greatly.

    Returns:
        torch.Tensor: [k, k, ntimesteps - 1] tensor of mjacs
    """
    k, ntimesteps, ndof = q.shape
    dqs = q[:, 1:, :].unsqueeze(1).expand(k, k, ntimesteps - 1, ndof) - q[:, :-1, :].unsqueeze(0).expand(
        k, k, ntimesteps - 1, ndof
    )  # [k, k, ntimesteps - 1, ndof]

    if robot.has_prismatic_joints:
        dqs[:, :, :, robot.prismatic_joint_idxs] *= prismatic_joint_scaling

    abs_dqs = torch.abs(torch.remainder(dqs + torch.pi, 2 * torch.pi) - torch.pi)  # [k, k, ntimesteps - 1, ndof]
    mjacs, _ = torch.max(abs_dqs, 3)  # [k, k, ntimesteps - 1]
    return mjacs


def dp_search(
    robot: Robot,
    q: torch.Tensor,
    self_collision_violations: torch.Tensor,
    env_collision_violations: torch.Tensor,
    use_cuda: bool = False,
    verbosity: int = 1,
):
    """
    q (torch.Tensor):  [k x ntimesteps x n_dofs] tensor of joint configurations
    """
    k, ntimesteps, ndof = q.shape
    q_device = "cpu" if not use_cuda else DEVICE
    q = q.to(q_device)

    with TimerContext("calculating configs near joint limits in dp_search()", enabled=verbosity > 0):
        jlimit_violations = joint_limit_almost_violations_3d(robot, q).to(q_device)
    costs = torch.zeros((k, ntimesteps), device=q_device, dtype=q.dtype)
    q_costs_external = (
        K_JLIM_COST * jlimit_violations
        + K_COLLISION_COST * env_collision_violations.to(q_device)
        + K_COLLISION_COST * self_collision_violations.to(q_device)
    )
    costs[:, 0] = q_costs_external[:, 0]

    # Calculate mjacs, run search
    mjacs = _get_mjacs(q, robot)
    memo = torch.zeros((k, ntimesteps), dtype=torch.int32)
    for t in range(1, ntimesteps):
        t_next_cost = torch.maximum(mjacs[:, :, t - 1], costs[:, t - 1])  # [k, k]
        t_next_cost += q_costs_external[:, t].unsqueeze(0).expand(k, k).transpose(0, 1)
        costs[:, t], memo[:, t] = torch.min(t_next_cost, 1)  # [k]

    best_path = torch.zeros((ntimesteps, ndof), dtype=q.dtype).to(q.device)
    _, i = torch.min(costs[:, -1], 0)
    if verbosity > 1:
        xs = []
        ys = []

    for t in range(ntimesteps - 1, -1, -1):
        if verbosity > 1:
            xs.append(t)
            ys.append(i.item())

        best_path[t, :] = q[i, t, :]
        i = memo[i, t]

    # Plot the path
    if verbosity > 2:
        warnings.warn("FYI: SAVING FIGURE. REMOVE THIS WHEN TIMING MATTERS")
        plt.figure(figsize=(10, 10))
        plt.tight_layout()
        plt.imshow(q_costs_external, vmin=0, vmax=K_COLLISION_COST * 2 + K_JLIM_COST)
        plt.plot(xs, ys, label="best path", color="red")
        plt.title("dp_search() cost landscape and returned path")
        plt.xlabel("timestep")
        plt.ylabel("k")
        plt.legend()
        plt.colorbar()
        plt.grid(True, which="both", axis="both")
        plt.savefig("debug__dp_search_path.png", bbox_inches="tight")
        plt.close()

    return best_path
