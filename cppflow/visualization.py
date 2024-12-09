from time import time, sleep
from typing import List, Optional, Union, Tuple

from klampt.model import coordinates, trajectory, create
from klampt import vis
from klampt.math import so3
import matplotlib.pyplot as plt
import numpy as np
from jrl.robot import Robot
from jrl.config import PT_NP_TYPE
import torch

from cppflow.evaluation_utils import angular_changes
from cppflow.data_types import Plan, PlanNp, plan_from_qpath, Constraints
from cppflow.problem import Problem
from cppflow.collision_detection import (
    env_colliding_links_klampt,
    env_colliding_links_capsule,
    env_colliding_configs_klampt,
)
from cppflow.utils import to_numpy, to_torch

PI = np.pi


def delay(t: float):
    # Start delay. for filming, etc
    if t > 0:
        t0 = time()
        while time() - t0 < t:
            sleep(0.5)
            print("...")


# TODO: Consider plotting qpaths from midway through the optimization
def plot_optimized_trajectory(
    problem: Problem,
    constraints: Constraints,
    qpath_0: PT_NP_TYPE,
    qpath_optimized: PT_NP_TYPE,
    xs: List[torch.Tensor] = [],
    figsize: Tuple = (13, 12),
    max_plot_ys: Tuple = (None, None, None),
    show: bool = True,
    save_filepath: Optional[str] = None,
    plot_obstacles: bool = True,
    suptitle: Optional[str] = None,
):
    """Detailed plot showing a qpath and its optimized version."""
    max_plot_ys = (None, None, None) if max_plot_ys is None else max_plot_ys
    n = qpath_0.shape[0]
    assert len(xs) == 0, "xs is unimplemented"
    target_path = to_numpy(problem.target_path)
    plan_0: Plan = PlanNp(plan_from_qpath(qpath_0, problem, constraints))
    plan_f: Plan = PlanNp(plan_from_qpath(qpath_optimized, problem, constraints))
    robot = problem.robot
    if (qpath_0 - qpath_optimized).abs().max() < 1e-8:
        print("warning: qpath_0 & qpath_optimized are the same")
    assert (
        qpath_0.shape == qpath_optimized.shape
    ), f"Error: qpath_0.shape != qpath_optimized.shape ({qpath_0.shape} != {qpath_optimized.shape})"
    assert (
        qpath_0.shape[0] == target_path.shape[0]
    ), f"Error: length qpath_0 != length target_path ({qpath_0.shape}[0] != {target_path.shape}[0])"

    # Setup subplots
    fig, axs = plt.subplots(4, 2, figsize=figsize)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=15)
    else:
        fig.suptitle(problem.full_name, fontweight="bold")
    fig.tight_layout()
    ylabels = [
        "degrees",
        "radians",
        "error (mm)",
        "error (deg)",
        "joint angle change (deg)",
        "joint angle change (cm)",
        "distance (m)",
        "distance (m)",
    ]
    titles = [
        "Joint values",
        "Trajectory length contribution",
        "Positional error",
        "Rotational error",
        "Per-timestep mjac - revolute",
        "Per-timestep mjac - prismatic",
        "Min. Distance to self-collision",
        "Distance to environment-collision",
    ]
    for i, (ax, ylabel, title) in enumerate(zip(axs.flatten(), ylabels, titles)):
        ax.grid(alpha=0.25)
        # ax.set_xlabel("timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    idxs = list(range(qpath_optimized.shape[0]))
    adelta_qpath_0 = angular_changes(qpath_0).abs()
    adelta_qpath_f = angular_changes(qpath_optimized).abs()

    optimized_colors = ["blue", "green", "black", "orange"]
    joint_limit_colors = [
        "tab:blue",
        "tab:orange",
        "tab:red",
        "maroon",
        "cyan",
        "midnightblue",
        "forestgreen",
        "dimgrey",
    ]

    # Joint values
    ax = axs[0, 0]
    eps = np.deg2rad(2)
    i_oc = 0
    something_plotted = False
    for joint_idx in range(robot.ndof):
        if robot.ndof == 8 and joint_idx == 0:
            continue
        l, u = robot.actuated_joints_limits[joint_idx]
        angles0 = qpath_0[:, joint_idx].cpu().numpy()
        angles_opt = qpath_optimized[:, joint_idx].cpu().numpy()
        close_to_lower = angles_opt.min() < l + eps
        close_to_upper = u - eps < angles_opt.max()
        if close_to_lower or close_to_upper:
            something_plotted = True
            ax.plot(idxs, np.rad2deg(angles0), color="red")
            p = ax.plot(
                idxs, np.rad2deg(angles_opt), label=f"joint {joint_idx} - optimized", color=joint_limit_colors[i_oc]
            )
            color = p[0].get_color()
            i_oc += 1
        if close_to_lower:
            ax.plot(idxs, np.rad2deg(l * np.ones(len(idxs))), color=color, alpha=0.5, linestyle="--")
        if close_to_upper:
            ax.plot(idxs, np.rad2deg(u * np.ones(len(idxs))), color=color, alpha=0.5, linestyle="--")

    if robot.ndof == 8:
        l, u = robot.actuated_joints_limits[0]
        ax.plot(idxs, l * np.ones(len(idxs)), color="darkblue", alpha=0.5, linestyle="--")
        ax.plot(idxs, u * np.ones(len(idxs)), color="darkblue", alpha=0.5, linestyle="--")
        ax.plot(idxs, qpath_0[:, 0].cpu().numpy(), color="red")
        ax.plot(idxs, qpath_optimized[:, 0].cpu().numpy(), label=f"joint 0 (prismatic)", color="darkblue")
        something_plotted = True

    if not something_plotted:
        joint_idx = 1
        ax.plot(idxs, np.rad2deg(qpath_0[:, joint_idx].cpu().numpy()), color="red")
        ax.plot(
            idxs,
            np.rad2deg(qpath_optimized[:, joint_idx].cpu().numpy()),
            label=f"joint {joint_idx} - (no close by joint limits)",
            color=joint_limit_colors[i_oc],
        )

    ax.legend(loc="upper right")

    # Trajectory length contribution
    ax = axs[0, 1]
    degs = []
    y_0s = []
    y_fs = []

    deg_max = int(torch.rad2deg(adelta_qpath_f.max()).item()) + 1

    for i in range(deg_max):
        l = np.deg2rad(i)
        u = np.deg2rad(i + 1)

        adeltas_0 = adelta_qpath_0[torch.logical_and(adelta_qpath_0 > l, adelta_qpath_f < u)]
        adeltas_f = adelta_qpath_f[torch.logical_and(adelta_qpath_0 > l, adelta_qpath_f < u)]

        degs.append(i + 1)
        y_0s.append(adeltas_0.sum().item())
        y_fs.append(adeltas_f.sum().item())

    ax.text(
        deg_max - 1, min(y_fs) + 3, f"TL_0: {adelta_qpath_0.sum().item():.2f}\nTL_f: {adelta_qpath_f.sum().item():.2f}"
    )
    ax.set_title(f"sum(|delta-q|(i < deg < i + 1)) by bucketed deg")
    ax.grid(alpha=0.25)
    ax.plot(degs, y_0s, label=f"initial", color="red")
    ax.scatter(degs, y_0s, color="red")
    ax.plot(degs, y_fs, label=f"final", color="black")
    ax.scatter(degs, y_fs, color="black")
    ax.legend(loc="upper right")

    # if robot.has_prismatic_joints:
    #     joint_idx = 0
    #     l, u = robot.actuated_joints_limits[joint_idx]
    #     ax.plot(idxs, 100 * l * np.ones(len(idxs)), color=color, alpha=0.5, linestyle="--")
    #     ax.plot(idxs, 100 * u * np.ones(len(idxs)), color=color, alpha=0.5, linestyle="--")
    #     ax.plot(idxs, 100 * qpath_0[:, joint_idx].cpu().numpy(), color="red")
    #     ax.plot(
    #         idxs,
    #         100 * qpath_optimized[:, joint_idx].cpu().numpy(),
    #         label=f"joint {joint_idx} - optimized",
    #         color=joint_limit_colors[i_oc],
    #     )
    #     ax.legend(loc="upper left")

    # Positional error
    ax = axs[1, 0]
    ax.grid(alpha=0.25)
    ax.plot(idxs, plan_0.positional_errors_mm, label="initial", color="red")
    ax.plot(idxs, plan_0.positional_errors_mm.max() * np.ones(len(idxs)), color="red", alpha=0.5, linestyle="--")
    ax.plot(idxs, plan_f.positional_errors_mm, label="final", color="black")
    ax.plot(idxs, plan_f.positional_errors_mm.max() * np.ones(len(idxs)), color="black", alpha=0.5, linestyle="--")
    ax.plot(
        idxs,
        SUCCESS_THRESHOLD_translation_ERR_MAX_MM * np.ones(len(idxs)),
        color="green",
        alpha=0.5,
        linestyle="--",
        label="max allowed",
    )
    if max_plot_ys[0] is None:
        ax.set_ylim(0, max(plan_f.positional_errors_mm.max(), SUCCESS_THRESHOLD_translation_ERR_MAX_MM) * 1.1)
    else:
        ax.set_ylim(0, max_plot_ys[0])
    # ax.set_ylim(0, None)
    ax.legend(loc="upper right")

    # Angular error
    ax = axs[1, 1]
    ax.plot(idxs, plan_0.rotational_errors_deg, label="initial", color="red")
    ax.plot(idxs, plan_0.rotational_errors_deg.max() * np.ones(len(idxs)), color="red", alpha=0.5, linestyle="--")
    ax.plot(idxs, plan_f.rotational_errors_deg, label="final", color="black")
    ax.plot(idxs, plan_f.rotational_errors_deg.max() * np.ones(len(idxs)), color="black", alpha=0.5, linestyle="--")
    ax.plot(
        idxs,
        constraints.max_allowed_rotation_error_deg * np.ones(len(idxs)),
        color="green",
        alpha=0.5,
        linestyle="--",
        label="max allowed",
    )
    ax.legend(loc="upper right")
    if max_plot_ys[0] is None:
        ax.set_ylim(0, max(plan_f.rotational_errors_deg.max(), constraints.max_allowed_rotation_error_deg) * 1.1)
    else:
        ax.set_ylim(0, max_plot_ys[1])

    # Maximum joint angle change - revolute
    ax = axs[2, 0]
    # ax.set_yscale("log")
    ax.plot(
        idxs,
        constraints.max_allowed_mjac_deg * np.ones(len(idxs)),
        label="max allowed",
        alpha=0.5,
        linestyle="--",
        color="green",
    )
    ax.plot(idxs[:-1], plan_0.mjac_per_timestep_deg, label="initial", color="red")
    ax.plot(idxs, plan_0.mjac_per_timestep_deg.max() * np.ones(len(idxs)), alpha=0.5, linestyle="--", color="red")
    ax.plot(idxs[:-1], torch.mean(adelta_qpath_f, dim=1).cpu().numpy(), label="final", color="grey")
    ax.plot(idxs[:-1], plan_f.mjac_per_timestep_deg, label="final", color="black")
    ax.plot(
        idxs,
        plan_f.mjac_per_timestep_deg.max() * np.ones(len(idxs)),
        alpha=0.5,
        linestyle="--",
        color="black",
    )
    ax.set_ylim(0, None)
    # ax.set_ylim(0, max(plan_f.mjac_per_timestep_deg.max(), constraints.max_allowed_mjac_deg) * 1.1)

    # Maximum joint angle change - prismatic
    ax = axs[2, 1]
    ax.plot(
        idxs,
        constraints.max_allowed_mjac_cm * np.ones(len(idxs)),
        label="max allowed - prismatic",
        alpha=0.5,
        linestyle="--",
        color="green",
    )
    ax.plot(idxs[:-1], plan_0.mjac_per_timestep_cm, label="initial", color="red")
    ax.plot(idxs, plan_0.mjac_per_timestep_cm.max() * np.ones(len(idxs)), alpha=0.5, linestyle="--", color="red")
    ax.plot(idxs[:-1], plan_f.mjac_per_timestep_cm, label="final - prismatic", color="black")
    ax.plot(
        idxs,
        plan_f.mjac_per_timestep_cm.max() * np.ones(len(idxs)),
        alpha=0.5,
        linestyle="--",
        color="black",
    )
    ax.legend(loc="upper right")
    ax.set_ylim(0, None)
    # ax.set_ylim(0, max(plan_f.mjac_per_timestep_cm.max(), constraints.max_allowed_mjac_cm) * 1.1)

    # Self-collision avoidance
    ax = axs[3, 0]
    self_collision_dists_start = robot.self_collision_distances(qpath_0)
    min_selfcol_distance_per_ts_start, _ = torch.min(self_collision_dists_start, dim=1)
    assert min_selfcol_distance_per_ts_start.numel() == n
    self_collision_dists_final = robot.self_collision_distances(qpath_optimized)
    min_selfcol_distance_per_ts_final, _ = torch.min(self_collision_dists_final, dim=1)
    assert min_selfcol_distance_per_ts_final.numel() == n
    ax.plot(idxs, min_selfcol_distance_per_ts_start.cpu().numpy(), color="red", label="initial")
    ax.plot(idxs, min_selfcol_distance_per_ts_final.cpu().numpy(), color="black", label="final")
    ax.legend()
    ax.set_ylim(
        min(
            min_selfcol_distance_per_ts_start.cpu().numpy().min(),
            min_selfcol_distance_per_ts_final.cpu().numpy().min(),
            0,
        ),
        None,
    )

    # Environment collision avoidance
    ax = axs[3, 1]
    if plot_obstacles and len(problem.obstacles_cuboids) > 0:
        colliding_klampt = env_colliding_configs_klampt(problem, qpath_optimized)
        ax.plot(idxs, 0.01 * colliding_klampt.cpu().numpy(), color="darkorange", label="klampt")

        for i, (obstacle_cuboid, obstacle_Tcuboid) in enumerate(
            zip(problem.obstacles_cuboids, problem.obstacles_Tcuboids)
        ):
            env_collision_dists_start = robot.env_collision_distances(qpath_0, obstacle_cuboid, obstacle_Tcuboid)
            min_envcol_distance_per_ts_start, _ = torch.min(env_collision_dists_start, dim=1)

            assert min_envcol_distance_per_ts_start.numel() == n
            env_collision_dists_final = robot.env_collision_distances(
                qpath_optimized, obstacle_cuboid, obstacle_Tcuboid
            )
            min_envcol_distance_per_ts_final, _ = torch.min(env_collision_dists_final, dim=1)
            assert min_envcol_distance_per_ts_final.numel() == n
            ax.plot(idxs, min_envcol_distance_per_ts_start.cpu().numpy(), color="red")
            ax.plot(
                idxs,
                min_envcol_distance_per_ts_final.cpu().numpy(),
                color=optimized_colors[i],
                label=f"min-dist. to obs_{i}",
            )
        ax.legend(loc="upper right")
    # ax.set_ylim(-0.02, 0.05)
    ax.set_ylim(-0.01, 0.01)

    if save_filepath is not None:
        plt.savefig(save_filepath, bbox_inches="tight")
    if not show:
        plt.close(fig)


# TODO: highlight portions of best_qpath that exceed smoothness constraints
def plot_pose_error_distribution(
    problem: Problem,
    qpath_or_plans_optimized: List[Union[PT_NP_TYPE, Plan]],
    qpath_or_plan_groups: List[List[Union[PT_NP_TYPE, Plan]]],
    group_names: List[str],
):
    """Plot the positional and rotational errors of a bunch of trajectories"""
    plan_groups = []
    for qpath_or_plan_group in qpath_or_plan_groups:
        plans = []
        for qpath_or_plan in qpath_or_plan_group:
            if isinstance(qpath_or_plan, (np.ndarray, torch.Tensor)):
                plans.append(plan_from_qpath(to_torch(qpath_or_plan), problem, constraints))
            else:
                plans.append(qpath_or_plan)
        plan_groups.append(plans)

    final_plans = []
    for qpath_or_plan in qpath_or_plans_optimized:
        if isinstance(qpath_or_plan, (np.ndarray, torch.Tensor)):
            final_plans.append(plan_from_qpath(to_torch(qpath_or_plan), problem, constraints))
        else:
            final_plans.append(qpath_or_plan)

    fig, axs = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    xs = list(range(problem.target_path.shape[0]))
    colors = ["tab:blue", "tab:orange", "tab:red", "maroon", "cyan", "midnightblue", "forestgreen", "dimgrey"]
    fig.suptitle(f"Timestep vs. pose error for '{problem.full_name}'", fontsize=15)

    ALPHA = 0.2
    S = 0.5

    def _setup_axis(ax, title, ylabel):
        ax.set_title(title)
        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)

    ax = axs[0, 0]
    _setup_axis(ax, "Translational error", "L2 error (cm)")
    for i, plans in enumerate(plan_groups):
        color = colors[i]
        for j, plan in enumerate(plans):
            ax.scatter(xs, plan.positional_errors_cm, alpha=ALPHA, s=S, color=color)

    ax.set_ylim(0, np.max(list(plan.positional_errors_cm for plan in final_plans)))

    # AX_2
    ax = axs[0, 1]
    _setup_axis(ax, "Rotational error", "Rotational error (deg)")
    for i, plans in enumerate(plan_groups):
        color = colors[i]
        for j, plan in enumerate(plans):
            label = group_names[i] if j == 0 else None
            ax.scatter(xs, plan.rotational_errors_deg, alpha=ALPHA, s=S, color=color, label=label)

    # AX_3
    ax = axs[1, 0]
    _setup_axis(ax, "Joint angle change", "Per-timestep max. change (deg)")
    for i, plan in enumerate(final_plans):
        ax.plot(xs[1:], plan.mjac_per_timestep_deg, alpha=1.0, label=group_names[i], color=colors[i])
    ax.legend()


# TODO: highlight portions of best_qpath that exceed smoothness constraints
def plot_trajectory_distribution(
    robot: Robot,
    left_qpaths: List[Union[PT_NP_TYPE, Plan]],
    right_qpaths: List[Union[PT_NP_TYPE, Plan]],
    left_name: str = "left",
    right_name: str = "right",
    left_qpath: Optional[Union[PT_NP_TYPE, Plan]] = None,
    right_qpath: Optional[Union[PT_NP_TYPE, Plan]] = None,
    same_color: bool = True,
):
    """Plot a bunch of trajectories on the same plot"""
    if len(left_qpaths) > 0 and isinstance(left_qpaths[0], Plan):
        left_qpaths = [plan.q_path for plan in left_qpaths]
    if len(right_qpaths) > 0 and isinstance(right_qpaths[0], Plan):
        right_qpaths = [plan.q_path for plan in right_qpaths]
    if left_qpath is not None and isinstance(left_qpath, Plan):
        left_qpath = left_qpath.q_path
    if right_qpath is not None and isinstance(right_qpath, Plan):
        right_qpath = right_qpath.q_path

    n_joints = robot.ndof
    n_cols = 2 if len(right_qpaths) > 0 else 1
    if n_cols == 2:
        figsize = (15, 4 * n_joints + 2)
    else:
        figsize = (8, 3 * n_joints + 2)
    fig, axs = plt.subplots(n_joints, n_cols, figsize=figsize, constrained_layout=True)
    # if n_cols == 1:
    #     fig.tight_layout()

    if len(left_qpaths) > 0:
        if isinstance(left_qpaths[0], torch.Tensor):
            left_qpaths = [traj.cpu().numpy() for traj in left_qpaths]
    if len(right_qpaths) > 0:
        if isinstance(right_qpaths[0], torch.Tensor):
            right_qpaths = [traj.cpu().numpy() for traj in right_qpaths]

    if len(left_qpaths) > 10:
        ALPHA = 0.2
    else:
        ALPHA = 1.0

    if same_color:
        COLOR_L = "blue"
        COLOR_R = "red"
    else:
        COLOR_L = None
        COLOR_R = None
    COLOR_BEST_QPATH = "black"

    S = 0.5
    xs = list(range(left_qpaths[0].shape[0]))

    def _format_ax(_ax, joint_i, description):
        _ax.set_title(f"Joint {joint_i+1} values - {description}")
        _ax.set_xlabel("timestep")
        _ax.set_ylabel("radians")
        _ax.grid(alpha=0.25)

    # -----------------
    # Fill out subplots
    #
    for joint_idx in range(n_joints):
        if n_cols == 2:
            ax = axs[joint_idx, 0]
        else:
            ax = axs[joint_idx]

        _format_ax(ax, joint_idx, left_name)
        color = COLOR_L
        if len(left_qpaths) < 7:
            color = None
        for qpath in left_qpaths:
            ax.scatter(xs, qpath[:, joint_idx], alpha=ALPHA, color=color, s=S)

        if left_qpath is not None:
            ax.plot(xs, left_qpath[:, joint_idx], alpha=1.0, color=COLOR_BEST_QPATH)

        if n_cols < 2:
            continue
        ax = axs[joint_idx, 1]
        _format_ax(ax, joint_idx, right_name)

        color = COLOR_R
        if len(right_qpaths) < 7:
            color = None
        for qpath in right_qpaths:
            ax.scatter(xs, qpath[:, joint_idx], alpha=ALPHA, color=color, s=S)

        if right_qpath is not None:
            ax.plot(xs, right_qpath[:, joint_idx], alpha=1.0, color=COLOR_BEST_QPATH)


def plot_plan(
    plan: Plan,
    problem: Problem,
    other_plans: List[Plan],
    other_plan_names: List[Plan],
):
    fig, axs = plt.subplots(4, 2, figsize=(16, 18))  # (width, height)
    n = plan.q_path.shape[0]
    xs = list(range(n))
    plan = PlanNp(plan)
    ikflow_plans = []  # [PlanNp(plan) for plan in ikflow_plans]
    other_plans = [PlanNp(plan) for plan in other_plans]
    are_prismatic_joints = plan.q_path_prismatic.size > 0
    for row in range(4):
        for col in range(2):
            axs[row, col].grid(alpha=0.25)

    _OTHER_PLAN_ALPHA = 0.9
    _OTHER_PLAN_LINE_STYLE = "dotted"

    ax: plt.Axes = axs[0, 0]
    ax.set_title("L2 Error")
    ax.set_xlabel("timestep")
    ax.set_ylabel("mm")
    ax.plot(plan.positional_errors_mm, label="plan")
    for i, other_plan in enumerate(other_plans):
        ax.plot(
            other_plan.positional_errors_mm,
            label=other_plan_names[i],
            alpha=_OTHER_PLAN_ALPHA,
            linestyle=_OTHER_PLAN_LINE_STYLE,
        )
    ax.plot(
        SUCCESS_THRESHOLD_translation_ERR_MAX_MM * np.ones(n),
        color="red",
        linestyle="--",
        label="max. allowed error",
    )
    ax.legend()

    # PLOT 2: ANGULAR ERROR
    ax = axs[1, 0]
    ax.set_title("Angular Error")
    ax.set_xlabel("timestep")
    ax.set_ylabel("degrees")
    ax.plot(plan.rotational_errors_deg, label="plan")
    for i, other_plan in enumerate(other_plans):
        ax.plot(
            other_plan.rotational_errors_deg,
            label=other_plan_names[i],
            alpha=_OTHER_PLAN_ALPHA,
            linestyle=_OTHER_PLAN_LINE_STYLE,
        )
    ax.plot(
        constraints.max_allowed_rotation_error_deg * np.ones(n),
        color="red",
        linestyle="--",
        label="max. allowed error",
    )
    ax.legend()

    # PLOT 3: JOINT ANGLES
    ax = axs[2, 0]
    ax.set_title("Joint angles")
    ax.set_xlabel("timestep")
    ax.set_ylabel("degrees")

    def plot_qs(_qpath, set_colors=None, linestyle=None, include_bounds=True, name=""):
        colors = []
        for i in range(problem.robot.ndof):
            l, u = problem.robot.actuated_joints_limits[i]
            l = np.rad2deg(l)
            u = np.rad2deg(u)
            qs = np.rad2deg(_qpath[:, i])
            close_to_lim = abs(np.max(qs) - u) < 2.0 or abs(np.min(qs) - l) < 2.0
            if not close_to_lim:
                colors.append(None)
                continue
            color = set_colors[i] if set_colors is not None else None
            p = ax.plot(qs, label=f"joint_{i} {name}", linestyle=linestyle, color=color)
            color = p[0].get_color()
            colors.append(color)
            if include_bounds:
                ax.plot(l * np.ones(n), alpha=0.25, color=color)
                ax.plot(u * np.ones(n), alpha=0.25, color=color)

            violating_idxs = np.logical_or((qs < l), (u < qs)).nonzero()[0]
            if len(violating_idxs) > 0:
                violating_vals = qs[violating_idxs]
                ax.scatter(violating_idxs, violating_vals, color="red", label=f"joint_{i} JL almost violated")
        return colors

    if problem.robot.ndof == 8:
        l, u = problem.robot.actuated_joints_limits[0]
        ax.plot(100 * l * np.ones(n), color="darkblue", alpha=0.5, linestyle="--")
        ax.plot(100 * u * np.ones(n), color="darkblue", alpha=0.5, linestyle="--")
        ax.plot(100 * plan.q_path[:, 0], label=f"joint 0 (prismatic)", color="darkblue")
        for i, other_plan in enumerate(other_plans):
            ax.plot(
                100 * other_plan.q_path[:, 0],
                label=other_plan_names[i],
                alpha=_OTHER_PLAN_ALPHA,
                linestyle=_OTHER_PLAN_LINE_STYLE,
            )
    else:
        colors = plot_qs(plan.q_path)
        for other_plan_name, other_plan in zip(other_plan_names, other_plans):
            plot_qs(
                other_plan.q_path,
                set_colors=colors,
                linestyle=_OTHER_PLAN_LINE_STYLE,
                name=other_plan_name,
            )
    ax.legend(loc="upper left")

    # PLOT 4: MJAC
    ax = axs[3, 0]
    ax.set_title("Max joint space change, at each timestep")
    ax.set_xlabel("timestep")
    ax.set_ylabel("degrees" if not are_prismatic_joints else "degrees / cm")

    for i, other_plan in enumerate(other_plans):
        ax.plot(
            other_plan.mjac_per_timestep_deg,
            label=other_plan_names[i],
            alpha=_OTHER_PLAN_ALPHA,
            linestyle=_OTHER_PLAN_LINE_STYLE,
        )
    ax.plot(plan.mjac_per_timestep_deg, label="plan", color="red")
    ax.plot(
        constraints.max_allowed_mjac_deg * np.ones(n - 1),
        color="red",
        linestyle="--",
        label="max. allowed revolute mjac",
    )
    if are_prismatic_joints:
        ax.plot(plan.mjac_per_timestep_cm, label="plan - prismatic", color="black")
        ax.plot(
            constraints.max_allowed_mjac_cm * np.ones(n - 1),
            color="black",
            linestyle="--",
            label="max. allowed prismatic mjac",
        )
    ax.legend()

    # RIGHT PLOT 1: IKFlow solution error (positional)
    ax = axs[0, 1]
    ax.set_title("IKFlow pose error - positional")
    ax.set_xlabel("timestep")
    ax.set_ylabel("positional error (cm)")
    for i, ikf_plan in enumerate(ikflow_plans):
        ax.scatter(xs, ikf_plan.positional_errors_cm, color="tab:orange", alpha=0.5, s=0.75)

    # RIGHT PLOT 2: IKFlow solution error (rotational)
    ax = axs[1, 1]
    ax.set_title("IKFlow pose error - rotational")
    ax.set_xlabel("timestep")
    ax.set_ylabel("rotational error (rad)")
    for i, ikf_plan in enumerate(ikflow_plans):
        ax.scatter(xs, ikf_plan.rotational_errors_deg, color="tab:orange", alpha=0.5, s=0.75)

    # RIGHT PLOT 3: Minimum distance to self collision
    ax = axs[2, 1]
    ax.set_title("Minimum distance to self collision")
    ax.set_xlabel("timestep")
    ax.set_ylabel("distance (cm)")
    x = plan.q_path
    dists = problem.robot.self_collision_distances(to_torch(x))
    min_dists, _ = torch.min(dists, dim=1)
    ax.plot(xs, 100 * min_dists.cpu().numpy(), color="tab:orange", alpha=0.5)
    ax.set_ylim(0, None)

    # RIGHT PLOT 4: Minimum distance to environment collision
    ax = axs[3, 1]
    ax.set_title("Minimum distance to environment collision")
    ax.set_xlabel("timestep")
    ax.set_ylabel("distance (cm)")
    if len(problem.obstacles_cuboids) > 0:
        x = plan.q_path
        dists = []
        for cuboid, Tcuboid in zip(problem.obstacles_cuboids, problem.obstacles_Tcuboids):
            dists.append(problem.robot.env_collision_distances(to_torch(x), cuboid, Tcuboid))
        dists = torch.cat(dists, dim=1)
        min_dists, _ = torch.min(dists, dim=1)
        ax.plot(xs, 100 * min_dists.cpu().numpy(), color="tab:orange", alpha=0.5)
        ax.set_ylim(0, None)

    fig.tight_layout(pad=0.1)
    plt.show()


def visualize_plan(plan: Plan, problem: Problem, time_p_loop: float = 0.1, start_delay: float = 3.0):
    plan = PlanNp(plan)
    q_path = plan.q_path.copy()
    pose_path = plan.pose_path.copy()
    robot = problem.robot
    background_color = (1, 1, 1, 0.7)

    size = 3
    for x0 in range(-size, size + 1):
        for y0 in range(-size, size + 1):
            vis.add(
                f"floor_{x0}_{y0}",
                trajectory.Trajectory([1, 0], [(-size, y0, 0), (size, y0, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )
            vis.add(
                f"floor_{x0}_{y0}2",
                trajectory.Trajectory([1, 0], [(x0, -size, 0), (x0, size, 0)]),
                color=(0.75, 0.75, 0.75, 1.0),
                width=2.0,
                hide_label=True,
                pointSize=0,
            )

    vis.add("world", robot.klampt_world_model)
    robot_klampt_name = vis.getItemName(robot.klampt_world_model.robot(0))[1]
    # vis.init("GLUT")  # Change to this if PyQt throws "g_main_context_push_thread_default: assertion 'acquired_context' failed"
    vis.setBackgroundColor(background_color[0], background_color[1], background_color[2], background_color[3])

    for i, obs in enumerate(problem.obstacles):
        R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        assert abs(obs["roll"]) < 1e-8 and abs(obs["pitch"]) < 1e-8 and abs(obs["yaw"]) < 1e-8
        box = create.box(
            width=obs["size_x"],
            height=obs["size_z"],
            depth=obs["size_y"],
            center=(0, 0, 0),
            t=(obs["x"], obs["y"], obs["z"]),
            R=R,
        )

        vis.add(f"box_{i}", box, color=(0.0, 1.0, 0.0, 0.75), hide_label=True)

    vis.add("coordinates", coordinates.manager())
    vis.add("x_axis", trajectory.Trajectory([1, 0], [[1, 0, 0], [0, 0, 0]]))
    vis.add("y_axis", trajectory.Trajectory([1, 0], [[0, 1, 0], [0, 0, 0]]))
    vis.add(
        "path",
        trajectory.Trajectory([1, 0], [waypoint[0:3] for waypoint in to_numpy(problem.target_path)]),
        color=(1.0, 0.0, 0.0, 1.0),
        width=3.0,
        hide_label=True,
        pointSize=1,
    )
    vis.addPlot("mjac-plot")
    vis.addPlot("self-col-plot")
    vis.addPlot("env-col-plot")
    vis.addPlot("positional-error-plot")
    vis.setPlotDuration("mjac-plot", 7)
    vis.setPlotDuration("self-col-plot", 7)
    vis.setPlotDuration("env-col-plot", 7)
    vis.setPlotDuration("positional-error-plot", 7)
    # vis.setPlotRange("mjac-plot", 0, PI)

    vis.resizeWindow(1600, 1200)
    vis.setWindowTitle(f"{robot.name} plan visualization")
    vis.show()

    def update_end_effector_tf(j):
        pose = pose_path[j]
        R = so3.from_quaternion(pose[3:7])
        t = pose[0:3]
        T = (R, t)
        vis.add("end_effector", T, length=0.2, width=0.01, fancy=False)
        # vis.add("end_effector", T, length=0.15, width=2, fancy=False) # width is only used for 'fancy' mode (I
        # believe)

    # Sleep delay, so can setup a camera/ move the window/ etc
    robot.set_klampt_robot_config(q_path[0])
    update_end_effector_tf(0)
    delay(start_delay)

    i = 0
    colorized_links = set()
    while vis.shown():
        # Modify the world here. Do not modify the internal state of any visualization items outside of the lock
        vis.lock()
        if i < len(q_path):
            q = q_path[i]
            robot.set_klampt_robot_config(q)
            update_end_effector_tf(i)
            i += 1

        vis.unlock()
        # Outside of the lock you can use any vis.X functions, including vis.setItemConfig() to modify the state of
        # objects

        # USE_KLAMPT_COLLISIONS = False
        USE_KLAMPT_COLLISIONS = True
        is_collision = False
        if i < len(q_path):
            if USE_KLAMPT_COLLISIONS:
                colliding_links = env_colliding_links_klampt(problem, q)
            else:
                colliding_links = env_colliding_links_capsule(problem, to_torch(q))

            # color colliding links
            for link in colliding_links:
                colorized_links.add(link)
                vis.setColor(("world", robot_klampt_name, link), 1, 0, 0)
            for link in colorized_links:
                if link not in colliding_links:
                    vis.setColor(("world", robot_klampt_name, link), 1.0, 1.0, 1.0)

        if is_collision:
            time_p_loop = 0.01
            sleep(2 * time_p_loop)
        else:
            sleep(0.5 * time_p_loop)

        # update mjac plot
        if i < len(q_path) - 1:
            vis.logPlot("self-col-plot", "self-collisions", float(plan.self_colliding_per_ts[i]))
            vis.logPlot("env-col-plot", "env-collisions", float(plan.env_colliding_per_ts[i]))
            vis.logPlot("mjac-plot", "mjac_deg", plan.mjac_per_timestep_deg[i])
            vis.logPlot("positional-error-plot", "positional-error, mm", plan.positional_errors_mm[i])
            if robot.ndof > 7:
                vis.logPlot("mjac-plot", "mjac_cm", plan.mjac_per_timestep_cm[i])

    vis.kill()
