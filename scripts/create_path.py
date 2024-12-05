from typing import List, Tuple
import argparse

import pandas as pd
import numpy as np
from pyquaternion import Quaternion

np.set_printoptions(suppress=True)

paths = {
    # there are 25 interpolated poses between each waypoint in the original path
    "rot_yz2": [
        ([0.0, 0.0, 0.0], [1.000000, 0.0, 0.0, 0.0], 30),
        ([0.0, 0.0, 0.0], [0.923880, 0.0, 0.382683, 0.0], 30),  # rot_y(45 deg)
        ([0.0, 0.0, 0.0], [1.000000, 0.0, 0.0, 0.0], 30),
        ([0.0, 0.0, 0.0], [0.923880, 0.0, -0.382683, 0.0], 30),
        ([0.0, 0.0, 0.0], [1.000000, 0.0, 0.0, 0.0], 30),
        ([0.0, 0.0, 0.0], [0.923880, 0.0, 0.0, 0.382683], 30),
        ([0.0, 0.0, 0.0], [1.000000, 0.0, 0.0, 0.0], 30),
        ([0.0, 0.0, 0.0], [0.923880, 0.0, 0.0, -0.382683], 30),
        ([0.0, 0.0, 0.0], [1.000000, 0.0, 0.0, 0.0], 30),
    ]
}


def circle_path(output_filepath: str):
    # Circle centered at (0, -0.25, 0) with radius 0.25 in the yz plane
    n_waypoints = 500
    xs = np.zeros(n_waypoints)
    thetas = np.linspace(0, 2 * np.pi, n_waypoints)
    ys = -0.25 + 0.25 * np.cos(thetas)
    zs = 0.25 * np.sin(thetas)

    time_out = np.arange(n_waypoints).reshape((n_waypoints, 1))
    rotations_out = np.zeros((n_waypoints, 4))
    rotations_out[:, 0] = 1.0
    poses_out = np.hstack(
        (
            time_out,
            xs.reshape((n_waypoints, 1)),
            ys.reshape((n_waypoints, 1)),
            zs.reshape((n_waypoints, 1)),
            rotations_out,
        )
    )

    # save result to a csv
    df = pd.DataFrame(poses_out, columns=["time", "x", "y", "z", "qw", "qx", "qy", "qz"])
    df.to_csv(output_filepath, index=False)


def path_from_spaced_waypoints(waypoints: List[Tuple[List, List]], output_filepath: str):
    positions_in = np.array([wp[0] for wp in waypoints])
    assert np.absolute(positions_in).max() < 1e-8, "positions must be 0 - non zero handling unimplemented"
    rotations_in = np.array([wp[1] for wp in waypoints])

    rotations_out = []

    for i in range(len(rotations_in) - 1):
        q0 = Quaternion(waypoints[i][1])
        qf = Quaternion(waypoints[i + 1][1])
        rotations_out.append([q0.w, q0.x, q0.y, q0.z])
        for q in Quaternion.intermediates(q0, qf, waypoints[i][2], include_endpoints=False):
            rotations_out.append([q.w, q.x, q.y, q.z])
        # timestamps_in = np.array([0, 1])
        # timestamps_out = np.linspace(0, 1, waypoint_in_0[2])
        # xs = np.interp(timestamps_out, timestamps_in, [waypoint_in_0[0][0], waypoint_in_f[0][0]])
        # ys = np.interp(timestamps_out, timestamps_in, [waypoint_in_0[0][0], waypoint_in_f[0][0]])
        # zs = np.interp(timestamps_out, timestamps_in, [waypoint_in_0[0][0], waypoint_in_f[0][0]])

    q_last = Quaternion(waypoints[-1][1])
    rotations_out.append([q_last.w, q_last.x, q_last.y, q_last.z])

    # format result
    n_poses_out = len(rotations_out)
    time_out = np.arange(n_poses_out).reshape((n_poses_out, 1))
    positions_out = np.zeros((n_poses_out, 3))
    rotations_out = np.array(rotations_out)
    poses_out = np.hstack((time_out, positions_out, rotations_out))

    # save result to a csv
    df = pd.DataFrame(poses_out, columns=["time", "x", "y", "z", "qw", "qx", "qy", "qz"])
    df.to_csv(output_filepath, index=False)


""" Example usage

python scripts/create_path.py rot_yz2 rot_yz2.csv
python scripts/create_path.py circle2 circle2.csv
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a path")
    parser.add_argument("path_name", type=str)
    parser.add_argument("output_filepath", type=str)
    args = parser.parse_args()
    if args.path_name == "rot_yz2":
        path_from_spaced_waypoints(paths[args.path_name], args.output_filepath)
    elif args.path_name == "circle2":
        circle_path(args.output_filepath)
