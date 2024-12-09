from typing import List

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Pose
from jrl.robot import Robot
import torch
import rclpy

from cppflow.data_types import Plan


def waypoints_to_se3_sequence(waypoints: List[Pose]) -> torch.Tensor:
    """Convert a list of waypoints into a tensor representing the SE(3) path.

    Args:
        waypoints (List[Pose]): List of N waypoints in the form of Pose messages.

    Returns:
        torch.Tensor: Tensor representing the target path with dimensions (N, 7) and format x, y, z, qw, qx, qy, qz.
    """
    target_path = torch.zeros((len(waypoints), 7))

    for i, waypoint in enumerate(waypoints):
        target_path[i, :] = torch.tensor(
            [
                waypoint.position.x,
                waypoint.position.y,
                waypoint.position.z,
                waypoint.orientation.w,
                waypoint.orientation.x,
                waypoint.orientation.y,
                waypoint.orientation.z,
            ]
        )
    return target_path


def plan_to_ros_trajectory(plan: Plan, robot: Robot) -> JointTrajectory:
    """Convert a CppFlow plan into a ROS JointTrajectory message."""
    trajectory = JointTrajectory()
    trajectory.header.stamp = rclpy.time.Time().to_msg()  # Converts current time to a `Time` message
    trajectory.joint_names = robot.actuated_joint_names

    zero_velocity = [0.0] * robot.ndof
    for i in range(plan.q_path.shape[0]):
        point = JointTrajectoryPoint()
        point.positions = plan.q_path[i].cpu().tolist()
        point.velocities = zero_velocity
        point.time_from_start.sec = i
        point.time_from_start.nanosec = 12
        trajectory.points.append(point)
    return trajectory
