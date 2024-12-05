from typing import Optional
import numpy as np
import importlib.resources as resources
import warnings

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from cppflow_msgs.srv import CppFlowQuery, CppFlowEnvironmentConfig
from cppflow_msgs.msg import CppFlowProblem
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from jrl.robot import Robot
from jrl.robots import Panda


def _get_initial_configuration(robot: Robot, target_pose: Pose):
    warnings.warn("No collision checking is performed against obstacles in the scene")
    for _ in range(25):
        target_pose_np = np.array(
            [
                target_pose.position.x,
                target_pose.position.y,
                target_pose.position.z,
                target_pose.orientation.w,
                target_pose.orientation.x,
                target_pose.orientation.y,
                target_pose.orientation.z,
            ]
        )
        initial_configuration = robot.inverse_kinematics_klampt(target_pose_np)[0]
        if not robot.config_self_collides(initial_configuration):
            print(f"Initial configuration {initial_configuration} is not self colliding free")
            return initial_configuration
    raise RuntimeError("Could not find collision free initial configuration")


class CppFlowQueryClient(Node):
    def __init__(self):
        super().__init__("cppflow_publisher")

        self.planning_client = self.create_client(CppFlowQuery, "/cppflow_planning_query")
        while not self.planning_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service /cppflow_planning_query to be available...")

        self.scene_configuration_client = self.create_client(
            CppFlowEnvironmentConfig, "/cppflow_environment_configuration"
        )
        while not self.scene_configuration_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service /cppflow_environment_configuration to be available...")

        self.panda: Optional[Panda] = None
        self.send_scene_configuration_request()
        # self.send_dummy_problem_planning_request()
        self.send_cached_problem_planning_request()

    def send_scene_configuration_request(self):

        # Example setup for the request fields
        request = CppFlowEnvironmentConfig.Request()
        request.base_frame = "panda_link0"
        request.end_effector_frame = "panda_hand"
        request.jrl_robot_name = "panda"

        # Send request
        future = self.scene_configuration_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
            self.get_logger().info(f"Received response: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def send_cached_problem_planning_request(self):
        # Load CppFlowQuery_request.bin
        with resources.open_binary("cppflow.ros2.resources", "CppFlowQuery_request.bin") as f:
            request = deserialize_message(f.read(), CppFlowQuery.Request)
        self.get_logger().info(f"Loaded cached problem request 'CppFlowQuery_request.bin'")
        self._send_planning_request(request)

    def send_dummy_problem_planning_request(self):

        if self.panda is None:
            self.panda = Panda()

        request = CppFlowQuery.Request()

        # Example setup for the request fields
        request.base_frame = "panda_link0"
        request.end_effector_frame = "panda_hand"
        request.jrl_robot_name = "panda"

        # Create an example CppFlowProblem with waypoints
        # This is the beginning of the panda__1cube problem
        xyz_offset = np.array([0, 0.5421984559194368, 0.7885155964931997])
        # x, y, z, qw, x, y, z
        target_path = np.array(
            [
                [0.45, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.44547737, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.44095477, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.43643215, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.43190953, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.4273869, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.42286432, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.4183417, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.41381907, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.40929648, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.40477386, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )

        problem = CppFlowProblem()
        problem.waypoints = []
        for waypoint in target_path:
            xyz = xyz_offset + waypoint[0:3]
            qxyz = waypoint[3:]
            waypoint = Pose(
                position=Point(x=xyz[0], y=xyz[1], z=xyz[2]),
                orientation=Quaternion(x=qxyz[1], y=qxyz[2], z=qxyz[3], w=qxyz[0]),
            )
            problem.waypoints.append(waypoint)

        request.problems = [problem]
        request.initial_configuration_is_set = True
        request.initial_configuration = JointState(
            position=_get_initial_configuration(self.panda, request.problems[0].waypoints[0])
        )

        # Send request
        self._send_planning_request(request)

    def _send_planning_request(self, request):
        future = self.planning_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
            self.get_logger().info(f"Received CppFlowQuery.Response")
            for i, (trajectory, success, error) in enumerate(
                zip(response.trajectories, response.success, response.errors)
            ):
                self.get_logger().info(f"Problem {i}: Success = {success}, Error = {error}")
                self.get_logger().info(
                    f"Trajectory {i}: {trajectory.header}, {trajectory.joint_names}, {len(trajectory.points)} points"
                )
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")


""" Usage
ros2 run cppflow ros2_subscriber    # terminal 1
ros2 run cppflow ros2_publisher     # terminal 2
"""


def main(args=None):
    rclpy.init(args=args)
    client = CppFlowQueryClient()
    client.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
