import numpy as np
import rclpy
from rclpy.node import Node
from cppflow_msgs.srv import CppFlowQuery, CppFlowEnvironmentConfig
from cppflow_msgs.msg import CppFlowProblem
from geometry_msgs.msg import Pose, Point, Quaternion


# visualization_msgs/MarkerArray obstacles
# string base_frame
# string end_effector_frame
# string jrl_robot_name
# ---
# bool succes
# string error


class CppFlowQueryClient(Node):
    def __init__(self):
        super().__init__("cppflow_query_client")

        self.planning_client = self.create_client(CppFlowQuery, "/cppflow_planning_query")
        while not self.planning_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service /cppflow_planning_query to be available...")

        self.scene_configuration_client = self.create_client(
            CppFlowEnvironmentConfig, "/cppflow_environment_configuration"
        )
        while not self.scene_configuration_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for service /cppflow_environment_configuration to be available...")

        self.send_scene_configuration_request()
        self.send_planning_request()

    def send_scene_configuration_request(self):
        request = CppFlowEnvironmentConfig.Request()

        # Example setup for the request fields
        request.base_frame = "panda_link0"
        request.end_effector_frame = "panda_hand"
        request.jrl_robot_name = "panda"
        #
        future = self.scene_configuration_client.call_async(request)

        # Wait for the response and then parse
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
            self.get_logger().info(f"Received response: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")

    def dummy_problem(self):

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
        return problem

    def send_planning_request(self):
        request = CppFlowQuery.Request()

        # Example setup for the request fields
        request.base_frame = "panda_link0"
        request.end_effector_frame = "panda_hand"
        request.jrl_robot_name = "panda"

        # Create an example CppFlowProblem with waypoints
        request.problems = [self.dummy_problem()]

        # Wait for the response and then parse
        future = self.planning_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        try:
            response = future.result()
            self.get_logger().info(f"Received response: {response}")
            for i, (trajectory, success, error) in enumerate(
                zip(response.trajectories, response.success, response.errors)
            ):
                self.get_logger().info(f"Problem {i}: Success = {success}, Error = {error}")
                self.get_logger().info(f"Trajectory {i}: {trajectory}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    client = CppFlowQueryClient()
    try:
        rclpy.spin(client)
    except KeyboardInterrupt:
        pass
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
