from typing import Optional
from time import time

import rclpy
from rclpy.node import Node
from cppflow_msgs.msg import CppFlowProblem
from cppflow_msgs.srv import CppFlowQuery, CppFlowEnvironmentConfig
from jrl.robot import Robot
from jrl.robots import get_robot

from cppflow.problem import Problem
from cppflow.ros2.ros2_utils import waypoints_to_se3_sequence, plan_to_ros_trajectory
from cppflow.planners import PlannerSearcher, CppFlowPlanner, Planner
from cppflow.utils import set_seed

set_seed()


# visualization_msgs/MarkerArray obstacles
# string base_frame
# string end_effector_frame
# string jrl_robot_name
# ---
# # response
# bool succes
# string error


PLANNERS = {
    "CppFlowPlanner": CppFlowPlanner,
    "PlannerSearcher": PlannerSearcher,
}


PLANNER_HPARAMS = {
    "CppFlowPlanner": {
        "k": 175,
        "verbosity": 2,
        "do_rerun_if_large_dp_search_mjac": True,
        "do_rerun_if_optimization_fails": True,
        "do_return_search_path_mjac": True,
    },
    "PlannerSearcher": {"k": 175, "verbosity": 2},
}
PLANNER = "CppFlowPlanner"


class SubscriberNode(Node):
    def __init__(self):
        super().__init__("cppflow_query_server")
        self.srv = self.create_service(CppFlowQuery, "/cppflow_planning_query", self.query_callback)
        self.environment_setup_srv = self.create_service(
            CppFlowEnvironmentConfig, "/cppflow_environment_configuration", self.environment_setup_callback
        )

        self.get_logger().info("CppFlowQuery service server started...")
        self.robot: Optional[Robot] = None
        self.planner: Optional[Planner] = None
        self.obstacles = []

    def environment_setup_callback(self, request, response):
        self.get_logger().info(f"Received a CppFlowEnvironmentConfig message: {request}")

        # Get robot
        if (self.robot is None) or (self.robot.name != request.jrl_robot_name):
            try:
                t0 = time()
                self.robot = get_robot(request.jrl_robot_name)
                self.get_logger().info(f"Loaded robot '{self.robot.name}' in {time() - t0:.3f} seconds")
            except ValueError:
                response.success = False
                response.error = f"Robot '{request.jrl_robot_name}' doesn't exist in the Jrl library"
                self.get_logger().info(f"Returning response: {response}")
                return response

        # end effector frame doesn't match
        if self.robot.end_effector_link_name != request.end_effector_frame:
            response.success = False
            response.error = f"The provided dnd-effector frame '{request.end_effector_frame}' does not match the robot's end-effector link '{self.robot.end_effector_link_name}"
            self.get_logger().info(f"Returning response: {response}")
            return response

        # base link doesn't match
        robot_base_link_name = self.robot._end_effector_kinematic_chain[0].parent
        if robot_base_link_name != request.base_frame:
            response.success = False
            response.error = f"The provided base frame '{request.base_frame}' does not match the robot's base link '{robot_base_link_name}"
            self.get_logger().info(f"Returning response: {response}")
            return response

        self.obstacles = request.obstacles
        self.planner = PLANNERS[PLANNER](self.robot)
        response.success = True
        self.get_logger().info(f"Returning response: {response}")
        return response

    def query_callback(self, request, response):
        self.get_logger().info(f"Received a CppFlowQuery message: {request}")

        if len(request.problems) == 0:
            response.is_malformed_query = True
            response.malformed_query_error = "No problems provided"
            self.get_logger().info(f"Returning response: {response}")
            return response

        if len(request.problems) > 1:
            response.is_malformed_query = True
            response.malformed_query_error = "Only 1 planning problem allowed per query currently"
            self.get_logger().info(f"Returning response: {response}")
            return response

        if self.planner is None:
            response.is_malformed_query = True
            response.malformed_query_error = "Planner has not been configured. Send a 'CppFlowEnvironmentConfig' message on the '/cppflow_environment_configuration' topic to configure the scene first."
            self.get_logger().info(f"Returning response: {response}")
            return response

        request_problem: CppFlowProblem = request.problems[0]

        if len(request_problem.waypoints) < 3:
            response.is_malformed_query = True
            response.malformed_query_error = (
                f"At least 3 waypoints are required per problem (only {len(request_problem.waypoints)} provided)"
            )
            self.get_logger().info(f"Returning response: {response}")
            return response

        # TODO: Add obstacles
        problem = Problem(
            target_path=waypoints_to_se3_sequence(request_problem.waypoints),
            robot=self.robot,
            name="QUERIED-PROBLEM",
            full_name="QUERIED-PROBLEM-full_name",
            obstacles=[],
            obstacles_Tcuboids=[],
            obstacles_cuboids=[],
            obstacles_klampt=[],
        )
        planner_result = self.planner.generate_plan(problem, **PLANNER_HPARAMS[PLANNER])

        # Write output to 'response'
        response.trajectories = [plan_to_ros_trajectory(planner_result.plan, self.robot)]
        response.success = [False] * len(request.problems)
        response.errors = ["unimplemented"] * len(request.problems)
        self.get_logger().info(f"Returning response: {response}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
