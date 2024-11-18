from typing import Optional
from time import time
import traceback

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from cppflow_msgs.msg import CppFlowProblem
from cppflow_msgs.srv import CppFlowQuery, CppFlowEnvironmentConfig
from jrl.robot import Robot
from jrl.robots import get_robot
from jrl.utils import to_torch

from cppflow.problem import Problem
from cppflow.ros2.ros2_utils import waypoints_to_se3_sequence, plan_to_ros_trajectory
from cppflow.planners import PlannerSearcher, CppFlowPlanner, Planner, PlannerSettings
from cppflow.utils import set_seed
from cppflow.collision_detection import qpaths_batched_self_collisions, qpaths_batched_env_collisions

set_seed()

SAVE_MESSAGES = True

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
    "CppFlowPlanner": PlannerSettings(
        k=175,
        tmax_sec=5.0,
        anytime_mode_enabled=False,
        do_rerun_if_large_dp_search_mjac=True,
        do_rerun_if_optimization_fails=True,
        do_return_search_path_mjac=True,
    ),
    "PlannerSearcher": {"k": 175, "verbosity": 0},
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

        if SAVE_MESSAGES:
            save_filepath = "/tmp/CppFlowEnvironmentConfig_request.bin"
            with open(save_filepath, "wb") as file:
                file.write(serialize_message(request))
            self.get_logger().info(f"Saved CppFlowEnvironmentConfig request to '{save_filepath}'")

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
            response.error = (
                f"The provided dnd-effector frame '{request.end_effector_frame}' does not match the robot's"
                f" end-effector link '{self.robot.end_effector_link_name}"
            )
            self.get_logger().info(f"Returning response: {response}")
            return response

        # base link doesn't match
        robot_base_link_name = self.robot._end_effector_kinematic_chain[0].parent
        if robot_base_link_name != request.base_frame:
            response.success = False
            response.error = (
                f"The provided base frame '{request.base_frame}' does not match the robot's base link"
                f" '{robot_base_link_name}"
            )
            self.get_logger().info(f"Returning response: {response}")
            return response

        self.obstacles = request.obstacles
        self.planner = PLANNERS[PLANNER](self.robot)
        response.success = True
        self.get_logger().info(f"Returning response: {response}")
        return response

    def query_callback(self, request, response):
        self.get_logger().info(f"Received a CppFlowQuery message")

        def specify_malformed_query(msg: str):
            response.is_malformed_query = True
            response.malformed_query_error = msg
            self.get_logger().info(f"Returning response: {response}")
            return response

        if SAVE_MESSAGES:
            save_filepath = "/tmp/CppFlowQuery_request.bin"
            with open(save_filepath, "wb") as file:
                file.write(serialize_message(request))
            self.get_logger().info(f"Saved a CppFlowQuery request to '{save_filepath}'")

        if len(request.problems) == 0:
            return specify_malformed_query("No problems provided")

        if len(request.problems) > 1:
            return specify_malformed_query("Only 1 planning problem allowed per query currently")

        if self.planner is None:
            return specify_malformed_query(
                "Planner has not been configured. Send a 'CppFlowEnvironmentConfig' message on the"
                " '/cppflow_environment_configuration' topic to configure the scene first."
            )

        request_problem: CppFlowProblem = request.problems[0]

        if len(request_problem.waypoints) < 3:
            return specify_malformed_query(
                f"At least 3 waypoints are required per problem (only {len(request_problem.waypoints)} provided)"
            )

        settings = PLANNER_HPARAMS[PLANNER]
        if settings.anytime_mode_enabled:
            return specify_malformed_query("Anytime mode not supported by the ros2 interface")
        settings.tmax_sec = request.max_planning_time_sec
        settings.verbosity = request.verbosity
        self.planner.set_settings(settings)

        # TODO: Add obstacles
        q0 = to_torch(request.initial_configuration.position) if request.initial_configuration_is_set else None

        # Check if initial configuration is valid
        if qpaths_batched_env_collisions(problem, q0.view(1, 1, self.planner.robot.ndof)).item():
            return specify_malformed_query("Initial configuration is in collision with environment")
        if qpaths_batched_self_collisions(problem, q0.view(1, 1, self.planner.robot.ndof)).item():
            return specify_malformed_query("Initial configuration is self-colliding")

        problem = Problem(
            target_path=waypoints_to_se3_sequence(request_problem.waypoints),
            initial_configuration=q0,
            robot=self.robot,
            name="ros2-queried-problem",
            full_name="ros2-queried-problem",
            obstacles=[],
            obstacles_Tcuboids=[],
            obstacles_cuboids=[],
            obstacles_klampt=[],
        )
        try:
            plan = self.planner.generate_plan(problem).plan
        except (RuntimeError, AttributeError) as e:
            tb = traceback.extract_tb(e.__traceback__)[-1]
            filename = tb.filename
            line_number = tb.lineno
            error_msg = f"{e} (File: {filename}, Line: {line_number})"
            response.trajectories = []
            response.success = [False]
            response.errors = [error_msg]
            self.get_logger().info(f"Planning failed with exception: '{error_msg}'")
            return response

        self.get_logger().info(f"plan: {plan}")

        # Write output to 'response'
        response.trajectories = [plan_to_ros_trajectory(plan, self.robot)]
        response.success = [plan.is_valid]
        response.errors = [""]
        self.get_logger().info("Planning complete - returning {} trajectories".format(len(response.trajectories)))
        return response



""" Usage

ros2 run cppflow ros2_subscriber
"""

if __name__ == "__main__":
    rclpy.init()
    node = SubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
