# listener.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Adjust to the message type you want to listen to
from cppflow.plan import Plan
from cppflow_msgs.srv import CppFlowQuery



# string base_frame
# string end_effector_frame
# CppFlowProblem[] problems
# ---
# trajectory_msgs/JointTrajectory trajectories
# bool[] success
# string[] errors


class SubscriberNode(Node):
    def __init__(self):
        super().__init__('cppflow_query_server')
        self.srv = self.create_service(CppFlowQuery, '/cppflow_planning_query', self.query_callback)
        self.get_logger().info("CppFlowQuery service server started...")
    
    def query_callback(self, request, response):
        # Log the incoming request data
        self.get_logger().info(f"Received request: base_frame={request.base_frame}, "
                               f"end_effector_frame={request.end_effector_frame}, "
                               f"number of problems={len(request.problems)}")

        # TODO: Run CppFlow and convert the result to a JointTrajectory
        response.trajectories = []
        response.success = [False] * len(request.problems)
        response.errors = ['unimplemented'] * len(request.problems)
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

if __name__ == '__main__':
    main()