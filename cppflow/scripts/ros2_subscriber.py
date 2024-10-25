# listener.py
import rclpy
from rclpy.node import Node
from std_msgs.msg import String  # Adjust to the message type you want to listen to


class SubscriberNode(Node):
    def __init__(self):
        super().__init__('listener_node')  # Node name
        self.subscription = self.create_subscription(
            String,         # Message type
            'topic_name',   # Topic name (update with the correct topic)
            self.listener_callback,
            10              # QoS (Queue size)
        )
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f"Received message: '{msg.data}'")


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