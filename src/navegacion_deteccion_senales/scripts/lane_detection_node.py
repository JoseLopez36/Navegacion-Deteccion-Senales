#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

from std_msgs.msg import Float32
from sensor_msgs.msg import Image


class LaneDetectionNode(Node):
    """
    Nodo de detección de carriles mediante OpenCV y transformada de Hough.

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image (sensor_msgs/Image)

    Publicaciones:
      - /lane_detection/lane_error      (std_msgs/Float32)  — error lateral en píxeles
    """

    def __init__(self):
        super().__init__('lane_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')

        self.image_topic       = self.get_parameter('image_topic').value
        self.lane_error_topic  = self.get_parameter('lane_error_topic').value

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Utilidades ---
        self.bridge = CvBridge()

        # --- Publicadores ---
        self.error_pub = self.create_publisher(Float32, self.lane_error_topic, reliable_qos)

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        self.get_logger().info('lane_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        lateral_error = self._detect_lane(frame)

        error_msg = Float32()
        error_msg.data = float(lateral_error)
        self.error_pub.publish(error_msg)

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _detect_lane(self, frame):
        """
        Detecta líneas de carril y devuelve el error lateral en píxeles
        """
        # TODO: implementar pipeline completo de detección de carril
        lateral_error = 0.0
        return lateral_error


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()