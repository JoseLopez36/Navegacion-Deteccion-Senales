#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

import json
import time

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image

import lane_detection as ld


class LaneDetectionNode(Node):
    """
    Nodo de detección de carriles mediante modelo de segmentación (VGG/U-Net).

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image (sensor_msgs/Image)

    Publicaciones:
      - /lane_detection/lane_error      (std_msgs/Float32)  — error lateral en píxeles
      - /lane_detection/lane_state      (std_msgs/String)   — JSON con estado del carril
    """

    def __init__(self):
        super().__init__('lane_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('lane_state_topic', '/lane_detection/lane_state')
        self.declare_parameter('model_path', '/home/jose-lopez/Documents/WORKSPACE/Cuatrimestre_2/Percepcion_Automatica_Robotica/Navegacion-Deteccion-Senales/src/navegacion_deteccion_senales/models/lane_model.onnx')

        self.image_topic       = self.get_parameter('image_topic').value
        self.lane_error_topic  = self.get_parameter('lane_error_topic').value
        self.lane_state_topic  = self.get_parameter('lane_state_topic').value
        self.model_path        = self.get_parameter('model_path').value

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
        self.get_logger().info(f'Loading model from: {self.model_path}')
        self.model = ld.load_model(self.model_path)
        self.get_logger().info(f'Model loaded successfully: {type(self.model).__name__}')

        # --- Publicadores ---
        self.error_pub = self.create_publisher(Float32, self.lane_error_topic, reliable_qos)
        self.state_pub = self.create_publisher(String, self.lane_state_topic, reliable_qos)

        # --- Suscriptores ---
        self._latest_frame = None
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        # Procesa el último frame disponible a 1 Hz (1000 ms)
        self.create_timer(1.0, self._process_latest_frame)
        self.get_logger().info('lane_detection_node iniciado (1 Hz, CPU).')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        """Almacena el frame más reciente; el procesamiento ocurre a 1 Hz."""
        try:
            self._latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')

    def _process_latest_frame(self):
        """Callback del timer a 1 Hz: detecta carriles en el último frame disponible."""
        if self._latest_frame is None:
            return
        frame = self._latest_frame
        h, w = frame.shape[:2]
        self.get_logger().debug(f'Processing frame: {w}x{h}')
        
        try:
            t0 = time.time()
            state = ld.detect_lane_state(frame, roi_start=0.55)
            inference_ms = (time.time() - t0) * 1000.0

            # 4. Publicar error lateral
            error_msg = Float32()
            error_msg.data = state['offset_px']
            self.error_pub.publish(error_msg)

            # 5. Publicar estado del carril en JSON
            state_msg = String()
            state_msg.data = json.dumps(state)
            self.state_pub.publish(state_msg)

            self.get_logger().info(
                f"Inference: {inference_ms:.1f} ms | "
                f"Error: {state['offset_px']:+.1f} px | "
                f"Steering: {state['steering']:+.3f} | "
                f"Zone: {state['zone']} | "
                f"Left: {state['left_detected']} Right: {state['right_detected']}"
            )
        except Exception as e:
            self.get_logger().error(f'Error during lane detection: {e}')


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