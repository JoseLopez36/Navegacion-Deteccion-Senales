#!/usr/bin/env python3

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

import json
import time

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image, CameraInfo

from lane_detection import LaneDetector


class LaneDetectionNode(Node):
    """
    Nodo de detección de carriles mediante modelo de segmentación (VGG/U-Net).

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image (sensor_msgs/Image)
      - /carla/ego_vehicle/rgb_front/camera_info (sensor_msgs/CameraInfo)

    Publicaciones:
      - /lane_detection/lane_error      (std_msgs/Float32)  — error lateral en píxeles
      - /lane_detection/lane_state      (std_msgs/String)   — JSON con estado del carril
      - /lane_detection/mask            (sensor_msgs/Image) — máscara de segmentación
    """

    def __init__(self):
        super().__init__('lane_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('camera_info_topic', '/carla/ego_vehicle/rgb_front/camera_info')
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('lane_state_topic', '/lane_detection/lane_state')
        self.declare_parameter('lane_mask_topic', '/lane_detection/mask')
        self.declare_parameter('model_path', '/home/jose-lopez/Documents/WORKSPACE/Cuatrimestre_2/Percepcion_Automatica_Robotica/Navegacion-Deteccion-Senales/src/navegacion_deteccion_senales/models/lane_model.onnx')
        self.declare_parameter('camera_height_m', 1.5)

        self.image_topic       = self.get_parameter('image_topic').value
        self.camera_info_topic  = self.get_parameter('camera_info_topic').value
        self.lane_error_topic  = self.get_parameter('lane_error_topic').value
        self.lane_state_topic  = self.get_parameter('lane_state_topic').value
        self.lane_mask_topic   = self.get_parameter('lane_mask_topic').value
        self.model_path        = self.get_parameter('model_path').value
        self._camera_height_m  = float(self.get_parameter('camera_height_m').value)

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

        # --- Intrínsecos de la cámara ---
        self._fx = None  # focal length en píxeles; None hasta recibir camera_info
        self._fy = None
        self._cy = None

        # --- Métricas de inferencia ---
        self._inf_count   = 0
        self._inf_max_ms  = 0.0
        self._inf_sum_ms  = 0.0

        # --- Utilidades ---
        self.bridge = CvBridge()
        self.get_logger().info(f'Loading model from: {self.model_path}')
        self.detector = LaneDetector(self.model_path)
        self.get_logger().info('Model loaded successfully.')

        # --- Publicadores ---
        self.error_pub = self.create_publisher(Float32, self.lane_error_topic, reliable_qos)
        self.state_pub = self.create_publisher(String, self.lane_state_topic, reliable_qos)
        self.mask_pub  = self.create_publisher(Image, self.lane_mask_topic, reliable_qos)

        # --- Suscriptores ---
        self._latest_frame = None
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)
        self.create_subscription(CameraInfo, self.camera_info_topic, self._on_camera_info, sensor_qos)

        # Procesa el último frame disponible cada 0.1 segundos (10 Hz)
        self.create_timer(0.1, self._process_latest_frame)
        self.get_logger().info('lane_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_camera_info(self, msg: CameraInfo):
        if self._fx is None:
            self._fx = msg.k[0]  # K = [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            self._fy = msg.k[4]
            self._cy = msg.k[5]
            self.get_logger().info(
                f'camera_info recibido: fx={self._fx:.2f} fy={self._fy:.2f} cy={self._cy:.2f} px'
            )
            
    def _on_image(self, msg: Image):
        """Almacena el frame más reciente."""
        try:
            self._latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')

    def _process_latest_frame(self):
        """Callback del timer: detecta carriles en el último frame disponible."""
        if self._latest_frame is None:
            return

        frame = self._latest_frame
        try:
            t0 = time.time()
            error, state, mask = self.detector.detect_lane_state(frame)
            inference_ms = (time.time() - t0) * 1000.0

            # Publicar error lateral
            error_msg = Float32()
            # Convertir a metros usando geometría de cámara pinhole:
            #   Z = camera_height * fy / (y_ref - cy)  con y_ref = fila inferior del ROI
            #   error_m = offset_px * Z / fx
            if self._fx is not None and self._fy is not None and self._cy is not None:
                orig_h, orig_w = frame.shape[:2]
                y_ref = orig_h * 0.95  # fila de referencia (cerca del vehículo)
                denom = y_ref - self._cy
                if abs(denom) > 1.0:
                    Z = self._camera_height_m * self._fy / denom
                    error_m = error * Z / self._fx
                else:
                    error_m = error / self._fx
            else:
                error_m = error  # fallback hasta recibir camera_info
            error_msg.data = float(error_m)
            self.error_pub.publish(error_msg)

            # Publicar estado del carril en JSON
            state_msg = String()
            state_msg.data = json.dumps({k: v for k, v in state.items()})
            self.state_pub.publish(state_msg)

            # Publicar máscara de detección (convertir float32 a uint8)
            mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
            mask_msg = self.bridge.cv2_to_imgmsg(mask_uint8, encoding='mono8')
            mask_msg.header.stamp = self.get_clock().now().to_msg()
            self.mask_pub.publish(mask_msg)

            self._inf_count  += 1
            self._inf_sum_ms += inference_ms
            if inference_ms > self._inf_max_ms:
                self._inf_max_ms = inference_ms
            mean_ms = self._inf_sum_ms / self._inf_count
            self.get_logger().info(
                f"Inference: {inference_ms:.1f} ms  "
                f"max={self._inf_max_ms:.1f} ms  "
                f"mean={mean_ms:.1f} ms | "
                f"Error: {state['error']:+.1f} px"
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