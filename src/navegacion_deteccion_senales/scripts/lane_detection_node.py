#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

import cv2
import json

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image

import lane_detection as ld
import numpy as np


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
        self.declare_parameter('model_path', '/home/ros/workspace/src/navegacion_deteccion_senales/models/model_VGG.keras')

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
        self.model = ld.load_model(self.model_path)

        # --- Publicadores ---
        self.error_pub = self.create_publisher(Float32, self.lane_error_topic, reliable_qos)
        self.state_pub = self.create_publisher(String, self.lane_state_topic, reliable_qos)

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        self._last_status_time = 0.0
        self._last_image_time = self.get_clock().now().nanoseconds / 1e9
        self.create_timer(2.0, self._watchdog_timer)
        self.get_logger().info('lane_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _watchdog_timer(self):
        now = self.get_clock().now().nanoseconds / 1e9
        idle = now - self._last_image_time
        if idle > 3.0:
            self.get_logger().warn(
                f'No images received for {idle:.1f}s — '
                f'check carla_ros_bridge and CARLA server'
            )

    def _on_image(self, msg: Image):
        self._last_image_time = self.get_clock().now().nanoseconds / 1e9
        self.get_logger().debug('Received image')
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]
            self.get_logger().debug(f'Frame: {w}x{h}')

            lane_state = self._detect_lane(frame)
            lateral_error = lane_state.get('offset_px', 0.0)

            error_msg = Float32()
            error_msg.data = float(lateral_error)
            self.error_pub.publish(error_msg)
            self.get_logger().debug(f'Published lane_error: {lateral_error}')

            state_msg = String()
            state_msg.data = json.dumps(lane_state)
            self.state_pub.publish(state_msg)
            self.get_logger().debug(f'Published lane_state: {state_msg.data[:100]}...')

            now = self.get_clock().now().nanoseconds / 1e9
            if now - self._last_status_time >= 2.0:
                self._last_status_time = now
                self.get_logger().info(
                    f'Lane error: {lateral_error:+.1f}px  '
                    f"detected L:{int(lane_state.get('left_detected', False))} "
                    f"R:{int(lane_state.get('right_detected', False))}"
                )
        except Exception as e:
            self.get_logger().error(f'Error in _on_image: {e}')

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _detect_lane(self, frame):
        """
        Detecta líneas de carril y devuelve diccionario con estado completo.
        """
        h, w = frame.shape[:2]

        # Preprocesamiento y predicción
        preprocessed = ld.preprocess_frame(frame)
        pred_mask = ld.predict_lane(preprocessed, self.model)
        mask = ld.mask_to_array(pred_mask)

        # El modelo devuelve máscara a 224×224; reescalar a dims originales
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Diagnóstico: estadísticas de la máscara cruda
        pos = int(np.sum(mask > 0.5))
        self.get_logger().debug(
            f'Mask stats — min: {mask.min():.3f}, max: {mask.max():.3f}, '
            f'mean: {mask.mean():.3f}, pos_px: {pos}'
        )

        # Cálculo manual del error lateral en píxeles (centro del carril vs centro de imagen)
        binary = (mask > 0.5).astype(np.uint8)
        roi_start = int(h * 0.5)
        roi = binary[roi_start:, :]
        active = np.column_stack(np.where(roi > 0))
        if len(active) > 0:
            lane_center_x = np.mean(active[:, 1])
            lateral_error_px = float(lane_center_x - w / 2.0)
        else:
            lateral_error_px = 0.0

        lane_cx = float(lane_center_x) if len(active) > 0 else w / 2.0
        left_line, right_line, left_detected, right_detected = \
            self._extract_lane_lines(mask, w, h, lane_center_x=lane_cx)

        zone = 'CENTER'
        if lateral_error_px < -30.0:
            zone = 'LEFT'
        elif lateral_error_px > 30.0:
            zone = 'RIGHT'

        self.get_logger().debug(
            f'Lane center x: {lane_cx:.1f}, error_px: {lateral_error_px:.1f}'
        )

        return {
            'left': left_line,
            'right': right_line,
            'left_detected': left_detected,
            'right_detected': right_detected,
            'image_width': w,
            'image_height': h,
            'zone': zone,
            'lateral': float(np.clip(lateral_error_px / (w / 2.0), -1.0, 1.0)),
            'offset_px': lateral_error_px
        }

    def _extract_lane_lines(self, mask, img_w, img_h, lane_center_x=None, threshold=0.5):
        """
        Extrae coordenadas de líneas izquierda y derecha desde la máscara.
        Usa el centro del carril para dividir y toma los píxeles de borde
        extremos; ajusta una recta por mínimos cuadrados.
        Retorna: ([x_bot, y_bot, x_top, y_top], [...], left_ok, right_ok)
        """
        binary = (mask > threshold).astype(np.uint8)
        roi_start = int(img_h * 0.5)
        roi = binary[roi_start:, :]

        h_roi, w_roi = roi.shape
        if h_roi == 0 or w_roi == 0:
            return None, None, False, False

        # Centro del carril: el proporcionado o el centro de la imagen
        if lane_center_x is None:
            all_active = np.column_stack(np.where(roi > 0))
            lane_cx = int(np.mean(all_active[:, 1])) if len(all_active) > 0 else w_roi // 2
        else:
            lane_cx = int(lane_center_x)

        # Muestreo fino de filas (de abajo hacia arriba)
        y_positions = np.linspace(h_roi - 1, max(1, h_roi * 0.2), 20, dtype=int)

        left_pts = []   # [(x, y_abs), ...]
        right_pts = []

        for y in y_positions:
            row = roi[y, :]
            active = np.where(row > 0)[0]
            if len(active) == 0:
                continue

            left_active = active[active < lane_cx]
            right_active = active[active >= lane_cx]

            if len(left_active) > 0:
                # Borde derecho de la parte izquierda → límite izquierdo del carril
                left_pts.append([float(left_active[-1]), float(roi_start + y)])
            if len(right_active) > 0:
                # Borde izquierdo de la parte derecha → límite derecho del carril
                right_pts.append([float(right_active[0]), float(roi_start + y)])

        def _fit_line(pts):
            if len(pts) < 3:
                return None
            pts = np.array(pts)
            # x = m * y + b  (líneas aprox. verticales en la imagen)
            A = np.vstack([pts[:, 1], np.ones(len(pts))]).T
            m, b = np.linalg.lstsq(A, pts[:, 0], rcond=None)[0]

            y_bot = float(img_h)
            y_top = float(roi_start + h_roi * 0.2)
            x_bot = float(np.clip(m * y_bot + b, 0.0, float(img_w)))
            x_top = float(np.clip(m * y_top + b, 0.0, float(img_w)))
            return [x_bot, y_bot, x_top, y_top]

        left_line = _fit_line(left_pts)
        right_line = _fit_line(right_pts)

        return left_line, right_line, left_line is not None, right_line is not None


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