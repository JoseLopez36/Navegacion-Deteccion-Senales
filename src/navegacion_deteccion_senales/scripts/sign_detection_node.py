#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import message_filters

from cv_bridge import CvBridge

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image, CameraInfo

from sign_classification import CNN, process_image


SEMANTIC_TRAFFIC_SIGN = np.array([0, 220, 220])  # BGR (CARLA semantic palette)
SEMANTIC_TOLERANCE = 10


class SignDetectionNode(Node):
    """
    Nodo de detección y reconocimiento de señales de tráfico mediante CNN (PyTorch).

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image  (sensor_msgs/Image)

    Publicaciones:
      - /sign_detection/sign_label          (std_msgs/String)  — etiqueta de la señal detectada
      - /sign_detection/speed_limit         (std_msgs/Float32) — límite de velocidad extraído (m/s), -1 si no aplica
    """

    _CLASS_NAMES = {0: 'speed_limit_30', 1: 'speed_limit_60', 2: 'speed_limit_90', 3: 'stop'}
    _CLASS_SPEED = {0: 30.0 / 3.6, 1: 60.0 / 3.6, 2: 90.0 / 3.6, 3: 0.0}

    def __init__(self):
        super().__init__('sign_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('rgb_camera_info_topic', '/carla/ego_vehicle/rgb_front/camera_info')
        self.declare_parameter('semantic_topic', '/carla/ego_vehicle/semantic_segmentation_front/image')
        self.declare_parameter('semantic_camera_info_topic', '/carla/ego_vehicle/semantic_segmentation_front/camera_info')
        self.declare_parameter('sign_label_topic', '/sign_detection/sign_label')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')
        self.declare_parameter('sign_bbox_topic', '/sign_detection/bbox')
        self.declare_parameter('sign_mask_topic', '/sign_detection/mask')
        self.declare_parameter('model_path', '')
        self.declare_parameter('min_sign_area', 200)
        self.declare_parameter('max_sign_area', 50000)

        self.image_topic                = self.get_parameter('image_topic').value
        self.rgb_camera_info_topic      = self.get_parameter('rgb_camera_info_topic').value
        self.semantic_topic             = self.get_parameter('semantic_topic').value
        self.semantic_camera_info_topic = self.get_parameter('semantic_camera_info_topic').value
        self.sign_label_topic           = self.get_parameter('sign_label_topic').value
        self.speed_limit_topic          = self.get_parameter('speed_limit_topic').value
        self.sign_bbox_topic            = self.get_parameter('sign_bbox_topic').value
        self.sign_mask_topic            = self.get_parameter('sign_mask_topic').value
        self.model_path                 = self.get_parameter('model_path').value
        self.min_sign_area              = int(self.get_parameter('min_sign_area').value)
        self.max_sign_area              = int(self.get_parameter('max_sign_area').value)

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
        self.model  = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Info de cámaras ---
        self.rgb_camera_info      = None
        self.semantic_camera_info = None

        self._load_model()

        # --- Publicadores ---
        self.label_pub = self.create_publisher(String,  self.sign_label_topic,  reliable_qos)
        self.speed_pub = self.create_publisher(Float32, self.speed_limit_topic, reliable_qos)
        self.bbox_pub  = self.create_publisher(String,  self.sign_bbox_topic,   reliable_qos)
        self.mask_pub  = self.create_publisher(Image,   self.sign_mask_topic,   reliable_qos)

        # --- Suscriptores (sincronizados RGB + semántico) ---
        self.rgb_sub      = message_filters.Subscriber(self, Image, self.image_topic,    qos_profile=sensor_qos)
        self.semantic_sub = message_filters.Subscriber(self, Image, self.semantic_topic, qos_profile=sensor_qos)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.semantic_sub], queue_size=10, slop=0.03
        )
        self.sync.registerCallback(self._on_images_sync)

        # --- Suscriptores de camera_info ---
        self.create_subscription(CameraInfo, self.rgb_camera_info_topic,      self._on_rgb_camera_info,      sensor_qos)
        self.create_subscription(CameraInfo, self.semantic_camera_info_topic, self._on_semantic_camera_info, sensor_qos)

        self.get_logger().info('sign_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Inicialización del modelo
    # ------------------------------------------------------------------

    def _load_model(self):
        if not self.model_path:
            self.get_logger().warn('model_path no configurado; clasificación deshabilitada.')
            return
        if not os.path.exists(self.model_path):
            self.get_logger().error(f'Modelo no encontrado: {self.model_path}')
            return
        self.model = CNN()
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info(f'Clasificador CNN cargado desde {self.model_path}')

    # ------------------------------------------------------------------
    # Camera info callbacks
    # ------------------------------------------------------------------

    def _on_rgb_camera_info(self, msg: CameraInfo):
        self.rgb_camera_info = msg

    def _on_semantic_camera_info(self, msg: CameraInfo):
        self.semantic_camera_info = msg

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_images_sync(self, rgb_msg: Image, semantic_msg: Image):
        try:
            frame         = self.bridge.imgmsg_to_cv2(rgb_msg,      desired_encoding='bgr8')
            semantic_frame = self.bridge.imgmsg_to_cv2(semantic_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return

        sign_label, speed_limit_ms, bbox, mask = self._detect_signs(frame, semantic_frame)

        label_msg = String()
        label_msg.data = sign_label
        self.label_pub.publish(label_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed_limit_ms)
        self.speed_pub.publish(speed_msg)

        bbox_msg = String()
        bbox_msg.data = json.dumps(bbox)
        self.bbox_pub.publish(bbox_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = rgb_msg.header
        self.mask_pub.publish(mask_msg)

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _bbox_from_semantic(self, frame, semantic_frame):
        """
        Obtiene el bounding box de la señal más grande detectada en la imagen
        semántica y lo reescala a las coordenadas de la imagen RGB.
        Returns (bx, by, bw, bh, mask_uint8) o None si no se detecta nada.
        """
        mask = cv2.inRange(
            semantic_frame,
            SEMANTIC_TRAFFIC_SIGN - SEMANTIC_TOLERANCE,
            SEMANTIC_TRAFFIC_SIGN + SEMANTIC_TOLERANCE,
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_cnt = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if self.min_sign_area < area < self.max_sign_area and area > best_area:
                best_area = area
                best_cnt = cnt

        if best_cnt is None:
            return None

        bx, by, bw, bh = cv2.boundingRect(best_cnt)
        bx, by, bw, bh = self._map_bbox_to_rgb(bx, by, bw, bh, semantic_frame.shape)

        return bx, by, bw, bh, mask

    def _map_bbox_to_rgb(self, x, y, w, h, semantic_frame_shape):
        """Map bounding box from semantic image to RGB image using camera intrinsics."""
        if self.rgb_camera_info is None or self.semantic_camera_info is None:
            sem_h, sem_w = semantic_frame_shape[:2]
            rgb_h = self.rgb_camera_info.height if self.rgb_camera_info else sem_h
            rgb_w = self.rgb_camera_info.width  if self.rgb_camera_info else sem_w
            scale_x = rgb_w / sem_w
            scale_y = rgb_h / sem_h
            return max(0, int(x * scale_x)), max(0, int(y * scale_y)), int(w * scale_x), int(h * scale_y)

        K_sem = np.array(self.semantic_camera_info.k).reshape(3, 3)
        K_rgb = np.array(self.rgb_camera_info.k).reshape(3, 3)
        rgb_h, rgb_w = self.rgb_camera_info.height, self.rgb_camera_info.width

        corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]], dtype=np.float32)
        norm_x = (corners[:, 0] - K_sem[0, 2]) / K_sem[0, 0]
        norm_y = (corners[:, 1] - K_sem[1, 2]) / K_sem[1, 1]
        proj_x = np.clip(K_rgb[0, 0] * norm_x + K_rgb[0, 2], 0, rgb_w - 1)
        proj_y = np.clip(K_rgb[1, 1] * norm_y + K_rgb[1, 2], 0, rgb_h - 1)

        x_new, y_new = int(proj_x.min()), int(proj_y.min())
        return x_new, y_new, int(proj_x.max()) - x_new, int(proj_y.max()) - y_new

    def _detect_signs(self, frame, semantic_frame):
        """
        Detecta y clasifica señales de tráfico usando el ground truth semántico.
        Returns (label, speed_m_s, bbox_dict, mask_uint8)
        """
        img_h, img_w = frame.shape[:2]
        black_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        result = self._bbox_from_semantic(frame, semantic_frame)
        if result is None:
            return 'none', -1.0, {}, black_mask

        bx, by, bw, bh, mask = result
        crop = frame[by:by + bh, bx:bx + bw]
        bbox = {'x': bx, 'y': by, 'w': bw, 'h': bh}

        if self.model is None:
            return 'detected', -1.0, bbox, mask

        try:
            tensor = process_image(crop).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                class_id = int(logits.argmax(1).item())

            label    = self._CLASS_NAMES.get(class_id, 'unknown')
            speed_ms = self._CLASS_SPEED.get(class_id, -1.0)
            return label, speed_ms, bbox, mask
        except Exception as e:
            self.get_logger().warn(f'Error en clasificación: {e}')
            return 'detected', -1.0, bbox, mask


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()