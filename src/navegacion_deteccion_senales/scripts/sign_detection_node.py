#!/usr/bin/env python3

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image

import sign_detection as sd
from sign_classification import CNN, process_image


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
        self.declare_parameter('sign_label_topic', '/sign_detection/sign_label')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')
        self.declare_parameter('sign_bbox_topic', '/sign_detection/bbox')
        self.declare_parameter('sign_mask_topic', '/sign_detection/mask')
        self.declare_parameter('model_path', '')

        self.image_topic       = self.get_parameter('image_topic').value
        self.sign_label_topic  = self.get_parameter('sign_label_topic').value
        self.speed_limit_topic = self.get_parameter('speed_limit_topic').value
        self.sign_bbox_topic   = self.get_parameter('sign_bbox_topic').value
        self.sign_mask_topic   = self.get_parameter('sign_mask_topic').value
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
        self.model  = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._load_model()

        # --- Publicadores ---
        self.label_pub = self.create_publisher(String,  self.sign_label_topic,  reliable_qos)
        self.speed_pub = self.create_publisher(Float32, self.speed_limit_topic, reliable_qos)
        self.bbox_pub  = self.create_publisher(String,  self.sign_bbox_topic,   reliable_qos)
        self.mask_pub  = self.create_publisher(Image,   self.sign_mask_topic,   reliable_qos)

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

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
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error decoding image: {e}')
            return

        sign_label, speed_limit_ms, bbox, mask = self._detect_signs(frame)

        label_msg = String()
        label_msg.data = sign_label
        self.label_pub.publish(label_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed_limit_ms)
        self.speed_pub.publish(speed_msg)

        bbox_msg = String()
        bbox_msg.data = json.dumps(bbox)
        self.bbox_pub.publish(bbox_msg)

        if mask is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _detect_signs(self, frame):
        """
        Detecta y clasifica señales de tráfico en el frame.
        Returns (label, speed_m_s, bbox_dict, mask_uint8|None)
        """
        result = sd.detection_with_bbox(frame)

        if result is None:
            return 'none', -1.0, {}, None

        crop, (bx, by, bw, bh) = result

        # Máscara de detección (rojo) recortada al bbox, redimensionada a uint8
        from sign_detection import _red_mask
        import cv2
        full_mask = _red_mask(frame)
        mask_out  = cv2.resize(full_mask, (frame.shape[1], frame.shape[0]))

        bbox = {'x': bx, 'y': by, 'w': bw, 'h': bh}

        if self.model is None:
            return 'detected', -1.0, bbox, mask_out

        try:
            tensor = process_image(crop).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                class_id = int(logits.argmax(1).item())

            label      = self._CLASS_NAMES.get(class_id, 'unknown')
            speed_ms   = self._CLASS_SPEED.get(class_id, -1.0)
            return label, speed_ms, bbox, mask_out
        except Exception as e:
            self.get_logger().warn(f'Error en clasificación: {e}')
            return 'detected', -1.0, bbox, mask_out


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