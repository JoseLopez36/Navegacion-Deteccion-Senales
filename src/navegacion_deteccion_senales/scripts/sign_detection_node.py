#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from cv_bridge import CvBridge

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image


class SignDetectionNode(Node):
    """
    Nodo de detección y reconocimiento de señales de tráfico mediante CNN (PyTorch).

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image  (sensor_msgs/Image)

    Publicaciones:
      - /sign_detection/sign_label          (std_msgs/String)  — etiqueta de la señal detectada
      - /sign_detection/speed_limit         (std_msgs/Float32) — límite de velocidad extraído (m/s), -1 si no aplica
    """

    def __init__(self):
        super().__init__('sign_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('sign_label_topic', '/sign_detection/sign_label')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')

        self.image_topic       = self.get_parameter('image_topic').value
        self.sign_label_topic  = self.get_parameter('sign_label_topic').value
        self.speed_limit_topic = self.get_parameter('speed_limit_topic').value

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
        self.model  = None  # TODO: cargar modelo CNN con PyTorch

        self._load_model()

        # --- Publicadores ---
        self.label_pub = self.create_publisher(String, self.sign_label_topic, reliable_qos)
        self.speed_pub = self.create_publisher(Float32, self.speed_limit_topic, reliable_qos)

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        self.get_logger().info('sign_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Inicialización del modelo
    # ------------------------------------------------------------------

    def _load_model(self):
        if not self.model_path:
            self.get_logger().warn('model_path no configurado; detección deshabilitada.')
            return
        # TODO: implementar
        #   import torch
        #   self.model = MyCNN(num_classes=self.num_classes)
        #   self.model.load_state_dict(torch.load(self.model_path))
        #   self.model.eval()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        sign_label, speed_limit_ms, debug_frame = self._detect_signs(frame)

        label_msg = String()
        label_msg.data = sign_label
        self.label_pub.publish(label_msg)

        speed_msg = Float32()
        speed_msg.data = float(speed_limit_ms)
        self.speed_pub.publish(speed_msg)

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _detect_signs(self, frame):
        """
        Detecta y clasifica señales de tráfico en el frame.
        """
        # TODO: implementar pipeline completo de detección de señales
        sign_label     = 'none'
        speed_limit_ms = -1.0
        debug_frame    = frame.copy()
        return sign_label, speed_limit_ms, debug_frame


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