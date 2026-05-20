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
      - /sign_detection/debug_image         (sensor_msgs/Image) — imagen con bounding-boxes (opcional)
    """

    def __init__(self):
        super().__init__('sign_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('sign_label_topic', '/sign_detection/sign_label')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')
        self.declare_parameter('debug_image_topic', '/sign_detection/debug_image')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('model_path', '')        # ruta al .pth del modelo CNN
        self.declare_parameter('confidence_threshold', 0.7)
        self.declare_parameter('input_width', 32)
        self.declare_parameter('input_height', 32)
        self.declare_parameter('num_classes', 43)       # GTSRB tiene 43 clases

        self.image_topic       = self.get_parameter('image_topic').value
        self.sign_label_topic  = self.get_parameter('sign_label_topic').value
        self.speed_limit_topic = self.get_parameter('speed_limit_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.publish_debug     = bool(self.get_parameter('publish_debug_image').value)
        self.model_path        = self.get_parameter('model_path').value
        self.conf_threshold    = float(self.get_parameter('confidence_threshold').value)
        self.input_width       = int(self.get_parameter('input_width').value)
        self.input_height      = int(self.get_parameter('input_height').value)
        self.num_classes       = int(self.get_parameter('num_classes').value)

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
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, sensor_qos)

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        self.get_logger().info('sign_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Inicialización del modelo
    # ------------------------------------------------------------------

    def _load_model(self):
        """
        Carga el modelo CNN desde self.model_path.
        TODO: implementar carga del modelo PyTorch.
        """
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

        if self.publish_debug:
            debug_ros = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_ros.header = msg.header
            self.debug_pub.publish(debug_ros)

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _detect_signs(self, frame):
        """
        Detecta y clasifica señales de tráfico en el frame.

        Returns:
            sign_label (str):       etiqueta de la señal de mayor confianza ('none' si no detecta nada).
            speed_limit_ms (float): velocidad máxima en m/s si la señal es de límite de velocidad, -1.0 si no aplica.
            debug_frame (np.ndarray): frame con bounding-boxes y etiquetas anotadas.
        """
        # TODO: implementar pipeline completo de detección de señales
        #   1. Preprocesar frame (resize, normalizar)
        #   2. Detectar regiones candidatas (sliding-window, selective search, etc.)
        #   3. Clasificar cada región con self.model
        #   4. Filtrar por self.conf_threshold
        #   5. Extraer velocidad si la clase corresponde a señal de límite de velocidad
        #   6. Dibujar bounding-boxes y etiquetas en debug_frame
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