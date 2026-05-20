#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy\

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
      - /lane_detection/debug_image     (sensor_msgs/Image) — imagen anotada (opcional)
    """

    def __init__(self):
        super().__init__('lane_detection_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('debug_image_topic', '/lane_detection/debug_image')
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter('roi_top_ratio', 0.55)       # fracción superior de la imagen a ignorar
        self.declare_parameter('canny_low', 50)
        self.declare_parameter('canny_high', 150)
        self.declare_parameter('hough_threshold', 50)
        self.declare_parameter('hough_min_line_length', 40)
        self.declare_parameter('hough_max_line_gap', 20)

        self.image_topic       = self.get_parameter('image_topic').value
        self.lane_error_topic  = self.get_parameter('lane_error_topic').value
        self.debug_image_topic = self.get_parameter('debug_image_topic').value
        self.publish_debug     = bool(self.get_parameter('publish_debug_image').value)
        self.roi_top_ratio     = float(self.get_parameter('roi_top_ratio').value)
        self.canny_low         = int(self.get_parameter('canny_low').value)
        self.canny_high        = int(self.get_parameter('canny_high').value)
        self.hough_threshold   = int(self.get_parameter('hough_threshold').value)
        self.hough_min_length  = int(self.get_parameter('hough_min_line_length').value)
        self.hough_max_gap     = int(self.get_parameter('hough_max_line_gap').value)

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
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, sensor_qos)

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)

        self.get_logger().info('lane_detection_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        lateral_error, debug_frame = self._detect_lane(frame)

        error_msg = Float32()
        error_msg.data = float(lateral_error)
        self.error_pub.publish(error_msg)

        if self.publish_debug:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

    # ------------------------------------------------------------------
    # Lógica de detección
    # ------------------------------------------------------------------

    def _detect_lane(self, frame):
        """
        Detecta líneas de carril y devuelve el error lateral en píxeles.

        Returns:
            lateral_error (float): desplazamiento del centro estimado respecto al centro de imagen.
                                   Positivo → vehículo desplazado a la derecha del carril.
            debug_frame (np.ndarray): imagen BGR con anotaciones.
        """
        # TODO: implementar pipeline completo de detección de carril
        #   1. Convertir a escala de grises y aplicar desenfoque gaussiano
        #   2. Aplicar Canny con self.canny_low / self.canny_high
        #   3. Recortar ROI usando self.roi_top_ratio
        #   4. Aplicar HoughLinesP con self.hough_* params
        #   5. Separar líneas izquierda/derecha por pendiente
        #   6. Ajustar líneas medias y calcular punto de convergencia
        #   7. Calcular lateral_error = x_centro_carril - frame.shape[1] / 2
        #   8. Dibujar anotaciones sobre debug_frame
        lateral_error = 0.0
        debug_frame = frame.copy()
        return lateral_error, debug_frame


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