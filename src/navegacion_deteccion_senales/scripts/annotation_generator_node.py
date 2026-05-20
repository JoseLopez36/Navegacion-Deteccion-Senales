#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Time

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation, TextAnnotation


class AnnotationGeneratorNode(Node):
    """
    Nodo que genera anotaciones visuales para Foxglove Studio.

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image  (sensor_msgs/Image)         — imagen de referencia para timestamp
      - /lane_detection/lane_error          (std_msgs/Float32)           — error lateral del carril
      - /lane_detection/debug_image         (sensor_msgs/Image)          — imagen anotada del carril
      - /sign_detection/sign_label          (std_msgs/String)            — etiqueta de señal detectada
      - /sign_detection/speed_limit         (std_msgs/Float32)           — límite de velocidad (m/s)

    Publicaciones:
      - /foxglove/annotations               (foxglove_msgs/ImageAnnotations) — anotaciones para Foxglove
    """

    def __init__(self):
        super().__init__('annotation_generator_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('sign_label_topic', '/sign_detection/sign_label')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')
        self.declare_parameter('annotations_topic', '/foxglove/annotations')
        self.declare_parameter('image_width', 800)
        self.declare_parameter('image_height', 600)

        self.image_topic       = self.get_parameter('image_topic').value
        self.lane_error_topic  = self.get_parameter('lane_error_topic').value
        self.sign_label_topic  = self.get_parameter('sign_label_topic').value
        self.speed_limit_topic = self.get_parameter('speed_limit_topic').value
        self.annotations_topic = self.get_parameter('annotations_topic').value
        self.image_width       = int(self.get_parameter('image_width').value)
        self.image_height      = int(self.get_parameter('image_height').value)

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

        # --- Estado interno ---
        self.lane_error  = 0.0
        self.sign_label  = 'none'
        self.speed_limit = -1.0

        # --- Publicadores ---
        self.annotations_pub = self.create_publisher(
            ImageAnnotations,
            self.annotations_topic,
            reliable_qos,
        )

        # --- Suscriptores ---
        self.create_subscription(Image, self.image_topic, self._on_image, sensor_qos)
        self.create_subscription(Float32, self.lane_error_topic, self._on_lane_error, reliable_qos)
        self.create_subscription(String, self.sign_label_topic, self._on_sign_label, reliable_qos)
        self.create_subscription(Float32, self.speed_limit_topic, self._on_speed_limit, reliable_qos)

        self.get_logger().info('annotation_generator_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks de estado
    # ------------------------------------------------------------------

    def _on_lane_error(self, msg: Float32):
        self.lane_error = msg.data

    def _on_sign_label(self, msg: String):
        self.sign_label = msg.data

    def _on_speed_limit(self, msg: Float32):
        self.speed_limit = msg.data

    # ------------------------------------------------------------------
    # Callback principal: se ejecuta por cada frame de cámara
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        annotations = self._build_annotations(msg.header.stamp)
        self.annotations_pub.publish(annotations)

    # ------------------------------------------------------------------
    # Construcción de anotaciones
    # ------------------------------------------------------------------

    def _build_annotations(self, stamp: Time) -> ImageAnnotations:
        """
        Construye un mensaje ImageAnnotations con:
          - Línea vertical indicando el error lateral estimado del carril.
          - Texto con la etiqueta de la señal detectada y el límite de velocidad.

        TODO: completar con bounding-boxes de señales, polilíneas del carril, etc.
        """
        ann = ImageAnnotations()

        # --- Anotación de error lateral (línea vertical centrada) ---
        cx = float(self.image_width) / 2.0
        cy = float(self.image_height) / 2.0
        lane_center_x = cx - self.lane_error  # píxeles

        lane_line = PointsAnnotation()
        lane_line.timestamp = stamp
        lane_line.type = PointsAnnotation.LINE_LIST
        lane_line.thickness = 2.0
        lane_line.outline_color.r = 0.0
        lane_line.outline_color.g = 1.0
        lane_line.outline_color.b = 0.0
        lane_line.outline_color.a = 1.0
        # TODO: añadir puntos reales del carril detectado
        ann.points.append(lane_line)

        # --- Anotación de texto: señal y velocidad ---
        sign_text = TextAnnotation()
        sign_text.timestamp = stamp
        sign_text.position.x = 10.0
        sign_text.position.y = 30.0
        speed_str = (
            f'{self.speed_limit * 3.6:.0f} km/h'
            if self.speed_limit > 0
            else 'N/A'
        )
        sign_text.text = f'Señal: {self.sign_label} | Vel. max: {speed_str}'
        sign_text.font_size = 14.0
        sign_text.text_color.r = 1.0
        sign_text.text_color.g = 1.0
        sign_text.text_color.b = 0.0
        sign_text.text_color.a = 1.0
        sign_text.background_color.a = 0.0
        ann.texts.append(sign_text)

        # --- Anotación de texto: error lateral ---
        error_text = TextAnnotation()
        error_text.timestamp = stamp
        error_text.position.x = 10.0
        error_text.position.y = 55.0
        error_text.text = f'Error lateral: {self.lane_error:.1f} px'
        error_text.font_size = 14.0
        error_text.text_color.r = 0.0
        error_text.text_color.g = 1.0
        error_text.text_color.b = 1.0
        error_text.text_color.a = 1.0
        error_text.background_color.a = 0.0
        ann.texts.append(error_text)

        return ann


def main(args=None):
    rclpy.init(args=args)
    node = AnnotationGeneratorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()