#!/usr/bin/env python3

import json

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from builtin_interfaces.msg import Time
from foxglove_msgs.msg import Point2

from std_msgs.msg import Float32, String
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaEgoVehicleControl
from foxglove_msgs.msg import ImageAnnotations, PointsAnnotation, TextAnnotation


class AnnotationGeneratorNode(Node):
    """
    Nodo que genera anotaciones visuales para Foxglove Studio.

    Suscripciones:
      - /carla/ego_vehicle/rgb_front/image       (sensor_msgs/Image)              — timestamp de referencia
      - /lane_detection/lane_error               (std_msgs/Float32)               — error lateral en píxeles
      - /lane_detection/lane_state               (std_msgs/String)                — JSON con líneas y estado
      - /carla/ego_vehicle/speedometer           (std_msgs/Float32)               — velocidad actual
      - /carla/ego_vehicle/vehicle_control_cmd   (carla_msgs/CarlaEgoVehicleControl) — comandos de control

    Publicaciones:
      - /foxglove/annotations  (foxglove_msgs/ImageAnnotations)  — anotaciones para Foxglove
    """

    def __init__(self):
        super().__init__('annotation_generator_node')

        # --- Parámetros ---
        self.declare_parameter('image_topic', '/carla/ego_vehicle/rgb_front/image')
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('lane_state_topic', '/lane_detection/lane_state')
        self.declare_parameter('annotations_topic', '/foxglove/annotations')
        self.declare_parameter('speedometer_topic',  '/carla/ego_vehicle/speedometer')
        self.declare_parameter('vehicle_control_topic', '/carla/ego_vehicle/vehicle_control_cmd')
        self.declare_parameter('image_width',  800)
        self.declare_parameter('image_height', 600)

        self.image_topic            = self.get_parameter('image_topic').value
        self.lane_error_topic       = self.get_parameter('lane_error_topic').value
        self.lane_state_topic       = self.get_parameter('lane_state_topic').value
        self.annotations_topic      = self.get_parameter('annotations_topic').value
        self.speedometer_topic      = self.get_parameter('speedometer_topic').value
        self.vehicle_control_topic  = self.get_parameter('vehicle_control_topic').value
        self.image_width            = int(self.get_parameter('image_width').value)
        self.image_height           = int(self.get_parameter('image_height').value)

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
        self.lane_error    = 0.0
        self.lane_state: dict = {}
        self.current_speed = 0.0
        self.cmd_throttle  = 0.0
        self.cmd_brake     = 0.0
        self.cmd_steer     = 0.0

        # --- Publicadores ---
        self.annotations_pub = self.create_publisher(
            ImageAnnotations,
            self.annotations_topic,
            reliable_qos,
        )

        # --- Suscriptores ---
        self.create_subscription(Image,   self.image_topic,           self._on_image,          sensor_qos)
        self.create_subscription(Float32, self.lane_error_topic,      self._on_lane_error,     reliable_qos)
        self.create_subscription(String,  self.lane_state_topic,      self._on_lane_state,     reliable_qos)
        self.create_subscription(Float32, self.speedometer_topic,     self._on_speedometer,    sensor_qos)
        self.create_subscription(
            CarlaEgoVehicleControl, self.vehicle_control_topic,
            self._on_vehicle_control, reliable_qos)

        self.get_logger().info('annotation_generator_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks de estado
    # ------------------------------------------------------------------

    def _on_lane_error(self, msg: Float32):
        self.lane_error = msg.data

    def _on_lane_state(self, msg: String):
        try:
            self.lane_state = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().debug('Failed to decode lane_state JSON')

    def _on_speedometer(self, msg: Float32):
        self.current_speed = msg.data

    def _on_vehicle_control(self, msg: CarlaEgoVehicleControl):
        self.cmd_throttle = msg.throttle
        self.cmd_brake    = msg.brake
        self.cmd_steer    = msg.steer

    # ------------------------------------------------------------------
    # Callback principal: se ejecuta por cada frame de cámara
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image):
        annotations = self._build_annotations(msg.header.stamp)
        self.annotations_pub.publish(annotations)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point2(x: float, y: float) -> Point2:
        p = Point2()
        p.x = float(x)
        p.y = float(y)
        return p

    @staticmethod
    def _points_annotation(stamp: Time, ann_type, thickness: float,
                           outline_r: float, outline_g: float,
                           outline_b: float, outline_a: float,
                           fill_r: float = 0.0, fill_g: float = 0.0,
                           fill_b: float = 0.0, fill_a: float = 0.0) -> PointsAnnotation:
        ann = PointsAnnotation()
        ann.timestamp = stamp
        ann.type = ann_type
        ann.thickness = thickness
        ann.outline_color.r = outline_r
        ann.outline_color.g = outline_g
        ann.outline_color.b = outline_b
        ann.outline_color.a = outline_a
        ann.fill_color.r = fill_r
        ann.fill_color.g = fill_g
        ann.fill_color.b = fill_b
        ann.fill_color.a = fill_a
        return ann

    @staticmethod
    def _text_annotation(stamp: Time, x: float, y: float, text: str,
                         font_size: float,
                         r: float, g: float, b: float, a: float) -> TextAnnotation:
        t = TextAnnotation()
        t.timestamp = stamp
        t.position.x = float(x)
        t.position.y = float(y)
        t.text = text
        t.font_size = font_size
        t.text_color.r = r
        t.text_color.g = g
        t.text_color.b = b
        t.text_color.a = a
        t.background_color.a = 0.0
        return t

    # ------------------------------------------------------------------
    # Construcción de anotaciones
    # ------------------------------------------------------------------

    def _build_annotations(self, stamp: Time) -> ImageAnnotations:
        ann = ImageAnnotations()
        s = self.lane_state

        w = float(s.get('image_width', self.image_width))
        h = float(s.get('image_height', self.image_height))
        cx = w / 2.0

        left = s.get('left')
        right = s.get('right')

        if left:
            line = self._points_annotation(
                stamp, PointsAnnotation.LINE_STRIP, 4.0,
                1.0, 0.39, 0.0, 1.0)
            line.points.extend([
                self._point2(left[0], left[1]),
                self._point2(left[2], left[3])])
            ann.points.append(line)

        if right:
            line = self._points_annotation(
                stamp, PointsAnnotation.LINE_STRIP, 4.0,
                0.0, 0.78, 1.0, 1.0)
            line.points.extend([
                self._point2(right[0], right[1]),
                self._point2(right[2], right[3])])
            ann.points.append(line)

        if left and right:
            poly = self._points_annotation(
                stamp, PointsAnnotation.LINE_LOOP, 2.0,
                0.0, 0.71, 0.0, 0.6,
                0.0, 0.71, 0.0, 0.15)
            poly.points.extend([
                self._point2(left[0], left[1]),
                self._point2(left[2], left[3]),
                self._point2(right[2], right[3]),
                self._point2(right[0], right[1])])
            ann.points.append(poly)

        deviation = self._points_annotation(
            stamp, PointsAnnotation.LINE_STRIP, 2.0,
            1.0, 1.0, 0.0, 1.0)
        lane_center_x = cx - self.lane_error
        deviation.points.extend([
            self._point2(cx, h * 0.7),
            self._point2(lane_center_x, h * 0.7)])
        ann.points.append(deviation)

        zone = s.get('zone', 'UNKNOWN')
        lateral = s.get('lateral', 0.5)
        offset = s.get('offset_px', self.lane_error)

        ann.texts.append(self._text_annotation(
            stamp, 10.0, 30.0, f'Zona: {zone}', 20.0,
            0.0, 1.0, 0.4, 1.0))
        ann.texts.append(self._text_annotation(
            stamp, 10.0, 55.0,
            f'Lateral: {lateral:.2f}  Offset: {int(offset):+d}px', 18.0,
            1.0, 1.0, 0.0, 1.0))
        ann.texts.append(self._text_annotation(
            stamp, 10.0, 75.0,
            f"L:{int(s.get('left_detected', False))}  R:{int(s.get('right_detected', False))}",
            16.0, 0.6, 0.6, 0.6, 1.0))

        speed_kmh = self.current_speed * 3.6
        ann.texts.append(self._text_annotation(
            stamp, w - 220.0, 30.0, f'Vel: {speed_kmh:.1f} km/h', 20.0,
            1.0, 1.0, 1.0, 1.0))
        ann.texts.append(self._text_annotation(
            stamp, w - 220.0, 55.0,
            f'Throttle: {self.cmd_throttle:.2f}  Brake: {self.cmd_brake:.2f}', 17.0,
            0.4, 1.0, 0.4, 1.0))
        ann.texts.append(self._text_annotation(
            stamp, w - 220.0, 75.0,
            f'Steer: {self.cmd_steer:+.3f} rad', 17.0,
            1.0, 0.65, 0.0, 1.0))

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