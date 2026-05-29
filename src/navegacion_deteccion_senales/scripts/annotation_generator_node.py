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
      - /sign_detection/sign_label              (std_msgs/String)                — etiqueta de la señal detectada
      - /sign_detection/speed_limit             (std_msgs/Float32)               — límite de velocidad (m/s)
      - /sign_detection/bbox                    (std_msgs/String)                — JSON con bbox {x,y,w,h}

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
        self.declare_parameter('sign_label_topic', '/sign_detection/sign_label')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')
        self.declare_parameter('sign_bbox_topic', '/sign_detection/bbox')
        self.declare_parameter('image_width',  800)
        self.declare_parameter('image_height', 600)

        self.image_topic            = self.get_parameter('image_topic').value
        self.lane_error_topic       = self.get_parameter('lane_error_topic').value
        self.lane_state_topic       = self.get_parameter('lane_state_topic').value
        self.annotations_topic      = self.get_parameter('annotations_topic').value
        self.speedometer_topic      = self.get_parameter('speedometer_topic').value
        self.vehicle_control_topic  = self.get_parameter('vehicle_control_topic').value
        self.sign_label_topic       = self.get_parameter('sign_label_topic').value
        self.speed_limit_topic      = self.get_parameter('speed_limit_topic').value
        self.sign_bbox_topic        = self.get_parameter('sign_bbox_topic').value
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
        self.sign_label    = 'none'
        self.sign_speed_ms = -1.0
        self.sign_bbox: dict = {}

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
        self.create_subscription(String,  self.sign_label_topic,      self._on_sign_label,     reliable_qos)
        self.create_subscription(Float32, self.speed_limit_topic,     self._on_sign_speed,     reliable_qos)
        self.create_subscription(String,  self.sign_bbox_topic,       self._on_sign_bbox,      reliable_qos)

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

    def _on_sign_label(self, msg: String):
        self.sign_label = msg.data

    def _on_sign_speed(self, msg: Float32):
        self.sign_speed_ms = msg.data

    def _on_sign_bbox(self, msg: String):
        try:
            self.sign_bbox = json.loads(msg.data)
        except json.JSONDecodeError:
            self.sign_bbox = {}

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
                           outline: tuple, fill: tuple = (0.0, 0.0, 0.0, 0.0)) -> PointsAnnotation:
        ann = PointsAnnotation()
        ann.timestamp = stamp
        ann.type = ann_type
        ann.thickness = thickness
        ann.outline_color.r, ann.outline_color.g, ann.outline_color.b, ann.outline_color.a = outline
        ann.fill_color.r, ann.fill_color.g, ann.fill_color.b, ann.fill_color.a = fill
        return ann

    @staticmethod
    def _text_annotation(stamp: Time, x: float, y: float, text: str,
                         font_size: float, color: tuple,
                         bg: tuple = (0.0, 0.0, 0.0, 0.55)) -> TextAnnotation:
        t = TextAnnotation()
        t.timestamp = stamp
        t.position.x = float(x)
        t.position.y = float(y)
        t.text = text
        t.font_size = font_size
        t.text_color.r, t.text_color.g, t.text_color.b, t.text_color.a = color
        t.background_color.r, t.background_color.g, t.background_color.b, t.background_color.a = bg
        return t

    # ------------------------------------------------------------------
    # Construcción de anotaciones
    # ------------------------------------------------------------------

    # Colores por etiqueta de señal
    _SIGN_COLOR = {
        'stop':           (1.0,  0.15, 0.15, 1.0),
        'speed_limit_30': (1.0,  0.6,  0.0,  1.0),
        'speed_limit_60': (1.0,  0.6,  0.0,  1.0),
        'speed_limit_90': (1.0,  0.6,  0.0,  1.0),
        'detected':       (0.8,  0.8,  0.0,  1.0),
    }
    _SIGN_FILL = {
        'stop':           (1.0,  0.15, 0.15, 0.12),
        'speed_limit_30': (1.0,  0.6,  0.0,  0.10),
        'speed_limit_60': (1.0,  0.6,  0.0,  0.10),
        'speed_limit_90': (1.0,  0.6,  0.0,  0.10),
        'detected':       (0.8,  0.8,  0.0,  0.08),
    }

    def _build_annotations(self, stamp: Time) -> ImageAnnotations:
        ann = ImageAnnotations()
        s = self.lane_state

        w = float(s.get('image_width', self.image_width))
        h = float(s.get('image_height', self.image_height))
        cx = w / 2.0

        left  = s.get('left')
        right = s.get('right')

        # ── Polígono de carril (relleno verde semitransparente) ─────────
        if left and right:
            poly = self._points_annotation(
                stamp, PointsAnnotation.LINE_LOOP, 1.0,
                outline=(0.0, 1.0, 0.4, 0.0),
                fill=(0.0, 1.0, 0.4, 0.18))
            poly.points.extend([
                self._point2(left[0],  left[1]),
                self._point2(left[2],  left[3]),
                self._point2(right[2], right[3]),
                self._point2(right[0], right[1])])
            ann.points.append(poly)

        # ── Línea izquierda (amarillo) ──────────────────────────────────
        if left:
            line = self._points_annotation(
                stamp, PointsAnnotation.LINE_STRIP, 6.0,
                outline=(1.0, 0.85, 0.0, 1.0))
            line.points.extend([
                self._point2(left[0], left[1]),
                self._point2(left[2], left[3])])
            ann.points.append(line)

        # ── Línea derecha (cian) ────────────────────────────────────────
        if right:
            line = self._points_annotation(
                stamp, PointsAnnotation.LINE_STRIP, 6.0,
                outline=(0.0, 0.9, 1.0, 1.0))
            line.points.extend([
                self._point2(right[0], right[1]),
                self._point2(right[2], right[3])])
            ann.points.append(line)

        # ── Línea de desviación: centro imagen → centro de carril ───────
        error         = float(s.get('error', 0.0))
        lane_center_x = cx - error
        y_dev         = h * 0.72

        dev_color = (0.2, 1.0, 0.3, 1.0) if abs(error) < 30 else (1.0, 0.3, 0.2, 1.0)
        deviation = self._points_annotation(
            stamp, PointsAnnotation.LINE_STRIP, 3.0,
            outline=dev_color)
        deviation.points.extend([
            self._point2(cx, y_dev),
            self._point2(lane_center_x, y_dev)])
        ann.points.append(deviation)

        # Punto en el centro del carril
        dot = self._points_annotation(
            stamp, PointsAnnotation.POINTS, 10.0,
            outline=dev_color, fill=dev_color)
        dot.points.append(self._point2(lane_center_x, y_dev))
        ann.points.append(dot)

        # Punto en el centro del vehículo (blanco)
        ego = self._points_annotation(
            stamp, PointsAnnotation.POINTS, 10.0,
            outline=(1.0, 1.0, 1.0, 0.9),
            fill=(1.0, 1.0, 1.0, 0.9))
        ego.points.append(self._point2(cx, y_dev))
        ann.points.append(ego)

        # ── Señal de tráfico: bounding box + etiqueta ──────────────────
        bbox = self.sign_bbox
        if bbox and self.sign_label not in ('none', ''):
            bx = float(bbox.get('x', 0))
            by = float(bbox.get('y', 0))
            bw = float(bbox.get('w', 0))
            bh = float(bbox.get('h', 0))

            sig_col  = self._SIGN_COLOR.get(self.sign_label,  (0.8, 0.8, 0.0, 1.0))
            sig_fill = self._SIGN_FILL.get(self.sign_label,   (0.8, 0.8, 0.0, 0.08))

            box = self._points_annotation(
                stamp, PointsAnnotation.LINE_LOOP, 3.5,
                outline=sig_col, fill=sig_fill)
            box.points.extend([
                self._point2(bx,      by),
                self._point2(bx + bw, by),
                self._point2(bx + bw, by + bh),
                self._point2(bx,      by + bh)])
            ann.points.append(box)

            # Esquinas decorativas (L-shapes)
            corner_len = min(bw, bh) * 0.2
            for cx_c, cy_c, dx, dy in [
                (bx,      by,      1, 1),
                (bx + bw, by,     -1, 1),
                (bx + bw, by + bh,-1,-1),
                (bx,      by + bh, 1,-1),
            ]:
                lc = self._points_annotation(stamp, PointsAnnotation.LINE_STRIP, 5.0, outline=sig_col)
                lc.points.extend([
                    self._point2(cx_c + dx * corner_len, cy_c),
                    self._point2(cx_c,                   cy_c),
                    self._point2(cx_c,                   cy_c + dy * corner_len)])
                ann.points.append(lc)

            speed_kmh_sign = self.sign_speed_ms * 3.6 if self.sign_speed_ms >= 0 else -1.0
            speed_str  = f'  {speed_kmh_sign:.0f} km/h' if speed_kmh_sign >= 0 else ''
            label_str  = self.sign_label.replace('_', ' ').upper()
            ann.texts.append(self._text_annotation(
                stamp, bx, max(0.0, by - 24.0),
                f'{label_str}{speed_str}', 16.0,
                (1.0, 1.0, 1.0, 1.0),
                bg=(sig_col[0] * 0.5, sig_col[1] * 0.5, sig_col[2] * 0.5, 0.75)))

        # ── HUD izquierdo: error lateral ────────────────────────────────
        err_col = (0.3, 1.0, 0.4, 1.0) if abs(error) < 30 else (1.0, 0.35, 0.2, 1.0)
        ann.texts.append(self._text_annotation(
            stamp, 8.0, 28.0,
            f'Error: {self.lane_error:+.4f} m  ({int(error):+d} px)', 16.0,
            err_col,
            bg=(0.0, 0.0, 0.0, 0.55)))

        # ── HUD derecho: velocidad + control ────────────────────────────
        speed_kmh = self.current_speed * 3.6
        rx = w - 148.0

        ann.texts.append(self._text_annotation(
            stamp, rx, 28.0, f'{speed_kmh:.1f} km/h', 22.0,
            (1.0, 1.0, 1.0, 1.0),
            bg=(0.0, 0.0, 0.0, 0.6)))
        ann.texts.append(self._text_annotation(
            stamp, rx, 54.0,
            f'T {self.cmd_throttle:.2f}  B {self.cmd_brake:.2f}', 15.0,
            (0.4, 1.0, 0.4, 1.0),
            bg=(0.0, 0.0, 0.0, 0.55)))
        ann.texts.append(self._text_annotation(
            stamp, rx, 76.0, f'Steer {self.cmd_steer:+.3f}', 15.0,
            (1.0, 0.7, 0.1, 1.0),
            bg=(0.0, 0.0, 0.0, 0.55)))

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