#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

import json

from std_msgs.msg import Float32, String
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaEgoVehicleStatus


class VehicleControlNode(Node):
    """
    Nodo de control del vehículo.

    Suscripciones:
      - /lane_detection/lane_error          (std_msgs/Float32)  — error lateral en píxeles
      - /lane_detection/lane_state          (std_msgs/String)   — JSON con estado del carril
      - /sign_detection/speed_limit         (std_msgs/Float32)  — límite de velocidad (m/s)
      - /carla/ego_vehicle/speedometer      (std_msgs/Float32)

    Publicaciones:
      - /carla/ego_vehicle/vehicle_control_cmd (carla_msgs/CarlaEgoVehicleControl)

    Control:
      - Dirección: PD sobre error lateral en píxeles
      - Velocidad:  PI sobre error de velocidad con braking activo
      - Seguridad:  frena si no se detecta carril durante > lane_loss_timeout s
    """

    def __init__(self):
        super().__init__('vehicle_control_node')

        # --- Parámetros ---
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('target_speed', 5.0)           # m/s
        self.declare_parameter('max_steering_angle', 0.5)     # rad
        self.declare_parameter('kp_steering', 0.005)          # P sobre error en píxeles
        self.declare_parameter('kd_steering', 0.001)          # D sobre derivada del error
        self.declare_parameter('kp_throttle', 0.3)            # P sobre error de velocidad
        self.declare_parameter('ki_throttle', 0.05)           # I sobre error de velocidad
        self.declare_parameter('max_integral', 1.0)           # anti-windup
        self.declare_parameter('lane_loss_timeout', 1.0)      # s sin carril → frena
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('lane_state_topic', '/lane_detection/lane_state')
        self.declare_parameter('speed_limit_topic', '/sign_detection/speed_limit')
        self.declare_parameter('speedometer_topic', '/carla/ego_vehicle/speedometer')
        self.declare_parameter('vehicle_control_topic', '/carla/ego_vehicle/vehicle_control_cmd')

        self.control_rate          = float(self.get_parameter('control_rate').value)
        self.target_speed          = float(self.get_parameter('target_speed').value)
        self.max_steering_angle    = float(self.get_parameter('max_steering_angle').value)
        self.kp_steering           = float(self.get_parameter('kp_steering').value)
        self.kd_steering           = float(self.get_parameter('kd_steering').value)
        self.kp_throttle           = float(self.get_parameter('kp_throttle').value)
        self.ki_throttle           = float(self.get_parameter('ki_throttle').value)
        self.max_integral          = float(self.get_parameter('max_integral').value)
        self.lane_loss_timeout     = float(self.get_parameter('lane_loss_timeout').value)
        self.lane_error_topic      = self.get_parameter('lane_error_topic').value
        self.lane_state_topic      = self.get_parameter('lane_state_topic').value
        self.speed_limit_topic     = self.get_parameter('speed_limit_topic').value
        self.speedometer_topic     = self.get_parameter('speedometer_topic').value
        self.vehicle_control_topic = self.get_parameter('vehicle_control_topic').value

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
        self.lane_error       = 0.0
        self.prev_lane_error  = 0.0
        self.speed_limit      = self.target_speed
        self.current_speed    = 0.0
        self.throttle_integral = 0.0
        self.lane_detected    = False
        self._last_lane_time  = self.get_clock().now().nanoseconds / 1e9
        self._last_control_time = self.get_clock().now().nanoseconds / 1e9

        # --- Publicadores ---
        self.control_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            self.vehicle_control_topic,
            reliable_qos,
        )

        # --- Suscriptores ---
        self.create_subscription(
            Float32,
            self.lane_error_topic,
            self._on_lane_error,
            reliable_qos,
        )
        self.create_subscription(
            String,
            self.lane_state_topic,
            self._on_lane_state,
            reliable_qos,
        )
        self.create_subscription(
            Float32,
            self.speed_limit_topic,
            self._on_speed_limit,
            reliable_qos,
        )
        self.create_subscription(
            Float32,
            self.speedometer_topic,
            self._on_speedometer,
            sensor_qos,
        )

        # --- Timer de control ---
        self.create_timer(1.0 / self.control_rate, self._control_loop)

        self.get_logger().info('vehicle_control_node iniciado.')

    # ------------------------------------------------------------------
    # Callbacks de suscripción
    # ------------------------------------------------------------------

    def _on_lane_error(self, msg: Float32):
        self.lane_error = msg.data

    def _on_lane_state(self, msg: String):
        try:
            state = json.loads(msg.data)
            detected = state.get('left_detected', False) or state.get('right_detected', False)
            if detected:
                self.lane_detected = True
                self._last_lane_time = self.get_clock().now().nanoseconds / 1e9
            else:
                self.lane_detected = False
        except Exception:
            pass

    def _on_speed_limit(self, msg: Float32):
        if msg.data > 0.0:
            self.speed_limit = msg.data

    def _on_speedometer(self, msg: Float32):
        self.current_speed = msg.data

    # ------------------------------------------------------------------
    # Bucle de control
    # ------------------------------------------------------------------

    def _control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self._last_control_time
        self._last_control_time = now
        dt = max(dt, 1e-3)

        # --- Seguridad: pérdida de carril ---
        lane_age = now - self._last_lane_time
        lane_lost = lane_age > self.lane_loss_timeout

        if lane_lost:
            cmd = CarlaEgoVehicleControl()
            cmd.hand_brake = False
            cmd.reverse = False
            cmd.manual_gear_shift = False
            cmd.throttle = 0.0
            cmd.brake = 0.7
            cmd.steer = 0.0
            self.control_pub.publish(cmd)
            return

        # --- PD de dirección sobre error lateral en píxeles ---
        d_error = (self.lane_error - self.prev_lane_error) / dt
        self.prev_lane_error = self.lane_error

        steering_raw = -(self.kp_steering * self.lane_error + self.kd_steering * d_error)
        steering = float(max(-self.max_steering_angle, min(self.max_steering_angle, steering_raw)))

        # --- PI de velocidad ---
        effective_speed = min(self.speed_limit, self.target_speed)
        speed_error = effective_speed - self.current_speed

        if speed_error > 0.0:
            self.throttle_integral += speed_error * dt
            self.throttle_integral = min(self.throttle_integral, self.max_integral)
        else:
            self.throttle_integral = max(0.0, self.throttle_integral + speed_error * dt)

        throttle_raw = self.kp_throttle * speed_error + self.ki_throttle * self.throttle_integral

        if speed_error >= 0.0:
            throttle = float(max(0.0, min(1.0, throttle_raw)))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(min(1.0, -throttle_raw * 0.5))

        # --- Aplicar control ---
        cmd = CarlaEgoVehicleControl()
        cmd.hand_brake = False
        cmd.reverse = False
        cmd.manual_gear_shift = False
        cmd.throttle = throttle
        cmd.brake = brake
        cmd.steer = steering

        self.control_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()