#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from std_msgs.msg import Float32
from carla_msgs.msg import CarlaEgoVehicleControl


class VehicleControlNode(Node):
    """
    Nodo de control del vehículo.

    Suscripciones:
      - /lane_detection/lane_error          (std_msgs/Float32)              — error lateral en metros
      - /carla/ego_vehicle/speedometer      (std_msgs/Float32)

    Publicaciones:
      - /carla/ego_vehicle/vehicle_control_cmd (carla_msgs/CarlaEgoVehicleControl)
    """

    def __init__(self):
        super().__init__('vehicle_control_node')

        # --- Parámetros ---
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('target_speed', 5.0) # m/s
        self.declare_parameter('max_steer', 1.0) # normalizado CARLA [-1, 1]
        self.declare_parameter('max_steer_rad', 0.6) # ángulo físico máximo del volante [rad]
        self.declare_parameter('kp_throttle', 0.3)
        self.declare_parameter('ki_throttle', 0.01)
        self.declare_parameter('kd_throttle', 0.05)
        self.declare_parameter('kp_steering', 0.002)
        self.declare_parameter('ki_steering', 0.001)
        self.declare_parameter('kd_steering', -1.07e-05)
        self.declare_parameter('lane_error_topic', '/lane_detection/lane_error')
        self.declare_parameter('speedometer_topic', '/carla/ego_vehicle/speedometer')
        self.declare_parameter('vehicle_control_topic', '/carla/ego_vehicle/vehicle_control_cmd')

        self.control_rate         = float(self.get_parameter('control_rate').value)
        self.target_speed         = float(self.get_parameter('target_speed').value)
        self.max_steer            = float(self.get_parameter('max_steer').value)
        self.max_steer_rad        = float(self.get_parameter('max_steer_rad').value)
        self.kp_throttle          = float(self.get_parameter('kp_throttle').value)
        self.ki_throttle          = float(self.get_parameter('ki_throttle').value)
        self.kd_throttle          = float(self.get_parameter('kd_throttle').value)
        self.kp_steering          = float(self.get_parameter('kp_steering').value)
        self.ki_steering          = float(self.get_parameter('ki_steering').value)
        self.kd_steering          = float(self.get_parameter('kd_steering').value)
        self.lane_error_topic     = self.get_parameter('lane_error_topic').value
        self.speedometer_topic    = self.get_parameter('speedometer_topic').value
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
        self.lane_error    = 0.0
        self.speed_limit   = self.target_speed
        self.current_speed = 0.0

        # --- Estado PID crucero ---
        self._throttle_integral  = 0.0
        self._throttle_prev_err  = 0.0

        # --- Estado PID dirección ---
        self._steering_integral  = 0.0
        self._steering_prev_err  = 0.0

        self._prev_time = self.get_clock().now()

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

    def _on_speedometer(self, msg: Float32):
        self.current_speed = msg.data

    # ------------------------------------------------------------------
    # Bucle de control
    # ------------------------------------------------------------------

    def _control_loop(self):
        now = self.get_clock().now()
        dt  = max((now - self._prev_time).nanoseconds * 1e-9, 1e-6)
        self._prev_time = now

        throttle, brake = self._cruise_control(dt)
        steering        = self._keep_lane(dt)

        self.get_logger().info(
            f'speed={self.current_speed:.2f}m/s  target={self.speed_limit:.1f}m/s  '
            f'throttle={throttle:.2f}  brake={brake:.2f}  '
            f'steer={steering:+.3f}  lane_err={self.lane_error:+.4f}m',
            throttle_duration_sec=1.0,
        )

        cmd = CarlaEgoVehicleControl()
        cmd.hand_brake        = False
        cmd.reverse           = False
        cmd.manual_gear_shift = False
        cmd.throttle          = throttle
        cmd.brake             = brake
        cmd.steer             = steering

        self.control_pub.publish(cmd)

    def _cruise_control(self, dt: float):
        """PID de velocidad. Devuelve (throttle, brake) en [0, 1]"""
        error = self.speed_limit - self.current_speed

        self._throttle_integral += error * dt
        self._throttle_integral  = max(-10.0, min(10.0, self._throttle_integral))  # anti-windup
        derivative = (error - self._throttle_prev_err) / dt
        self._throttle_prev_err = error

        output = (self.kp_throttle * error
                  + self.ki_throttle * self._throttle_integral
                  + self.kd_throttle * derivative)

        throttle = float(max(0.0, min(1.0,  output)))
        brake    = float(max(0.0, min(1.0, -output)))
        return throttle, brake

    def _keep_lane(self, dt: float) -> float:
        """PID de mantenimiento de carril.

        Entrada : lane_error en metros.
        Salida  : steer normalizado CARLA en [-1, 1].

        El PID opera en radianes (kp [rad/m], ki [rad/(m·s)], kd [rad·s/m]).
        La salida en rad se normaliza por max_steer_rad antes de enviarse a CARLA.

        Convención de signo:
          lane_error > 0 → vehículo desplazado a la derecha → girar izquierda (steer < 0)
        """
        error = -self.lane_error  # negado: error positivo → corrección hacia la izquierda

        self._steering_integral += error * dt
        self._steering_integral  = max(-10.0, min(10.0, self._steering_integral))  # anti-windup
        derivative = (error - self._steering_prev_err) / dt
        self._steering_prev_err = error

        steer_rad = (self.kp_steering * error
                     + self.ki_steering * self._steering_integral
                     + self.kd_steering * derivative)

        steer_norm = steer_rad / self.max_steer_rad
        return float(max(-self.max_steer, min(self.max_steer, steer_norm)))


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