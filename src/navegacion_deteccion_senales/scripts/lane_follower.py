#!/usr/bin/env python3

import rclpy
from autoware_control_msgs.msg import Control
from autoware_vehicle_msgs.msg import GearCommand, HazardLightsCommand, TurnIndicatorsCommand
from autoware_vehicle_msgs.srv import ControlModeCommand
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from tier4_vehicle_msgs.msg import VehicleEmergencyStamped

class LaneFollower(Node):
    def __init__(self):
        super().__init__('lane_follower')

        # Declarar parámetros
        self.declare_parameter('update_rate', 10.0)  # 10 Hz
        self.declare_parameter('control_rate', 60.0)
        self.declare_parameter('target_speed', 3.0)
        self.declare_parameter('target_acceleration', 1.0)
        self.declare_parameter('image_topic', '/sensing/camera/traffic_light/image_raw')
        self.declare_parameter('control_topic', '/control/command/control_cmd')
        self.declare_parameter('gear_topic', '/control/command/gear_cmd')
        self.declare_parameter('turn_indicator_topic', '/control/command/turn_indicators_cmd')
        self.declare_parameter('hazard_light_topic', '/control/command/hazard_lights_cmd')
        self.declare_parameter('emergency_topic', '/control/command/emergency_cmd')
        self.declare_parameter('control_mode_service', 'input/control_mode_request')
        self.declare_parameter('publish_debug_image', True)

        # Obtener valores de los parámetros
        self.update_rate = float(self.get_parameter('update_rate').value)
        self.control_rate = float(self.get_parameter('control_rate').value)
        self.target_speed = float(self.get_parameter('target_speed').value)
        self.target_acceleration = float(self.get_parameter('target_acceleration').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.control_topic = self.get_parameter('control_topic').value
        self.gear_topic = self.get_parameter('gear_topic').value
        self.turn_indicator_topic = self.get_parameter('turn_indicator_topic').value
        self.hazard_light_topic = self.get_parameter('hazard_light_topic').value
        self.emergency_topic = self.get_parameter('emergency_topic').value
        self.control_mode_service = self.get_parameter('control_mode_service').value
        self.publish_debug_image = bool(self.get_parameter('publish_debug_image').value)

        # Configurar QoS
        command_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Inicializar variables
        self.bridge = CvBridge()
        self.last_steering_angle = 0.0

        # Crear publishers
        self.control_pub = self.create_publisher(
            Control,
            self.control_topic,
            command_qos
        )
        self.gear_pub = self.create_publisher(
            GearCommand, 
            self.gear_topic, 
            command_qos
        )
        self.turn_pub = self.create_publisher(
            TurnIndicatorsCommand,
            self.turn_indicator_topic,
            command_qos
        )
        self.hazard_pub = self.create_publisher(
            HazardLightsCommand,
            self.hazard_light_topic,
            command_qos
        )
        self.emergency_pub = self.create_publisher(
            VehicleEmergencyStamped,
            self.emergency_topic,
            command_qos
        )
        self.debug_pub = self.create_publisher(
            Image,
            '~/debug_image',
            sensor_qos
        )
        self.control_mode_client = self.create_client(ControlModeCommand, self.control_mode_service)

        # Crear subscriber
        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.on_image,
            sensor_qos
        )
        
        # Crear timer
        self.status_timer = self.create_timer(1.0 / self.update_rate, self.publish_vehicle_state_commands)
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.publish_last_control)
        self.control_mode_timer = self.create_timer(1.0, self.request_autonomous_mode)

    def request_autonomous_mode(self):
        # Solicitar modo autónomo a AWSIM
        if not self.control_mode_client.service_is_ready():
            self.get_logger().warn('Esperando al servicio de modo de control de AWSIM...', throttle_duration_sec=5.0)
            return

        request = ControlModeCommand.Request()
        request.stamp = self.get_clock().now().to_msg()
        request.mode = ControlModeCommand.Request.AUTONOMOUS
        future = self.control_mode_client.call_async(request)
        future.add_done_callback(self.on_control_mode_response)
        self.control_mode_timer.cancel()

    def on_control_mode_response(self, future):
        # Procesar respuesta del servicio de modo de control
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f'Error solicitando modo autónomo: {exc}')
            self.control_mode_timer = self.create_timer(1.0, self.request_autonomous_mode)
            return

        if response.success:
            self.get_logger().info('AWSIM está en modo autónomo.')
        else:
            self.get_logger().warn('AWSIM rechazó el modo autónomo; se reintentará.')
            self.control_mode_timer = self.create_timer(1.0, self.request_autonomous_mode)

    def publish_vehicle_state_commands(self):
        # Publicar comando de marcha
        gear = GearCommand()
        gear.stamp = self.get_clock().now().to_msg()
        gear.command = GearCommand.DRIVE
        self.gear_pub.publish(gear)

        # Publicar comando de indicadores
        turn = TurnIndicatorsCommand()
        turn.stamp = self.get_clock().now().to_msg()
        turn.command = TurnIndicatorsCommand.DISABLE
        self.turn_pub.publish(turn)

        # Publicar comando de luces de emergencia
        hazard = HazardLightsCommand()
        hazard.stamp = self.get_clock().now().to_msg()
        hazard.command = HazardLightsCommand.DISABLE
        self.hazard_pub.publish(hazard)
        
        # Publicar comando de luces de emergencia
        emergency = VehicleEmergencyStamped()
        emergency.stamp = self.get_clock().now().to_msg()
        emergency.emergency = False
        self.emergency_pub.publish(emergency)

    def on_image(self, msg):
        # Convertir mensaje ROS a imagen OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Aplicar detección de carril
        steering_angle, debug_frame = self.detect_lane(frame)

        # Publicar control
        self.publish_control(steering_angle)

        # Actualizar último ángulo
        self.last_steering_angle = steering_angle

        # Publicar imagen de debug
        if self.publish_debug_image:
            debug_msg = self.bridge.cv2_to_imgmsg(debug_frame, encoding='bgr8')
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

    def detect_lane(self, frame):
        # TODO: Implementar lógica de detección de carril
        return 0.0, frame

    def publish_control(self, steering_angle):
        # Publicar acciones de control
        control = Control()
        control.stamp = self.get_clock().now().to_msg()
        control.lateral.stamp = control.stamp
        control.lateral.steering_tire_angle = steering_angle
        control.lateral.steering_tire_rotation_rate = 0.0
        control.longitudinal.stamp = control.stamp
        control.longitudinal.velocity = self.target_speed
        control.longitudinal.acceleration = self.target_acceleration
        control.longitudinal.jerk = 0.0
        if self.target_acceleration != 0.0:
            control.longitudinal.is_defined_acceleration = True
        else:
            control.longitudinal.is_defined_acceleration = False
        if self.target_acceleration != 0.0:
            control.longitudinal.is_defined_jerk = True
        else:
            control.longitudinal.is_defined_jerk = False
        self.control_pub.publish(control)

    def publish_last_control(self):
        self.publish_control(self.last_steering_angle)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()