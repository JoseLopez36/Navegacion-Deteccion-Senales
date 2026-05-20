from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'navegacion_deteccion_senales'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'params.yaml',
    ])

    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='error',
        description='Logging level for all nodes (debug, info, warn, error, fatal)',
    )
    log_level = LaunchConfiguration('log_level')

    def make_node(executable, name):
        return Node(
            package=package_name,
            executable=executable,
            name=name,
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', log_level],
        )

    return LaunchDescription([
        log_level_arg,
        make_node('lane_detection_node.py',      'lane_detection_node'),
        make_node('sign_detection_node.py',      'sign_detection_node'),
        make_node('vehicle_control_node.py',     'vehicle_control_node'),
        make_node('annotation_generator_node.py','annotation_generator_node'),
    ])