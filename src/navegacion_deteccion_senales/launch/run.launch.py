from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    package_name = 'navegacion_deteccion_senales'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'params.yaml',
    ])

    return LaunchDescription([
        Node(
            package=package_name,
            executable='vehicle_control_node.py',
            name='vehicle_control_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', 'warn'],
        ),
        Node(
            package=package_name,
            executable='lane_detection_node.py',
            name='lane_detection_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        Node(
            package=package_name,
            executable='sign_detection_node.py',
            name='sign_detection_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', 'info'],
        ),
        Node(
            package=package_name,
            executable='annotation_generator_node.py',
            name='annotation_generator_node',
            output='screen',
            parameters=[params_file],
            arguments=['--ros-args', '--log-level', 'warn'],
        )
    ])