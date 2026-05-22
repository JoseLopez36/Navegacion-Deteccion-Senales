from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch para recolección de dataset de carriles.
    """
    package_name = 'navegacion_deteccion_senales'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'params.yaml',
    ])

    return LaunchDescription([
        Node(
            package=package_name,
            executable='lane_dataset_node.py',
            name='lane_dataset_node',
            output='screen',
            parameters=[params_file],
        ),
    ])