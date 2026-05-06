from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    package_name = 'navegacion_deteccion_senales'
    params_file = PathJoinSubstitution([
        FindPackageShare(package_name),
        'config',
        'lane_follower.yaml',
    ])

    return LaunchDescription([
        Node(
            package=package_name,
            executable='lane_follower.py',
            name='lane_follower',
            output='screen',
            parameters=[params_file],
        ),
    ])