import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory("pa_planner"), "config", "planner.yaml")
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("pusher", default_value="",
                              description='"paddles" | "point"; "" = scene.yaml pusher_mode'),
        Node(
            package="pa_planner",
            executable="planner_node",
            name="planner_node",
            output="screen",
            parameters=[params, {"use_sim_time": LaunchConfiguration("use_sim_time"),
                                 "pusher": ParameterValue(LaunchConfiguration("pusher"), value_type=str)}],
        ),
    ])
