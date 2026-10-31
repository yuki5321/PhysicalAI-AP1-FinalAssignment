"""Simulation for tools/eval_recovery.py --backend gazebo (docs/recovery_eval.md).

    ros2 launch pa_gazebo eval.launch.py [gui:=false] [method:=classic|cnn|hybrid] [hand:=true]

= sim.launch.py without the scene manager and without hand_actor (the eval tool
places the letters and drives the hand itself), plus the letter detector.
planner_node is NOT started: the tool runs the planner state machine.
pusher defaults to "paddles" here because the eval tool sends /push_cmd.
The hand model is still spawned (parked out of view) when hand:=true, for
`--disturb hand`.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pa_gazebo.hand_logic import HandConfig


def generate_launch_description():
    pa_gazebo = get_package_share_directory("pa_gazebo")
    pa_perception = get_package_share_directory("pa_perception")
    pa_description = get_package_share_directory("pa_description")
    park = HandConfig().park

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pa_gazebo, "launch", "sim.launch.py")),
        launch_arguments={
            "gui": LaunchConfiguration("gui"),
            "verbose": LaunchConfiguration("verbose"),
            "scene_manager": "false",
            "hand_actor": "false",
            "pusher": LaunchConfiguration("pusher"),
        }.items(),
    )
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pa_perception, "launch", "perception.launch.py")),
        launch_arguments={
            "method": LaunchConfiguration("method"),
            "model_path": LaunchConfiguration("model_path"),
            "min_confidence": LaunchConfiguration("min_confidence"),
            "publish_debug": "false",
            "use_sim_time": "true",
        }.items(),
    )
    spawn_hand = Node(
        package="gazebo_ros", executable="spawn_entity.py", output="screen",
        arguments=["-file", os.path.join(pa_description, "models", "hand", "model.sdf"), "-entity", "hand",
                   "-x", str(park[0]), "-y", str(park[1]), "-z", str(park[2]), "-timeout", "120"],
        condition=IfCondition(LaunchConfiguration("hand")),
    )

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="false", description="start gzclient"),
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("method", default_value="classic", description="detector: classic | cnn | hybrid"),
        DeclareLaunchArgument("model_path", default_value="",
                              description="ONNX model for cnn / hybrid; empty = installed default"),
        DeclareLaunchArgument("min_confidence", default_value="0.5"),
        DeclareLaunchArgument("hand", default_value="true", description="spawn the hand model (for --disturb hand)"),
        DeclareLaunchArgument("pusher", default_value="paddles",
                              description="tools/recovery_gazebo.py drives the paddles (/push_cmd); "
                                          '"point" or "" (scene.yaml) for the point pusher'),
        sim,
        perception,
        spawn_hand,
    ])
