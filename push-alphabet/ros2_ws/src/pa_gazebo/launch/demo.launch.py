"""Full Push-Alphabet demo in one command (for the submission video).

    ros2 launch pa_gazebo demo.launch.py [gui:=true] [seed:=-1] [method:=hybrid]
        [record:=true] [out_dir:=...] [auto_hand:=false|auto|always] [pusher:=point|paddles]
        [sequence:=IUYK] [display_word:=YUKI] [rosbridge:=true] [port:=9090] [label:="..."]

Starts:
  sim.launch.py (Gazebo + table/camera + pusher, hand model + hand_actor)
      pusher:=point   orange point pusher + point_push_executor (/point_push_cmd)
      pusher:=paddles 4 paddles + ros2_control + push_executor (/push_cmd)
      "" (default) = scene.yaml pusher_mode; the same value goes to planner_node.
  scene_manager (with seed / sequence / display_word)
  letter_detector (method:=hybrid falls back to classic if onnxruntime / the model is missing)
  planner_node
  rosbridge_websocket on `port` (host/hand_bridge.py connects here)
  demo_recorder (record:=true writes <out_dir>/demo_<time>.mp4 + _raw.mp4;
      out_dir "" = /ws/src/push-alphabet/videos, i.e. <repo>/videos on the host)
  auto_hand (default false = not started; auto_hand:=auto): scripted hand sweeps through each letter while it
      is being recovered, but only while no webcam bridge publishes /hand/state;
      always = ignore the webcam; false = do not start it.
"""

import os

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.conditions import IfCondition, LaunchConfigurationNotEquals
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pa_gazebo = get_package_share_directory("pa_gazebo")
    pa_perception = get_package_share_directory("pa_perception")
    pa_planner = get_package_share_directory("pa_planner")
    use_sim_time = {"use_sim_time": True}

    def arg(name):
        return LaunchConfiguration(name)

    def as_str(name):
        return ParameterValue(LaunchConfiguration(name), value_type=str)

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pa_gazebo, "launch", "sim.launch.py")),
        launch_arguments={
            "gui": arg("gui"),
            "verbose": arg("verbose"),
            "seed": arg("seed"),
            "scene_manager": "false",   # started below with sequence / display_word
            "hand_actor": arg("hand_actor"),
            "pusher": arg("pusher"),
        }.items(),
    )
    scene_manager = Node(
        package="pa_gazebo", executable="scene_manager", output="screen",
        parameters=[use_sim_time, {
            "seed": ParameterValue(arg("seed"), value_type=int),
            "sequence": as_str("sequence"),
            "display_word": as_str("display_word"),
        }],
    )
    perception = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pa_perception, "launch", "perception.launch.py")),
        launch_arguments={
            "method": arg("method"),
            "model_path": arg("model_path"),
            "min_confidence": arg("min_confidence"),
            "publish_debug": "false",
            "use_sim_time": "true",
        }.items(),
    )
    planner = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pa_planner, "launch", "planner.launch.py")),
        launch_arguments={"use_sim_time": "true", "pusher": arg("pusher")}.items(),
    )
    try:
        rb = os.path.join(get_package_share_directory("rosbridge_server"), "launch",
                          "rosbridge_websocket_launch.xml")
        rosbridge = IncludeLaunchDescription(
            AnyLaunchDescriptionSource(rb),
            launch_arguments={"port": arg("port")}.items(),
            condition=IfCondition(arg("rosbridge")),
        )
    except PackageNotFoundError:
        rosbridge = LogInfo(msg="rosbridge_server is not installed: the webcam hand bridge cannot connect "
                                "(sudo apt install ros-humble-rosbridge-server)")
    recorder = Node(
        package="pa_gazebo", executable="demo_recorder", output="screen",
        parameters=[use_sim_time, {
            "record": ParameterValue(arg("record"), value_type=bool),
            "out_dir": as_str("out_dir"),
            "fps": ParameterValue(arg("fps"), value_type=float),
            "write_raw": ParameterValue(arg("write_raw"), value_type=bool),
            "word": as_str("display_word"),
            "label": as_str("label"),
        }],
    )
    auto_hand = Node(
        package="pa_gazebo", executable="auto_hand", name="auto_hand", output="screen",
        parameters=[use_sim_time, {
            "mode": as_str("auto_hand"),
            "seed": ParameterValue(arg("seed"), value_type=int),
            "knocks_per_letter": ParameterValue(arg("knocks_per_letter"), value_type=int),
        }],
        condition=LaunchConfigurationNotEquals("auto_hand", "false"),
    )

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="start gzclient"),
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("seed", default_value="-1", description="letter start poses / hand script (-1 = random)"),
        DeclareLaunchArgument("sequence", default_value="", description='work order, "" = scene.yaml (IUYK)'),
        DeclareLaunchArgument("display_word", default_value="", description='shelf word, "" = scene.yaml (YUKI)'),
        DeclareLaunchArgument("method", default_value="hybrid",
                              description="detector: classic | cnn | hybrid (falls back to classic)"),
        DeclareLaunchArgument("model_path", default_value="", description="ONNX model, empty = installed default"),
        DeclareLaunchArgument("min_confidence", default_value="0.5"),
        DeclareLaunchArgument("record", default_value="true", description="write the overlay / raw MP4s"),
        DeclareLaunchArgument("out_dir", default_value="",
                              description='video directory, "" = /ws/src/push-alphabet/videos (or ~/pa_demo)'),
        DeclareLaunchArgument("fps", default_value="15.0", description="video fps in sim time (camera: 15 Hz)"),
        DeclareLaunchArgument("write_raw", default_value="true", description="also write plain camera frames"),
        DeclareLaunchArgument("label", default_value="",
                              description='extra HUD text, e.g. "detector: hybrid (CNN)"'),
        DeclareLaunchArgument("auto_hand", default_value="false",
                              description="false | auto (scripted unless a webcam bridge publishes) | always"),
        DeclareLaunchArgument("hand_actor", default_value="true",
                              description="spawn the hand model + hand_actor (parked unless /hand/state says present)"),
        DeclareLaunchArgument("pusher", default_value="",
                              description='"point" | "paddles"; "" = scene.yaml pusher_mode (sim + planner)'),
        DeclareLaunchArgument("knocks_per_letter", default_value="1"),
        DeclareLaunchArgument("rosbridge", default_value="true"),
        DeclareLaunchArgument("port", default_value="9090", description="rosbridge websocket port"),
        sim,
        scene_manager,
        perception,
        planner,
        rosbridge,
        recorder,
        auto_hand,
    ])
