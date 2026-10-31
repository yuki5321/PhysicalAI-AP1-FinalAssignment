"""Start the letter detector. Examples:
ros2 launch pa_perception perception.launch.py threshold:=-1 sides:=auto
ros2 launch pa_perception perception.launch.py method:=hybrid
ros2 launch pa_perception perception.launch.py method:=cnn model_path:=/path/to/letter_cnn.onnx

method:=cnn|hybrid needs onnxruntime and the model (default: the installed
share/pa_perception/models/letter_cnn.onnx); if either is missing the node
warns and runs the classic detector.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("threshold", default_value="-1",
                              description="gray threshold 0..255, -1 = automatic"),
        DeclareLaunchArgument("sides", default_value="auto",
                              description="silhouette model: auto | top | extruded"),
        DeclareLaunchArgument("method", default_value="classic",
                              description="detector: classic | cnn | hybrid (cnn + classic IoU refinement)"),
        DeclareLaunchArgument("model_path", default_value="",
                              description="ONNX model; empty = share/pa_perception/models/letter_cnn.onnx"),
        DeclareLaunchArgument("min_confidence", default_value="0.5",
                              description="cnn: minimum softmax confidence"),
        DeclareLaunchArgument("publish_debug", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
    ]
    node = Node(
        package="pa_perception",
        executable="letter_detector",
        name="letter_detector",
        output="screen",
        parameters=[{
            "threshold": ParameterValue(LaunchConfiguration("threshold"), value_type=int),
            "sides": LaunchConfiguration("sides"),
            "method": LaunchConfiguration("method"),
            "model_path": ParameterValue(LaunchConfiguration("model_path"), value_type=str),
            "min_confidence": ParameterValue(LaunchConfiguration("min_confidence"), value_type=float),
            "publish_debug": ParameterValue(LaunchConfiguration("publish_debug"), value_type=bool),
            "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
        }],
    )
    return LaunchDescription(args + [node])
