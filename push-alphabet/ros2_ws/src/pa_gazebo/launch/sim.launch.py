"""Push-Alphabet simulation: Gazebo Classic + table/pusher/camera + letters + hand.

    ros2 launch pa_gazebo sim.launch.py [gui:=false] [seed:=0] [scene_manager:=false] [hand_actor:=false]
        [pusher:=point|paddles]

The robot description and the ros2_control parameter file are generated here
from pa_core/scene.yaml (pa_description.urdf_gen), so geometry lives in one place.

pusher (default: scene.yaml pusher_mode):
  point    URDF without paddles and without ros2_control; point_push_executor
           spawns the orange 'point_pusher' model and executes /point_push_cmd.
  paddles  4 paddles + ros2_control (joint_state_broadcaster, paddle_controller)
           + push_executor for /push_cmd (as before).

gazebo_params (default: a generated file) makes gazebo_ros publish /clock at
CLOCK_HZ instead of 10 Hz, so sim-time timers (pusher motion) tick smoothly.
"""

import os
import tempfile

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, LogInfo,
                            OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable,
                            SetLaunchConfiguration)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pa_core.config import load_scene
from pa_description.urdf_gen import build_controllers_yaml, build_urdf
from pa_gazebo.hand_logic import HandConfig

PUSHERS = ("point", "paddles")
CLOCK_HZ = 100.0


def default_pusher() -> str:
    return str(load_scene().get("pusher_mode", "paddles")).strip().lower()


def pusher_actions(context, use_sim_time):
    """Robot description + pusher nodes for the `pusher` launch argument."""
    pusher = LaunchConfiguration("pusher").perform(context).strip().lower() or default_pusher()
    if pusher not in PUSHERS:
        raise RuntimeError(f"pusher:={pusher!r}: expected one of {PUSHERS}")
    paddles = pusher == "paddles"
    if paddles:
        # Generated ros2_control parameters (joint names from scene.yaml).
        fd, controllers = tempfile.mkstemp(prefix="pa_controllers_", suffix=".yaml")
        with os.fdopen(fd, "w") as f:
            f.write(build_controllers_yaml())
        robot_description = build_urdf(controllers_yaml=controllers)
    else:
        robot_description = build_urdf(paddles=False)

    rsp = Node(
        package="robot_state_publisher", executable="robot_state_publisher", output="screen",
        parameters=[{"robot_description": robot_description}, use_sim_time],
    )
    spawn_robot = Node(
        package="gazebo_ros", executable="spawn_entity.py", output="screen",
        arguments=["-topic", "robot_description", "-entity", "push_alphabet", "-timeout", "120"],
    )
    actions = [LogInfo(msg=f"[sim.launch] pusher: {pusher}"), rsp, spawn_robot]
    if paddles:
        jsb_spawner = Node(
            package="controller_manager", executable="spawner", output="screen",
            arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        )
        paddle_spawner = Node(
            package="controller_manager", executable="spawner", output="screen",
            arguments=["paddle_controller", "--controller-manager", "/controller_manager"],
        )
        actions += [
            RegisterEventHandler(OnProcessExit(target_action=spawn_robot, on_exit=[jsb_spawner])),
            RegisterEventHandler(OnProcessExit(target_action=jsb_spawner, on_exit=[paddle_spawner])),
            Node(package="pa_gazebo", executable="push_executor", output="screen", parameters=[use_sim_time]),
        ]
    else:
        actions.append(Node(package="pa_gazebo", executable="point_push_executor", output="screen",
                            parameters=[use_sim_time]))
    return actions


def generate_launch_description():
    gazebo_ros = get_package_share_directory("gazebo_ros")
    pa_gazebo = get_package_share_directory("pa_gazebo")
    pa_description = get_package_share_directory("pa_description")
    models = os.path.join(pa_description, "models")

    # gazebo_ros_init (node "gazebo") publishes /clock at 10 Hz by default.
    fd, gazebo_params = tempfile.mkstemp(prefix="pa_gazebo_params_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        f.write(f"gazebo:\n  ros__parameters:\n    publish_rate: {CLOCK_HZ}\n")

    gui = LaunchConfiguration("gui")
    verbose = LaunchConfiguration("verbose")
    seed = LaunchConfiguration("seed")
    use_sim_time = {"use_sim_time": True}
    park = HandConfig().park

    defaults = [d for d in ("/usr/share/gazebo-11/models", os.path.expanduser("~/.gazebo/models"))
                if os.path.isdir(d)]
    model_path = os.pathsep.join(p for p in [models, os.environ.get("GAZEBO_MODEL_PATH", "")] + defaults if p)
    # Gazebo/* material scripts live under /usr/share/gazebo-11; keep it on the
    # resource path even if the shell did not source gazebo's setup.sh.
    resources = [d for d in ("/usr/share/gazebo-11",) if os.path.isdir(d)]
    resource_path = os.pathsep.join(p for p in resources + [os.environ.get("GAZEBO_RESOURCE_PATH", "")] if p)

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(gazebo_ros, "launch", "gazebo.launch.py")),
        launch_arguments={
            "world": os.path.join(pa_gazebo, "worlds", "push_alphabet.world"),
            "gui": gui,
            "verbose": verbose,
            "pause": "false",
            "params_file": LaunchConfiguration("gazebo_params"),
        }.items(),
    )
    # gazebo_ros's launch files read launch configurations by name, and included
    # launch files share them. Our own `seed` (letter poses, -1 = random) and the
    # demo's `record` (mp4 recorder) would otherwise reach gzserver as
    # `--seed -1` (rejected: "Error. Invalid arguments") and `--record`.
    # Clear them inside a scoped group so our nodes still see the real values.
    gazebo = GroupAction(scoped=True, actions=[
        SetLaunchConfiguration("seed", ""),
        SetLaunchConfiguration("record", "false"),
        gazebo_launch,
    ])

    spawn_hand = Node(
        package="gazebo_ros", executable="spawn_entity.py", output="screen",
        arguments=["-file", os.path.join(models, "hand", "model.sdf"), "-entity", "hand",
                   "-x", str(park[0]), "-y", str(park[1]), "-z", str(park[2]), "-timeout", "120"],
        condition=IfCondition(LaunchConfiguration("hand_actor")),
    )
    scene_manager = Node(package="pa_gazebo", executable="scene_manager", output="screen",
                         parameters=[use_sim_time, {"seed": seed}],
                         condition=IfCondition(LaunchConfiguration("scene_manager")))
    hand_actor = Node(package="pa_gazebo", executable="hand_actor", output="screen",
                      parameters=[use_sim_time],
                      condition=IfCondition(LaunchConfiguration("hand_actor")))

    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="start gzclient"),
        DeclareLaunchArgument("verbose", default_value="false"),
        DeclareLaunchArgument("seed", default_value="-1", description="letter start-pose seed (-1 = random)"),
        DeclareLaunchArgument("scene_manager", default_value="true"),
        DeclareLaunchArgument("hand_actor", default_value="true"),
        DeclareLaunchArgument("pusher", default_value=default_pusher(),
                              description='"point" (Push-T style rod) | "paddles" (4 paddles + ros2_control); '
                                          'default = scene.yaml pusher_mode'),
        DeclareLaunchArgument("gazebo_params", default_value=gazebo_params,
                              description=f'gzserver ROS params file; default sets /clock to {CLOCK_HZ:.0f} Hz '
                                          '("" = gazebo_ros defaults, 10 Hz)'),
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path),
    ] + ([SetEnvironmentVariable("GAZEBO_RESOURCE_PATH", resource_path)] if resource_path else []) + [
        gazebo,
        OpaqueFunction(function=pusher_actions, args=[use_sim_time]),
        spawn_hand,
        scene_manager,
        hand_actor,
    ])
