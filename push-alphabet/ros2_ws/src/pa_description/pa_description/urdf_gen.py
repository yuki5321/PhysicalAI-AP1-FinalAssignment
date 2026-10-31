"""Generate the robot description (table, walls, 4 paddles, overhead camera)
from pa_core's scene.yaml, so no dimension is written twice.

`paddles=False` (pusher_mode "point") leaves the paddles and the whole
ros2_control block out; the point pusher is a separate SDF model
(pa_description.pusher_model) moved by pa_gazebo point_push_executor. The
walls stay where they are in both modes (same world as pa_core.pushsim).

Paddle color: in the user's first Gazebo run the paddles rendered white
(Gazebo/Orange did not take effect) and perception took a resting paddle for
an "I". The paddle links therefore also get explicit ambient/diffuse through a
<visual> blob in their <gazebo reference> block (sdformat copies it into the
link's visuals); the table is left as it was (perception is tuned to it).

World frame (scene.yaml): table center at the origin, the table top surface at
z = 0, +y = "up" of the letters. The robot is fixed to the Gazebo world.

Used at launch time (pa_gazebo/launch/sim.launch.py) and by the `gen_urdf`
console script:  ros2 run pa_description gen_urdf > /tmp/pa.urdf
"""

import argparse
import math
import sys

from pa_core.config import load_scene

PLATE_THICKNESS = 0.02     # table plate, below z = 0
WALL_THICKNESS = 0.01
WALL_HEIGHT = 0.02         # taller than a letter, so letters cannot climb over
WALL_MARGIN = 0.01         # gap between the farthest paddle sweep and the walls
ARM_GAP = 0.0015           # paddle bottom above the table top
PADDLE_MASS = 0.2         # heavy enough that letters barely deflect the driven paddle
CAMERA_UPDATE_HZ = 15.0
PADDLE_RGB = (0.95, 0.42, 0.05)   # orange: perception's occluder mask hides it

# Overhead camera link orientation. A Gazebo camera looks along its link +x,
# with image "up" = link +z and image "right" = link -y. Pitch +90 deg then
# yaw +90 deg gives: link x -> world -z (looking down), link z -> world +y
# (image up), link -y -> world +x (image right), as docs/interfaces.md requires.
CAMERA_RPY = (0.0, math.pi / 2, math.pi / 2)
# REP-103 optical frame (z forward, x right, y down) relative to the camera link.
OPTICAL_RPY = (-math.pi / 2, 0.0, -math.pi / 2)


def joint_name(pusher_name: str) -> str:
    return f"pusher_{pusher_name}_joint"


def paddle_link_name(pusher_name: str) -> str:
    return f"pusher_{pusher_name}"


def joint_names(scene=None) -> list:
    scene = scene or load_scene()
    return [joint_name(p["name"]) for p in scene["pushers"]["list"]]


def max_sweep_extent(scene=None, step_deg: float = 1.0) -> float:
    """Largest |x| or |y| reached by any paddle (with its thickness) over 0..max_angle."""
    scene = scene or load_scene()
    sp = scene["pushers"]
    L, r = sp["arm_length"], sp["arm_thickness"] / 2
    ext = 0.0
    n = int(sp["max_angle"] / step_deg) + 1
    for p in sp["list"]:
        px, py = p["pivot"]
        for i in range(n):
            a = math.radians(p["rest_heading"] + i * step_deg)
            c, s = math.cos(a), math.sin(a)
            for u, v in ((0, -r), (0, r), (L, -r), (L, r)):
                x = px + u * c - v * s
                y = py + u * s + v * c
                ext = max(ext, abs(x), abs(y))
    return ext


def wall_inner_half(scene=None) -> float:
    """Inner half-size of the perimeter walls: outside the table plate and every paddle sweep."""
    scene = scene or load_scene()
    return max(scene["workspace"]["table_half_size"], max_sweep_extent(scene) + WALL_MARGIN)


def _f(v: float) -> str:
    return f"{v:.6g}"


def _xyz(x, y, z) -> str:
    return f"{_f(x)} {_f(y)} {_f(z)}"


def _box_inertia(m, sx, sy, sz):
    return (m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12)


def _inertial(m, sx, sy, sz, xyz=(0, 0, 0)) -> str:
    ixx, iyy, izz = _box_inertia(m, sx, sy, sz)
    return (f'    <inertial>\n      <origin xyz="{_xyz(*xyz)}" rpy="0 0 0"/>\n      <mass value="{_f(m)}"/>\n'
            f'      <inertia ixx="{ixx:.4e}" ixy="0" ixz="0" iyy="{iyy:.4e}" iyz="0" izz="{izz:.4e}"/>\n'
            f'    </inertial>\n')


def _box(name, xyz, size, material=None, collision=True) -> str:
    mat = f'\n      <material name="{material}"/>' if material else ""
    out = (f'    <visual name="{name}_visual">\n      <origin xyz="{_xyz(*xyz)}" rpy="0 0 0"/>\n'
           f'      <geometry><box size="{_xyz(*size)}"/></geometry>{mat}\n    </visual>\n')
    if collision:
        out += (f'    <collision name="{name}_collision">\n      <origin xyz="{_xyz(*xyz)}" rpy="0 0 0"/>\n'
                f'      <geometry><box size="{_xyz(*size)}"/></geometry>\n    </collision>\n')
    return out


def _rgba(c) -> str:
    return f"{c[0]:.3f} {c[1]:.3f} {c[2]:.3f} 1"


def _gazebo_color(rgb, emissive=None) -> str:
    """<visual> blob for a <gazebo reference> block: explicit colors that do not
    depend on Gazebo's material scripts being found."""
    em = f"<emissive>{_rgba(emissive)}</emissive>" if emissive else ""
    return (f'    <visual>\n      <material><ambient>{_rgba(rgb)}</ambient><diffuse>{_rgba(rgb)}</diffuse>'
            f'<specular>0.1 0.1 0.1 1</specular>{em}</material>\n    </visual>')


def build_urdf(scene=None, controllers_yaml: str | None = None, paddles: bool = True) -> str:
    """URDF string. `controllers_yaml` is the ros2_control parameter file passed to
    the gazebo_ros2_control plugin; without it the plugin block is omitted (RViz use).
    `paddles=False`: no paddle links/joints and no ros2_control at all (point pusher)."""
    scene = scene or load_scene()
    ws = scene["workspace"]
    sp = scene["pushers"]
    cam = scene["camera"]

    inner = wall_inner_half(scene)
    outer = inner + WALL_THICKNESS
    T = PLATE_THICKNESS

    parts = ['<?xml version="1.0"?>',
             '<!-- Generated by pa_description.urdf_gen from pa_core/scene.yaml. Do not edit. -->',
             '<robot name="push_alphabet">',
             '  <material name="table_dark"><color rgba="0.18 0.18 0.2 1"/></material>',
             '  <material name="table_top"><color rgba="0.12 0.13 0.16 1"/></material>',
             '  <material name="paddle"><color rgba="0.85 0.4 0.1 1"/></material>',
             '  <material name="hub"><color rgba="0.1 0.1 0.1 1"/></material>',
             '',
             '  <link name="world"/>',
             '  <joint name="world_to_table" type="fixed">',
             '    <parent link="world"/>',
             '    <child link="table"/>',
             '    <origin xyz="0 0 0" rpy="0 0 0"/>',
             '  </joint>',
             '']

    # Table: plate (top at z = 0) + four low walls.
    table = ['  <link name="table">', _inertial(5.0, 2 * outer, 2 * outer, T, (0, 0, -T / 2)).rstrip("\n")]
    table.append(_box("plate", (0, 0, -T / 2), (2 * outer, 2 * outer, T), "table_dark").rstrip("\n"))
    # Visible table area (table_half_size), visual only, a hair above the plate.
    table.append(_box("table_top", (0, 0, 0.0002), (2 * ws["table_half_size"], 2 * ws["table_half_size"], 0.0004),
                      "table_top", collision=False).rstrip("\n"))
    zc = WALL_HEIGHT / 2
    c = inner + WALL_THICKNESS / 2
    walls = {
        "wall_south": ((0, -c, zc), (2 * outer, WALL_THICKNESS, WALL_HEIGHT)),
        "wall_north": ((0, c, zc), (2 * outer, WALL_THICKNESS, WALL_HEIGHT)),
        "wall_west": ((-c, 0, zc), (WALL_THICKNESS, 2 * inner, WALL_HEIGHT)),
        "wall_east": ((c, 0, zc), (WALL_THICKNESS, 2 * inner, WALL_HEIGHT)),
    }
    for name, (xyz, size) in walls.items():
        table.append(_box(name, xyz, size, "table_dark").rstrip("\n"))
    table.append('  </link>')
    parts += table
    parts.append('  <gazebo reference="table">\n    <material>Gazebo/DarkGrey</material>\n'
                 '    <mu1>0.3</mu1>\n    <mu2>0.3</mu2>\n  </gazebo>')

    # Paddles: revolute about z at the pivot, 0 rad = rest_heading, + = CCW into the workspace.
    L, th, h = sp["arm_length"], sp["arm_thickness"], sp["arm_height"]
    upper = math.radians(sp["max_angle"])
    for p in (sp["list"] if paddles else []):
        link = paddle_link_name(p["name"])
        px, py = p["pivot"]
        yaw = math.radians(p["rest_heading"])
        parts += [
            '',
            f'  <link name="{link}">',
            _inertial(PADDLE_MASS, L, th, h, (L / 2, 0, 0)).rstrip("\n"),
            _box("arm", (L / 2, 0, 0), (L, th, h), "paddle").rstrip("\n"),
            f'    <visual name="hub_visual">\n      <origin xyz="0 0 {_f(h / 2 + 0.003)}" rpy="0 0 0"/>\n'
            f'      <geometry><cylinder radius="{_f(1.6 * th)}" length="0.006"/></geometry>\n'
            f'      <material name="hub"/>\n    </visual>',
            '  </link>',
            f'  <joint name="{joint_name(p["name"])}" type="revolute">',
            '    <parent link="table"/>',
            f'    <child link="{link}"/>',
            f'    <origin xyz="{_xyz(px, py, ARM_GAP + h / 2)}" rpy="0 0 {_f(yaw)}"/>',
            '    <axis xyz="0 0 1"/>',
            f'    <limit lower="-0.05" upper="{_f(upper + 0.05)}" effort="10.0" velocity="20.0"/>',
            '    <dynamics damping="0.0" friction="0.0"/>',
            '  </joint>',
            f'  <gazebo reference="{link}">\n    <material>Gazebo/Orange</material>\n'
            + _gazebo_color(PADDLE_RGB, (0.3, 0.12, 0.0)) + '\n'
            '    <mu1>0.5</mu1>\n    <mu2>0.5</mu2>\n    <kp>1e5</kp>\n    <kd>10</kd>\n'
            '    <minDepth>0.0005</minDepth>\n    <maxVel>0.1</maxVel>\n  </gazebo>',
        ]

    # Overhead camera.
    hfov = math.radians(cam["hfov_deg"])
    parts += [
        '',
        '  <link name="overhead_camera_link">',
        _inertial(0.01, 0.02, 0.02, 0.02).rstrip("\n"),
        '  </link>',
        '  <joint name="overhead_camera_joint" type="fixed">',
        '    <parent link="table"/>',
        '    <child link="overhead_camera_link"/>',
        f'    <origin xyz="0 0 {_f(cam["height"])}" rpy="{_xyz(*CAMERA_RPY)}"/>',
        '  </joint>',
        '  <link name="overhead_camera_optical"/>',
        '  <joint name="overhead_camera_optical_joint" type="fixed">',
        '    <parent link="overhead_camera_link"/>',
        '    <child link="overhead_camera_optical"/>',
        f'    <origin xyz="0 0 0" rpy="{_xyz(*OPTICAL_RPY)}"/>',
        '  </joint>',
        '  <gazebo reference="overhead_camera_link">',
        '    <sensor type="camera" name="overhead_camera">',
        '      <always_on>true</always_on>',
        f'      <update_rate>{_f(CAMERA_UPDATE_HZ)}</update_rate>',
        '      <visualize>false</visualize>',
        '      <camera name="overhead">',
        f'        <horizontal_fov>{hfov:.6f}</horizontal_fov>',
        '        <image>',
        f'          <width>{int(cam["width"])}</width>',
        f'          <height>{int(cam["height_px"])}</height>',
        '          <format>B8G8R8</format>',
        '        </image>',
        '        <clip><near>0.05</near><far>3.0</far></clip>',
        '      </camera>',
        '      <plugin name="overhead_camera_controller" filename="libgazebo_ros_camera.so">',
        '        <ros><namespace>/</namespace></ros>',
        '        <camera_name>overhead_camera</camera_name>',
        '        <frame_name>overhead_camera_optical</frame_name>',
        '      </plugin>',
        '    </sensor>',
        '  </gazebo>',
    ]

    # ros2_control: velocity command per paddle. The trajectory controller closes the
    # position loop (PID + velocity feed-forward). Gazebo's position interface teleports
    # the joint (zero link velocity), which makes contact pushing unreliable; a velocity
    # interface moves the paddle like the kinematic arm of the 2D model.
    if not paddles:
        parts.append('</robot>')
        return "\n".join(parts) + "\n"
    parts += ['', '  <ros2_control name="PaddleSystem" type="system">',
              '    <hardware>', '      <plugin>gazebo_ros2_control/GazeboSystem</plugin>', '    </hardware>']
    for jn in joint_names(scene):
        parts += [f'    <joint name="{jn}">',
                  '      <command_interface name="velocity">',
                  '        <param name="min">-20.0</param>',
                  '        <param name="max">20.0</param>',
                  '      </command_interface>',
                  '      <state_interface name="position">', '        <param name="initial_value">0.0</param>',
                  '      </state_interface>',
                  '      <state_interface name="velocity"/>',
                  '    </joint>']
    parts.append('  </ros2_control>')
    if controllers_yaml:
        parts += ['  <gazebo>',
                  '    <plugin name="gazebo_ros2_control" filename="libgazebo_ros2_control.so">',
                  f'      <parameters>{controllers_yaml}</parameters>',
                  '    </plugin>',
                  '  </gazebo>']
    parts.append('</robot>')
    return "\n".join(parts) + "\n"


def build_controllers_yaml(scene=None, update_rate: int = 500, kp: float = 20.0) -> str:
    """ros2_control parameter file for the paddles (joint names come from scene.yaml)."""
    joints = joint_names(scene)
    jl = "\n".join(f"      - {j}" for j in joints)
    gains = "\n".join(f"      {j}: {{p: {kp}, i: 0.0, d: 0.0, i_clamp: 0.0, ff_velocity_scale: 1.0}}" for j in joints)
    return f"""# Generated by pa_description.urdf_gen from pa_core/scene.yaml. Do not edit.
controller_manager:
  ros__parameters:
    update_rate: {update_rate}
    use_sim_time: true
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    paddle_controller:
      type: joint_trajectory_controller/JointTrajectoryController

paddle_controller:
  ros__parameters:
    use_sim_time: true
    joints:
{jl}
    command_interfaces:
      - velocity
    state_interfaces:
      - position
      - velocity
    allow_partial_joints_goal: true
    state_publish_rate: 50.0
    action_monitor_rate: 20.0
    constraints:
      stopped_velocity_tolerance: 0.0
      goal_time: 0.0
    gains:
{gains}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description="Print the Push-Alphabet URDF (or controllers yaml).")
    ap.add_argument("--controllers", default=None, help="path of the ros2_control yaml to reference")
    ap.add_argument("--print-controllers", action="store_true", help="print the controllers yaml instead")
    ap.add_argument("--no-paddles", action="store_true", help="point pusher: no paddles, no ros2_control")
    args = ap.parse_args(argv)
    if args.print_controllers:
        sys.stdout.write(build_controllers_yaml())
    else:
        sys.stdout.write(build_urdf(controllers_yaml=args.controllers, paddles=not args.no_paddles))


if __name__ == "__main__":
    main()
