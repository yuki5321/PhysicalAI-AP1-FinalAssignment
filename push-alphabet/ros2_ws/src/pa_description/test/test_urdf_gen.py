import math
import xml.etree.ElementTree as ET

import numpy as np
import yaml

from pa_core.config import load_scene
from pa_description import urdf_gen


def rpy_matrix(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def test_urdf_is_well_formed_and_has_paddle_joints():
    root = ET.fromstring(urdf_gen.build_urdf(controllers_yaml="/tmp/c.yaml"))
    assert root.tag == "robot"
    joints = {j.get("name"): j for j in root.findall("joint")}
    scene = load_scene()
    for p in scene["pushers"]["list"]:
        j = joints[f"pusher_{p['name']}_joint"]
        assert j.get("type") == "revolute"
        assert j.find("axis").get("xyz") == "0 0 1"
        x, y, _ = map(float, j.find("origin").get("xyz").split())
        assert (x, y) == tuple(p["pivot"])
        yaw = float(j.find("origin").get("rpy").split()[2])
        assert math.isclose(yaw, math.radians(p["rest_heading"]), abs_tol=1e-5)
        assert float(j.find("limit").get("upper")) >= math.radians(scene["pushers"]["max_angle"])
    rc = root.find("ros2_control")
    assert {j.get("name") for j in rc.findall("joint")} == set(urdf_gen.joint_names())
    assert root.find("gazebo/plugin[@filename='libgazebo_ros2_control.so']/parameters").text == "/tmp/c.yaml"


def test_camera_sensor_matches_scene():
    root = ET.fromstring(urdf_gen.build_urdf())
    cam = load_scene()["camera"]
    sensor = root.find("gazebo[@reference='overhead_camera_link']/sensor")
    assert sensor.get("type") == "camera"
    c = sensor.find("camera")
    assert math.isclose(float(c.find("horizontal_fov").text), math.radians(cam["hfov_deg"]), rel_tol=1e-5)
    assert int(c.find("image/width").text) == cam["width"]
    assert int(c.find("image/height").text) == cam["height_px"]
    assert c.find("image/format").text == "B8G8R8"
    assert sensor.find("plugin/camera_name").text == "overhead_camera"
    j = [j for j in root.findall("joint") if j.get("name") == "overhead_camera_joint"][0]
    assert float(j.find("origin").get("xyz").split()[2]) == cam["height"]


def test_camera_orientation_image_right_is_world_x_and_up_is_world_y():
    R = rpy_matrix(*urdf_gen.CAMERA_RPY)
    look, link_y, link_z = R[:, 0], R[:, 1], R[:, 2]
    np.testing.assert_allclose(look, [0, 0, -1], atol=1e-9)    # looks down
    np.testing.assert_allclose(link_z, [0, 1, 0], atol=1e-9)   # image up = world +y
    np.testing.assert_allclose(-link_y, [1, 0, 0], atol=1e-9)  # image right = world +x


def test_optical_frame_is_rep103():
    R = rpy_matrix(*urdf_gen.CAMERA_RPY) @ rpy_matrix(*urdf_gen.OPTICAL_RPY)
    np.testing.assert_allclose(R[:, 0], [1, 0, 0], atol=1e-9)   # optical x = image right
    np.testing.assert_allclose(R[:, 1], [0, -1, 0], atol=1e-9)  # optical y = image down
    np.testing.assert_allclose(R[:, 2], [0, 0, -1], atol=1e-9)  # optical z = view direction


def test_walls_clear_every_paddle_sweep_and_rest_paddles_clear_workspace():
    scene = load_scene()
    assert urdf_gen.wall_inner_half(scene) > urdf_gen.max_sweep_extent(scene)
    ws = scene["workspace"]["half_size"]
    r = scene["pushers"]["arm_thickness"] / 2
    for p in scene["pushers"]["list"]:
        a = math.radians(p["rest_heading"])
        # The resting arm's inner face lies outside the workspace square.
        n = np.array([-math.sin(a), math.cos(a)])  # CCW normal = sweep direction
        px, py = p["pivot"]
        inner_face = np.array([px, py]) + n * r
        assert max(abs(inner_face[0]), abs(inner_face[1])) >= ws


def test_controllers_yaml_lists_joints():
    data = yaml.safe_load(urdf_gen.build_controllers_yaml())
    pc = data["paddle_controller"]["ros__parameters"]
    assert pc["joints"] == urdf_gen.joint_names()
    assert pc["command_interfaces"] == ["velocity"]
    assert set(pc["gains"]) == set(urdf_gen.joint_names())
    assert all(g["ff_velocity_scale"] == 1.0 and g["p"] > 0 for g in pc["gains"].values())
    types = data["controller_manager"]["ros__parameters"]
    assert types["paddle_controller"]["type"] == "joint_trajectory_controller/JointTrajectoryController"


def test_urdf_without_paddles_has_no_paddles_and_no_ros2_control():
    text = urdf_gen.build_urdf(controllers_yaml="/tmp/c.yaml", paddles=False)
    root = ET.fromstring(text)
    assert root.tag == "robot"
    assert root.find("ros2_control") is None
    assert "gazebo_ros2_control" not in text and "PaddleSystem" not in text
    links = {link.get("name") for link in root.findall("link")}
    assert {"world", "table", "overhead_camera_link", "overhead_camera_optical"} <= links
    assert not any(n.startswith("pusher_") for n in links)
    assert all(j.get("type") == "fixed" for j in root.findall("joint"))
    # camera and walls are the same as with paddles
    full = ET.fromstring(urdf_gen.build_urdf())
    assert ET.tostring(root.find("gazebo[@reference='overhead_camera_link']/sensor")) == \
        ET.tostring(full.find("gazebo[@reference='overhead_camera_link']/sensor"))
    t1 = [ET.tostring(v) for v in root.find("link[@name='table']")]
    t2 = [ET.tostring(v) for v in full.find("link[@name='table']")]
    assert t1 == t2


def test_paddles_have_explicit_orange_for_gazebo():
    root = ET.fromstring(urdf_gen.build_urdf())
    for p in load_scene()["pushers"]["list"]:
        g = root.find(f"gazebo[@reference='{urdf_gen.paddle_link_name(p['name'])}']")
        mat = g.find("visual/material")
        amb = [float(v) for v in mat.find("ambient").text.split()]
        dif = [float(v) for v in mat.find("diffuse").text.split()]
        for c in (amb, dif):
            assert c[0] > 0.8 and 0.25 < c[1] < 0.6 and c[2] < 0.2   # orange, saturated


def test_main_no_paddles(capsys):
    urdf_gen.main(["--no-paddles"])
    root = ET.fromstring(capsys.readouterr().out)
    assert root.find("ros2_control") is None
