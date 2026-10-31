"""Pure-python parts of tools/recovery_gazebo.py (no ROS needed).

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_gazebo:ros2_ws/src/pa_perception \\
        python3 -m pytest tools/test_recovery_gazebo.py
"""

import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import recovery_gazebo as rg  # noqa: E402
from pa_core.config import load_scene  # noqa: E402
from pa_core.letters import is_disturbed, letter_polygons  # noqa: E402

SCENE = load_scene()
WS = SCENE["workspace"]["half_size"]
TABLE = SCENE["workspace"]["table_half_size"]


def test_import_does_not_pull_ros():
    assert "rclpy" not in sys.modules


# ------------------------------------------------------------ quaternion -> yaw
@pytest.mark.parametrize("yaw", [0.0, 0.3, -1.2, math.pi / 2, 3.0, -3.0])
def test_yaw_roundtrip(yaw):
    q = rg.yaw_to_quaternion(yaw)
    assert rg.quaternion_to_yaw(*q) == pytest.approx(yaw, abs=1e-9)
    assert rg.quaternion_tilt(*q) == pytest.approx(0.0, abs=1e-7)


def test_yaw_of_tilted_quaternion_and_tilt():
    # yaw 0.7 then a 0.5 rad roll: yaw stays 0.7 (ZYX), tilt = 0.5
    cy, sy = math.cos(0.35), math.sin(0.35)
    cr, sr = math.cos(0.25), math.sin(0.25)
    q = (sr * cy, sr * sy, cr * sy, cr * cy)  # q_yaw * q_roll, (x, y, z, w)
    assert rg.quaternion_to_yaw(*q) == pytest.approx(0.7, abs=1e-9)
    assert rg.quaternion_tilt(*q) == pytest.approx(0.5, abs=1e-9)


def test_letter_status():
    assert rg.letter_status(0.0, 0.0, 0.0, 0.0, WS, TABLE) == "ok"
    assert rg.letter_status(WS + 0.01, 0.0, 0.0, 0.0, WS, TABLE) == "outside"
    assert rg.letter_status(0.0, TABLE + 0.01, 0.0, 0.0, WS, TABLE) == "off_table"
    assert rg.letter_status(0.0, 0.0, -0.02, 0.0, WS, TABLE) == "fell"
    assert rg.letter_status(0.0, 0.0, 0.01, 1.4, WS, TABLE) == "tipped"
    assert set(rg.UNRECOVERABLE) == {"off_table", "fell", "tipped"}


# ------------------------------------------------------------ pose sampling
@pytest.mark.parametrize("rng", [random.Random(3), np.random.default_rng(3)], ids=["random", "numpy"])
@pytest.mark.parametrize("letter", ["I", "U", "Y", "K"])
def test_teleport_pose_is_disturbed_and_inside(letter, rng):
    for _ in range(20):
        x, y, th = rg.sample_teleport_pose(letter, rng)
        assert is_disturbed(letter, x, y, th)
        pts = np.vstack(letter_polygons(letter, x, y, th))
        assert np.all(np.abs(pts) <= WS)
        assert all(isinstance(v, float) for v in (x, y, th))


def test_teleport_pose_deterministic_for_seed():
    a = [rg.sample_teleport_pose("K", np.random.default_rng(7)) for _ in range(2)]
    assert a[0] == a[1]


@pytest.mark.parametrize("letter", ["I", "U", "Y", "K"])
def test_hand_stroke_goes_through_the_letter(letter):
    rng = np.random.default_rng(11)
    r = rg.letter_radius(letter)
    for _ in range(50):
        pose = (rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(-math.pi, math.pi))
        s = rg.sample_hand_stroke(letter, pose, rng, WS)
        sx, sy = s.start
        ex, ey = s.end
        assert max(abs(sx), abs(sy), abs(ex), abs(ey)) <= WS + 1e-12
        # lands clear of the letter: palm leading edge behind the bounding circle
        u = (math.cos(s.heading), math.sin(s.heading))
        along_start = (sx - pose[0]) * u[0] + (sy - pose[1]) * u[1]
        assert along_start + rg.PALM_HALF_Y < -r
        # ends past the letter origin, line misses it by at most max_offset
        along_end = (ex - pose[0]) * u[0] + (ey - pose[1]) * u[1]
        assert along_end > 0.0
        assert abs(s.offset) <= 0.02
        # fingers (+y body axis) trail the motion
        fy = (-math.sin(s.yaw), math.cos(s.yaw))
        assert fy[0] * u[0] + fy[1] * u[1] == pytest.approx(-1.0, abs=1e-9)


def test_hand_stroke_near_edge_stays_in_limit():
    rng = random.Random(0)
    for _ in range(30):
        s = rg.sample_hand_stroke("U", (0.07, -0.07, 0.4), rng, WS)
        assert max(abs(v) for v in s.start + s.end) <= WS + 1e-12


def test_hand_trajectory_phases():
    cfg = rg.stroke_cfg(WS, 0.15)
    stroke = rg.HandStroke((-0.09, 0.0), (0.03, 0.0), math.pi / 2, 0.0, 0.0)
    dt = 1 / 30
    traj = rg.hand_trajectory(cfg.park, stroke, dt, cfg)
    pos = [p for p, _ in traj]
    assert pos[-1] == pytest.approx(cfg.park)
    # never faster than vmax
    prev = cfg.park
    for p, v in traj:
        assert math.dist(p, prev) <= cfg.vmax * dt + 1e-9
        assert math.hypot(*v) <= cfg.vmax + 1e-6
        prev = p
    # only on the table between start and end, and it reaches both
    down = [p for p in pos if p[2] <= cfg.z_down + 1e-9]
    assert down[0][:2] == pytest.approx(stroke.start)
    assert any(p[:2] == pytest.approx(stroke.end) for p in down)
    assert all(-0.09 - 1e-9 <= p[0] <= 0.03 + 1e-9 and abs(p[1]) < 1e-9 for p in down)
    # comes down from above: no point below z_up outside the stroke area
    for p in pos:
        if p[2] < cfg.z_up - 1e-4:
            assert abs(p[1]) < 1e-9 and -0.09 - 1e-9 <= p[0] <= 0.03 + 1e-9


# ------------------------------------------------------------ settle detection
def _stream(poses, dt=0.05, t0=0.0):
    return [(t0 + i * dt, p) for i, p in enumerate(poses)]


def test_settle_still_stream():
    d = rg.SettleDetector(window_s=0.4)
    out = [d.update(t, p) for t, p in _stream([(0.01, 0.02, 0.3)] * 12)]
    assert not any(out[:8]) and all(out[8:])  # 8 * 0.05 s = 0.4 s after the first sample


def test_settle_moving_then_still_with_noise():
    poses = [(0.001 * i, 0.0, 0.0) for i in range(20)]           # 1 mm per sample
    poses += [(0.02 + 1e-4 * ((-1) ** i), 0.0, 1e-3 * ((-1) ** i)) for i in range(20)]
    d = rg.SettleDetector(window_s=0.4)
    out = [d.update(t, p) for t, p in _stream(poses)]
    assert not any(out[:20])
    first = out.index(True)
    assert 20 + 8 <= first <= 20 + 9


def test_settle_restarts_on_rotation_and_wraps():
    d = rg.SettleDetector(window_s=0.2, ang_tol=0.01)
    s = _stream([(0, 0, math.pi - 0.001)] * 3 + [(0, 0, -math.pi + 0.001)] * 3 + [(0, 0, 0.5)] * 6)
    out = [d.update(t, p) for t, p in s]
    assert out[4] and out[5]          # +-pi is the same angle
    assert not out[6] and not out[9] and out[10]  # 0.5 rad jump restarts the window


def test_settle_missing_pose_and_clock_jump():
    d = rg.SettleDetector(window_s=0.1)
    assert not d.update(0.0, (0, 0, 0))
    assert d.update(0.2, (0, 0, 0))
    assert not d.update(0.25, None)
    assert not d.update(0.3, (0, 0, 0))
    assert not d.update(0.0, (0, 0, 0))   # sim reset: time went backwards
    assert d.update(0.15, (0, 0, 0))


# ------------------------------------------------------------ push id matching
def test_push_tracker_matches_ids_only():
    t = rg.PushTracker(start=10)
    a = t.next_id()
    b = t.next_id()
    assert (a, b) == (11, 12)
    t.on_done(11)
    assert t.is_done(a) and not t.is_done(b)
    t.on_done(999)  # someone else's push
    assert not t.is_done(b)
    t.on_done(12)
    assert t.is_done(b)


def test_push_tracker_wraps_uint32_and_stays_bounded():
    t = rg.PushTracker(start=2 ** 32 - 1)
    assert t.next_id() == 0
    for i in range(3000):
        t.on_done(t.next_id())
    assert len(t.done) <= 1001
    assert t.is_done(t.last_id)


def test_push_tracker_default_start_differs():
    assert 0 <= rg.PushTracker().last_id < 2 ** 31


# ------------------------------------------------------------ observation freshness
def test_observation_freshness():
    b = rg.ObservationBuffer()
    assert b.latest_after(0.0) is None
    b.add(1.0, 1.05, [("I", 0.0, 0.0, 0.0, 0.9)])
    b.add(1.1, 1.15, [("I", 0.01, 0.0, 0.0, 0.9)])
    assert b.latest_after(1.05)[0][1] == 0.01      # newest after the threshold
    assert b.latest_after(0.5)[0][1] == 0.01       # newest overall
    assert b.latest_after(1.2) is None             # push settled at 1.2: nothing fresh yet
    b.add(1.3, 1.32, [])
    assert b.latest_after(1.2) == []               # fresh frame with no letter -> [] (not None)


def test_observation_zero_stamp_uses_receive_time_and_out_of_order():
    b = rg.ObservationBuffer(maxlen=3)
    b.add(0.0, 2.0, [("U", 0, 0, 0, 1)])
    assert b.latest_after(1.9) is not None and b.latest_after(2.1) is None
    b.add(3.0, 3.1, [("U", 1, 0, 0, 1)])
    b.add(2.5, 3.2, [("U", 2, 0, 0, 1)])          # older image received later
    assert b.latest_after(2.8)[0][1] == 1
    b.add(4.0, 4.1, [])
    assert len(b.items) == 3
