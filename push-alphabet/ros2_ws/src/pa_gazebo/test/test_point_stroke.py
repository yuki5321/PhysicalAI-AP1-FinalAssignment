"""Tests for pa_gazebo.point_stroke (the ROS-free part of point_push_executor)."""

import math

import pytest

from pa_core.config import load_scene
from pa_description.urdf_gen import WALL_HEIGHT, wall_inner_half
from pa_gazebo.point_stroke import PointPusherConfig, plan_stroke, valid_command


@pytest.fixture
def cfg():
    return PointPusherConfig.from_scene()


def close(a, b, tol=1e-9):
    return all(abs(u - v) <= tol for u, v in zip(a, b))


def test_config_from_scene(cfg):
    scene = load_scene()
    pp = scene["point_pusher"]
    assert cfg.radius == pp["radius"] and cfg.speed == pp["speed"]
    assert cfg.park == tuple(pp["park"])
    assert cfg.z_travel > WALL_HEIGHT and cfg.z_travel > scene["letter"]["thickness"]
    assert 0 < cfg.z_down < scene["letter"]["thickness"] / 2
    # pusher stays inside the walls with a gap
    assert cfg.limit + cfg.radius < wall_inner_half(scene)
    assert cfg.limit > scene["workspace"]["half_size"]


def test_phases_waypoints_and_timing(cfg):
    plan = plan_stroke(cfg.park, -0.05, 0.01, 0.0, 0.01, 0.0, cfg)
    phases = [s.phase for s in plan.segments]
    assert phases == ["approach", "lower", "stroke", "lift", "park"]   # starts at park (lifted)
    pt = plan.phase_times()
    # contiguous segments
    for a, b in zip(plan.segments, plan.segments[1:]):
        assert a.t1 == pytest.approx(b.t0) and close(a.p1, b.p0)
    assert close(plan.pose_at(0.0), cfg.park)
    assert close(plan.pose_at(pt["approach"][1]), (-0.05, 0.01, cfg.z_travel))
    assert close(plan.pose_at(pt["lower"][1]), (-0.05, 0.01, cfg.z_down))
    assert close(plan.pose_at(pt["stroke"][1]), (0.0, 0.01, cfg.z_down))
    assert close(plan.pose_at(pt["lift"][1]), (0.0, 0.01, cfg.z_travel))
    assert close(plan.pose_at(plan.total_time + 1.0), cfg.park)
    assert plan.phase_at(plan.total_time + 0.1) == "done"
    # stroke at the scene speed (msg.speed 0), lifted moves at travel speed
    assert plan.speed == cfg.speed
    assert pt["stroke"][1] - pt["stroke"][0] == pytest.approx(0.05 / cfg.speed)
    assert pt["lower"][1] - pt["lower"][0] == pytest.approx((cfg.z_travel - cfg.z_down) / cfg.vertical_speed)
    d = math.dist(cfg.park, (-0.05, 0.01, cfg.z_travel))
    assert pt["approach"][1] - pt["approach"][0] == pytest.approx(d / cfg.travel_speed)
    # well under the planner's push timeout (20 s)
    assert plan.total_time < 8.0
    assert not plan.clamped


def test_stroke_is_straight_at_constant_height_and_speed(cfg):
    plan = plan_stroke(cfg.park, 0.02, -0.03, 0.02, 0.02, 0.04, cfg)
    t0, t1 = plan.phase_times()["stroke"]
    for i in range(1, 20):
        t = t0 + (t1 - t0) * i / 20
        x, y, z = plan.pose_at(t)
        assert x == pytest.approx(0.02) and z == pytest.approx(cfg.z_down)
        assert y == pytest.approx(-0.03 + 0.04 * (t - t0))
        vx, vy, vz = plan.velocity_at(t, 0.01)
        if t + 0.01 < t1:
            assert (vx, vy, vz) == pytest.approx((0.0, 0.04, 0.0), abs=1e-9)
    assert plan.speed == 0.04


def test_lifted_moves_never_go_below_travel_height(cfg):
    plan = plan_stroke(cfg.park, 0.05, 0.05, -0.05, -0.05, 0.0, cfg)
    for s in plan.segments:
        if s.phase in ("approach", "park"):
            assert min(s.p0[2], s.p1[2]) >= cfg.z_travel - 1e-12
        if s.phase in ("lower", "lift"):
            assert close(s.p0[:2], s.p1[:2])   # vertical only


def test_velocity_lookahead_does_not_overshoot_corner(cfg):
    plan = plan_stroke(cfg.park, -0.03, 0.0, 0.0, 0.0, 0.0, cfg)
    t0, t1 = plan.phase_times()["stroke"]
    h = 0.1   # 10 Hz clock: pose + h*v lands on the path one tick later
    for t in (t1 - 0.05, t1 - 0.01, t1):
        p = plan.pose_at(t)
        v = plan.velocity_at(t, h)
        nxt = tuple(a + h * b for a, b in zip(p, v))
        assert close(nxt, plan.pose_at(t + h), 1e-12)
        assert nxt[0] <= 0.0 + 1e-12   # never past the stroke end


def test_clamping_to_table(cfg):
    plan = plan_stroke(cfg.park, -0.5, 0.0, 0.5, 0.3, 0.0, cfg)
    assert plan.clamped
    assert plan.start == (-cfg.limit, 0.0) and plan.end == (cfg.limit, cfg.limit)
    for s in plan.segments:
        if s.phase in ("lower", "stroke", "lift"):
            for p in (s.p0, s.p1):
                assert abs(p[0]) <= cfg.limit + 1e-12 and abs(p[1]) <= cfg.limit + 1e-12


def test_speed_is_clamped(cfg):
    assert plan_stroke(cfg.park, 0, 0, 0.01, 0, 100.0, cfg).speed == cfg.max_speed
    p = plan_stroke(cfg.park, 0, 0, 0.01, 0, 1e-6, cfg)
    assert p.speed == cfg.min_speed and p.clamped
    assert plan_stroke(cfg.park, 0, 0, 0.01, 0, -1.0, cfg).speed == cfg.speed   # <= 0: default


def test_starting_below_travel_height_rises_first(cfg):
    plan = plan_stroke((0.01, 0.0, cfg.z_down), 0.03, 0.0, 0.05, 0.0, 0.0, cfg)
    assert plan.segments[0].phase == "rise"
    assert close(plan.segments[0].p1, (0.01, 0.0, cfg.z_travel))


def test_zero_length_stroke_still_lowers_and_parks(cfg):
    plan = plan_stroke(cfg.park, 0.0, 0.0, 0.0, 0.0, 0.0, cfg)
    assert [s.phase for s in plan.segments] == ["approach", "lower", "lift", "park"]


def test_valid_command():
    assert valid_command(0, 0, 0.01, 0, 0.0)
    assert not valid_command(float("nan"), 0, 0, 0)
    assert not valid_command(0, 0, float("inf"), 0)
