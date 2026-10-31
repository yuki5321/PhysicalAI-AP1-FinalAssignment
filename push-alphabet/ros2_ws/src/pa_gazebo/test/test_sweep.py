import math
from types import SimpleNamespace

import pytest

from pa_core.config import load_scene
from pa_core.pushsim import sweep_profile
from pa_gazebo.sweep import PushQueue, clamp_angle, plan_sweep


def cmd(i, side, angle):
    return SimpleNamespace(id=i, side=side, angle_deg=angle)


def test_plan_matches_2d_model_timing():
    sp = load_scene()["pushers"]
    p = plan_sweep(45.0, settle_s=0.3)
    t, hold = sweep_profile(45.0)
    assert p.t_out == p.t_back == t == pytest.approx(45.0 / sp["speed_deg_s"])
    assert p.hold == hold == sp["hold_s"]
    assert p.total_time == pytest.approx(2 * t + hold + 0.3)
    wp = p.waypoints(dt=0.02)
    times = [w[0] for w in wp]
    assert times == sorted(times) and times[-1] == pytest.approx(2 * t + hold)
    gaps = [b - a for a, b in zip([0.0] + times, times)]
    assert sum(g > 0.02 + 1e-9 for g in gaps) <= 1  # only the hold segment is longer
    a = math.radians(45)
    for tt, q, qd in wp:
        assert q == pytest.approx(p.angle_at(tt - 1e-12) if tt < p.motion_time else 0.0, abs=1e-6)
        assert abs(qd) <= a / t + 1e-9
    assert (t, a, 0.0) == pytest.approx(wp[[w[0] for w in wp].index(pytest.approx(t))])
    assert wp[-1][1:] == (0.0, 0.0)
    assert any(qd == pytest.approx(a / t) for _, _, qd in wp)
    assert any(qd == pytest.approx(-a / t) for _, _, qd in wp)


def test_small_angles_use_minimum_sweep_time():
    p = plan_sweep(2.0)
    assert p.t_out == pytest.approx(0.15)


def test_angle_profile_is_piecewise_linear():
    p = plan_sweep(90.0, settle_s=0.0)
    a = math.radians(90.0)
    assert p.angle_at(0.0) == 0.0
    assert p.angle_at(p.t_out / 2) == pytest.approx(a / 2)
    assert p.angle_at(p.t_out + p.hold / 2) == pytest.approx(a)
    assert p.angle_at(p.t_out + p.hold + p.t_back / 2) == pytest.approx(a / 2)
    assert p.angle_at(p.motion_time + 1) == 0.0


def test_clamp():
    mx = load_scene()["pushers"]["max_angle"]
    assert clamp_angle(-10) == 0.0
    assert clamp_angle(mx + 50) == mx
    assert clamp_angle(float("nan")) == 0.0


def test_queue_runs_commands_in_order_one_at_a_time():
    q = PushQueue(4, settle_s=0.3)
    q.push(cmd(1, 0, 30.0))
    q.push(cmd(2, 3, 60.0))
    started, finished = q.tick(0.0)
    assert started[0].id == 1 and started[1] == 0 and finished is None
    t1 = started[2].total_time
    assert q.tick(t1 - 0.01) == (None, None)
    started, finished = q.tick(t1)
    assert finished.id == 1
    assert started[0].id == 2 and started[1] == 3
    t2 = started[2].total_time
    started, finished = q.tick(t1 + t2)
    assert finished.id == 2 and started is None
    assert not q.busy


def test_queue_waits_for_paddle_back_at_rest_with_timeout():
    q = PushQueue(4, settle_s=0.0, rest_tol_deg=3.0, rest_timeout=1.0)
    q.push(cmd(7, 1, 40.0))
    _, _ = q.tick(0.0)
    T = plan_sweep(40.0, settle_s=0.0).total_time
    away = lambda side: math.radians(20.0)  # noqa: E731
    assert q.tick(T, away) == (None, None)
    assert q.tick(T + 0.5, away) == (None, None)
    _, finished = q.tick(T + 0.5, lambda side: math.radians(1.0))
    assert finished.id == 7
    q.push(cmd(8, 1, 40.0))
    q.tick(10.0)
    _, finished = q.tick(10.0 + T + 1.0, away)  # timeout reached
    assert finished.id == 8


def test_invalid_side():
    q = PushQueue(4)
    assert not q.valid(cmd(1, 4, 10))
    assert q.valid(cmd(1, 3, 10))
