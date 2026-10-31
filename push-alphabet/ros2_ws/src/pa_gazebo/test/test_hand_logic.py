import math

from pa_gazebo.hand_logic import HandConfig, HandTracker, hand_target, step_hand

CFG = HandConfig(limit=0.10)


def run(cur, target, seconds, dt=1 / 30):
    path = [cur]
    for _ in range(int(seconds / dt)):
        cur, _ = step_hand(cur, target, dt, CFG)
        path.append(cur)
    return cur, path


def test_target_mapping_and_clipping():
    assert hand_target(True, 0.02, -0.03, CFG) == (0.02, -0.03, CFG.z_down)
    assert hand_target(True, 0.5, -0.5, CFG) == (0.10, -0.10, CFG.z_down)
    assert hand_target(False, 0.0, 0.0, CFG) == CFG.park
    assert hand_target(True, float("nan"), 0.0, CFG) == CFG.park


def test_enters_from_above_then_descends():
    target = hand_target(True, 0.03, 0.02, CFG)
    end, path = run(CFG.park, target, 5.0)
    assert end == target
    # While far (horizontally) from the target the hand never dips below travel height.
    for p in path:
        if math.hypot(p[0] - target[0], p[1] - target[1]) > CFG.xy_tol + 1e-9:
            assert p[2] >= CFG.z_up - 1e-9


def test_slides_along_table_when_already_down():
    start = (0.0, 0.0, CFG.z_down)
    target = hand_target(True, 0.08, 0.0, CFG)
    end, path = run(start, target, 2.0)
    assert end == target
    assert all(p[2] == CFG.z_down for p in path)


def test_leaves_by_rising_first():
    start = (0.05, 0.05, CFG.z_down)
    end, path = run(start, CFG.park, 5.0)
    assert end == CFG.park
    rising = [p for p in path if p[2] < CFG.z_up - 1e-9]
    assert all(p[0] == 0.05 and p[1] == 0.05 for p in rising)


def test_speed_limit_and_velocity():
    dt = 0.05
    nxt, vel = step_hand(CFG.park, (0.0, 0.0, CFG.z_up), dt, CFG)
    d = math.dist(nxt, CFG.park)
    assert d <= CFG.vmax * dt + 1e-12
    assert math.isclose(math.sqrt(sum(v * v for v in vel)), d / dt)


def test_tracker_goes_stale():
    tr = HandTracker(CFG)
    assert tr.target(0.0) == CFG.park
    tr.update(True, 0.01, 0.02, now=10.0)
    assert tr.target(10.5)[:2] == (0.01, 0.02)
    assert tr.target(10.0 + CFG.stale_s + 0.1) == CFG.park
