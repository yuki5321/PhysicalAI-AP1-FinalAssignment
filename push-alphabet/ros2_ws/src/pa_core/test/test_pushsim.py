import math

import numpy as np
import pytest

from pa_core.config import load_scene
from pa_core.pushsim import PushSim, pushers, simulate_push, sweep_profile

# Each paddle sweeps counter-clockwise from its side, so it pushes the letter
# inward (and along the side, in the sweep direction).
INWARD = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}
START = {0: (0.0, -0.06), 1: (0.06, 0.0), 2: (0.0, 0.06), 3: (-0.06, 0.0)}


@pytest.fixture(scope="module", params=list("IUYK"))
def sim(request):
    return PushSim(request.param)


def test_pushers_match_scene():
    ps = pushers()
    assert [p.name for p in ps] == [p["name"] for p in load_scene()["pushers"]["list"]]
    assert ps[1].rest_heading == pytest.approx(math.pi / 2)


def test_sweep_profile():
    sp = load_scene()["pushers"]
    t, hold = sweep_profile(90.0)
    assert t == pytest.approx(90.0 / sp["speed_deg_s"])
    assert hold == sp["hold_s"]
    assert sweep_profile(1.0)[0] == pytest.approx(0.15)  # minimum sweep time


def test_set_pose_roundtrip(sim):
    sim.set_pose(0.012, -0.034, 0.5)
    assert sim.pose() == pytest.approx((0.012, -0.034, 0.5))


def test_zero_angle_push_is_noop(sim):
    for pose in [(0.0, 0.0, 0.0), (0.01, 0.02, 0.3), (-0.03, 0.05, -2.0)]:
        sim.set_pose(*pose)
        assert sim.push(0, 0.0) == pytest.approx(pose, abs=1e-9)


def test_letter_far_from_paddle_is_untouched(sim):
    # A short sweep of the south paddle does not reach a letter in the upper half.
    sim.set_pose(0.0, 0.03, 0.0)
    assert sim.push(0, 15.0) == pytest.approx((0.0, 0.03, 0.0), abs=1e-6)


@pytest.mark.parametrize("side", range(4))
def test_push_moves_letter_inward(sim, side):
    x0, y0 = START[side]
    sim.set_pose(x0, y0, 0.0)
    x, y, _ = sim.push(side, 40.0)
    d = np.array([x - x0, y - y0])
    assert float(d @ np.array(INWARD[side])) > 0.02  # at least 2 cm inward


def test_larger_angle_pushes_further(sim):
    x0, y0 = START[0]
    sim.set_pose(x0, y0, 0.0)
    y20 = sim.push(0, 20.0)[1]
    sim.set_pose(x0, y0, 0.0)
    y40 = sim.push(0, 40.0)[1]
    assert y40 > y20 > y0


def test_rollout_independent_of_history(sim):
    sim.set_pose(0.0, -0.06, 0.2)
    a = sim.push(0, 40.0)
    sim.set_pose(0.03, 0.01, 1.0)
    sim.push(1, 70.0)
    sim.set_pose(0.0, -0.06, 0.2)
    b = sim.push(0, 40.0)
    assert b == pytest.approx(a, abs=1e-6)


def test_paddles_return_to_rest(sim):
    sim.set_pose(0.0, -0.06, 0.0)
    sim.push(0, 60.0)
    for arm, p in zip(sim.arms, sim.pushers):
        assert arm.angle == pytest.approx(p.rest_heading)


def test_four_fold_symmetry():
    # The paddle layout is a pinwheel: rotating the scene by 90 deg maps side k to k+1.
    pose = (0.01, -0.055, 0.3)
    a = simulate_push("U", pose, 0, 35.0)
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    rot = (c * pose[0] - s * pose[1], s * pose[0] + c * pose[1], pose[2] + math.pi / 2)
    b = simulate_push("U", rot, 1, 35.0)
    a_rot = (c * a[0] - s * a[1], s * a[0] + c * a[1])
    assert b[:2] == pytest.approx(a_rot, abs=1e-3)
    assert math.remainder(b[2] - a[2] - math.pi / 2, 2 * math.pi) == pytest.approx(0, abs=0.02)


def test_rollout_independent_of_history_when_wedged():
    # A push that wedges the letter between paddle and wall is very sensitive;
    # set_pose must forget everything simulated before (cached arbiters too).
    pose, side, angle = (0.0032, 0.0150, 1.318), 2, 110.0
    a = simulate_push("I", pose, side, angle)
    sim = PushSim("I")
    for p, s, ang in [((0.0, -0.03, 0.0), 0, 60.0), ((0.02, 0.01, 1.0), 1, 90.0)]:
        sim.set_pose(*p)
        sim.push(s, ang)
    sim.set_pose(*pose)
    assert sim.push(side, angle) == pytest.approx(a, abs=1e-6)


def test_walls_keep_letter_on_table(sim):
    from pa_core.pushsim import wall_inner_half
    w = wall_inner_half()
    assert w > load_scene()["workspace"]["table_half_size"] - 1e-9
    # Throw the letter toward the east wall as hard as possible.
    sim.set_pose(-0.05, 0.03, 0.0)
    x, y, _ = sim.push(3, 120.0)
    assert abs(x) < w and abs(y) < w


def test_contact_angle(sim):
    sim.set_pose(0.0, -0.06, 0.0)
    a = sim.contact_angle(0)
    assert a is not None and 0.0 < a < 60.0
    # No contact just before it, contact just after it.
    sim.set_pose(0.0, -0.06, 0.0)
    assert sim.push(0, a - 1.0, fast=False) == pytest.approx((0.0, -0.06, 0.0), abs=1e-6)
    sim.set_pose(0.0, -0.06, 0.0)
    assert sim.push(0, a + 2.0, fast=False) != pytest.approx((0.0, -0.06, 0.0), abs=1e-5)
    sim.set_pose(0.5, 0.5, 0.0)
    assert sim.contact_angle(0) is None


def test_fast_push_matches_full_simulation():
    # Skipping the parts of a sweep in which nothing moves must not change
    # the outcome (beyond the chaos of pushes that throw the letter).
    rng = np.random.default_rng(0)
    for letter in "IUYK":
        sim = PushSim(letter)
        errs = []
        for _ in range(40):
            pose = (*rng.uniform(-0.06, 0.06, 2), rng.uniform(-math.pi, math.pi))
            side, angle = int(rng.integers(4)), float(rng.choice(np.arange(5, 65, 5)))
            sim.set_pose(*pose)
            f = sim.push(side, angle)
            sim.set_pose(*pose)
            g = sim.push(side, angle, fast=False)
            errs.append(math.hypot(f[0] - g[0], f[1] - g[1]))
        assert np.median(errs) < 1e-4, letter
        assert np.percentile(errs, 90) < 2e-3, letter
