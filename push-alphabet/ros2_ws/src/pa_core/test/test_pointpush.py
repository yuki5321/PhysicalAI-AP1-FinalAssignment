import math

import pytest

from pa_core.config import load_scene
from pa_core.letters import is_aligned, pose_error
from pa_core.planner import cost
from pa_core.pointplanner import PointPlan, PointPushPlanner, contact_points
from pa_core.pointpush import PointPushSim, simulate_stroke


@pytest.fixture(scope="module", params=list("IUYK"))
def sim(request):
    return PointPushSim(request.param)


def test_no_paddles_in_world(sim):
    sim.set_pose(0.0, 0.0, 0.0)
    assert sim.arms == [] and sim.arm_shapes == []
    # only the letter's shapes + 4 walls
    assert len(sim.space.shapes) == len(sim.shapes) + 4


def test_push_moves_letter_in_push_direction(sim):
    r = load_scene()["point_pusher"]["radius"]
    # strokes that end inside the letter (it has to make way)
    for (x0, y0, x1, y1), axis, sign in [((0.0, -0.07, 0.0, 0.0), 1, 1),
                                         ((0.0, 0.07, 0.0, -0.03), 1, -1),  # U: through its gap
                                         ((-0.07, 0.0, 0.0, 0.0), 0, 1),
                                         ((0.07, 0.0, 0.0, 0.0), 0, -1)]:
        sim.set_pose(0.0, 0.0, 0.0)
        p = sim.push(x0, y0, x1, y1)
        assert sign * p[axis] > 0.005, (x0, y0, p)
        assert abs(p[1 - axis]) < 0.03
        # the disc ends at x1/y1: the letter does not overlap it any more (up to settling)
        assert not sim.overlaps(x1, y1, -0.0005 - 0.1 * r)


def test_zero_length_and_miss_are_noops(sim):
    for pose in [(0.0, 0.0, 0.0), (0.01, 0.02, 0.3)]:
        sim.set_pose(*pose)
        assert sim.push(0.05, 0.05, 0.05, 0.05) == pytest.approx(pose, abs=1e-9)
        sim.set_pose(*pose)
        # a stroke far away from the letter
        assert sim.push(0.14, -0.14, 0.14, -0.09) == pytest.approx(pose, abs=1e-9)


def test_deterministic_and_history_free():
    pose, st = (0.01, -0.005, 0.4), (-0.06, -0.03, 0.0, 0.0)
    a = simulate_stroke("K", pose, st)
    s = PointPushSim("K")
    s.set_pose(0.02, 0.03, -1.0)
    s.push(0.0, 0.08, 0.02, 0.02)  # something else first
    s.set_pose(*pose)
    assert s.push(*st) == a
    assert simulate_stroke("K", pose, st) == a


def test_push_depths_matches_single_pushes():
    s = PointPushSim("U")
    pose = (0.005, 0.0, 0.3)
    s.set_pose(*pose)
    lens = [0.02, 0.035, 0.06]
    multi = s.push_depths(-0.06, -0.01, 1.0, 0.0, lens)
    for L, m in zip(lens, multi):
        s.set_pose(*pose)
        assert s.push(-0.06, -0.01, -0.06 + L, -0.01) == pytest.approx(m, abs=1e-12)


def test_contact_points_on_outline():
    for L in "IUYK":
        pts = contact_points(L)
        assert len(pts) >= 4
        for qx, qy, nx, ny in pts:
            assert math.hypot(nx, ny) == pytest.approx(1.0)
    assert len(contact_points("I")) == 8  # 3 per long side, 1 per end


@pytest.mark.parametrize("letter", list("IUYK"))
def test_plan_starts_outside_and_reduces_cost(letter):
    pl = PointPushPlanner(letter)
    pose = (0.03, -0.02, math.radians(25))
    plan = pl.plan(pose)
    assert isinstance(plan, PointPlan)
    pl.sim.set_pose(*pose)
    assert not pl.sim.overlaps(plan.x0, plan.y0)  # lowered beside the letter, not on it
    L = math.hypot(plan.x1 - plan.x0, plan.y1 - plan.y0)
    assert 0 < L <= load_scene()["point_pusher"]["max_stroke"] + 1e-9
    lim = load_scene()["workspace"]["table_half_size"]
    assert max(abs(plan.x0), abs(plan.y0), abs(plan.x1), abs(plan.y1)) <= lim
    after = simulate_stroke(letter, pose, plan.stroke)
    assert after == pytest.approx(plan.predicted, abs=1e-9)
    assert cost(letter, after) < cost(letter, pose) - 1.0


def test_closed_loop_U_from_rotated_pose():
    pl = PointPushPlanner("U")
    world = PointPushSim("U")
    world.set_pose(0.04, 0.03, math.radians(120))
    for k in range(12):
        if is_aligned("U", *world.pose()):
            break
        world.push(*pl.plan(world.pose()).stroke)
    assert is_aligned("U", *world.pose()), (k, pose_error("U", *world.pose()))
    assert k <= 10
