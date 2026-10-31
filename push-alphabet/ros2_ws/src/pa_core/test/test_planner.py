import math

import pytest

from pa_core.config import load_scene
from pa_core.letters import is_aligned
from pa_core.planner import Plan, PushPlanner, cost
from pa_core.pushsim import PushSim


def test_cost_zero_at_target():
    t = load_scene()["target"]
    assert cost("U", (t["x"], t["y"], math.radians(t["theta"]))) == pytest.approx(0)


def test_cost_grows_with_error_and_penalizes_leaving_workspace():
    assert cost("U", (0.01, 0, 0)) < cost("U", (0.02, 0, 0))
    assert cost("U", (0, 0, 0.1)) < cost("U", (0, 0, 0.2))
    assert cost("I", (0, 0, math.pi)) == pytest.approx(0, abs=1e-9)
    ws = load_scene()["workspace"]["half_size"]
    assert cost("U", (ws + 0.01, 0, 0)) > 100


@pytest.fixture(scope="module", params=list("IU"))
def planner(request):
    return PushPlanner(request.param)


def test_plan_is_valid(planner):
    p = planner.plan((0.0, -0.03, 0.2))
    assert isinstance(p, Plan)
    assert p.side in range(4)
    assert 0 < p.angle_deg <= load_scene()["pushers"]["max_angle"]
    assert len(p.predicted) == 3


@pytest.mark.parametrize("pose", [(0.0, -0.03, 0.0), (0.025, 0.0, 0.0), (-0.02, 0.02, 0.0)])
def test_plan_matches_model_and_moves_letter(planner, pose):
    letter = planner.letter
    p = planner.plan(pose)
    sim = PushSim(letter)
    sim.set_pose(*pose)
    after = sim.push(p.side, p.angle_deg)
    # The model is deterministic, so executing the plan gives its prediction.
    assert after == pytest.approx(p.predicted, abs=1e-6)
    assert p.predicted != pytest.approx(pose, abs=1e-4)  # never a no-contact push


@pytest.mark.parametrize("pose", [(0.0, -0.03, 0.0), (-0.02, 0.02, 0.0)])
def test_plan_predicts_improvement(planner, pose):
    # Within its lookahead the planner expects to end closer than it started.
    assert planner.plan(pose).cost < cost(planner.letter, pose)


@pytest.mark.parametrize("pose", [(0.0, -0.03, 0.0), (-0.02, 0.02, 0.0)])
def test_two_planned_pushes_reduce_cost(planner, pose):
    # The planner may take a push that is worse on its own when the two-push
    # sequence ends better (and single pushes that only look good from the
    # exact pose are ranked by their worst perturbed outcome), so the
    # guarantee is over its lookahead: two planned pushes end closer.
    p1 = planner.plan(pose)
    p2 = planner.plan(p1.predicted)
    assert cost(planner.letter, p2.predicted) < cost(planner.letter, pose)


def test_unreachable_letter_returns_noop_plan(planner):
    far = (0.5, 0.5, 0.0)
    p = planner.plan(far)
    assert p.side in range(4)
    assert p.predicted == pytest.approx(far)


def test_explore_returns_valid_plan(planner):
    p = planner.plan((0.0, -0.03, 0.0), explore=1.0)
    assert p.side in range(4) and 0 < p.angle_deg <= max(planner.angles)
    # Exploration only picks pushes that keep the letter inside the workspace.
    ws = load_scene()["workspace"]["half_size"]
    assert abs(p.predicted[0]) <= ws and abs(p.predicted[1]) <= ws


def test_closed_loop_offset_converges():
    letter = "I"
    planner = PushPlanner(letter)
    world = PushSim(letter)
    pose = (0.0, -0.04, 0.0)
    for _ in range(10):
        if is_aligned(letter, *pose):
            break
        p = planner.plan(pose)
        world.set_pose(*pose)
        pose = world.push(p.side, p.angle_deg)
    assert is_aligned(letter, *pose)


def test_edge_penalty():
    # Beyond the edge margin, distance costs more than the plain position term.
    from pa_core.planner import EDGE_MARGIN, EDGE_WEIGHT
    assert EDGE_WEIGHT > 0
    tol = load_scene()["target"]["pos_tol"]
    inside = (EDGE_MARGIN - 0.005, 0.0, 0.0)
    near_edge = (EDGE_MARGIN + 0.02, 0.0, 0.0)
    assert cost("I", inside) == pytest.approx(inside[0] / tol)
    assert cost("I", near_edge) > near_edge[0] / tol + 1.0


def test_stalled_and_fine_options_give_valid_plans():
    # Deeper search when stalled, and the (off by default) fine contact angles.
    pl = PushPlanner("I", deltas=(1, 2, 3), beam3=2)
    ws = load_scene()["workspace"]["half_size"]
    pose = (0.0, -0.03, 0.4)
    for explore in (0.0, 1e-9):  # 1e-9: deeper search, (almost) never random
        p = pl.plan(pose, explore=explore)
        assert p.side in range(4) and 0 < p.angle_deg <= pl.max_angle
        assert abs(p.predicted[0]) <= ws and abs(p.predicted[1]) <= ws
        assert p.predicted != pytest.approx(pose, abs=1e-4)
