"""Point pusher (Push-T style) support in the state machine; no ROS. Run:
PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner python3 -m pytest ros2_ws/src/pa_planner
"""

from dataclasses import dataclass
import math
import random

import pytest

from pa_core.letters import is_aligned
from pa_planner.state_machine import (
    FAILED, LOST, PLANNING, PUSHING, SETTLING, LetterDone, LetterObs, Log, Params,
    PlannerPool, PlannerStateMachine, PlanRequest, PushCmd, run_plan_sync)

DISTURBED = (0.04, -0.03, 0.8)


def of(actions, kind):
    return [a for a in actions if isinstance(a, kind)]


def obs(pose, letter="I"):
    return [LetterObs(letter, *pose, 1.0)]


@dataclass
class FakePointPlan:  # same fields as pa_core.pointplanner.PointPlan
    x0: float
    y0: float
    x1: float
    y1: float
    predicted: tuple
    cost: float


class FakePool:
    def __init__(self, predicted=(0.0, 0.0, 0.0)):
        self.predicted = predicted
        self.calls = []

    def plan(self, req):
        self.calls.append(req)
        return FakePointPlan(0.05, 0.0, 0.02, 0.0, self.predicted, 1.0)


def recovering(t=0.0):
    sm = PlannerStateMachine(Params(settle_s=0.5, hand_clear_s=1.0, max_pushes=20))
    sm.on_active_letter("I", t)
    sm.on_letters(obs(DISTURBED), t + 0.1)  # no hand ever seen -> straight to RECOVER
    assert sm.rec.phase != PLANNING
    return sm, t + 0.2


def push_once(sm, pool, observed, t):
    """observe -> plan (via pool) -> push -> done -> settle. Returns (actions, t)."""
    acts = run_plan_sync(sm, pool, sm.on_letters(obs(observed), t), t)
    cmds = of(acts, PushCmd)
    if cmds:
        sm.on_push_done(cmds[0].id, t + 1.0)
        acts += sm.tick(t + 1.6)
    return acts, t + 2.0


def test_point_plan_result_emits_stroke():
    sm, t = recovering()
    req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
    out = sm.on_plan_result(req.req_id, -1, 0.0, t + 0.1, stroke=(0.05, -0.01, 0.02, 0.0))
    cmd = of(out, PushCmd)[0]
    assert cmd.stroke == (0.05, -0.01, 0.02, 0.0)
    assert cmd.side == -1 and cmd.angle_deg == 0.0 and cmd.id == 1
    assert sm.rec.phase == PUSHING
    sm.on_push_done(cmd.id, t + 1.0)
    assert sm.rec.phase == SETTLING


def test_paddle_push_has_no_stroke():
    sm, t = recovering()
    req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
    cmd = of(sm.on_plan_result(req.req_id, 2, 40, t), PushCmd)[0]
    assert cmd.stroke is None and cmd.side == 2


@pytest.mark.parametrize("bad", [(0.0, 0.0, 0.0), (0.0, math.nan, 0.0, 0.0), "abcd"])
def test_invalid_stroke_is_rejected(bad):
    sm, t = recovering()
    req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
    out = sm.on_plan_result(req.req_id, -1, 0.0, t, stroke=bad)
    assert not of(out, PushCmd) and of(out, Log)[0].level == "warn"
    assert sm.state != PUSHING and sm.rec.phase != PUSHING


def test_run_plan_sync_passes_stroke_through():
    sm, t = recovering()
    pool = FakePool()
    acts = run_plan_sync(sm, pool, sm.on_letters(obs(DISTURBED), t), t)
    cmd = of(acts, PushCmd)[0]
    assert cmd.stroke == (0.05, 0.0, 0.02, 0.0)
    assert sm.rec.predicted == (0.0, 0.0, 0.0)


def mismatch_warnings(acts):
    return [a for a in of(acts, Log) if a.level == "warn" and "prediction" in a.msg]


def test_repeated_prediction_mismatch_warns():
    sm, t = recovering()
    pool = FakePool(predicted=(0.0, 0.0, 0.0))  # planner always expects the target
    warns = []
    for _ in range(6):
        acts, t = push_once(sm, pool, DISTURBED, t)  # ...but the letter never moves
        warns += mismatch_warnings(acts)
    # 6 plans; observations 2..6 are compared (5 mismatches) -> one warning at the 3rd
    assert len(warns) == 1 and "3 pushes in a row" in warns[0].msg


def test_prediction_match_does_not_warn():
    sm, t = recovering()
    pool = FakePool(predicted=DISTURBED)  # predicted == what is observed next
    warns = []
    for _ in range(6):
        acts, t = push_once(sm, pool, DISTURBED, t)
        warns += mismatch_warnings(acts)
    assert warns == [] and sm.rec.mismatches == 0


def test_mismatch_respects_symmetry():
    # "I" is 180 deg symmetric: a prediction off by pi is the same pose.
    sm, t = recovering()
    pool = FakePool(predicted=(DISTURBED[0], DISTURBED[1], DISTURBED[2] + math.pi))
    for _ in range(4):
        _, t = push_once(sm, pool, DISTURBED, t)
    assert sm.rec.mismatches == 0


def test_pool_rejects_unknown_pusher():
    with pytest.raises(ValueError):
        PlannerPool("hammer")
    assert PlannerPool().pusher == "paddles"


# ---------------------------------------------------------------- closed loop

def run_point_closed_loop(letter, start, max_pushes=30, seed=0):
    """State machine + PointPushPlanner against PointPushSim as the world."""
    pytest.importorskip("pa_core.pointplanner")
    pointpush = pytest.importorskip("pa_core.pointpush")

    random.seed(seed)
    sm = PlannerStateMachine(Params(settle_s=0.4, hand_clear_s=1.0, max_pushes=max_pushes))
    pool = PlannerPool("point")
    world = pointpush.PointPushSim(letter)
    world.set_pose(*start)
    t = 0.0
    pending, done, logs = [], [], []

    def handle(actions):
        for a in run_plan_sync(sm, pool, actions, t):
            if isinstance(a, PushCmd):
                assert a.stroke is not None and len(a.stroke) == 4
                pending.append(a)
            elif isinstance(a, LetterDone):
                done.append(a.letter)
            elif isinstance(a, Log):
                logs.append(a)

    handle(sm.on_active_letter(letter, t))
    handle(sm.on_letters([LetterObs(letter, 0.0, 0.0, 0.0)], t))
    handle(sm.on_hand(True, 0.5))
    handle(sm.on_letters([LetterObs(letter, *world.pose())], 0.6))
    handle(sm.on_hand(False, 0.8))
    t = 0.8
    for _ in range(3000):
        t += 0.1
        handle(sm.tick(t))
        if pending:
            cmd = pending.pop(0)
            world.push(*cmd.stroke)
            t += 2.0
            handle(sm.on_push_done(cmd.id, t))
        handle(sm.on_letters([LetterObs(letter, *world.pose())], t))
        if done or sm.state in (FAILED, LOST):
            break
    return sm, done, world.pose(), logs


def test_point_closed_loop_aligns_I():
    sm, done, pose, logs = run_point_closed_loop("I", (0.04, -0.03, 0.9))
    assert done == ["I"], (sm.state, pose, sm.rec.pushes)
    assert is_aligned("I", *pose)
    assert sm.rec.pushes <= 30
    # the planner's model is the world here: its predictions should hold
    assert not [m for m in logs if "prediction" in m.msg], logs
