"""State machine tests (no ROS). Run:
PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner python3 -m pytest ros2_ws/src/pa_planner
"""

import math
import random

import pytest

from pa_core.letters import is_aligned
from pa_planner.state_machine import (
    DONE, FAILED, IDLE, LOST, RECOVER, WAIT_HAND, WAIT_LETTER, PLANNING, PUSHING, SETTLING,
    LetterDone, LetterObs, Params, PlannerPool, PlannerStateMachine, PlanRequest, PushCmd,
    Status, run_plan_sync)

ALIGNED = (0.0, 0.0, 0.0)
DISTURBED = (0.04, -0.03, 0.8)


def of(actions, kind):
    return [a for a in actions if isinstance(a, kind)]


def obs(pose, letter="I", conf=1.0):
    return [LetterObs(letter, *pose, conf)]


@pytest.fixture
def sm():
    return PlannerStateMachine(Params(settle_s=0.5, hand_clear_s=1.0, max_pushes=5,
                                      lost_timeout_s=3.0, push_timeout_s=10.0))


def start_recovery(sm, t=0.0):
    """Active I, seen aligned, then disturbed while the hand is over the table."""
    sm.on_active_letter("I", t)
    sm.on_letters(obs(ALIGNED), t + 0.1)
    sm.on_hand(True, t + 0.2)
    out = sm.on_letters(obs(DISTURBED), t + 0.3)
    assert sm.state == WAIT_HAND
    assert Status(WAIT_HAND) in out
    sm.on_hand(False, t + 0.5)
    assert sm.state == WAIT_HAND
    sm.on_hand(False, t + 1.0)
    assert sm.state == WAIT_HAND  # only 0.8 s since the hand was last seen
    sm.tick(t + 1.25)
    assert sm.state == RECOVER
    return t + 1.25


def test_waits_for_letter_then_idle(sm):
    assert sm.state == WAIT_LETTER
    assert sm.on_letters(obs(DISTURBED), 0.0) == []
    out = sm.on_active_letter("I", 0.1)
    assert sm.state == IDLE and Status(IDLE) in out
    sm.on_letters(obs(ALIGNED), 0.2)
    sm.on_letters(obs((0.005, 0.0, math.radians(5))), 0.3)  # small error: not "disturbed"
    assert sm.state == IDLE


def test_misaligned_new_letter_is_recovered(sm):
    sm.on_active_letter("I", 0.0)
    sm.on_letters(obs((0.01, 0.0, 0.0)), 0.1)  # off target but below disturb threshold
    assert sm.state == RECOVER  # no hand ever seen -> straight through WAIT_HAND


def test_full_cycle_with_fake_planner(sm):
    t = start_recovery(sm)
    out = sm.on_letters(obs(DISTURBED), t + 0.1)
    req = of(out, PlanRequest)[0]
    assert req.letter == "I" and req.explore == 0.0 and sm.rec.phase == PLANNING
    # stale result is ignored
    assert sm.on_plan_result(req.req_id + 99, 1, 30, t + 0.2) == []
    out = sm.on_plan_result(req.req_id, 1, 500.0, t + 0.3)
    cmd = of(out, PushCmd)[0]
    assert cmd.id == 1 and cmd.side == 1 and cmd.angle_deg == 120.0  # clamped to max_angle
    assert sm.rec.phase == PUSHING
    sm.on_push_done(cmd.id + 1, t + 1.0)  # wrong id
    assert sm.rec.phase == PUSHING
    sm.on_push_done(cmd.id, t + 2.0)
    assert sm.rec.phase == SETTLING
    # observation during settling is not used
    assert of(sm.on_letters(obs(ALIGNED), t + 2.2), LetterDone) == []
    sm.tick(t + 2.5)
    out = sm.on_letters(obs(ALIGNED), t + 2.6)
    assert of(out, LetterDone) == [LetterDone("I")]
    assert sm.state == WAIT_LETTER
    # stale repeat of the same letter is ignored, next letter starts IDLE
    sm.on_active_letter("I", t + 3.0)
    assert sm.state == WAIT_LETTER
    sm.on_active_letter("U", t + 3.1)
    assert sm.state == IDLE and sm.active == "U"
    sm.on_active_letter("", t + 4.0)
    assert sm.state == DONE


def test_push_ids_increment(sm):
    t = start_recovery(sm)
    ids = []
    for k in range(3):
        out = sm.on_letters(obs(DISTURBED), t)
        req = of(out, PlanRequest)[0]
        cmd = of(sm.on_plan_result(req.req_id, 0, 30, t), PushCmd)[0]
        ids.append(cmd.id)
        sm.on_push_done(cmd.id, t + 1)
        sm.tick(t + 1.6)
        t += 2
    assert ids == [1, 2, 3]


def test_hand_during_push_pauses_after_push(sm):
    t = start_recovery(sm)
    req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
    cmd = of(sm.on_plan_result(req.req_id, 2, 40, t), PushCmd)[0]
    sm.on_hand(True, t + 0.5)
    assert sm.state == RECOVER and sm.rec.phase == PUSHING  # the swing finishes
    sm.on_hand(True, t + 0.8)
    sm.on_hand(False, t + 0.9)
    sm.on_push_done(cmd.id, t + 1.0)
    sm.tick(t + 1.5)
    assert sm.state == WAIT_HAND
    sm.tick(t + 2.0)
    assert sm.state == RECOVER  # hand last seen at t+0.8 (>= 1 s ago)
    assert sm.rec.pushes == 0   # fresh budget after a new disturbance


def test_hand_during_planning_drops_plan(sm):
    t = start_recovery(sm)
    req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
    sm.on_hand(True, t + 0.1)
    assert sm.state == WAIT_HAND
    assert sm.on_plan_result(req.req_id, 0, 30, t + 0.4) == []


def test_stale_hand_message_counts_as_absent(sm):
    sm.on_active_letter("I", 0.0)
    sm.on_letters(obs(ALIGNED), 0.1)
    sm.on_hand(True, 0.2)  # bridge dies while the hand is present
    sm.on_letters(obs(DISTURBED), 0.3)
    sm.tick(1.5)
    assert sm.state == WAIT_HAND
    sm.tick(2.3)  # timeout 1 s (-> absent at 1.2) + clear 1 s
    assert sm.state == RECOVER


def test_push_timeout(sm):
    t = start_recovery(sm)
    req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
    sm.on_plan_result(req.req_id, 0, 30, t)
    sm.tick(t + 5)
    assert sm.rec.phase == PUSHING
    sm.tick(t + 10.5)
    assert sm.rec.phase == SETTLING


def test_letter_not_visible(sm):
    t = start_recovery(sm)
    sm.on_letters(obs(DISTURBED, letter="U"), t + 1)  # other letters do not count
    sm.on_letters(obs(DISTURBED, conf=0.1), t + 1.5)  # low confidence does not count
    sm.tick(t + 3.5)
    assert sm.state == LOST
    out = sm.on_letters(obs(DISTURBED), t + 4)
    assert sm.state == RECOVER and of(out, PlanRequest) == []  # via WAIT_HAND; fresh obs next
    out = sm.on_letters(obs(DISTURBED), t + 4.1)
    assert of(out, PlanRequest)


def test_lost_in_idle_and_back(sm):
    sm.on_active_letter("I", 0.0)
    sm.on_letters(obs(ALIGNED), 0.1)
    sm.tick(3.5)
    assert sm.state == LOST
    sm.on_letters(obs(ALIGNED), 4.0)
    assert sm.state == IDLE


def test_max_pushes_then_failed(sm):
    t = start_recovery(sm)
    for _ in range(5):
        req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
        cmd = of(sm.on_plan_result(req.req_id, 0, 30, t), PushCmd)[0]
        sm.on_push_done(cmd.id, t + 1)
        sm.tick(t + 1.6)
        t += 2
    out = sm.on_letters(obs(DISTURBED), t)
    assert sm.state == FAILED and not of(out, PlanRequest)
    # a person helps: hand comes and goes -> retry with a fresh budget
    sm.on_hand(True, t + 1)
    sm.on_hand(False, t + 2)
    sm.tick(t + 3.1)
    assert sm.state == RECOVER and sm.rec.pushes == 0


def test_stall_turns_on_explore(sm):
    sm.p.max_pushes = 20
    t = start_recovery(sm)
    explores = []
    for _ in range(5):
        req = of(sm.on_letters(obs(DISTURBED), t), PlanRequest)[0]
        explores.append(req.explore)
        cmd = of(sm.on_plan_result(req.req_id, 0, 30, t), PushCmd)[0]
        sm.on_push_done(cmd.id, t + 1)
        sm.tick(t + 1.6)
        t += 2
    assert explores == [0.0, 0.0, 0.0, 0.3, 0.3]


# ---------------------------------------------------------------- closed loop

def run_closed_loop(letter, start, max_pushes=25, seed=0):
    """Drive the state machine against PushSim as the world, with a fake clock."""
    from pa_core.pushsim import PushSim

    random.seed(seed)
    params = Params(settle_s=0.4, hand_clear_s=1.0, max_pushes=max_pushes)
    sm = PlannerStateMachine(params)
    pool = PlannerPool()
    world = PushSim(letter)
    world.set_pose(*start)
    t = 0.0
    pending = []  # PushCmds to execute
    done = []
    states = []

    def handle(actions):
        for a in run_plan_sync(sm, pool, actions, t):
            if isinstance(a, PushCmd):
                pending.append(a)
            elif isinstance(a, LetterDone):
                done.append(a.letter)
            elif isinstance(a, Status):
                states.append(a.state)

    handle(sm.on_active_letter(letter, t))
    pose = (0.0, 0.0, 0.0)
    handle(sm.on_letters([LetterObs(letter, *pose)], t))
    # the hand knocks the letter over, then leaves
    handle(sm.on_hand(True, 0.5))
    handle(sm.on_letters([LetterObs(letter, *world.pose())], 0.6))
    handle(sm.on_hand(False, 0.8))
    t = 0.8
    for _ in range(2000):
        t += 0.1
        handle(sm.tick(t))
        if pending:
            cmd = pending.pop(0)
            world.push(cmd.side, cmd.angle_deg)
            t += 2.0
            handle(sm.on_push_done(cmd.id, t))
        handle(sm.on_letters([LetterObs(letter, *world.pose())], t))  # camera at 10 Hz
        if done or sm.state in (FAILED, LOST):
            break
    return sm, done, world.pose(), states


def test_closed_loop_aligns_I():
    sm, done, pose, states = run_closed_loop("I", (0.04, -0.03, 0.9))
    assert done == ["I"], (sm.state, pose, sm.rec.pushes)
    assert is_aligned("I", *pose)
    assert sm.state == WAIT_LETTER
    assert WAIT_HAND in states and RECOVER in states
    assert sm.rec.pushes <= 25


def test_done_needs_margin_inside_tolerance(sm):
    # 7 mm is inside the scene tolerance (8 mm) but not inside 0.8x of it:
    # perception noise could hide a true error just above 8 mm, so keep pushing.
    assert sm.p.done_margin == pytest.approx(0.8)
    t = start_recovery(sm)
    borderline = (0.007, 0.0, 0.0)
    assert is_aligned("I", *borderline)
    out = sm.on_letters(obs(borderline), t + 0.1)
    assert of(out, LetterDone) == [] and of(out, PlanRequest)
    req = of(out, PlanRequest)[0]
    cmd = of(sm.on_plan_result(req.req_id, 0, 10, t + 0.2), PushCmd)[0]
    sm.on_push_done(cmd.id, t + 1.0)
    sm.tick(t + 1.6)
    out = sm.on_letters(obs((0.005, 0.0, math.radians(4))), t + 1.7)
    assert of(out, LetterDone) == [LetterDone("I")]


def test_prefers_detection_inside_workspace_over_paddle_lookalike():
    # A resting paddle (y ~ 0.115, just outside the workspace) detected as a confident "I"
    # must not be chosen over the real letter inside the workspace.
    sm = PlannerStateMachine(Params())
    sm.on_active_letter("I", 0.0)
    real = LetterObs("I", -0.045, -0.017, 0.47, 0.6)
    paddle = LetterObs("I", 0.050, 0.116, -1.57, 0.95)
    sm.on_letters([paddle, real], 0.1)
    assert sm.pose == (real.x, real.y, real.theta)
