"""Planner state machine (pure Python, no ROS).

The ROS node (`planner_node`) feeds it events and executes the actions it
returns; the tests drive it with a fake clock and `pa_core.pushsim.PushSim`
as the world. All times are seconds on one monotonic clock chosen by the caller
(the node uses its ROS clock, so `use_sim_time` works).

States (published on /planner/status as the bare name):

    WAIT_LETTER  no active letter yet, or /letter_done sent and waiting for the next one
    IDLE         watching the active letter; it is in place (or not yet disturbed)
    WAIT_HAND    disturbance seen (or hand appeared mid-recovery); wait until the
                 hand has been gone for `hand_clear_s`
    RECOVER      observe -> plan -> push -> wait /push_done + settle -> observe ...
    LOST         the active letter has not been seen for `lost_timeout_s`
    FAILED       `max_pushes` used up without aligning; waits for a hand to come
                 and go (a person helps) or for the letter to become aligned
    DONE         /active_letter is "" (all letters finished)

Planning is not done here: RECOVER emits a `PlanRequest` and waits for
`on_plan_result` with the same `req_id`, so the caller can run
`PushPlanner.plan` (hundreds of ms) off its executor thread. `run_plan_sync`
is a helper for callers that do not care.

Two pushers (docs/point_pusher.md): the 4 paddles (`PushCmd.side/angle_deg`,
`pa_core.planner.PushPlanner`) or the Push-T style point pusher
(`PushCmd.stroke = (x0, y0, x1, y1)`, `pa_core.pointplanner.PointPushPlanner`).
The machine itself does not care which: a plan result that carries a `stroke`
becomes a point push, otherwise a paddle push.

Prediction check: when the plan result carries the planner's `predicted` pose,
the next observation after the push is compared with it. If they disagree by
more than `mismatch_pos` / `mismatch_ang_deg` for `mismatch_count` pushes in a
row, a warning is logged (typically a perceived pose that does not match the real
letter, or a pusher that is not where the planner thinks). Only a warning: the
machine keeps recovering from what it sees.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from pa_core.config import load_scene
from pa_core.letters import angle_error_deg, is_disturbed, pose_error

# ---------------------------------------------------------------- states

WAIT_LETTER = "WAIT_LETTER"
IDLE = "IDLE"
WAIT_HAND = "WAIT_HAND"
RECOVER = "RECOVER"
LOST = "LOST"
FAILED = "FAILED"
DONE = "DONE"

# Margin around the workspace when choosing among detections of the active letter [m].
WORKSPACE_SLACK = 0.008

# RECOVER sub-phases (reported in Status.detail, not in the state name)
OBSERVE = "observe"
PLANNING = "planning"
PUSHING = "pushing"
SETTLING = "settling"


# ---------------------------------------------------------------- inputs / outputs

@dataclass(frozen=True)
class LetterObs:
    letter: str
    x: float
    y: float
    theta: float
    confidence: float = 1.0


@dataclass(frozen=True)
class PlanRequest:
    req_id: int
    letter: str
    pose: tuple  # (x, y, theta)
    explore: float


@dataclass(frozen=True)
class PushCmd:
    id: int
    side: int            # paddle index; -1 for a point push
    angle_deg: float     # paddle swing; 0 for a point push
    stroke: tuple | None = None  # point pusher: (x0, y0, x1, y1) [m, world]; None for paddles


@dataclass(frozen=True)
class LetterDone:
    letter: str


@dataclass(frozen=True)
class Status:
    state: str
    detail: str = ""


@dataclass(frozen=True)
class Log:
    level: str  # "info" | "warn"
    msg: str


@dataclass
class Params:
    settle_s: float = 0.5          # after /push_done, wait this long, then use the next /letters
    hand_clear_s: float = 1.0      # hand must be absent this long before recovering
    hand_timeout_s: float = 1.0    # no /hand/state for this long -> treat as absent
    max_pushes: int = 30
    push_timeout_s: float = 10.0   # no /push_done for this long -> assume it finished
    lost_timeout_s: float = 3.0    # active letter unseen this long -> LOST
    min_confidence: float = 0.3
    stall_pushes: int = 3          # no improvement for this many pushes -> explore
    explore: float = 0.3           # PushPlanner.plan(explore=...) while stalled
    start_if_misaligned: bool = True  # recover a newly active letter that is not aligned
    done_margin: float = 0.8       # "aligned" here = within this fraction of scene tolerances,
                                   # so perception noise does not end a recovery just short of them
    mismatch_pos: float = 0.015    # observed vs. predicted pose after a push: position gap [m]
    mismatch_ang_deg: float = 20.0  # ... or angle gap [deg] counted as a mismatch
    mismatch_count: int = 3        # this many mismatching pushes in a row -> warning


@dataclass
class _Recovery:
    phase: str = OBSERVE
    obs_after: float = 0.0
    phase_t0: float = 0.0
    pushes: int = 0
    best: float = math.inf
    stall: int = 0
    req_id: int = -1
    push_id: int = -1
    pause_for_hand: bool = False
    predicted: tuple | None = None  # planner's predicted pose after the current push
    mismatches: int = 0             # consecutive pushes whose outcome disagreed with it


def aligned_within(letter: str, pose, margin: float = 1.0) -> bool:
    """`is_aligned` with both tolerances scaled by `margin`."""
    t = load_scene()["target"]
    dp, da = pose_error(letter, *pose)
    return dp < margin * t["pos_tol"] and da < margin * t["ang_tol"]


def progress_cost(letter: str, pose) -> float:
    """Normalized error used for stall detection (same units as the planner cost)."""
    t = load_scene()["target"]
    dp, da = pose_error(letter, *pose)
    return dp / t["pos_tol"] + da / t["ang_tol"]


# ---------------------------------------------------------------- machine

class PlannerStateMachine:
    def __init__(self, params: Params | None = None, max_angle: float | None = None,
                 n_sides: int | None = None):
        self.p = params or Params()
        sc = load_scene()["pushers"]
        self.max_angle = float(sc["max_angle"] if max_angle is None else max_angle)
        self.n_sides = int(len(sc["list"]) if n_sides is None else n_sides)

        self.state = WAIT_LETTER
        self.detail = ""
        self.active = None          # active letter name, or None before the first message
        self.done_letter = None     # last letter we reported done
        self.pose = None            # last pose of the active letter
        self.pose_t = -math.inf     # time it was observed
        self.state_t0 = 0.0
        self.hand_present = False
        self.hand_last_present = -math.inf
        self.hand_msg_t = -math.inf
        self.hand_seen_since_fail = False
        self.fresh_letter = False   # active letter just changed, not observed yet
        self.rec = _Recovery()
        self.lost_from = None
        self._next_push_id = 1
        self._next_req_id = 1
        self._out: list = []

    # ------------------------------------------------------------ helpers

    def _emit(self, *actions):
        self._out.extend(actions)

    def _flush(self):
        out, self._out = self._out, []
        return out

    def _set(self, state, now, detail=""):
        if state != self.state or detail != self.detail:
            self.state, self.detail = state, detail
            self.state_t0 = now
            self._emit(Status(state, detail))

    def _hand_now(self, now) -> bool:
        """Hand over the table (stale messages count as absent)."""
        return self.hand_present and now - self.hand_msg_t <= self.p.hand_timeout_s

    def _hand_clear(self, now) -> bool:
        return not self._hand_now(now) and now - self._last_present(now) >= self.p.hand_clear_s

    def _last_present(self, now):
        if self.hand_present:  # stale "present": count it as present until it timed out
            return min(now, self.hand_msg_t + self.p.hand_timeout_s)
        return self.hand_last_present

    def _start_wait_hand(self, now):
        self._set(WAIT_HAND, now)
        self._check_hand_clear(now)

    def _check_hand_clear(self, now):
        if self.state == WAIT_HAND and self._hand_clear(now):
            self._start_recover(now)

    def _start_recover(self, now):
        self.rec = _Recovery(phase=OBSERVE, obs_after=now, phase_t0=now)
        self._set(RECOVER, now, OBSERVE)

    def _phase(self, phase, now):
        self.rec.phase = phase
        self.rec.phase_t0 = now
        self._set(RECOVER, now, phase)

    def _aligned(self, name, pose) -> bool:
        return aligned_within(name, pose, self.p.done_margin)

    def _visible_recently(self, now):
        return now - self.pose_t <= self.p.lost_timeout_s

    # ------------------------------------------------------------ events

    def on_active_letter(self, name: str, now: float):
        name = (name or "").strip()
        if name == "":
            self.active = ""
            self._set(DONE, now)
            return self._flush()
        if self.state == WAIT_LETTER and name == self.done_letter:
            return self._flush()  # stale repeat of the letter we just finished
        if name != self.active or self.state in (WAIT_LETTER, DONE):
            self.active = name
            self.pose, self.pose_t = None, -math.inf
            self.fresh_letter = True
            self.rec = _Recovery()
            self._set(IDLE, now)
            self._emit(Log("info", f"active letter {name}"))
        return self._flush()

    def on_hand(self, present: bool, now: float):
        self.hand_msg_t = now
        self.hand_present = bool(present)
        if present:
            self.hand_last_present = now
            self.hand_seen_since_fail = True
            if self.state == RECOVER:
                if self.rec.phase in (OBSERVE, PLANNING):
                    # Nothing is moving yet: drop the pending plan and wait.
                    self.rec.req_id = -1
                    self._start_wait_hand(now)
                else:
                    self.rec.pause_for_hand = True  # finish the current push first
        else:
            self._check_hand_clear(now)
        return self._flush()

    def on_letters(self, letters, now: float):
        if self.active in (None, ""):
            return self._flush()
        # Prefer candidates inside the workspace: a paddle at rest lies just outside
        # it and can look like an "I" to the detector.
        half = load_scene()["workspace"]["half_size"] + WORKSPACE_SLACK
        best, best_key = None, None
        for o in letters:
            if o.letter == self.active and o.confidence >= self.p.min_confidence:
                key = (abs(o.x) <= half and abs(o.y) <= half, o.confidence)
                if best is None or key > best_key:
                    best, best_key = o, key
        if best is None:
            self._tick(now)
            return self._flush()
        self.pose = (best.x, best.y, best.theta)
        self.pose_t = now
        pose = self.pose
        name = self.active
        if self.state == RECOVER and self.rec.phase == SETTLING:
            self._tick(now)  # settle time may have just run out

        if self.state == LOST:
            if not self._aligned(name, pose):
                self._start_wait_hand(now)
            elif self.lost_from in (RECOVER, WAIT_HAND):
                self._letter_done(now)  # it came back in place
            else:
                self._set(IDLE, now)
        elif self.state == IDLE:
            fresh, self.fresh_letter = self.fresh_letter, False
            if is_disturbed(name, *pose) or (
                    fresh and self.p.start_if_misaligned and not self._aligned(name, pose)):
                self._emit(Log("info", f"{name} disturbed at "
                                       f"({pose[0]:.3f}, {pose[1]:.3f}, {math.degrees(pose[2]):.0f} deg)"))
                self._start_wait_hand(now)
        elif self.state == FAILED:
            if self._aligned(name, pose):
                self._letter_done(now)
        elif self.state == RECOVER and self.rec.phase == OBSERVE and now >= self.rec.obs_after:
            self._observe(pose, now)
        return self._flush()

    def on_push_done(self, push_id: int, now: float):
        if self.state == RECOVER and self.rec.phase == PUSHING and push_id == self.rec.push_id:
            self._push_finished(now)
        return self._flush()

    def on_plan_result(self, req_id: int, side: int, angle_deg: float, now: float,
                       stroke=None, predicted=None):
        """Plan for `req_id`. Paddles: `side`, `angle_deg`. Point pusher: pass
        `stroke=(x0, y0, x1, y1)` (side/angle are then ignored). `predicted`:
        optional (x, y, theta) the planner expects after the push."""
        if self.state != RECOVER or self.rec.phase != PLANNING or req_id != self.rec.req_id:
            return self._flush()  # stale (hand appeared, letter changed, ...)
        if stroke is not None:
            try:
                stroke = tuple(float(v) for v in stroke)
            except (TypeError, ValueError):
                stroke = ()
            if len(stroke) != 4 or not all(math.isfinite(v) for v in stroke):
                return self._invalid_plan(f"planner returned invalid stroke {stroke!r}", now)
            cmd_args = (-1, 0.0, stroke)
        else:
            side = int(side)
            angle = float(min(max(angle_deg, 0.0), self.max_angle))
            if not 0 <= side < self.n_sides:
                return self._invalid_plan(f"planner returned invalid side {side}", now)
            cmd_args = (side, angle, None)
        pid = self._next_push_id
        self._next_push_id += 1
        self.rec.push_id = pid
        self.rec.pushes += 1
        self.rec.predicted = tuple(predicted) if predicted is not None else None
        self._phase(PUSHING, now)
        self._emit(PushCmd(pid, *cmd_args))
        return self._flush()

    def _invalid_plan(self, msg, now):
        self._emit(Log("warn", msg))
        self._phase(OBSERVE, now)
        self.rec.obs_after = now
        return self._flush()

    def tick(self, now: float):
        self._tick(now)
        return self._flush()

    def _tick(self, now: float):
        s = self.state
        if s == WAIT_HAND:
            self._check_hand_clear(now)
        elif s == IDLE:
            if not self._visible_recently(now) and now - self.state_t0 > self.p.lost_timeout_s:
                self._emit(Log("warn", f"{self.active} not visible"))
                self._lost(now)
        elif s == FAILED:
            if self.hand_seen_since_fail and self._hand_clear(now):
                self._emit(Log("info", "hand came and went; retrying"))
                self._start_recover(now)
        elif s == RECOVER:
            r = self.rec
            if r.phase == PUSHING and now - r.phase_t0 > self.p.push_timeout_s:
                self._emit(Log("warn", f"no /push_done for id {r.push_id}; assuming it finished"))
                self._push_finished(now)
            elif r.phase == SETTLING and now >= r.obs_after:
                if r.pause_for_hand or self._hand_now(now):
                    r.pause_for_hand = False
                    self._start_wait_hand(now)
                else:
                    self._phase(OBSERVE, now)
            elif r.phase == OBSERVE and now - r.obs_after > self.p.lost_timeout_s:
                self._emit(Log("warn", f"{self.active} not visible"))
                self._lost(now)

    def _lost(self, now):
        self.lost_from = self.state
        self._set(LOST, now)

    # ------------------------------------------------------------ recovery steps

    def _push_finished(self, now):
        self.rec.obs_after = now + self.p.settle_s
        self._phase(SETTLING, now)
        if self.p.settle_s <= 0:
            self._tick(now)

    def _letter_done(self, now):
        self._emit(LetterDone(self.active), Log("info", f"{self.active} aligned"))
        self.done_letter = self.active
        self._set(WAIT_LETTER, now)

    def _check_prediction(self, name, pose):
        """Compare the observation after a push with the planner's prediction."""
        r = self.rec
        pred, r.predicted = r.predicted, None
        if pred is None:
            return
        dp = math.hypot(pose[0] - pred[0], pose[1] - pred[1])
        da = abs(angle_error_deg(name, pose[2], pred[2]))
        if dp <= self.p.mismatch_pos and da <= self.p.mismatch_ang_deg:
            r.mismatches = 0
            return
        r.mismatches += 1
        if self.p.mismatch_count > 0 and r.mismatches % self.p.mismatch_count == 0:
            self._emit(Log("warn", (
                f"{name}: observed pose disagrees with the planner's prediction for "
                f"{r.mismatches} pushes in a row (last: seen ({pose[0]:.3f}, {pose[1]:.3f}, "
                f"{math.degrees(pose[2]):.0f} deg), predicted ({pred[0]:.3f}, {pred[1]:.3f}, "
                f"{math.degrees(pred[2]):.0f} deg), gap {dp * 1000:.0f} mm / {da:.0f} deg); "
                f"check the detected outline / pose and the pusher calibration")))

    def _observe(self, pose, now):
        name, r = self.active, self.rec
        self._check_prediction(name, pose)
        if self._aligned(name, pose):
            self._letter_done(now)
            return
        if r.pushes >= self.p.max_pushes:
            self._emit(Log("warn", f"{name}: {r.pushes} pushes without aligning"))
            self.hand_seen_since_fail = False
            self._set(FAILED, now)
            return
        c = progress_cost(name, pose)
        r.stall = 0 if c < r.best - 1e-3 else r.stall + 1
        r.best = min(r.best, c)
        explore = self.p.explore if r.stall >= self.p.stall_pushes else 0.0
        r.req_id = self._next_req_id
        self._next_req_id += 1
        self._phase(PLANNING, now)
        self._emit(PlanRequest(r.req_id, name, tuple(pose), explore))


# ---------------------------------------------------------------- planning helper

PUSHERS = ("paddles", "point")


def default_pusher() -> str:
    """`pusher_mode` from scene.yaml ("paddles" if missing)."""
    return str(load_scene().get("pusher_mode", "paddles"))


class PlannerPool:
    """Caches one planner per letter: `pa_core.planner.PushPlanner` for
    pusher="paddles", `pa_core.pointplanner.PointPushPlanner` for "point".
    Not thread-safe: call `plan` from one worker thread only."""

    def __init__(self, pusher: str = "paddles"):
        if pusher not in PUSHERS:
            raise ValueError(f"pusher must be one of {PUSHERS}, got {pusher!r}")
        self.pusher = pusher
        self._planners = {}

    def get(self, letter):
        if letter not in self._planners:
            if self.pusher == "point":
                from pa_core.pointplanner import PointPushPlanner
                self._planners[letter] = PointPushPlanner(letter)
            else:
                from pa_core.planner import PushPlanner
                self._planners[letter] = PushPlanner(letter)
        return self._planners[letter]

    def plan(self, req: PlanRequest):
        return self.get(req.letter).plan(req.pose, explore=req.explore)


def feed_plan(sm: PlannerStateMachine, req: PlanRequest, plan, now):
    """Pass a `Plan` (paddles) or `PointPlan` (point pusher) to `sm.on_plan_result`."""
    predicted = getattr(plan, "predicted", None)
    if hasattr(plan, "x0"):
        return sm.on_plan_result(req.req_id, -1, 0.0, now,
                                 stroke=(plan.x0, plan.y0, plan.x1, plan.y1), predicted=predicted)
    return sm.on_plan_result(req.req_id, plan.side, plan.angle_deg, now, predicted=predicted)


def run_plan_sync(sm: PlannerStateMachine, pool: PlannerPool, actions, now):
    """Execute PlanRequests inline (tests / simple callers). Returns the other actions."""
    out = []
    queue = list(actions)
    while queue:
        a = queue.pop(0)
        if isinstance(a, PlanRequest):
            queue.extend(feed_plan(sm, a, pool.plan(a), now))
        else:
            out.append(a)
    return out
