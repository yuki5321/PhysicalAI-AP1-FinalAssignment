"""Paddle sweep timing and the push command queue (pure python, no ROS).

One push = rotate out from rest (0 rad) to `angle_deg` at pushers.speed_deg_s,
hold `hold_s`, rotate back, then let the letter settle. The timing comes from
pa_core.pushsim.sweep_profile, so Gazebo moves the paddles exactly like the
2D model the planner rolls out.
"""

from collections import deque
from dataclasses import dataclass
import math

from pa_core.config import load_scene
from pa_core.pushsim import sweep_profile


@dataclass(frozen=True)
class SweepPlan:
    angle_deg: float   # clamped command
    t_out: float       # seconds rest -> angle
    hold: float        # seconds at angle
    t_back: float      # seconds angle -> rest
    settle: float      # seconds after returning before reporting done

    @property
    def motion_time(self) -> float:
        return self.t_out + self.hold + self.t_back

    @property
    def total_time(self) -> float:
        return self.motion_time + self.settle

    def waypoints(self, dt: float = 0.02) -> list:
        """[(time_from_start [s], position [rad], velocity [rad/s])] for a trajectory
        controller: constant angular speed out and back (like pa_core.pushsim), sampled
        every ~dt so cubic interpolation between points stays on the straight ramp.
        Velocity is zero at the turning points and at the end."""
        a = math.radians(self.angle_deg)
        pts = []

        def ramp(t0, T, q0, q1):
            n = max(1, int(math.ceil(T / dt - 1e-9)))
            w = (q1 - q0) / T
            for k in range(1, n + 1):
                last = k == n
                pts.append((t0 + T * k / n, q0 + (q1 - q0) * k / n, 0.0 if last else w))

        ramp(0.0, self.t_out, 0.0, a)
        pts.append((self.t_out + self.hold, a, 0.0))
        ramp(self.t_out + self.hold, self.t_back, a, 0.0)
        return pts

    def angle_at(self, t: float) -> float:
        """Commanded joint angle [rad] t seconds after the start (piecewise linear)."""
        a = math.radians(self.angle_deg)
        if t <= 0:
            return 0.0
        if t < self.t_out:
            return a * t / self.t_out
        if t < self.t_out + self.hold:
            return a
        if t < self.motion_time:
            return a * (self.motion_time - t) / self.t_back
        return 0.0


def clamp_angle(angle_deg: float, scene=None) -> float:
    scene = scene or load_scene()
    mx = float(scene["pushers"]["max_angle"])
    if not math.isfinite(angle_deg):
        return 0.0
    return min(max(float(angle_deg), 0.0), mx)


def plan_sweep(angle_deg: float, settle_s: float = 0.3, scene=None) -> SweepPlan:
    a = clamp_angle(angle_deg, scene)
    t, hold = sweep_profile(a)
    return SweepPlan(a, t, hold, t, settle_s)


@dataclass
class _Active:
    cmd: object
    plan: SweepPlan
    start: float


class PushQueue:
    """FIFO of push commands, executed one at a time.

    tick(now, joint_angle) -> (started, finished):
      started  = (cmd, side, plan) for a sweep that must be sent to the controller now, or None
      finished = the command whose sweep is over (to echo on /push_done), or None
    `joint_angle(side)` returns the measured joint angle [rad] or None if unknown.
    A sweep is finished once its nominal time (+settle) has elapsed and the paddle
    is back near rest, or `rest_timeout` seconds later at the latest.
    """

    def __init__(self, n_sides: int, settle_s: float = 0.3, rest_tol_deg: float = 3.0,
                 rest_timeout: float = 2.0, scene=None):
        self.n_sides = n_sides
        self.settle_s = settle_s
        self.rest_tol = math.radians(rest_tol_deg)
        self.rest_timeout = rest_timeout
        self.scene = scene
        self.pending = deque()
        self.active = None

    def valid(self, cmd) -> bool:
        return 0 <= int(cmd.side) < self.n_sides

    def push(self, cmd) -> None:
        self.pending.append(cmd)

    @property
    def busy(self) -> bool:
        return self.active is not None or bool(self.pending)

    def tick(self, now: float, joint_angle=lambda side: None):
        finished = None
        if self.active is not None:
            act = self.active
            elapsed = now - act.start
            if elapsed >= act.plan.total_time:
                q = joint_angle(int(act.cmd.side))
                at_rest = q is None or abs(q) <= self.rest_tol
                if at_rest or elapsed >= act.plan.total_time + self.rest_timeout:
                    finished = act.cmd
                    self.active = None
        started = None
        if self.active is None and self.pending:
            cmd = self.pending.popleft()
            plan = plan_sweep(cmd.angle_deg, self.settle_s, self.scene)
            self.active = _Active(cmd, plan, now)
            started = (cmd, int(cmd.side), plan)
        return started, finished
