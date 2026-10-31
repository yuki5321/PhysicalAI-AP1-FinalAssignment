"""Point-pusher stroke trajectory (pure python, no ROS).

One PointPushCommand (x0, y0) -> (x1, y1) becomes a piecewise-linear path of
the pusher model origin (= bottom center of the cylinder, world frame):

    rise      straight up to z_travel (only if the pusher is below it)
    approach  line to (x0, y0, z_travel), at travel_speed
    lower     down to (x0, y0, z_down), at vertical_speed
    stroke    straight to (x1, y1, z_down), at the commanded speed
    lift      up to (x1, y1, z_travel), at vertical_speed
    park      line to the park pose, at travel_speed

`z_travel` clears the letters and the perimeter walls, so lifted moves never
touch anything. Start/end points are clamped to stay inside the walls.
The executor samples `pose_at(t)` against sim time and gives Gazebo the
velocity toward the pose one tick ahead (`velocity_at`), so the kinematic body
follows the path between ticks without overshooting corners.
"""

from dataclasses import dataclass, field
import math

WALL_HEIGHT = 0.02          # pa_description.urdf_gen.WALL_HEIGHT
WALL_CLEARANCE = 0.002      # pusher surface to wall inner face, at least


@dataclass
class PointPusherConfig:
    radius: float = 0.006
    height: float = 0.03
    speed: float = 0.05             # default stroke speed [m/s] (msg.speed == 0)
    max_stroke: float = 0.06
    approach: float = 0.01
    park: tuple = (0.14, 0.14, 0.10)
    z_down: float = 0.001           # bottom of the pusher above the table while pushing
    z_travel: float = 0.03          # bottom height for lifted moves (above letters and walls)
    travel_speed: float = 0.2       # lifted moves [m/s]
    vertical_speed: float = 0.1     # lower / lift [m/s]
    min_speed: float = 0.005
    max_speed: float = 0.3
    limit: float = 0.15             # |x|, |y| of the pusher center while down (inside the walls)

    @classmethod
    def from_scene(cls, scene=None, wall_inner_half=None, **kw):
        """Dimensions from scene.yaml point_pusher; `wall_inner_half` [m] from pa_description."""
        if scene is None:
            from pa_core.config import load_scene
            scene = load_scene()
        if wall_inner_half is None:
            from pa_description.urdf_gen import wall_inner_half as wih
            wall_inner_half = wih(scene)
        pp = scene["point_pusher"]
        z_travel = max(WALL_HEIGHT, float(scene["letter"]["thickness"])) + 0.01
        args = dict(
            radius=float(pp["radius"]), height=float(pp["height"]), speed=float(pp["speed"]),
            max_stroke=float(pp["max_stroke"]), approach=float(pp["approach"]),
            park=tuple(float(v) for v in pp["park"]),
            z_travel=z_travel,
            limit=float(wall_inner_half) - float(pp["radius"]) - WALL_CLEARANCE,
        )
        args.update(kw)
        return cls(**args)


@dataclass
class Segment:
    t0: float
    t1: float
    p0: tuple
    p1: tuple
    phase: str

    def at(self, t: float) -> tuple:
        if self.t1 <= self.t0:
            return tuple(self.p1)
        a = min(1.0, max(0.0, (t - self.t0) / (self.t1 - self.t0)))
        return tuple(u + a * (v - u) for u, v in zip(self.p0, self.p1))


@dataclass
class StrokePlan:
    segments: list = field(default_factory=list)
    start: tuple = (0.0, 0.0)       # stroke start actually used (after clamping)
    end: tuple = (0.0, 0.0)
    speed: float = 0.0              # stroke speed actually used
    clamped: bool = False           # start/end/speed had to be changed
    final: tuple = (0.0, 0.0, 0.0)  # pose after the plan (park)

    @property
    def total_time(self) -> float:
        return self.segments[-1].t1 if self.segments else 0.0

    def segment_at(self, t: float):
        for s in self.segments:
            if t < s.t1:
                return s
        return None

    def pose_at(self, t: float) -> tuple:
        s = self.segment_at(t)
        return s.at(t) if s is not None else tuple(self.final)

    def phase_at(self, t: float) -> str:
        s = self.segment_at(t)
        return s.phase if s is not None else "done"

    def velocity_at(self, t: float, h: float) -> tuple:
        """Mean velocity from pose_at(t) to pose_at(t + h): the twist that brings a
        kinematic body to where the path is one tick later."""
        if h <= 0:
            return (0.0, 0.0, 0.0)
        a, b = self.pose_at(t), self.pose_at(t + h)
        return tuple((v - u) / h for u, v in zip(a, b))

    def phase_times(self) -> dict:
        return {s.phase: (s.t0, s.t1) for s in self.segments}


def _clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def _dist(a, b) -> float:
    return math.sqrt(sum((v - u) ** 2 for u, v in zip(a, b)))


def valid_command(x0, y0, x1, y1, speed=0.0) -> bool:
    return all(math.isfinite(float(v)) for v in (x0, y0, x1, y1, speed))


def plan_stroke(cur, x0, y0, x1, y1, speed, cfg: PointPusherConfig) -> StrokePlan:
    """Path from the current pose `cur` (x, y, z) through one straight push and back to park."""
    sx, sy = _clip(x0, cfg.limit), _clip(y0, cfg.limit)
    ex, ey = _clip(x1, cfg.limit), _clip(y1, cfg.limit)
    clamped = (sx, sy, ex, ey) != (x0, y0, x1, y1)
    v = float(speed) if speed and speed > 0 else cfg.speed
    vs = min(cfg.max_speed, max(cfg.min_speed, v))
    clamped = clamped or (speed and speed > 0 and vs != v)
    zt = max(cfg.z_travel, cfg.z_down)

    plan = StrokePlan(start=(sx, sy), end=(ex, ey), speed=vs, clamped=bool(clamped), final=tuple(cfg.park))
    t = 0.0
    p = tuple(float(c) for c in cur)

    def go(q, vel, phase):
        nonlocal t, p
        d = _dist(p, q)
        if d < 1e-9:
            return
        dt = d / vel
        plan.segments.append(Segment(t, t + dt, p, tuple(q), phase))
        t += dt
        p = tuple(q)

    if p[2] < zt - 1e-6:
        go((p[0], p[1], zt), cfg.vertical_speed, "rise")
    go((sx, sy, zt), cfg.travel_speed, "approach")
    go((sx, sy, cfg.z_down), cfg.vertical_speed, "lower")
    go((ex, ey, cfg.z_down), vs, "stroke")
    go((ex, ey, zt), cfg.vertical_speed, "lift")
    go(tuple(cfg.park), cfg.travel_speed, "park")
    return plan
