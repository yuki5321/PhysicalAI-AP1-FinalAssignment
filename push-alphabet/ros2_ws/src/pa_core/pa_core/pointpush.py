"""2D model of one point-pusher stroke (Push-T style), see docs/point_pusher.md.

The pusher is a thin vertical cylinder (`scene.yaml` point_pusher.radius): it
is lowered at (x0, y0), slides in a straight line to (x1, y1) at `speed`, is
lifted, and the letter settles. Top view, it is a kinematic disc.

The world (letter shapes, planar friction, perimeter walls at
`wall_inner_half()`, rebuilt on every `set_pose`) is `pa_core.pushsim.PushSim`
without the paddles, so both pushers share one model of the table.

Like `PushSim.push`, a stroke starts from a fresh world at the current pose,
so its outcome depends only on (pose, stroke): the planner's rollouts and the
evaluation world agree exactly. The part of the stroke before the disc first
touches the letter is not simulated (the letter is at rest then; the contact
distance comes from the geometry). After the stroke the world is rebuilt once
more at the letter's pose and velocity and settles until rest; this makes a
stroke of length L give the same result as the first L of a longer one
(`push_depths`, used by the planner to roll out several lengths at once).
"""

from __future__ import annotations

import math

import numpy as np
import pymunk

from pa_core.config import load_scene
from pa_core.pushsim import MM, PushSim, _outline

_PRE_CONTACT = 0.0005  # m: start the simulated stroke this far before the first contact


def point_pusher_cfg() -> dict:
    return dict(load_scene()["point_pusher"])


def stroke_duration(length: float, speed: float | None = None) -> float:
    """Seconds the disc slides for a stroke of `length` [m] (lowering / lifting not included)."""
    v = float(speed or point_pusher_cfg()["speed"])
    return abs(length) / v


class PointPushSim(PushSim):
    """`set_pose(x, y, theta)`, `push(x0, y0, x1, y1, speed=None) -> (x, y, theta)`, `pose()`."""

    def __init__(self, letter: str, dt: float = 1 / 240):
        pp = point_pusher_cfg()
        self.radius = float(pp["radius"])
        self.speed = float(pp["speed"])
        self.max_stroke = float(pp["max_stroke"])
        self.approach = float(pp["approach"])
        self.disc_friction = 0.5  # pymunk multiplies it with the letter's 0.5
        super().__init__(letter, dt)

    def _build(self, x, y, theta):
        super()._build(x, y, theta)
        # No paddles in point mode.
        self.space.remove(*self.arms, *self.arm_shapes)
        self.arms, self.arm_shapes = [], []

    # ------------------------------------------------------------ geometry

    def contact_distance(self, x0, y0, dx, dy) -> float | None:
        """Distance [m] the disc travels from (x0, y0) along the unit vector
        (dx, dy) until it first touches the letter at its current pose (0 if it
        already overlaps it), or None if the line misses it. Uses the outline
        samples (1 mm), about 0.02 mm accurate."""
        x, y, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        pts = _outline(self.letter) @ np.array([[c, s], [-s, c]]) + (x, y)
        w = pts - (x0, y0)
        along = w[:, 0] * dx + w[:, 1] * dy
        perp = np.abs(w[:, 0] * dy - w[:, 1] * dx)
        r = self.radius
        hit = perp <= r
        if not hit.any():
            return None
        d = along[hit] - np.sqrt(r * r - perp[hit] ** 2)
        d = d[along[hit] >= -r]
        if d.size == 0:
            return None
        return max(0.0, float(d.min()))

    def overlaps(self, x, y, margin: float = 0.0) -> bool:
        """The disc at (x, y) (radius + margin) intersects the letter."""
        lx, ly, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        # point in body frame
        px = (x - lx) * c + (y - ly) * s
        py = -(x - lx) * s + (y - ly) * c
        r = self.radius + margin
        from pa_core.letters import letter_rects
        for rc in letter_rects(self.letter):
            cr, sr = math.cos(rc.angle), math.sin(rc.angle)
            u = (px - rc.cx) * cr + (py - rc.cy) * sr
            v = -(px - rc.cx) * sr + (py - rc.cy) * cr
            du = max(abs(u) - rc.w / 2, 0.0)
            dv = max(abs(v) - rc.h / 2, 0.0)
            if du * du + dv * dv <= r * r:
                return True
        return False

    # ------------------------------------------------------------ simulation

    def push(self, x0, y0, x1, y1, speed=None, settle: float = 0.5):
        """Lower the disc at (x0, y0), slide it to (x1, y1), lift it, let the
        letter settle. Returns the new pose (x, y, theta)."""
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1e-6:
            self.set_pose(*self.pose())
            return self.pose()
        return self.push_depths(x0, y0, (x1 - x0) / L, (y1 - y0) / L, [L], speed, settle)[0]

    def push_depths(self, x0, y0, dx, dy, lengths, speed=None, settle: float = 0.5):
        """Poses after strokes from (x0, y0) along unit (dx, dy) of each length
        in `lengths` (ascending), from one simulation. Identical to calling
        `push` once per length from the same pose."""
        pose0 = self.pose()
        self.set_pose(*pose0)
        v = float(speed or self.speed)
        out = [None] * len(lengths)
        sc = self.contact_distance(x0, y0, dx, dy)
        if sc is None or sc >= lengths[-1]:
            todo = [i for i, L in enumerate(lengths)]
        else:
            todo = [i for i, L in enumerate(lengths) if L <= sc]
        for i in todo:
            out[i] = pose0  # the disc never reaches the letter
        if all(o is not None for o in out):
            return out
        s0 = max(0.0, sc - _PRE_CONTACT)
        step = v * self.dt
        disc = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        disc.position = ((x0 + dx * s0) * MM, (y0 + dy * s0) * MM)
        shape = pymunk.Circle(disc, self.radius * MM)
        shape.friction = self.disc_friction
        self.space.add(disc, shape)
        disc.velocity = (dx * v * MM, dy * v * MM)
        k = 0
        for i, L in enumerate(lengths):
            if out[i] is not None:
                continue
            n = max(0, int(round((L - s0) / step)))
            while k < n:
                self.space.step(self.dt)
                k += 1
            out[i] = self._settled(settle)
        self.space.remove(disc, shape)
        self.set_pose(*out[-1])
        return out

    def _settled(self, settle):
        """Pose after lifting the disc now and letting the letter come to rest.
        Done in a fresh world (pose + velocity), so the stroke's world goes on
        unchanged for longer lengths."""
        b = self.body
        state = (b.position, b.angle, b.velocity, b.angular_velocity)
        space, body = self.space, self.body
        keep = (self.space, self.body, self.shapes, self.friction, self.arms, self.arm_shapes,
                self.walls)
        self._build(state[0].x / MM, state[0].y / MM, state[1])
        self.body.velocity = state[2]
        self.body.angular_velocity = state[3]
        for _ in range(max(1, int(round(settle / self.dt)))):
            self.space.step(self.dt)
            if self._at_rest():
                break
        p = self.pose()
        (self.space, self.body, self.shapes, self.friction, self.arms, self.arm_shapes,
         self.walls) = keep
        assert self.space is space and self.body is body
        return p


def simulate_stroke(letter, pose, stroke, speed=None):
    sim = PointPushSim(letter)
    sim.set_pose(*pose)
    return sim.push(*stroke, speed=speed)
