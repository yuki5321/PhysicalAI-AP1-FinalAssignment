"""Fast top-down 2D model of one paddle push (pymunk), used by the planner.

Same idea as the Push-T environment: the block slides on a table, and planar
friction is emulated by a pivot joint and a gear joint to a static body whose
max forces equal the Coulomb friction force / torque. Internally millimeters.

The table has the same low perimeter walls as the Gazebo world
(`wall_inner_half`), so a letter pushed off the workspace stops at a wall
instead of sliding away forever.

`push` only simulates the part of a sweep that can matter (`fast=True`, the
default): the paddle starts just before the angle where it first touches the
letter (`contact_angle`, from the geometry), the return sweep stops once the
paddle has left the letter, and settling stops once the letter is at rest.
A sweep that never reaches the letter is not simulated at all. The result
matches the full simulation (`fast=False`) to a fraction of a millimeter and
is several times cheaper, which the planner spends on more candidates.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
import pymunk

from pa_core.config import load_scene
from pa_core.letters import _union_mask, letter_area, letter_rects

MM = 1000.0
G = 9810.0  # mm/s^2


@dataclass
class Pusher:
    name: str
    pivot: tuple  # m
    rest_heading: float  # rad


WALL_MARGIN = 0.01  # gap between the farthest paddle sweep and the walls (as pa_description)
_FF_MARGIN = math.radians(3.0)  # fast push: start this far before the contact angle
_BOUNDARY_STEP = 0.001  # m, spacing of letter outline samples for contact_angle


@lru_cache(maxsize=None)
def _mean_radius(letter: str) -> float:
    """Mean distance of the letter's area from its centroid [m] (friction torque arm)."""
    pts, inside, _ = _union_mask(letter_rects(letter))
    return float(np.linalg.norm(pts[inside], axis=1).mean())


def pushers() -> list:
    sc = load_scene()["pushers"]
    return [Pusher(p["name"], tuple(p["pivot"]), math.radians(p["rest_heading"])) for p in sc["list"]]


def max_sweep_extent(step_deg: float = 1.0) -> float:
    """Largest |x| or |y| [m] reached by any paddle (with its thickness) over 0..max_angle."""
    sp = load_scene()["pushers"]
    L, r = sp["arm_length"], sp["arm_thickness"] / 2
    ext = 0.0
    for p in sp["list"]:
        px, py = p["pivot"]
        for i in range(int(sp["max_angle"] / step_deg) + 1):
            a = math.radians(p["rest_heading"] + i * step_deg)
            c, s = math.cos(a), math.sin(a)
            for u, v in ((0, -r), (0, r), (L, -r), (L, r)):
                ext = max(ext, abs(px + u * c - v * s), abs(py + u * s + v * c))
    return ext


@lru_cache(maxsize=1)
def wall_inner_half() -> float:
    """Inner half-size [m] of the perimeter walls; same rule as
    pa_description.urdf_gen.wall_inner_half (outside the plate and every sweep)."""
    sc = load_scene()
    return max(sc["workspace"]["table_half_size"], max_sweep_extent() + WALL_MARGIN)


@lru_cache(maxsize=None)
def _outline(letter: str) -> np.ndarray:
    """Points [m] every ~1 mm along the outline of each rectangle (body frame)."""
    pts = []
    for r in letter_rects(letter):
        c = r.corners()
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            n = max(2, int(math.ceil(np.linalg.norm(b - a) / _BOUNDARY_STEP)) + 1)
            pts.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
    return np.concatenate(pts)


def sweep_profile(angle_deg: float):
    """(sweep seconds, hold seconds) for one push; shared with the Gazebo driver."""
    sc = load_scene()["pushers"]
    t = max(0.15, abs(angle_deg) / sc["speed_deg_s"])
    return t, sc["hold_s"]


class PushSim:
    def __init__(self, letter: str, dt: float = 1 / 240):
        self.letter = letter
        self.dt = dt
        self.scene = load_scene()
        sl = self.scene["letter"]
        mass = sl["mass"] * 1000.0  # grams, any consistent unit works
        mu = sl["friction"]
        area = letter_area(letter)
        self._polys = [([tuple(v) for v in r.corners() * MM], mass * r.w * r.h / area)
                       for r in letter_rects(letter)]
        self._f_max = mu * mass * G
        self._tau_max = mu * mass * G * _mean_radius(letter) * MM
        sp = self.scene["pushers"]
        self.arm_len = sp["arm_length"] * MM
        self.arm_r = sp["arm_thickness"] * MM / 2
        self.pushers = pushers()
        w = wall_inner_half() * MM
        self._wall_corners = [(-w, -w), (w, -w), (w, w), (-w, w)]
        self.space = None
        self._build(0.0, 0.0, 0.0)

    def _build(self, x, y, theta):
        """A new pymunk world with the letter at (x, y, theta) [m, rad].

        Everything is created anew, so that a rollout depends only on (pose,
        action): reusing the space (or even only the bodies) kept solver
        state that wedging contacts amplified into centimeters of difference
        in where a push throws the letter."""
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.space.iterations = 20

        self.body = pymunk.Body()  # mass and moment come from the shapes
        self.shapes = []
        for verts, m in self._polys:
            sh = pymunk.Poly(self.body, verts)
            sh.friction = 0.5
            sh.mass = m
            self.shapes.append(sh)
        # Angle first: pymunk places the body through its center of gravity
        # using the current angle.
        self.body.angle = theta
        self.body.position = (x * MM, y * MM)
        self.space.add(self.body, *self.shapes)

        # Planar friction: velocity-only constraints to the static body (no
        # position correction), with the Coulomb force / torque as their limits.
        static = self.space.static_body
        pivot = pymunk.PivotJoint(static, self.body, (0, 0), (0, 0))
        pivot.max_bias = 0
        pivot.max_force = self._f_max
        gear = pymunk.GearJoint(static, self.body, 0.0, 1.0)
        gear.max_bias = 0
        gear.max_force = self._tau_max
        self.friction = (pivot, gear)
        self.space.add(*self.friction)

        self.arms = []
        self.arm_shapes = []
        for p in self.pushers:
            b = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            b.position = (p.pivot[0] * MM, p.pivot[1] * MM)
            b.angle = p.rest_heading
            seg = pymunk.Segment(b, (0, 0), (self.arm_len, 0), self.arm_r)
            seg.friction = 0.5
            self.space.add(b, seg)
            self.arms.append(b)
            self.arm_shapes.append(seg)

        # Perimeter walls (static), as in the Gazebo world.
        c = self._wall_corners
        self.walls = []
        for i in range(4):
            seg = pymunk.Segment(static, c[i], c[(i + 1) % 4], 1.0)
            seg.friction = 0.5
            self.walls.append(seg)
        self.space.add(*self.walls)

    def set_pose(self, x, y, theta):
        """Place the letter at rest at (x, y, theta), paddles at rest. Starts a
        fresh world; objects a caller added to `space` (e.g. a hand) are kept."""
        old = self.space
        ours = {id(o) for o in [self.body, *self.shapes, *self.friction, *self.arms,
                                 *self.arm_shapes, *self.walls]}
        extra = [o for o in list(old.bodies) + list(old.shapes) + list(old.constraints)
                 if id(o) not in ours]
        self._build(x, y, theta)
        if extra:
            old.remove(*extra)
            self.space.add(*extra)

    def pose(self):
        return self.body.position.x / MM, self.body.position.y / MM, self.body.angle

    def contact_angle(self, side: int):
        """Sweep angle [deg] at which paddle `side` first touches the letter at
        its current pose (0 if it already touches it at rest), or None if no
        angle up to 180 deg reaches it. Geometric, about 0.2 deg accurate."""
        p = self.pushers[side]
        x, y, th = self.pose()
        c, s = math.cos(th), math.sin(th)
        pts = _outline(self.letter) * MM @ np.array([[c, s], [-s, c]]) + (x * MM, y * MM)
        d = pts - (p.pivot[0] * MM, p.pivot[1] * MM)
        dist = np.hypot(d[:, 0], d[:, 1])
        L, r = self.arm_len, self.arm_r
        ok = (dist > r) & (dist <= L + r)
        if not ok.any():
            return None
        dist, d = dist[ok], d[ok]
        phi = np.arctan2(d[:, 1], d[:, 0]) - p.rest_heading
        # Leading face of the arm (|point| <= L) or its rounded tip (beyond L).
        face = np.arcsin(r / dist)
        cos_tip = np.clip((dist ** 2 + L * L - r * r) / (2 * dist * L), -1.0, 1.0)
        lead = np.where(dist <= L, face, np.arccos(cos_tip))
        a = np.degrees(np.mod(phi - lead + math.radians(5.0), 2 * math.pi)) - 5.0
        a = a[a <= 180.0]
        if a.size == 0:
            return None
        return max(0.0, float(a.min()))

    def _overlapping(self) -> bool:
        """The letter already intersects a paddle at rest or a wall."""
        return any(q.shape.body is not self.body
                   for sh in self.shapes for q in self.space.shape_query(sh))

    def _at_rest(self) -> bool:
        return abs(self.body.velocity) < 0.05 and abs(self.body.angular_velocity) < 1e-3

    def _run(self, seconds, arm=None, omega=0.0):
        if arm is not None:
            arm.angular_velocity = omega
        for _ in range(max(1, int(round(seconds / self.dt)))):
            self.space.step(self.dt)
        if arm is not None:
            arm.angular_velocity = 0.0

    def push(self, side: int, angle_deg: float, settle: float = 0.3, fast: bool = True):
        """Sweep paddle `side` out to `angle_deg` and back. Returns the new pose.

        `fast=False` simulates the whole profile (sweep out, hold, sweep back,
        settle); the default skips the parts in which nothing can move.

        The letter is at rest between pushes, so the push starts from a fresh
        world at the current pose: its outcome depends only on (pose, side,
        angle), as in the planner's rollouts, not on solver state left over
        from the previous push."""
        self.set_pose(*self.pose())
        if not fast:
            return self._push_full(side, angle_deg, settle)
        arm = self.arms[side]
        rest = self.pushers[side].rest_heading
        target = math.radians(angle_deg)
        if self._overlapping():
            return self._push_full(side, angle_deg, settle)  # resolve the overlap as usual
        a0 = self.contact_angle(side)
        if a0 is None or angle_deg <= 0 or math.radians(a0) - _FF_MARGIN >= target:
            return self.pose()  # the paddle never reaches the letter
        t, hold = sweep_profile(angle_deg)
        omega = target / t
        n = max(1, int(round(t / self.dt)))
        # Skip whole steps before the contact: the letter does not move then,
        # and the paddle is where the full simulation would have put it.
        k0 = min(n, max(0, int((math.radians(a0) - _FF_MARGIN) / (omega * self.dt))))
        arm.angle = rest + omega * self.dt * k0
        self.space.reindex_shapes_for_body(arm)
        arm.angular_velocity = omega
        for _ in range(n - k0):
            self.space.step(self.dt)
        arm.angular_velocity = 0.0
        self._run(hold)
        # Back: stop once the letter is at rest and the paddle is below the
        # angle where it would touch the letter (it only moves away from there).
        arm.angular_velocity = -omega
        for k in range(n):
            self.space.step(self.dt)
            if k % 8 == 7 and self._at_rest():
                a = self.contact_angle(side)
                if a is None or arm.angle - rest < math.radians(a) - _FF_MARGIN:
                    break
        arm.angular_velocity = 0.0
        arm.angle = rest
        self.space.reindex_shapes_for_body(arm)
        # Settle: until the letter is at rest (or `settle` seconds).
        for _ in range(max(1, int(round(settle / self.dt)))):
            self.space.step(self.dt)
            if self._at_rest():
                break
        return self.pose()

    def _push_full(self, side, angle_deg, settle):
        arm = self.arms[side]
        t, hold = sweep_profile(angle_deg)
        omega = math.radians(angle_deg) / t
        self._run(t, arm, omega)
        self._run(hold)
        self._run(t, arm, -omega)
        arm.angle = self.pushers[side].rest_heading
        self._run(settle)
        return self.pose()


def simulate_push(letter, pose, side, angle_deg, fast=True):
    sim = PushSim(letter)
    sim.set_pose(*pose)
    return sim.push(side, angle_deg, fast=fast)
