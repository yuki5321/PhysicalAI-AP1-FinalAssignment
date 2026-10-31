"""Choose a point-pusher stroke by simulating candidates (docs/point_pusher.md).

Candidates come from the letter's geometry at the current pose: the letter is
a union of rectangles, and on every rectangle edge that is part of the
letter's outline a few contact points are taken (1 to 3 per edge, by length).
From each, the disc pushes along the inward normal and +-30 deg; its path
passes the point where it touches the edge, and it is lowered `approach`
before that (a start that would land on the letter is dropped). Each
(start, direction) is simulated once in `PointPushSim` for several push
depths (`push_depths`). Strokes whose start or end leaves the table are
dropped.

Search (same scheme as `pa_core.planner.PushPlanner`): the first-step
candidates are ranked by `pa_core.planner.cost`; the best ROBUST_TOP are
re-ranked by their worst outcome under small pose perturbations; the best
BEAM are expanded one more step with a coarser candidate set (2-step
lookahead). Near the target ("fine" mode) shorter depths are added to the
first step. `explore > 0`: with that probability a random action among the
better first steps that keep the letter in the workspace.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import random

import numpy as np

from pa_core.config import load_scene
from pa_core.letters import letter_rects, pose_error
from pa_core.planner import _moved, _outside, aligned, cost
from pa_core.pointpush import PointPushSim

ANGLES = (0.0, math.radians(30.0), math.radians(-30.0))
DEPTHS = (0.01, 0.02, 0.04)          # + (max_stroke - approach)
FINE_DEPTHS = (0.003, 0.006)
FINE_POS = 0.02                      # fine mode below this position error [m] ...
FINE_ANG = 20.0                      # ... and this angle error [deg]
COARSE_DEPTHS = (0.01, 0.03)         # lookahead step (+ max)
COARSE_ANGLES = (0.0,)               # lookahead step: inward normal only
BEAM = 4
ROBUST_TOP = 8
PERTURB = ((0.0005, -0.0005, math.radians(1.0)), (-0.0005, 0.0005, math.radians(-1.0)))
START_CLEARANCE = 0.001              # m between the lowered disc and the letter
SEQ_PENALTY = 0.5                    # a 2-step sequence must beat stopping now by this


@dataclass
class PointPlan:
    x0: float
    y0: float
    x1: float
    y1: float
    predicted: tuple
    cost: float

    @property
    def stroke(self):
        return (self.x0, self.y0, self.x1, self.y1)


def _edge_points(n_edge: int) -> list:
    return [(i + 0.5) / n_edge for i in range(n_edge)]


@lru_cache(maxsize=None)
def contact_points(letter: str, coarse: bool = False) -> tuple:
    """((qx, qy, nx, ny), ...) in the body frame: points on the letter's outline
    (on rectangle edges, not inside another rectangle) and their outward normals."""
    rects = letter_rects(letter)
    out = []
    eps = 0.0005
    for ri, r in enumerate(rects):
        c = r.corners()
        for i in range(4):
            a, b = c[i], c[(i + 1) % 4]
            e = b - a
            L = float(np.hypot(*e))
            # corners are CCW: outward normal is the edge rotated by -90 deg
            n = np.array([e[1], -e[0]]) / L
            k = 1 if coarse else int(min(3, max(1, round(L / 0.022))))
            for f in _edge_points(k):
                q = a + f * e
                probe = (q + n * eps)[None, :]
                if any(o.contains(probe)[0] for o in rects):
                    continue
                out.append((float(q[0]), float(q[1]), float(n[0]), float(n[1])))
    return tuple(out)


class PointPushPlanner:
    def __init__(self, letter: str, beam: int = BEAM, robust_top: int = ROBUST_TOP):
        self.letter = letter
        self.beam = beam
        self.robust_top = robust_top
        self.sim = PointPushSim(letter)
        sc = load_scene()
        self.r = self.sim.radius
        self.approach = self.sim.approach
        self.max_stroke = self.sim.max_stroke
        self.limit = float(sc["workspace"]["table_half_size"]) - self.r
        t = sc["target"]
        self._tgt = (t["x"], t["y"])

    # ------------------------------------------------------------ candidates

    def strokes(self, pose, depths, coarse=False, angles=ANGLES):
        """[(x0, y0, dx, dy, [total lengths ascending])] at `pose`."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        self.sim.set_pose(*pose)
        lim = self.limit
        out = []
        dep = sorted(set(min(d, self.max_stroke - self.approach) for d in depths))
        for qx, qy, nx, ny in contact_points(self.letter, coarse):
            wq = (x + c * qx - s * qy, y + s * qx + c * qy)
            wn = (c * nx - s * ny, s * nx + c * ny)
            px, py = wq[0] + wn[0] * self.r, wq[1] + wn[1] * self.r  # disc center at contact
            for a in angles:
                ca, sa = math.cos(a), math.sin(a)
                dx = -(wn[0] * ca - wn[1] * sa)
                dy = -(wn[0] * sa + wn[1] * ca)
                x0, y0 = px - dx * self.approach, py - dy * self.approach
                if abs(x0) > lim or abs(y0) > lim:
                    continue
                if self.sim.overlaps(x0, y0, START_CLEARANCE):
                    continue
                lens = []
                for d in dep:
                    L = self.approach + d
                    if abs(x0 + dx * L) <= lim and abs(y0 + dy * L) <= lim:
                        lens.append(L)
                if lens:
                    out.append((x0, y0, dx, dy, lens))
        return out

    def _expand(self, pose, depths, coarse=False):
        """[(cost, (x0, y0, x1, y1), predicted)] of strokes that move the letter, best first."""
        res = []
        angles = COARSE_ANGLES if coarse else ANGLES
        for x0, y0, dx, dy, lens in self.strokes(pose, depths, coarse, angles):
            self.sim.set_pose(*pose)
            preds = self.sim.push_depths(x0, y0, dx, dy, lens)
            for L, p in zip(lens, preds):
                if _moved(pose, p):
                    res.append((cost(self.letter, p), (x0, y0, x0 + dx * L, y0 + dy * L), p))
        res.sort(key=lambda t: t[0])
        return res

    def rollout(self, pose, stroke):
        self.sim.set_pose(*pose)
        return self.sim.push(*stroke)

    def _fine(self, pose) -> bool:
        dp, da = pose_error(self.letter, *pose)
        return dp < FINE_POS and da < FINE_ANG

    def _first_step(self, pose):
        depths = DEPTHS + (self.max_stroke,)
        if self._fine(pose):
            depths = FINE_DEPTHS + depths
        first = self._expand(pose, depths)
        out = []
        for i, (c, st, p) in enumerate(first):
            if i < self.robust_top:
                for dx, dy, dth in PERTURB:
                    q = (pose[0] + dx, pose[1] + dy, pose[2] + dth)
                    c = max(c, cost(self.letter, self.rollout(q, st)))
            out.append((c, st, p))
        out.sort(key=lambda t: t[0])
        return out

    # ------------------------------------------------------------ plan

    def plan(self, pose, explore: float = 0.0) -> PointPlan:
        pose = tuple(float(v) for v in pose)
        first = self._first_step(pose)
        if not first:
            # Nothing reachable: a zero-length stroke at the letter, prediction unchanged.
            return PointPlan(pose[0], pose[1], pose[0], pose[1], pose, cost(self.letter, pose))

        if explore > 0 and random.random() < explore:
            safe = [f for f in first if _outside(f[2]) == 0.0][: max(3 * self.beam, 10)]
            c, st, p = random.choice(safe or first)
            return PointPlan(*st, p, c)

        c1, st, p = first[0]
        best = PointPlan(*st, p, c1)
        if aligned(self.letter, p):
            return best
        coarse_depths = COARSE_DEPTHS + (self.max_stroke,)
        for c1, st, p in first[: self.beam]:
            nxt = self._expand(p, coarse_depths, coarse=True)
            c2 = nxt[0][0] if nxt else math.inf
            score = min(c1, c2 + SEQ_PENALTY)
            if score < best.cost:
                best = PointPlan(*st, p, score)
        return best
