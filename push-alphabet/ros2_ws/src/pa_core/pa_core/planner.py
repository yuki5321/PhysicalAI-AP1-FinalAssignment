"""Choose which paddle to swing, and how far, by simulating candidates.

A small model-predictive search: every (side, angle) candidate is rolled out
in the 2D push model, the best few are expanded one more step, and the first
action of the cheapest sequence is executed. The loop closes through the
camera, so model error is corrected on the next step.

Details that matter in closed loop:
- Candidates that do not touch the letter are dropped: "do nothing, then X"
  is never better than "X", and choosing it would repeat forever. Grid angles
  below each paddle's contact angle (`PushSim.contact_angle`, geometric) are
  not even simulated.
- The second step uses a coarser angle grid (10 deg): it only has to rank
  the beam, and the time saved pays for a wider beam (8 instead of 5).
- If one push already aligns the letter (within ALIGN_MARGIN of the
  tolerances, as the state machine requires), it is returned without lookahead.
- Exploration (when the caller detects a stall) picks a random action among
  the better first-step candidates that keep the letter in the workspace,
  instead of any action, which often threw the letter off the table.

Refinements measured with tools/eval_recovery.py (docs/recovery_results.md):
- EDGE_WEIGHT: extra cost beyond EDGE_MARGIN from the center. Next to a
  paddle pivot no paddle can push the letter back toward the center, so a
  step that trades position for angle near the edge can end in a corner trap.
- ROBUST_TOP / PERTURB: the best first steps are ranked by their worst
  outcome under small pose / servo-angle perturbations. Pushes that throw the
  letter (it slides along the paddle, spins around its tip, hits a wall) are
  chaotic, and the nominal rollout can promise anything.
- BEAM3: when stalled (`explore > 0`), the best two-step sequences are
  expanded once more (three-step lookahead).
  Together these were the only setting at least as good as the old planner on
  both seeds of both disturbance modes (they help most after hand strokes,
  which leave the letter near the edge).
- DELTAS1 / DELTAS2 (off): extra angles just past each paddle's contact angle
  (fine pushes). They made the planner dither near the target.
"""

from dataclasses import dataclass
import math
import random

from pa_core.config import load_scene
from pa_core.letters import angle_error_deg, pose_error
from pa_core.pushsim import PushSim

# A push whose predicted motion is below this is treated as "no contact".
_NOOP_POS = 1e-4  # m
_NOOP_ANG = math.radians(0.1)

# Degrees past the contact angle (first step / lookahead steps).
DELTAS1 = ()
DELTAS2 = ()

# Edge cost: beyond EDGE_MARGIN [m] from the center (Chebyshev, |x| or |y|),
# each millimeter costs EDGE_WEIGHT millimeters of position error extra.
EDGE_MARGIN = 0.05
EDGE_WEIGHT = 2.0

# Robustness check of the best first-step candidates: the same push from
# slightly different poses / angles (perception and servo error). Pushes that
# throw the letter (it slides along the paddle, spins around its tip, bounces
# off a wall) are chaotic: the nominal rollout can promise the target while
# the real outcome lands anywhere. Their cost is the worst of these rollouts.
PERTURB = ((0.0005, -0.0005, math.radians(1.0), 1.0),
           (-0.0005, 0.0005, math.radians(-1.0), -1.0))
ROBUST_TOP = 16

# Stalled (explore > 0): expand this many of the best two-step sequences once more.
BEAM3 = 4


@dataclass
class Plan:
    side: int
    angle_deg: float
    predicted: tuple
    cost: float


def _outside(pose) -> float:
    """How far (m) the letter centroid is outside the workspace (0 inside)."""
    ws = load_scene()["workspace"]["half_size"]
    return max(0.0, abs(pose[0]) - ws, abs(pose[1]) - ws)


def cost(letter, pose) -> float:
    sc = load_scene()
    t = sc["target"]
    x, y, th = pose
    dp = math.hypot(x - t["x"], y - t["y"])
    da = abs(angle_error_deg(letter, th, math.radians(t["theta"])))
    c = dp / t["pos_tol"] + da / t["ang_tol"]
    edge = max(abs(x), abs(y)) - EDGE_MARGIN
    if edge > 0:
        c += EDGE_WEIGHT * edge / t["pos_tol"]
    out = _outside(pose)
    if out > 0:
        c += 100.0 + out / t["pos_tol"]
    return c


# A prediction counts as "aligned" (no lookahead needed) within this fraction
# of the scene tolerances; the state machine uses the same margin
# (Params.done_margin) before it declares a letter done.
ALIGN_MARGIN = 0.8


def aligned(letter, pose, margin: float = ALIGN_MARGIN) -> bool:
    t = load_scene()["target"]
    dp, da = pose_error(letter, *pose)
    return dp < margin * t["pos_tol"] and da < margin * t["ang_tol"]


def _moved(a, b) -> bool:
    return (math.hypot(a[0] - b[0], a[1] - b[1]) > _NOOP_POS
            or abs(math.remainder(a[2] - b[2], 2 * math.pi)) > _NOOP_ANG)


class PushPlanner:
    def __init__(self, letter: str, angles=None, beam: int = 8, angles2=None,
                 deltas=DELTAS1, deltas2=DELTAS2, beam3: int | None = None):
        self.letter = letter
        sc = load_scene()["pushers"]
        self.max_angle = float(sc["max_angle"])
        max_angle = int(self.max_angle)
        # Fixed grids (only used when given explicitly; the default is
        # contact-relative angles, see the module docstring).
        self.angles = list(angles or range(5, max_angle + 1, 5))
        self.angles2 = list(angles2 or range(10, max_angle + 1, 10))
        self.deltas = tuple(deltas)
        self.deltas2 = tuple(deltas2)
        self.n_sides = len(sc["list"])
        self.beam = beam
        self.beam3 = BEAM3 if beam3 is None else beam3
        self.sim = PushSim(letter)

    def rollout(self, pose, side, angle):
        self.sim.set_pose(*pose)
        return self.sim.push(side, angle)

    def contact_angles(self, pose):
        """Per side: sweep angle [deg] of first contact, or None if out of reach."""
        self.sim.set_pose(*pose)
        out = []
        for s in range(self.n_sides):
            a = self.sim.contact_angle(s)
            out.append(a if a is not None and a < self.max_angle else None)
        return out

    def candidates(self, angles=None, pose=None, deltas=None):
        """(side, angle) pairs: the angle grid for every side, or with `pose`
        only the grid angles that reach the letter plus `deltas` past contact."""
        grid = list(angles or self.angles)
        if pose is None:
            return [(s, a) for s in range(self.n_sides) for a in grid]
        out = []
        for s, a0 in enumerate(self.contact_angles(pose)):
            if a0 is None:
                continue
            seen = set()
            fine = [round(a0 + d, 1) for d in (deltas or ())]
            for a in fine + [a for a in grid if a > a0]:
                if a <= self.max_angle and a not in seen:
                    seen.add(a)
                    out.append((s, a))
        return out

    def _cands(self, pose, step):
        if step == 1:
            return self.candidates(self.angles, pose, self.deltas)
        return self.candidates(self.angles2, pose, self.deltas2)

    def _expand(self, pose, step):
        """[(cost, side, angle, predicted)] of the pushes that move the letter, best first."""
        out = []
        for s, a in self._cands(pose, step):
            p = self.rollout(pose, s, a)
            if _moved(pose, p):
                out.append((cost(self.letter, p), s, a, p))
        out.sort(key=lambda t: t[0])
        return out

    def _first_step(self, pose):
        first = self._expand(pose, 1)
        if not PERTURB or ROBUST_TOP <= 0:
            return first
        out = []
        for i, (c, s, a, p) in enumerate(first):
            if i < ROBUST_TOP:
                for dx, dy, dth, da in PERTURB:
                    q = (pose[0] + dx, pose[1] + dy, pose[2] + dth)
                    a2 = min(max(a + da, 0.0), self.max_angle)
                    c = max(c, cost(self.letter, self.rollout(q, s, a2)))
            out.append((c, s, a, p))
        out.sort(key=lambda t: t[0])
        return out

    def plan(self, pose, explore: float = 0.0) -> Plan:
        first = self._first_step(pose)
        if not first:
            # No paddle reaches the letter (e.g. it left the table). Return a
            # valid Plan whose prediction is "unchanged" so callers can detect it.
            s, a = 0, self.max_angle
            return Plan(s, a, tuple(pose), cost(self.letter, pose))

        if explore > 0 and random.random() < explore:
            safe = [f for f in first if _outside(f[3]) == 0.0][: max(3 * self.beam, 10)]
            c, s, a, p = random.choice(safe or first)
            return Plan(s, a, p, c)

        c1, s, a, p = first[0]
        best = Plan(s, a, p, c1)
        if aligned(self.letter, p):
            return best

        # Two-step lookahead over the beam.
        seconds = []  # (c2, index in first, second-step pose)
        for i, (c1, s, a, p) in enumerate(first[: self.beam]):
            nxt = self._expand(p, 2)
            if nxt:
                c2, _, _, p2 = nxt[0]
                seconds.append((c2, i, p2))
            else:
                c2 = math.inf
            # A two-step sequence is only worth it if it ends better than stopping now.
            score = min(c1, c2 + 0.5)
            if score < best.cost:
                best = Plan(s, a, p, score)

        if explore > 0 and self.beam3 > 0:
            # Stalled: look one step further from the best two-step sequences.
            seconds.sort(key=lambda t: t[0])
            for c2, i, p2 in seconds[: self.beam3]:
                if aligned(self.letter, p2):
                    continue
                nxt = self._expand(p2, 2)
                if not nxt:
                    continue
                score = nxt[0][0] + 1.0
                if score < best.cost:
                    c1, s, a, p = first[i]
                    best = Plan(s, a, p, score)
        return best
