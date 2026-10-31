"""ROS-free backend for `tools/eval_recovery.py` (see docs/recovery_eval.md).

World = `pa_core.pushsim.PushSim` (the same 2D model the planner uses, so this
measures the planner / state machine / perception loop, not model error).
Observation = `--perception`:

    truth     true pose (+ optional Gaussian noise `obs_noise=(sigma_pos_m, sigma_ang_rad)`);
              while the hand is in view a letter more than half hidden under it is not seen
    classic   render the true pose with `pa_perception.synth` (paddles at rest, the hand
    cnn       drawn on top while it is in view) and run the corresponding detector on
    hybrid    the image

The clock is virtual: `push` advances it by the sweep duration (`sweep_profile`,
out + hold + back) plus the PushSim settle time, `wait(dt)` by `dt`.

`pusher="point"` (docs/point_pusher.md): the world is `pa_core.pointpush.PointPushSim`
(no paddles) and pushes are `push_stroke(x0, y0, x1, y1)`; the clock advances by
POINT_OVERHEAD_S (move over the start, lower, lift) + stroke length / speed + settle.

Disturbances (`disturb(mode, rng)`):

    teleport  random disturbed pose in the workspace
    hand      the Gazebo hand (pa_description/models/hand: palm + finger boxes, as
              kinematic pymunk polygons) driven along a stroke exactly like
              recovery_gazebo / hand_actor: fly in from the park pose at z_up, come
              down behind the letter, slide through it (random heading / offset /
              overshoot, speed limited to `hand_vmax`), rise and fly away
              (`hand_logic.step_hand`). It touches the letter only while its boxes
              are below the letter top. Up to `hand_tries` strokes (each from the
              same start pose, not seen by the robot), then a teleport fallback.
              The stroke is also recorded as a timeline of /hand/state-like events
              and observations (`last_disturb["events"]`) that eval_recovery replays
              into the state machine.
    disc      the old model: a kinematic disc (one finger) swept through the letter;
              no hand events (blocking, like teleport)

`plan_hand_stroke` / `run_hand_stroke` do one stroke from the current state, for
a hand that comes back mid-recovery (`--rehand-prob`).

Hand presence follows /hand/state as hand_actor sees it (the person's hand over
the table): present from the moment the hand leaves the park pose until it starts
to lift off; the model is still in camera view for a while after that (rising
and flying away), and is drawn / occludes until it leaves the view.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pymunk

from pa_core.config import load_scene
from pa_core.letters import _union_mask, is_disturbed, letter_rects, transform
from pa_core.pointpush import PointPushSim, stroke_duration
from pa_core.pushsim import MM, PushSim, sweep_profile

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("ros2_ws/src/pa_gazebo",):
    _full = os.path.join(_HERE, "..", _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.append(_full)

from recovery_gazebo import hand_trajectory, sample_hand_stroke, stroke_cfg  # noqa: E402

PERCEPTIONS = ("truth", "classic", "cnn", "hybrid")
DISTURBS = ("teleport", "hand", "disc")
PUSHERS = ("paddles", "point")
# Point pusher: time to move over the start, lower, and lift after the stroke [s]
# (added to the stroke time; the park -> start travel overlaps the settle wait).
POINT_OVERHEAD_S = 0.6
PADDLE_BGR = (20, 100, 220)  # orange paddles, as in the perception tests
HAND_BGR = (140, 170, 215)   # skin, inside gen_dataset's hand color ranges
HAND_SIDE_BGR = (105, 128, 162)

# pa_description/models/hand/model.sdf collision boxes, link frame (palm center at
# the bottom of the palm, fingers toward +y): (cx, cy, size_x, size_y, yaw, z_bottom, z_top).
HAND_BOXES = (
    (0.0, 0.0, 0.075, 0.08, 0.0, 0.000, 0.024),     # palm
    (-0.026, 0.068, 0.016, 0.058, 0.0, 0.001, 0.019),  # index
    (-0.008, 0.073, 0.017, 0.066, 0.0, 0.001, 0.019),  # middle
    (0.01, 0.07, 0.016, 0.06, 0.0, 0.001, 0.019),      # ring
    (0.027, 0.06, 0.014, 0.045, 0.0, 0.002, 0.018),    # little
    (-0.05, 0.01, 0.018, 0.05, 0.6, 0.001, 0.019),     # thumb
)
HAND_DT = 1.0 / 30.0     # hand update rate (recovery_gazebo hand_rate_hz)
OBS_EVERY = 3            # one observation every 3 hand steps (10 Hz camera)
OCCLUDED_FRAC = 0.5      # truth perception: hidden above this fraction -> not seen


def _box_local_corners(cx, cy, w, h, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    loc = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return loc @ np.array([[c, -s], [s, c]]).T + [cx, cy]


_HAND_LOCAL = [(_box_local_corners(*b[:5]), b[5], b[6]) for b in HAND_BOXES]


def hand_world_boxes(x, y, yaw):
    """[(corners (4, 2) world xy, z_bottom_offset, z_top_offset)] of the hand at (x, y, yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return [(loc @ rot.T + [x, y], zb, zt) for loc, zb, zt in _HAND_LOCAL]


def _inside_convex(pts, corners):
    """Points inside a convex quad (either winding)."""
    e0, e1 = corners[1] - corners[0], corners[2] - corners[1]
    sign = 1.0 if e0[0] * e1[1] - e0[1] * e1[0] > 0 else -1.0
    ok = np.ones(len(pts), dtype=bool)
    for i in range(len(corners)):
        a, b = corners[i], corners[(i + 1) % len(corners)]
        cr = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
        ok &= cr * sign >= 0
    return ok


class Sim2DBackend:
    name = "sim2d"

    def __init__(self, perception: str = "truth", obs_noise=(0.0, 0.0), settle_s: float = 0.3,
                 image_noise: float = 2.0, teleport_range: float = 0.07,
                 hand_radius: float = 0.015, hand_speed: float = 0.25, model_path=None,
                 hand_vmax: float = 0.15, hand_tries: int = 4, pusher: str = "paddles"):
        if perception not in PERCEPTIONS:
            raise ValueError(f"perception must be one of {PERCEPTIONS}, got {perception!r}")
        if pusher not in PUSHERS:
            raise ValueError(f"pusher must be one of {PUSHERS}, got {pusher!r}")
        # paddles: PushSim world, push(side, angle); point: PointPushSim (no
        # paddles in the world), push_stroke(x0, y0, x1, y1).
        self.pusher = pusher
        self.perception = perception
        self.obs_noise = (float(obs_noise[0]), float(obs_noise[1]))
        self.settle_s = float(settle_s)
        self.image_noise = float(image_noise)
        self.teleport_range = float(teleport_range)
        self.hand_radius = float(hand_radius)   # disc mode
        self.hand_speed = float(hand_speed)     # disc mode
        self.scene = load_scene()
        self.ws_half = float(self.scene["workspace"]["half_size"])
        self.thickness = float(self.scene["letter"]["thickness"])
        self.hand_cfg = stroke_cfg(self.ws_half, float(hand_vmax))
        self.hand_tries = int(hand_tries)
        cam = self.scene["camera"]
        self.view_half = cam["height"] * math.tan(math.radians(cam["hfov_deg"]) / 2)
        self.sim = None
        self.letter = None
        self._t = 0.0
        self._rng = np.random.default_rng(0)
        self._hand = None          # (x, y, z, yaw) while the hand model is in camera view
        self._pts = None           # letter area samples (body frame), for occlusion
        self.last_disturb = {}
        self._no_obs = False
        self.detector = None
        self._cam = None
        if perception != "truth":
            self.detector = self._make_detector(perception, model_path)

    # ------------------------------------------------------------ setup

    def _new_sim(self, letter):
        return PointPushSim(letter) if self.pusher == "point" else PushSim(letter)

    @staticmethod
    def _make_detector(method, model_path):
        from pa_perception.cnn_detector import default_model_path, make_detector
        if method in ("cnn", "hybrid"):
            path = model_path or default_model_path()
            try:
                import onnxruntime  # noqa: F401
            except ImportError as e:
                raise SystemExit(f"--perception {method} needs onnxruntime ({e})")
            if not os.path.isfile(path):
                raise SystemExit(f"--perception {method} needs the CNN model: {path} not found "
                                 f"(train it with tools/train_cnn.py)")
        det, used = make_detector(method, model_path=model_path, threads=1, fallback=False)
        assert used == method
        return det

    def reset(self, letter: str, pose: tuple, rng) -> None:
        # A fresh PushSim per trial: pymunk keeps some state across set_pose()
        # (results can depend on earlier pushes), which would make a trial
        # depend on the trials run before it in the same process.
        self.sim = self._new_sim(letter)
        self._t = 0.0  # each trial starts at t=0 (fresh state machine too)
        self.letter = letter
        self._hand = None
        self.last_disturb = {}
        pts, inside, _ = _union_mask(letter_rects(letter), res=0.002)
        self._pts = pts[inside]
        self.sim.set_pose(*pose)
        # Observation noise is drawn from its own stream, seeded by the trial rng.
        self._rng = np.random.default_rng(int(rng.integers(2 ** 63)) if hasattr(rng, "integers")
                                          else rng.getrandbits(63))

    # ------------------------------------------------------------ disturbance

    def disturb(self, mode: str, rng) -> tuple:
        if mode == "teleport":
            r = self.teleport_range
            for _ in range(1000):
                pose = (rng.uniform(-r, r), rng.uniform(-r, r), rng.uniform(-math.pi, math.pi))
                if is_disturbed(self.letter, *pose):
                    break
            self.sim.set_pose(*pose)
            self._t += 0.5
            self.last_disturb = {"mode": "teleport", "tries": 1}
            return self.true_pose()
        if mode == "hand":
            return self._disturb_hand(rng)
        if mode == "disc":
            target = self.true_pose()
            for k in range(200):
                self.sim.set_pose(*target)
                self._disc_sweep(rng)
                p = self.true_pose()
                if self._disturbed_ok(p):
                    break
            self.last_disturb = {"mode": "disc", "tries": k + 1}
            return self.true_pose()
        raise ValueError(f"unknown disturb mode {mode!r} (one of {DISTURBS})")

    def _disturbed_ok(self, p) -> bool:
        return is_disturbed(self.letter, *p) and abs(p[0]) < self.ws_half and abs(p[1]) < self.ws_half

    def _disturb_hand(self, rng) -> tuple:
        """Gazebo-like hand stroke; retried from the same start (fresh world, same
        clock, unseen by the robot) up to `hand_tries` times, then teleport."""
        start = self.true_pose()
        t0 = self._t
        for k in range(self.hand_tries):
            self.sim = self._new_sim(self.letter)
            self.sim.set_pose(*start)
            self._t = t0
            plan = self.plan_hand_stroke(rng, keep_in_workspace=False)
            events = self.run_hand_stroke(plan)
            p = self.true_pose()
            if self._disturbed_ok(p):
                self.last_disturb = {"mode": "hand", "tries": k + 1, "events": events,
                                     "stroke": plan["stroke"]}
                return p
        # fall back to a teleport (like the gazebo backend), from the start pose
        self.sim = self._new_sim(self.letter)
        self.sim.set_pose(*start)
        self._t = t0
        p = self.disturb("teleport", rng)
        self.last_disturb = {"mode": "teleport", "tries": self.hand_tries + 1, "fallback_from": "hand"}
        return p

    def _in_view(self, pos) -> bool:
        """Some part of the hand model is inside the camera view (table plane)."""
        return max(abs(pos[0]), abs(pos[1])) < self.view_half + 0.06

    def plan_hand_stroke(self, rng, keep_in_workspace: bool = True) -> dict:
        """Sample a stroke through the letter at its current pose and the hand
        trajectory (from the park pose, hand_dt steps). `air_s` = time from the
        hand leaving the park pose to the first contact with the letter layer.

        keep_in_workspace: re-draw (up to `hand_tries` strokes, tried on a copy
        of the world) until the letter ends inside the workspace; the last one
        is kept if none does."""
        plan = None
        for _ in range(max(1, self.hand_tries)):
            plan = self._sample_plan(rng)
            if not keep_in_workspace:
                break
            p = self._try_plan(plan)
            if abs(p[0]) < self.ws_half and abs(p[1]) < self.ws_half:
                break
        return plan

    def _try_plan(self, plan) -> tuple:
        """Final letter pose of a planned stroke, on a fresh copy of the world."""
        saved = (self.sim, self._t, self._hand)
        self.sim = self._new_sim(self.letter)
        self.sim.set_pose(*saved[0].pose())
        self._no_obs = True
        try:
            self.run_hand_stroke(plan)
            return self.true_pose()
        finally:
            self._no_obs = False
            self.sim, self._t, self._hand = saved

    def _sample_plan(self, rng) -> dict:
        stroke = sample_hand_stroke(self.letter, self.true_pose(), rng, self.hand_cfg.limit)
        traj = hand_trajectory(tuple(self.hand_cfg.park), stroke, HAND_DT, self.hand_cfg)
        # present (/hand/state) until the lift-off starts: last step at z_down
        z_down = self.hand_cfg.z_down
        down = [i for i, (pos, _) in enumerate(traj) if pos[2] <= z_down + 1e-9]
        lift = (down[-1] + 1) if down else len(traj)
        # cut the flight back to the park pose once it is out of view
        end = len(traj)
        for i in range(lift, len(traj)):
            if not self._in_view(traj[i][0]):
                end = i + 1
                break
        traj = traj[:end]
        first_contact = next((i for i, (pos, _) in enumerate(traj) if pos[2] < self.thickness), len(traj))
        return {"stroke": stroke, "traj": traj, "lift": lift, "contact": first_contact,
                "air_s": first_contact * HAND_DT}

    def run_hand_stroke(self, plan: dict, t_start: float | None = None) -> list:
        """Run a planned stroke. The hand leaves the park pose at `t_start`
        (default: now). Steps that lie in the past (t_start < now: the hand came
        in during a push) must be in the air; they only move the hand.

        The world is not stepped before the hand first reaches the letter layer
        (the letter is at rest), and a fresh PushSim is used from the current
        pose, so a stroke tried on a copy (`_try_plan`) gives the same result.

        Returns the timeline [(t, present, hand (x, y, z, yaw) | None, obs | None)]:
        one entry per hand step (30 Hz) with an observation every OBS_EVERY steps,
        and a final entry when the hand has left the view. Advances the clock to
        its end."""
        pose0 = self.true_pose()
        self.sim = self._new_sim(self.letter)
        self.sim.set_pose(*pose0)
        sim, stroke = self.sim, plan["stroke"]
        t0 = self._t if t_start is None else float(t_start)
        body, shapes = None, []
        events = []
        n_sub = max(1, int(round(HAND_DT / sim.dt)))
        for i, (pos, vel) in enumerate(plan["traj"]):
            t = t0 + (i + 1) * HAND_DT
            present = i < plan["lift"]
            prev = plan["traj"][i - 1][0] if i else tuple(self.hand_cfg.park)
            if t > self._t + 1e-9 and i < plan["contact"]:
                self._t = t  # hand in the air, letter at rest: nothing to simulate
            elif t > self._t + 1e-9:
                contact = min(prev[2], pos[2]) < self.thickness
                if contact and body is None:
                    body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
                    body.angle = stroke.yaw
                    body.position = (prev[0] * MM, prev[1] * MM)
                    for loc, zb, zt in _HAND_LOCAL:
                        s = pymunk.Poly(body, [tuple(v) for v in loc * MM])
                        s.friction = 0.8
                        shapes.append(s)
                    sim.space.add(body, *shapes)
                elif not contact and body is not None:
                    sim.space.remove(body, *shapes)
                    body, shapes = None, []
                if body is not None:
                    body.position = (prev[0] * MM, prev[1] * MM)
                    body.velocity = (vel[0] * MM, vel[1] * MM)
                for _ in range(n_sub):
                    sim.space.step(sim.dt)
                self._t = t
            self._hand = (pos[0], pos[1], pos[2], stroke.yaw) if self._in_view(pos) else None
            obs = self.observe() if (i + 1) % OBS_EVERY == 0 and not self._no_obs else None
            events.append((t, present, self._hand, obs))
        if body is not None:
            sim.space.remove(body, *shapes)
        self._hand = None
        # let the letter come to rest, clear residual solver state
        for _ in range(int(round(0.2 / sim.dt))):
            sim.space.step(sim.dt)
        sim.set_pose(*self.true_pose())
        self._t = max(self._t, events[-1][0] if events else self._t) + 0.2
        events.append((self._t, False, None, [] if self._no_obs else self.observe()))
        return events

    def _disc_sweep(self, rng):
        """Sweep a kinematic disc (a finger) through the letter at a random
        heading and offset, then remove it and let the letter settle."""
        sim = self.sim
        x0, y0, _ = self.true_pose()
        heading = rng.uniform(-math.pi, math.pi)
        d = np.array([math.cos(heading), math.sin(heading)])
        n = np.array([-d[1], d[0]])
        offset = rng.uniform(-0.03, 0.03)          # misses the centroid -> also rotates it
        travel = rng.uniform(0.02, 0.06)           # how far past the centroid the finger goes
        speed = self.hand_speed * rng.uniform(0.6, 1.4)
        start = np.array([x0, y0]) - d * 0.075 + n * offset
        dist = 0.075 + travel
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = tuple(start * MM)
        shape = pymunk.Circle(body, self.hand_radius * MM)
        shape.friction = 0.8
        sim.space.add(body, shape)
        body.velocity = tuple(d * speed * MM)
        dur = dist / speed
        steps = max(1, int(round(dur / sim.dt)))
        for _ in range(steps):
            sim.space.step(sim.dt)
        body.velocity = (0, 0)
        # lift the hand off (not dragged back through the letter)
        sim.space.remove(body, shape)
        for _ in range(int(round(0.5 / sim.dt))):
            sim.space.step(sim.dt)
        # clear residual velocity / solver state
        sim.set_pose(*self.true_pose())
        self._t += dur + 0.5

    # ------------------------------------------------------------ interface

    def occlusion(self) -> float:
        """Fraction of the letter's area under the hand footprint (top view, no parallax)."""
        if self._hand is None:
            return 0.0
        pts = transform(self._pts, *self.true_pose())
        hidden = np.zeros(len(pts), dtype=bool)
        for corners, _, _ in hand_world_boxes(self._hand[0], self._hand[1], self._hand[3]):
            hidden |= _inside_convex(pts, corners)
        return float(hidden.mean())

    def observe(self) -> list:
        x, y, th = self.true_pose()
        if self.perception == "truth":
            if self._hand is not None and self.occlusion() > OCCLUDED_FRAC:
                return []
            sp, sa = self.obs_noise
            if sp > 0:
                x += self._rng.normal(0, sp)
                y += self._rng.normal(0, sp)
            if sa > 0:
                th += self._rng.normal(0, sa)
            return [(self.letter, x, y, th, 1.0)]
        from pa_perception.synth import render_scene
        img = render_scene([(self.letter, x, y, th)], scene=self.scene, pusher_value=PADDLE_BGR,
                           noise_sigma=0.0 if self._hand else self.image_noise, rng=self._rng)
        if self._hand is not None:
            self._draw_hand(img)
            if self.image_noise > 0:
                noisy = img.astype(np.float32) + self._rng.normal(0.0, self.image_noise, img.shape)
                img = np.clip(noisy, 0, 255).astype(np.uint8)
        return [(d.letter, float(d.x), float(d.y), float(d.theta), float(d.confidence))
                for d in self.detector.detect(img)]

    def _draw_hand(self, img):
        from pa_perception.camera_model import CameraModel
        from pa_perception.synth import box_pixel_polys, fill_polys
        if self._cam is None:
            self._cam = CameraModel.from_scene(self.scene)
        hx, hy, hz, yaw = self._hand
        for corners, zb, zt in hand_world_boxes(hx, hy, yaw):
            top, hull = box_pixel_polys(self._cam, corners, hz + zt, hz + zb, sides=True)
            fill_polys(img, [hull], HAND_SIDE_BGR)
            fill_polys(img, [top], HAND_BGR)

    def push(self, side: int, angle_deg: float) -> None:
        if self.pusher != "paddles":
            raise RuntimeError("push(side, angle) needs pusher='paddles'; use push_stroke")
        self.sim.push(int(side), float(angle_deg), settle=self.settle_s)
        t, hold = sweep_profile(angle_deg)
        self._t += 2 * t + hold + self.settle_s

    def push_stroke(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Point pusher: lower at (x0, y0), slide to (x1, y1), lift; letter settles."""
        if self.pusher != "point":
            raise RuntimeError("push_stroke needs pusher='point'")
        self.sim.push(float(x0), float(y0), float(x1), float(y1), settle=self.settle_s)
        self._t += POINT_OVERHEAD_S + stroke_duration(math.hypot(x1 - x0, y1 - y0)) + self.settle_s

    def wait(self, dt: float) -> None:
        self._t += dt

    def true_pose(self) -> tuple:
        return self.sim.pose()

    def now(self) -> float:
        return self._t

    def close(self) -> None:
        self.sim = None
