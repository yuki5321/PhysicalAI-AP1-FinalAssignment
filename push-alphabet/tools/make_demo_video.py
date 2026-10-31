"""Render the 2D demo video of the whole Push-Alphabet story (no ROS, no Gazebo).

Everything shown is computed by the project code, frame by frame on the sim
clock: the world is `pa_core.pushsim.PushSim` (pymunk), the hand is the Gazebo
hand model driven along the `recovery_sim2d` hand stroke, the camera image is
`pa_perception.synth.render_scene` (+ the swept paddles and the hand drawn the
way `recovery_sim2d` draws them for the detector), observations come from the
selected detector (truth | classic | hybrid), and decisions from the production
state machine `pa_planner.state_machine.PlannerStateMachine` + `PushPlanner`,
run through `eval_recovery.run_trial` (the recovery-evaluation trial loop).

For each letter (scene `sequence`, I -> U -> Y -> K): the letter sits at the
target, the hand knocks it, the robot waits for the hand to leave
(WAIT_HAND + `hand_clear_s`), then pushes it back (RECOVER). Each paddle sweep
is animated with intermediate PushSim poses (the push is stepped exactly like
`PushSim.push`). When the state machine reports the letter done, it slides to
the display shelf (display order YUKI); after the last one, snow falls.
Outcomes are not edited: a letter that fails is shown failing. `--search`
tries seeds from `--seed` upward until every letter succeeds with a real hand
stroke (the seed used is printed and shown in the video).

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner:ros2_ws/src/pa_perception \\
        python3 tools/make_demo_video.py --out data/video/demo_2d.mp4 \\
        --sheet data/video/demo_2d_sheet.png

Speed: the hand, the wait and the first `--realtime-pushes` pushes of each
letter play in real time (sim clock = video clock); later pushes play at
`--speedup` (raised automatically, up to `--max-speedup`, default x2, to fit
`--max-seconds`), with a visible "×2 早送り" label (the course allows at most 2x).

Pusher (`--pusher`, default point; docs/point_pusher.md):

    point    the Push-T style point pusher: world `pa_core.pointpush.PointPushSim`,
             plans from `PointPushPlanner`, pushes `push_stroke(x0, y0, x1, y1)`.
             The orange pusher (top view: a cylinder of `point_pusher.radius`)
             flies from its park pose over the stroke start, is lowered, slides
             along the stroke (the letter follows the recorded PointPushSim
             steps), is lifted and parks. No paddles are drawn, and the detector
             sees the same image (no paddles, the pusher as an orange occluder),
             as in Gazebo with `pusher:=point`.
    paddles  the four servo paddles (PushSim), as before.

Disturbance: by default there is no hand: each letter starts at a random
disturbed pose (`--disturb teleport`, like Gazebo's scene_manager). `--hand`
adds the hand knock (the letter starts at the target and the hand knocks it).
"""

from __future__ import annotations

import argparse
import bisect
import math
from multiprocessing import Pool
import os
import shutil
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
for _p in (_HERE, *(os.path.join(_ROOT, "ros2_ws", "src", d)
                    for d in ("pa_core", "pa_planner", "pa_perception"))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pymunk  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import eval_recovery as ev  # noqa: E402
from pa_core.config import load_scene  # noqa: E402
from pa_core.letters import letter_polygons, pose_error  # noqa: E402
from pa_core.pointpush import point_pusher_cfg, stroke_duration  # noqa: E402
from pa_core.pushsim import MM, sweep_profile  # noqa: E402
from pa_perception.camera_model import CameraModel  # noqa: E402
from pa_perception.synth import box_pixel_polys, fill_polys, render_scene  # noqa: E402
from pa_planner.state_machine import Params  # noqa: E402
from recovery_sim2d import (HAND_BGR, HAND_SIDE_BGR, PADDLE_BGR, POINT_OVERHEAD_S,  # noqa: E402
                            Sim2DBackend, hand_world_boxes)

W, H = 1280, 720
CAM_X, CAM_Y, CAM_S = 16, 52, 532          # camera view on screen
PANEL_X = CAM_X + CAM_S + 24               # info panel
SHELF_Y0, SHELF_Y1 = 598, 708              # shelf strip
SIDE_JA = {"south": "南", "east": "東", "north": "北", "west": "西"}
STATE_JA = {"IDLE": "見守り中", "WAIT_HAND": "手が離れるのを待つ", "RECOVER": "押して直す",
            "LOST": "見失った", "FAILED": "失敗", "WAIT_LETTER": "そろった！ 次の文字へ",
            "DONE": "完成", "": ""}
DETAIL_JA = {"observe": "観測", "planning": "押し方を計画", "pushing": "押している",
             "settling": "止まるのを待つ"}
STATE_RGB = {"IDLE": (120, 130, 145), "WAIT_HAND": (230, 160, 30), "RECOVER": (50, 120, 220),
             "LOST": (200, 80, 160), "FAILED": (210, 60, 50), "WAIT_LETTER": (40, 170, 90),
             "DONE": (40, 170, 90)}
BG = (18, 22, 32)
FG = (232, 236, 242)
DIM = (150, 158, 172)
ACCENT = (120, 200, 255)
DET_RGB = (255, 214, 40)
TARGET_RGB = (90, 220, 130)
CAPTION_EN = "2D model simulation (pymunk) - not Gazebo"
CAPTION_JA = "2D モデルのシミュレーション（pymunk）― Gazebo ではありません"
PERCEPTION_DESC = {"truth": "truth (true pose, hidden under the hand)",
                   "classic": "classic CV (segmentation + template IoU)",
                   "cnn": "CNN (ONNX)",
                   "hybrid": "hybrid (CNN + classic angle fit)"}


# ================================================================ fonts / text

def find_font():
    """(path, has_cjk). Prefers a Japanese font (IPA / Noto CJK), else DejaVu, else None."""
    cands = ["/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
             "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
             "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
             "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"]
    for c in cands:
        if os.path.isfile(c):
            return c, True
    if shutil.which("fc-match"):
        try:
            p = subprocess.run(["fc-match", "-f", "%{file}", ":lang=ja"], capture_output=True,
                               text=True, timeout=10).stdout.strip()
            if p and os.path.isfile(p) and "DejaVu" not in p:
                return p, True
        except (OSError, subprocess.SubprocessError):
            pass
    for c in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.isfile(c):
            return c, False
    return None, False


class Text:
    """Queue text during a frame, draw it all with PIL at the end (CJK capable)."""

    def __init__(self):
        self.path, self.cjk = find_font()
        self._fonts = {}
        self.ops = []

    def font(self, size):
        if size not in self._fonts:
            self._fonts[size] = (ImageFont.truetype(self.path, size) if self.path
                                 else ImageFont.load_default())
        return self._fonts[size]

    def ja(self, ja, en):
        return ja if self.cjk else en

    def add(self, xy, s, size=20, color=FG, anchor="la"):
        self.ops.append((xy, s, size, color, anchor))

    def width(self, s, size):
        return self.font(size).getlength(s)

    def flush(self, img):
        if not self.ops:
            return img
        pil = Image.fromarray(img)
        d = ImageDraw.Draw(pil)
        for xy, s, size, color, anchor in self.ops:
            d.text(xy, s, font=self.font(size), fill=tuple(int(c) for c in color), anchor=anchor)
        self.ops = []
        return np.asarray(pil).copy()


# ================================================================ recording backend

class RecordingBackend(Sim2DBackend):
    """Sim2DBackend that also records the world for rendering:
    `phys` = [(t, x, y, theta, hand (x, y, z, yaw) | None, paddle (side, delta_rad) | None,
    pin (x, y, z_bottom) | None)] at 30 Hz during hand strokes and pushes,
    `obs` = [(t, detections)]. pin None = the point pusher is at its park pose.

    Physics is unchanged: `push` / `push_stroke` run Sim2DBackend's push (the
    same PushSim / PointPushSim calls), only sampling the state in between.

    Point mode: the detector image has no paddles and shows the parked pusher
    (orange, an occluder for the detector), like Gazebo with pusher:=point."""

    REC_HZ = 30.0

    def __init__(self, *a, **kw):
        self.phys, self.obs = [], []
        self._hand_v = None
        super().__init__(*a, **kw)
        pp = point_pusher_cfg()
        self.pin_park = tuple(float(v) for v in pp["park"])

    # `_hand` is assigned once per hand step by run_hand_stroke: record there.
    @property
    def _hand(self):
        return self._hand_v

    @_hand.setter
    def _hand(self, v):
        self._hand_v = v
        if getattr(self, "sim", None) is not None and not getattr(self, "_no_obs", False):
            self._rec(v, None)

    def _rec(self, hand, paddle, t=None, pin=None):
        x, y, th = self.sim.pose()
        self.phys.append((self._t if t is None else t, x, y, th, hand, paddle, pin))

    def _truncate(self, t0):
        self.phys = [p for p in self.phys if p[0] <= t0 + 1e-9]
        self.obs = [o for o in self.obs if o[0] <= t0 + 1e-9]

    def reset(self, letter, pose, rng):
        super().reset(letter, pose, rng)
        self.phys, self.obs = [], []
        self._rec(None, None)

    def disturb(self, mode, rng):
        t0 = self._t
        p = super().disturb(mode, rng)
        if self.last_disturb.get("fallback_from"):
            self._truncate(t0)  # the hand strokes that did not disturb it are not kept
        if mode == "teleport" and not self.last_disturb.get("fallback_from"):
            # The letter starts at the disturbed pose (as Gazebo's scene_manager
            # places it): it is shown there from t = 0, with no earlier frames.
            self.phys = [(0.0, *p, None, None, None)]
            self.obs = []
        self._rec(None, None)
        return p

    def run_hand_stroke(self, plan, t_start=None):
        # A retried stroke restarts from the same clock: drop the earlier try.
        self._truncate(self._t if t_start is None else t_start)
        events = super().run_hand_stroke(plan, t_start)
        self._rec(None, None)
        return events

    def observe(self):
        if self.pusher == "point" and self.perception != "truth":
            out = self._observe_point()
        else:
            out = super().observe()
        if not self._no_obs:
            self.obs.append((self._t, out))
        return out

    def _observe_point(self):
        """Sim2DBackend.observe for the point pusher: no paddles in the image, the
        parked pusher drawn (it is out of the way, as when the robot looks)."""
        from pa_perception.synth import render_scene
        x, y, th = self.true_pose()
        img = render_scene([(self.letter, x, y, th)], scene=self.scene, pusher_value=None,
                           noise_sigma=0.0, rng=self._rng)
        if self._cam is None:
            self._cam = CameraModel.from_scene(self.scene)
        draw_pin(img, self._cam, self.pin_park)
        if self._hand is not None:
            self._draw_hand(img)
        if self.image_noise > 0:
            noisy = img.astype(np.float32) + self._rng.normal(0.0, self.image_noise, img.shape)
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        return [(d.letter, float(d.x), float(d.y), float(d.theta), float(d.confidence))
                for d in self.detector.detect(img)]

    def push(self, side, angle_deg):
        """Sim2DBackend.push unchanged (PushSim.push, whatever shortcuts it takes),
        with every physics step of it recorded through a wrapper on space.step.
        The steps are then laid out on the backend clock along the sweep profile
        (`sweep_profile`: out, hold, back, settle), see `push_samples`."""
        side, angle_deg = int(side), float(angle_deg)
        sim = self.sim
        space, arm = sim.space, sim.arms[side]
        rest = sim.pushers[side].rest_heading
        pose0 = sim.pose()
        steps = []
        orig = space.step

        def step(dt):
            orig(dt)
            steps.append((arm.angular_velocity, arm.angle - rest, *sim.pose()))

        space.step = step
        t0 = self._t
        try:
            super().push(side, angle_deg)
        finally:
            if space.__dict__.get("step") is step:
                del space.step
        for t, pose, paddle in push_samples(steps, pose0, sim.pose(), side, angle_deg,
                                            self.settle_s, sim.dt, 1.0 / self.REC_HZ):
            self.phys.append((t0 + t, *pose, None, paddle, None))
        self._rec(None, None)

    def push_stroke(self, x0, y0, x1, y1):
        """Sim2DBackend.push_stroke unchanged (PointPushSim.push), with every
        physics step recorded: the stroke steps (disc in the world: its travel
        along the stroke) and the settle steps (fresh world after the lift).
        Laid out on the backend clock by `stroke_samples`."""
        sim = self.sim
        pose0 = sim.pose()
        steps = []
        L = math.hypot(x1 - x0, y1 - y0)
        d = ((x1 - x0) / L, (y1 - y0) / L) if L > 1e-9 else (1.0, 0.0)
        cls_build = type(sim)._build

        def wrap(space):
            orig = space.step

            def step(dt):
                orig(dt)
                disc = next((sh.body for sh in space.shapes if isinstance(sh, pymunk.Circle)
                             and sh.body.body_type == pymunk.Body.KINEMATIC), None)
                if disc is not None:
                    px, py = disc.position.x / MM, disc.position.y / MM
                    steps.append(("stroke", (px - x0) * d[0] + (py - y0) * d[1], *sim.pose()))
                else:
                    steps.append(("settle", None, *sim.pose()))
            space.step = step

        def build(*a, **kw):
            cls_build(sim, *a, **kw)
            wrap(sim.space)

        wrap(sim.space)
        sim._build = build
        t0 = self._t
        try:
            super().push_stroke(x0, y0, x1, y1)
        finally:
            sim.__dict__.pop("_build", None)
            sim.space.__dict__.pop("step", None)
        for t, pose, pin in stroke_samples(steps, pose0, sim.pose(), (x0, y0, x1, y1), self.pin_park,
                                           self.settle_s, sim.dt, 1.0 / self.REC_HZ, sim.speed):
            self.phys.append((t0 + t, *pose, None, None, pin))
        self._rec(None, None)


PIN_BGR = PADDLE_BGR            # orange, as the Gazebo point_pusher model
PIN_SIDE_BGR = (10, 70, 165)
PIN_Z_DOWN = 0.001              # bottom of the lowered pusher above the table [m]
PIN_APPROACH_S = 0.4            # park -> over the start (0.25 s) + lower (0.15 s)
PIN_LIFT_S = 0.2                # lift after the stroke; then back to park


def draw_pin(img, cam, pin, radius=None, height=None):
    """Draw the point pusher (vertical cylinder, bottom at pin[2]) into a BGR image."""
    pp = point_pusher_cfg()
    r = float(radius or pp["radius"])
    h = float(height or pp["height"])
    a = np.linspace(0, 2 * math.pi, 20, endpoint=False)
    ring = np.stack([pin[0] + r * np.cos(a), pin[1] + r * np.sin(a)], axis=1)
    top, hull = box_pixel_polys(cam, ring, pin[2] + h, pin[2], sides=True)
    fill_polys(img, [hull], PIN_SIDE_BGR)
    fill_polys(img, [top], PIN_BGR)


def stroke_samples(steps, pose0, pose1, stroke, park, settle_s, dt, every, speed):
    """[(t, letter pose, pin (x, y, z_bottom) | None)] every `every` seconds over one
    point-pusher stroke, on the Sim2DBackend clock (POINT_OVERHEAD_S + stroke
    time + settle_s).

    Pin: park -> over the stroke start (in the air) -> lowered -> slides along
    the stroke at `speed` -> lifted -> back to park (None at the very end).
    Letter: during the slide, the last recorded stroke step whose disc travel is
    not past the drawn pin's travel (pose0 before the first contact: the part
    before contact is not simulated); after it, the settle step at the same
    time since the lift (the last one once the letter is at rest)."""
    x0, y0, x1, y1 = stroke
    L = math.hypot(x1 - x0, y1 - y0)
    dx, dy = ((x1 - x0) / L, (y1 - y0) / L) if L > 1e-9 else (0.0, 0.0)
    t_str = stroke_duration(L, speed)
    total = POINT_OVERHEAD_S + t_str + settle_s
    lift_s = max(0.0, POINT_OVERHEAD_S - PIN_APPROACH_S)
    zp = float(park[2])
    st = [s for s in steps if s[0] == "stroke"]
    se = [s for s in steps if s[0] == "settle"]
    stroke_end = tuple(st[-1][2:]) if st else tuple(pose0)
    res = []
    n = int(math.floor(total / every + 1e-9))
    for j in range(1, n + 1):
        t = j * every
        pose = tuple(pose0)
        if t < PIN_APPROACH_S:
            u = t / PIN_APPROACH_S
            if u < 0.625:   # fly over the start, in the air
                v = ease(u / 0.625)
                pin = (park[0] + (x0 - park[0]) * v, park[1] + (y0 - park[1]) * v, zp)
            else:           # lower
                v = ease((u - 0.625) / 0.375)
                pin = (x0, y0, zp + (PIN_Z_DOWN - zp) * v)
        elif t < PIN_APPROACH_S + t_str:
            s = speed * (t - PIN_APPROACH_S)
            pin = (x0 + dx * s, y0 + dy * s, PIN_Z_DOWN)
            for k in st:
                if k[1] <= s + 1e-9:
                    pose = tuple(k[2:])
                else:
                    break
        else:
            tau = t - PIN_APPROACH_S - t_str
            if se:
                i = int(round(tau / dt)) - 1
                pose = tuple(se[min(len(se) - 1, i)][2:]) if i >= 0 else stroke_end
            else:
                pose = stroke_end
            if tau < lift_s:
                pin = (x1, y1, PIN_Z_DOWN + (zp - PIN_Z_DOWN) * ease(tau / lift_s))
            else:
                v = ease((tau - lift_s) / max(1e-6, settle_s))
                pin = (x1 + (park[0] - x1) * v, y1 + (park[1] - y1) * v, zp)
        res.append((t, pose, pin))
    res.append((total, tuple(pose1), None))
    return res


def push_samples(steps, pose0, pose1, side, angle_deg, settle_s, dt, every):
    """[(t, pose, (side, paddle delta angle) | None)] every `every` seconds over one push.

    `steps` = [(arm angular velocity, arm delta angle, x, y, theta)] after each
    physics step of the push. The paddle angle follows the sweep profile
    (out at constant speed, hold, back, settle). The letter pose is the
    simulated one at the same phase: during the sweep out, the last step whose
    paddle angle is not past the drawn angle (steps skipped before contact:
    the letter has not moved yet); during hold / back / settle the step at the
    same time into that phase (or the last one, if the simulation stopped early
    because the letter was at rest)."""
    t_sw, hold = sweep_profile(angle_deg)
    target = math.radians(angle_deg)
    omega = target / t_sw
    out = [s for s in steps if s[0] > 0]
    i_back = next((i for i, s in enumerate(steps) if s[0] < 0), len(steps))
    i_out_end = max((i for i, s in enumerate(steps) if s[0] > 0), default=-1)
    hold_s = [s for s in steps[i_out_end + 1:i_back] if s[0] == 0]
    back = [s for s in steps[i_back:] if s[0] < 0]
    i_back_end = max((i for i, s in enumerate(steps) if s[0] < 0), default=len(steps) - 1)
    sett = [s for s in steps[i_back_end + 1:] if s[0] == 0] if back else []
    last = (steps[-1][2:] if steps else tuple(pose1))
    total = 2 * t_sw + hold + settle_s
    res = []
    n = int(math.floor(total / every + 1e-9))

    def at(seq, tau, fallback):
        if not seq:
            return fallback
        k = min(len(seq) - 1, max(0, int(round(tau / dt)) - 1))
        return tuple(seq[k][2:]) if tau / dt <= len(seq) + 0.5 else tuple(seq[-1][2:])

    for j in range(1, n + 1):
        t = j * every
        if t < t_sw:
            a = omega * t
            p = tuple(pose0)
            for s in out:
                if s[1] <= a + 1e-9:
                    p = tuple(s[2:])
                else:
                    break
            paddle = (side, a)
        elif t < t_sw + hold:
            base = tuple(out[-1][2:]) if out else tuple(pose0)
            p = at(hold_s, t - t_sw, base)
            paddle = (side, target)
        elif t < 2 * t_sw + hold:
            base = tuple(hold_s[-1][2:]) if hold_s else (tuple(out[-1][2:]) if out else tuple(pose0))
            p = at(back, t - t_sw - hold, base)
            paddle = (side, target - omega * (t - t_sw - hold))
        else:
            base = tuple(back[-1][2:]) if back else last
            p = at(sett, t - 2 * t_sw - hold, base)
            paddle = None
        res.append((t, p, paddle))
    res.append((total, tuple(pose1), None))
    return res


# ================================================================ episodes

def default_perception():
    try:
        import onnxruntime  # noqa: F401
        from pa_perception.cnn_detector import default_model_path
        if os.path.isfile(default_model_path()):
            return "hybrid"
    except ImportError:
        pass
    return "classic"


def simulate_letter(job):
    """One disturb -> recover episode with the eval trial loop. Returns a dict.

    job = (letter, seed, perception, max_pushes, hand_tries[, pusher[, hand]]);
    pusher defaults to "paddles" and hand to True (the hand knock) when omitted."""
    letter, seed, perception, max_pushes, hand_tries = job[:5]
    pusher = job[5] if len(job) > 5 else "paddles"
    hand = job[6] if len(job) > 6 else True
    be = RecordingBackend(perception=perception, hand_tries=hand_tries, pusher=pusher)
    trace = []
    w0 = time.perf_counter()
    try:
        row = ev.run_trial(be, letter, 0, seed, disturb="hand" if hand else "teleport",
                           params=Params(), max_pushes=max_pushes, perception=perception,
                           trace=trace, pusher=pusher)
        t_end = be.now()
    finally:
        be.close()
    return dict(letter=letter, seed=seed, perception=perception, row=row, trace=trace,
                phys=be.phys, obs=be.obs, t_end=t_end, wall=time.perf_counter() - w0,
                pusher=pusher, hand=hand)


def episode_ok(ep, hand_tries) -> bool:
    r = ep["row"]
    return bool(r["success"]) and r["disturb_tries"] == 1 and not r["hand_violations"] \
        and r["disturb_tries"] <= hand_tries


def run_episodes(letters, seed, perception, max_pushes=30, hand_tries=4, jobs=4, search=1,
                 log=print, pusher="paddles", hand=True):
    """Episodes for `letters` at `seed`; with search > 1 try seeds seed..seed+search-1 until
    every letter succeeds (first hand stroke disturbs it, no push with the hand present).
    Returns (episodes, seed, all_ok)."""
    eps = None
    for s in range(seed, seed + max(1, search)):
        jobs_ = [(L, s, perception, max_pushes, hand_tries, pusher, hand) for L in letters]
        if jobs > 1 and len(jobs_) > 1:
            with Pool(min(jobs, len(jobs_))) as pool:
                eps = pool.map(simulate_letter, jobs_, chunksize=1)
        else:
            eps = [simulate_letter(j) for j in jobs_]
        summ = "  ".join(f"{e['letter']}:{'ok' if e['row']['success'] else e['row']['fail_reason']}"
                         f"/{e['row']['pushes']}p/tries{e['row']['disturb_tries']}" for e in eps)
        ok = all(episode_ok(e, hand_tries) for e in eps)
        log(f"seed {s} perception={perception} pusher={pusher} hand={hand}: {summ}  -> {'ALL OK' if ok else 'rejected'}")
        if ok:
            return eps, s, True
    return eps, s, False


class Timeline:
    """Queries on one recorded episode at sim time t."""

    def __init__(self, ep):
        self.ep = ep
        self.phys = sorted(ep["phys"], key=lambda p: p[0])
        self.pt = [p[0] for p in self.phys]
        self.obs = ep["obs"]
        self.ot = [o[0] for o in self.obs]
        self.states, self.hands, self.pushes = [], [], []
        for t, s in ep["trace"]:
            if s.startswith("hand "):
                self.hands.append((t, s == "hand present"))
            elif s.startswith("push stroke "):
                st = tuple(float(v) for v in s.split()[2:6])
                self.pushes.append((t, -1, math.hypot(st[2] - st[0], st[3] - st[1]) * 1000, st))
            elif s.startswith("push ") and s != "push_done":
                _, side, ang = s.split()
                self.pushes.append((t, int(side), float(ang), None))
            elif s != "push_done":
                parts = s.split(" ", 1)
                self.states.append((t, parts[0], parts[1] if len(parts) > 1 else ""))
        self.st = [s[0] for s in self.states]
        self.ht = [h[0] for h in self.hands]
        self.pst = [p[0] for p in self.pushes]
        self.t_end = float(ep["t_end"])

    @staticmethod
    def _last(ts, items, t):
        i = bisect.bisect_right(ts, t + 1e-9) - 1
        return items[i] if i >= 0 else None

    def world(self, t):
        p = self._last(self.pt, self.phys, t) or self.phys[0]
        return (p[1], p[2], p[3]), p[4], p[5]

    def pin(self, t):
        """Point pusher (x, y, z_bottom) while it moves, None when parked (or paddles)."""
        p = self._last(self.pt, self.phys, t) or self.phys[0]
        return p[6] if len(p) > 6 else None

    def state(self, t):
        s = self._last(self.st, self.states, t)
        return (s[1], s[2]) if s else ("IDLE", "")

    def hand_present(self, t):
        h = self._last(self.ht, self.hands, t)
        return bool(h and h[1])

    def hand_left_t(self, t):
        h = self._last(self.ht, self.hands, t)
        return h[0] if h and not h[1] else None

    def n_pushes(self, t):
        return bisect.bisect_right(self.pst, t + 1e-9)

    def last_push(self, t):
        return self._last(self.pst, self.pushes, t)

    def detections(self, t):
        i = bisect.bisect_right(self.ot, t + 1e-9) - 1
        return (self.obs[i][0], self.obs[i][1]) if i >= 0 else (None, None)

    def push_start(self, k):
        """Start time of push k (1-based), or inf."""
        return self.pushes[k - 1][0] if 0 < k <= len(self.pushes) else math.inf


# ================================================================ drawing

class Renderer:
    def __init__(self, scene, perception, seed, fps, text: Text, pusher="paddles", hand=True):
        self.sc = scene
        self.pusher = pusher
        self.hand = hand
        self.pin_park = tuple(float(v) for v in scene["point_pusher"]["park"])
        self.cam = CameraModel.from_scene(scene)
        self.k = CAM_S / self.cam.width_px
        self.thick = float(scene["letter"]["thickness"])
        self.perception = perception
        self.seed = seed
        self.fps = fps
        self.tx = text
        self.word = scene["display_word"]
        self.seq = scene["sequence"]
        self.tgt = scene["target"]
        self.pushers = scene["pushers"]
        self.slots = {}
        n = len(self.word)
        gap = 150
        x0 = W / 2 + 120 - (n - 1) * gap / 2
        for i, L in enumerate(self.word):
            self.slots[L] = (x0 + i * gap, (SHELF_Y0 + SHELF_Y1) / 2 + 2)
        self.shelf_scale = 900.0  # px / m on the shelf
        self.noise_rng = np.random.default_rng(12345)

    # ---------------------------------------------------------- camera image

    def camera_image(self, letter, pose, hand, paddle, noise=2.0, pin=None):
        """Synthetic camera image (BGR, 640x640) as recovery_sim2d renders it for the
        detector, with the swept paddle (paddles) or the point pusher at `pin`
        (None: parked) and the hand. letter None: empty table."""
        letters = [(letter, *pose)] if letter else []
        img = render_scene(letters, cam=self.cam, scene=self.sc, pusher_value=None)
        p = self.pushers
        L, w, zt = p["arm_length"], p["arm_thickness"], p["arm_height"]
        if self.pusher == "point":
            draw_pin(img, self.cam, pin if pin is not None else self.pin_park)
        for i, pu in enumerate(p["list"] if self.pusher != "point" else []):
            a = math.radians(pu["rest_heading"])
            if paddle is not None and paddle[0] == i:
                a += paddle[1]
            d = np.array([math.cos(a), math.sin(a)])
            n = np.array([-d[1], d[0]])
            c0 = np.asarray(pu["pivot"], dtype=float)
            corners = np.array([c0 - n * w / 2, c0 + d * L - n * w / 2,
                                c0 + d * L + n * w / 2, c0 + n * w / 2])
            top, _ = box_pixel_polys(self.cam, corners, zt, 0.0, sides=False)
            fill_polys(img, [top], PADDLE_BGR)
        if hand is not None and pin is not None and pin[2] < hand[2]:
            draw_pin(img, self.cam, pin)  # the hand passes over / under: draw in z order
        if hand is not None:
            hx, hy, hz, yaw = hand
            for corners, zb, ztop in hand_world_boxes(hx, hy, yaw):
                top, hull = box_pixel_polys(self.cam, corners, hz + ztop, hz + zb, sides=True)
                fill_polys(img, [hull], HAND_SIDE_BGR)
                fill_polys(img, [top], HAND_BGR)
        if noise > 0:
            nz = self.noise_rng.normal(0.0, noise, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + nz, 0, 255).astype(np.uint8)
        return img

    def to_screen(self, pts_world, z):
        uv = self.cam.world_to_pixel(pts_world, z) * self.k
        return uv + [CAM_X, CAM_Y]

    def letter_contour(self, name, pose, z=None):
        """Outline (screen px) of the union of the letter's top faces."""
        z = self.thick if z is None else z
        polys = [self.to_screen(c, z) for c in letter_polygons(name, *pose)]
        allp = np.vstack(polys)
        x0, y0 = np.floor(allp.min(axis=0)).astype(int) - 3
        x1, y1 = np.ceil(allp.max(axis=0)).astype(int) + 3
        sc = 4
        m = np.zeros(((y1 - y0) * sc, (x1 - x0) * sc), np.uint8)
        for p in polys:
            cv2.fillConvexPoly(m, np.round((p - [x0, y0]) * sc).astype(np.int32), 255)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c.reshape(-1, 2) / sc + [x0, y0] for c in cs]

    @staticmethod
    def polyline(img, pts, color, th=2, closed=True):
        cv2.polylines(img, [np.round(pts * 16).astype(np.int32)], closed, color, th, cv2.LINE_AA,
                      shift=4)

    def dashed(self, img, pts, color, th=1, dash=7):
        pts = np.vstack([pts, pts[:1]])
        acc, on = 0.0, True
        for a, b in zip(pts[:-1], pts[1:]):
            seg = np.linalg.norm(b - a)
            s = 0.0
            while s < seg:
                e = min(seg, s + dash - acc)
                if on:
                    p0 = a + (b - a) * (s / seg)
                    p1 = a + (b - a) * (e / seg)
                    self.polyline(img, np.array([p0, p1]), color, th, closed=False)
                acc += e - s
                s = e
                if acc >= dash - 1e-9:
                    acc, on = 0.0, not on

    def draw_shape(self, img, name, cx, cy, scale, theta, fill, edge=None):
        """Filled letter (flat, top view) centered at screen (cx, cy)."""
        for c in letter_polygons(name, 0.0, 0.0, theta):
            p = np.stack([cx + c[:, 0] * scale, cy - c[:, 1] * scale], axis=1)
            cv2.fillConvexPoly(img, np.round(p * 16).astype(np.int32), fill, cv2.LINE_AA, shift=4)
        if edge is not None:
            pass

    # ---------------------------------------------------------- frame parts

    def frame_base(self):
        img = np.empty((H, W, 3), np.uint8)
        img[:] = BG
        return img

    def title_bar(self, img, letter_idx=None):
        tx = self.tx
        tx.add((CAM_X, 14), "Push-Alphabet", 26, FG)
        tx.add((CAM_X + 200, 19), tx.ja("崩されても自分で直す、雪の文字ロボ",
                                        "a letter robot that fixes itself"), 19, DIM)
        # sequence progress I -> U -> Y -> K
        x = W - 16
        items = []
        for i, L in enumerate(self.seq):
            items.append((L, i))
        xs = W - 16 - 44 * len(items)
        for L, i in items:
            col = (DIM if letter_idx is None or i > letter_idx else
                   (ACCENT if i == letter_idx else TARGET_RGB))
            cx = xs + 44 * i + 18
            cv2.rectangle(img, (int(cx - 16), 12), (int(cx + 16), 42), col, 2 if i != letter_idx else 3,
                          cv2.LINE_AA)
            tx.add((cx, 27), L, 22, col, "mm")
        tx.add((xs - 10, 27), tx.ja("順番", "order"), 16, DIM, "rm")
        del x

    def caption(self, img):
        tx = self.tx
        s = tx.ja(CAPTION_JA, CAPTION_EN) if False else CAPTION_EN
        w = tx.width(s, 15)
        x0, y0 = CAM_X + 6, CAM_Y + CAM_S - 28
        sub = img[y0:y0 + 22, x0:x0 + int(w) + 12]
        sub[:] = (sub * 0.35).astype(np.uint8)
        tx.add((x0 + 6, y0 + 3), s, 15, (255, 255, 255))

    def speed_label(self, img, speed):
        tx = self.tx
        if speed <= 1.0:
            s, col = tx.ja("×1 実時間", "x1 real time"), (200, 210, 220)
        else:
            s, col = f"×{speed:g} " + tx.ja("早送り", "fast"), (255, 200, 60)
        w = tx.width(s, 22 if speed > 1 else 16)
        x1, y0 = CAM_X + CAM_S - 8, CAM_Y + 8
        sub = img[y0:y0 + 32, int(x1 - w - 16):x1]
        sub[:] = (sub * 0.35).astype(np.uint8)
        tx.add((x1 - 8, y0 + 16), s, 22 if speed > 1 else 16, col, "rm")

    def camera_view(self, img, cam_bgr):
        v = cv2.resize(cam_bgr, (CAM_S, CAM_S), interpolation=cv2.INTER_AREA)
        img[CAM_Y:CAM_Y + CAM_S, CAM_X:CAM_X + CAM_S] = v[:, :, ::-1]
        cv2.rectangle(img, (CAM_X - 1, CAM_Y - 1), (CAM_X + CAM_S, CAM_Y + CAM_S), (70, 78, 96), 1)
        self.tx.add((CAM_X + 8, CAM_Y + 10), self.tx.ja("真上カメラ（知覚への入力画像）",
                                                         "overhead camera (detector input)"),
                    15, (230, 230, 230))

    def overlays(self, img, letter, dets, det_age, hand_present, moving=False):
        tx = self.tx
        tg = (self.tgt["x"], self.tgt["y"], math.radians(self.tgt["theta"]))
        for c in self.letter_contour(letter, tg, 0.0):
            self.dashed(img, c, TARGET_RGB, 1)
        if dets is None:
            return
        fresh = det_age is not None and det_age < 0.35
        if not fresh or moving:  # no camera frame is used while a paddle moves
            return
        if not dets:
            tx.add((CAM_X + CAM_S / 2, CAM_Y + 44), tx.ja("検出なし（手で隠れている）" if hand_present
                                                          else "検出なし", "no detection"),
                   18, (255, 140, 120), "mm")
            return
        for d in dets:
            name, x, y, th, conf = d
            if not name:
                continue
            col = DET_RGB if name == letter else (255, 120, 120)
            for c in self.letter_contour(name, (x, y, th)):
                self.polyline(img, c, col, 2)
            ctr = self.to_screen([(x, y)], self.thick)[0]
            up = self.to_screen([(x - 0.03 * math.sin(th), y + 0.03 * math.cos(th))], self.thick)[0]
            cv2.circle(img, tuple(np.round(ctr).astype(int)), 3, col, -1, cv2.LINE_AA)
            cv2.arrowedLine(img, tuple(np.round(ctr).astype(int)), tuple(np.round(up).astype(int)),
                            col, 2, cv2.LINE_AA, tipLength=0.25)
            lab = f"{name}  {conf:.2f}"
            pos = f"x {x * 1000:+.1f}  y {y * 1000:+.1f} mm  θ {math.degrees(th):+.0f}°"
            # label beside the letter, on the side toward the image center
            right = ctr[0] < CAM_X + CAM_S / 2
            lx = ctr[0] + 70 if right else ctr[0] - 70 - 206
            ly = ctr[1] + 50 if ctr[1] < CAM_Y + CAM_S / 2 else ctr[1] - 95
            lx = float(np.clip(lx, CAM_X + 6, CAM_X + CAM_S - 210))
            ly = float(np.clip(ly, CAM_Y + 40, CAM_Y + CAM_S - 76))
            sub = img[int(ly) - 2:int(ly) + 44, int(lx) - 4:int(lx) + 206]
            sub[:] = (sub * 0.4).astype(np.uint8)
            tx.add((lx, ly), lab, 18, col)
            tx.add((lx, ly + 23), pos, 14, col)

    def paddle_label(self, img, paddle, angle_cmd):
        if paddle is None:
            return
        pu = self.pushers["list"][paddle[0]]
        uv = self.to_screen([pu["pivot"]], self.pushers["arm_height"])[0]
        cv2.circle(img, tuple(np.round(uv).astype(int)), 7, (255, 170, 60), 2, cv2.LINE_AA)
        s = f"{SIDE_JA.get(pu['name'], pu['name']) if self.tx.cjk else pu['name']} " \
            f"{math.degrees(paddle[1]):.0f}°/{angle_cmd:.0f}°"
        x = float(np.clip(uv[0] + (14 if uv[0] < CAM_X + CAM_S / 2 else -14), CAM_X + 4, CAM_X + CAM_S - 4))
        y = float(np.clip(uv[1], CAM_Y + 60, CAM_Y + CAM_S - 50))
        self.tx.add((x, y), s, 17, (255, 190, 90), "lm" if uv[0] < CAM_X + CAM_S / 2 else "rm")

    def stroke_label(self, img, stroke, pin):
        """Stroke arrow (start -> end on the table) and a ring around the moving pusher."""
        if stroke is None or pin is None:
            return
        x0, y0, x1, y1 = stroke
        a = self.to_screen([(x0, y0)], 0.0)[0]
        b = self.to_screen([(x1, y1)], 0.0)[0]
        col = (255, 190, 90)
        cv2.arrowedLine(img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), col, 2,
                        cv2.LINE_AA, tipLength=min(0.4, 12.0 / max(1.0, float(np.linalg.norm(b - a)))))
        cv2.circle(img, tuple(np.round(a).astype(int)), 4, col, -1, cv2.LINE_AA)
        if pin[2] > 0.02:
            return  # lifted / flying back to park: no ring (it may be out of the view)
        uv = self.to_screen([pin[:2]], pin[2] + float(self.sc["point_pusher"]["height"]))[0]
        cv2.circle(img, tuple(np.round(uv).astype(int)), 12, col, 2, cv2.LINE_AA)
        L = math.hypot(x1 - x0, y1 - y0) * 1000
        x = float(np.clip(uv[0] + (18 if uv[0] < CAM_X + CAM_S / 2 else -18), CAM_X + 4, CAM_X + CAM_S - 4))
        y = float(np.clip(uv[1] - 18, CAM_Y + 60, CAM_Y + CAM_S - 50))
        s = self.tx.ja(f"押し棒 {L:.0f} mm", f"pusher {L:.0f} mm")
        self.tx.add((x, y), s, 17, col, "lm" if uv[0] < CAM_X + CAM_S / 2 else "rm")

    def panel(self, img, info):
        tx = self.tx
        x = PANEL_X
        xr = W - 16
        # state badge
        state, detail = info["state"], info.get("detail", "")
        col = STATE_RGB.get(state, (120, 120, 120))
        y = CAM_Y
        cv2.rectangle(img, (x, y), (xr, y + 92), col, -1, cv2.LINE_AA)
        tx.add((x + 14, y + 8), tx.ja("状態機械 state", "state machine"), 15, (255, 255, 255))
        st = state + (f" · {detail}" if detail else "")
        tx.add((x + 14, y + 28), st, 30, (255, 255, 255))
        ja = STATE_JA.get(state, "") + (f"（{DETAIL_JA.get(detail, detail)}）" if detail else "")
        if tx.cjk:
            tx.add((x + 14, y + 66), ja, 19, (255, 255, 255))
        y += 108
        # active letter
        L = info.get("letter")
        cv2.rectangle(img, (x, y), (x + 132, y + 132), (34, 40, 56), -1)
        if L:
            self.draw_shape(img, L, x + 66, y + 66, 1250.0, 0.0, (235, 238, 245))
        rows = [
            (tx.ja("対象の文字", "active letter"),
             f"{L}  ({info.get('idx', 0) + 1}/{len(self.seq)})" if L else "-"),
            (tx.ja("押した回数", "pushes"), f"{info.get('pushes', 0)}"),
            (tx.ja("最後の押し（始点 → 終点）" if self.pusher == "point" else "最後の指令",
                   "last stroke (from -> to)" if self.pusher == "point" else "last command"),
             info.get("cmd", "-")),
        ]
        yy = y
        for k, v in rows:
            tx.add((x + 152, yy), k, 15, DIM)
            tx.add((x + 152, yy + 18), v, 22, FG)
            yy += 46
        y += 148
        # hand
        hp = info.get("hand_present", False)
        hcol = (240, 150, 60) if hp else (90, 200, 120)
        cv2.circle(img, (x + 10, y + 13), 8, hcol, -1, cv2.LINE_AA)
        hs = tx.ja("あり（台の上）", "present") if hp else tx.ja("なし", "absent")
        if not hp and info.get("hand_visible"):
            hs = tx.ja("なし（持ち上げて去るところ）", "absent (lifting away)")
        tx.add((x + 26, y + 2), tx.ja("手", "hand") + ": " + hs, 20, FG)
        clr = info.get("clear")
        if clr is not None:
            bx0, bx1 = x + 26, xr - 200
            cv2.rectangle(img, (bx0, y + 34), (bx1, y + 48), (60, 66, 82), -1)
            f = min(1.0, clr[0] / clr[1])
            cv2.rectangle(img, (bx0, y + 34), (int(bx0 + f * (bx1 - bx0)), y + 48),
                          (90, 200, 120) if f >= 1 else (230, 160, 30), -1)
            tx.add((bx1 + 12, y + 41), tx.ja(f"手が離れて {clr[0]:.1f} / {clr[1]:.1f} s",
                                             f"hand gone {clr[0]:.1f} / {clr[1]:.1f} s"), 15, DIM, "lm")
        y += 66
        kv = [
            (tx.ja("知覚", "perception"), PERCEPTION_DESC.get(self.perception, self.perception)),
            (tx.ja("検出", "detection"), info.get("det", "-")),
            (tx.ja("真値の誤差", "true error"), info.get("err", "-")),
            (tx.ja("シミュ時刻", "sim time"), info.get("time", "-")),
        ]
        for k, v in kv:
            tx.add((x, y), k, 15, DIM)
            tx.add((x + 120, y - 2), v, 18, FG)
            y += 30
        y += 6
        if self.pusher == "point":
            tx.add((x, y), tx.ja("判断: 本番の状態機械 + PointPushPlanner（2D 押しモデルで先読み）",
                                 "decisions: production state machine + PointPushPlanner"), 14, DIM)
        else:
            tx.add((x, y), tx.ja("判断: 本番の状態機械 + PushPlanner（2D 押しモデルで先読み）",
                                 "decisions: production state machine + PushPlanner"), 14, DIM)
        tx.add((x, y + 20), f"seed {self.seed}  ·  " + (
            "hand = Gazebo hand model, stroke as hand_actor" if self.hand else
            tx.ja("開始: ランダムに崩した姿勢（Gazebo の scene_manager と同じ）",
                  "start: random disturbed pose (as Gazebo scene_manager)")), 14, DIM)

    def shelf(self, img, done, highlight=None, hide=None):
        tx = self.tx
        cv2.rectangle(img, (CAM_X, SHELF_Y0), (W - 16, SHELF_Y1), (40, 46, 62), -1)
        cv2.rectangle(img, (CAM_X, SHELF_Y1 - 8), (W - 16, SHELF_Y1), (120, 92, 60), -1)
        tx.add((CAM_X + 14, SHELF_Y0 + 12), tx.ja("展示棚", "display shelf"), 22, FG)
        tx.add((CAM_X + 14, SHELF_Y0 + 42), tx.ja("そろった文字を YUKI の順に並べる",
                                                  "finished letters, in YUKI order"), 14, DIM)
        tx.add((CAM_X + 14, SHELF_Y0 + 62), tx.ja(f"{len(done)} / {len(self.word)} 完成",
                                                  f"{len(done)} / {len(self.word)} done"), 18, ACCENT)
        for L, (cx, cy) in self.slots.items():
            r = (int(cx - 58), SHELF_Y0 + 6, int(cx + 58), SHELF_Y1 - 12)
            if L in done and L != hide:
                cv2.rectangle(img, r[:2], r[2:], (58, 66, 88), -1)
                fill = (255, 255, 255) if highlight else (236, 240, 246)
                self.draw_shape(img, L, cx, cy - 4, self.shelf_scale, 0.0, fill)
            else:
                self.dashed(img, np.array([[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]],
                                          float), (90, 98, 120), 1, 6)
                tx.add((cx, cy - 4), L, 30, (80, 88, 108), "mm")


# ================================================================ schedule

def build_schedule(timelines, fps, speedup, realtime_pushes, lead_s=1.0, ok_s=0.9, slide_s=1.3,
                   intro_s=3.0, finale_s=6.5):
    """[(kind, a, b, speed)]: ('intro', u), ('sim', ep, t), ('ok'|'slide'|'fail', ep, u), ('finale', u)."""
    out = []
    n = int(round(intro_s * fps))
    out += [("intro", i / max(1, n - 1), None, 1.0) for i in range(n)]
    for e, tl in enumerate(timelines):
        out += [("sim", e, 0.0, 1.0)] * int(round(lead_s * fps))
        fast_from = tl.push_start(realtime_pushes + 1)
        t = 0.0
        while t <= tl.t_end + 1e-9:
            sp = speedup if t >= fast_from - 1e-9 else 1.0
            out.append(("sim", e, t, sp))
            t += sp / fps
        out.append(("sim", e, tl.t_end, 1.0))
        if tl.ep["row"]["success"]:
            n = int(round(ok_s * fps))
            out += [("ok", e, i / max(1, n - 1), 1.0) for i in range(n)]
            n = int(round(slide_s * fps))
            out += [("slide", e, i / max(1, n - 1), 1.0) for i in range(n)]
        else:
            n = int(round(2.0 * fps))
            out += [("fail", e, i / max(1, n - 1), 1.0) for i in range(n)]
    n = int(round(finale_s * fps))
    out += [("finale", i / max(1, n - 1), None, 1.0) for i in range(n)]
    return out


def choose_speedup(timelines, fps, speedup, realtime_pushes, max_seconds, max_speedup=8.0):
    """Smallest speed-up >= `speedup` (and <= `max_speedup`) that fits `max_seconds`;
    `max_speedup` if none does."""
    speedup = min(float(speedup), float(max_speedup))
    cands = [s for s in sorted({speedup, 2.0, 3.0, 4.0, 6.0, 8.0, float(max_speedup)})
             if speedup <= s <= max_speedup]
    for s in cands:
        sched = build_schedule(timelines, fps, s, realtime_pushes)
        if len(sched) / fps <= max_seconds or s >= cands[-1]:
            return s, sched
    return speedup, build_schedule(timelines, fps, speedup, realtime_pushes)


# ================================================================ frames

def ease(u):
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, u)))


class Snow:
    def __init__(self, n=220, seed=7):
        r = np.random.default_rng(seed)
        self.x = r.uniform(0, W, n)
        self.y0 = r.uniform(-H, H, n)
        self.v = r.uniform(70, 170, n)
        self.r = r.uniform(1.5, 4.5, n)
        self.ph = r.uniform(0, 2 * math.pi, n)
        self.amp = r.uniform(5, 25, n)

    def draw(self, img, t):
        over = img.copy()
        y = (self.y0 + self.v * t) % (H + 40) - 20
        x = self.x + self.amp * np.sin(self.ph + t * 1.7)
        for xi, yi, ri in zip(x, y, self.r):
            cv2.circle(over, (int(xi * 4), int(yi * 4)), int(ri * 4), (255, 255, 255), -1,
                       cv2.LINE_AA, shift=2)
        cv2.addWeighted(over, 0.85, img, 0.15, 0, dst=img)


def render_frame(R: Renderer, spec, timelines, snow, fps):
    kind, a, b, speed = spec
    tx = R.tx
    img = R.frame_base()
    done = [tl.ep["letter"] for e, tl in enumerate(timelines)
            if tl.ep["row"]["success"] and kind in ("sim", "ok", "slide", "fail") and e < a]
    if kind == "finale":
        done = [tl.ep["letter"] for tl in timelines if tl.ep["row"]["success"]]
    if kind == "intro":
        R.camera_view(img, R.camera_image(None, (0, 0, 0), None, None))
        R.title_bar(img, None)
        R.shelf(img, [])
        over = img.copy()
        cv2.rectangle(over, (0, 0), (W, H), BG, -1)
        cv2.addWeighted(over, 0.78, img, 0.22, 0, dst=img)
        tx.add((W / 2, 200), "Push-Alphabet", 64, FG, "mm")
        tx.add((W / 2, 270), tx.ja("崩されても自分で直す、雪の文字ロボ",
                                   "the snow-letter robot that fixes what you knock over"), 30, ACCENT, "mm")
        lines = [tx.ja("真上カメラで文字を見て、押し棒（点）で押して元に戻す",
                       "overhead camera -> a point pusher pushes the letters back")
                 if R.pusher == "point" else
                 tx.ja("真上カメラで文字を見て、4本のヘラで押して元に戻す",
                       "overhead camera -> 4 servo paddles push the letters back"),
                 tx.ja("手で崩す → 手が離れるのを待つ → 押して直す → 棚へ → YUKI 完成",
                       "hand knocks -> wait for the hand -> push back -> shelf -> YUKI")
                 if R.hand else
                 tx.ja("崩れた文字を見つける → 押して直す → 棚へ → YUKI 完成",
                       "finds the disturbed letter -> pushes it back -> shelf -> YUKI"),
                 f"perception: {PERCEPTION_DESC.get(R.perception, R.perception)}   ·   seed {R.seed}",
                 CAPTION_EN]
        for i, s in enumerate(lines):
            tx.add((W / 2, 350 + 42 * i), s, 22 if i < 2 else 18, FG if i < 2 else DIM, "mm")
        return tx.flush(img)

    if kind == "finale":
        R.camera_view(img, R.camera_image(None, (0, 0, 0), None, None))
        R.title_bar(img, len(R.seq))
        R.panel(img, dict(state="DONE", detail="", letter=None, pushes=sum(
            len(tl.pushes) for tl in timelines), cmd="-",
            time=f"{sum(tl.t_end for tl in timelines):.1f} s " + tx.ja("（合計）", "(total)")))
        R.shelf(img, done, highlight=True)
        R.caption(img)
        ok = len(done) == len(R.seq)
        if ok:
            snow.draw(img, a * 6.5)
        bx = (CAM_X + 40, 210, CAM_X + CAM_S - 40, 420)
        over = img.copy()
        cv2.rectangle(over, bx[:2], bx[2:], (20, 40, 80), -1)
        cv2.addWeighted(over, 0.75, img, 0.25, 0, dst=img)
        cy = (bx[1] + bx[3]) / 2
        if ok:
            tx.add(((bx[0] + bx[2]) / 2, cy - 30), tx.ja("YUKI 完成！", "YUKI complete!"), 64,
                   (255, 255, 255), "mm")
            tx.add(((bx[0] + bx[2]) / 2, cy + 40), tx.ja("4 文字すべて自分で直せた", "all 4 letters recovered"),
                   24, ACCENT, "mm")
        else:
            tx.add(((bx[0] + bx[2]) / 2, cy - 20), f"{len(done)} / {len(R.seq)}", 64, (255, 255, 255), "mm")
            tx.add(((bx[0] + bx[2]) / 2, cy + 40), tx.ja("一部の文字は直せなかった", "not all letters recovered"),
                   24, (255, 160, 140), "mm")
        return tx.flush(img)

    e = a
    tl = timelines[e]
    L = tl.ep["letter"]
    t = b if kind == "sim" else tl.t_end
    pose, hand, paddle = tl.world(t)
    pin = tl.pin(t)
    state, detail = tl.state(t)
    if kind in ("ok", "slide") and state == "RECOVER":
        state, detail = "WAIT_LETTER", ""
    show_letter = kind != "slide"
    cam = R.camera_image(L if show_letter else None, pose, hand, paddle, pin=pin)
    R.title_bar(img, e)
    R.camera_view(img, cam)
    ot, dets = tl.detections(t)
    hp = tl.hand_present(t)
    if show_letter:
        R.overlays(img, L, dets, (t - ot) if ot is not None else None, hp,
                   paddle is not None or pin is not None)
    lp = tl.last_push(t)
    if paddle is not None and lp is not None:
        R.paddle_label(img, paddle, lp[2])
    if pin is not None and lp is not None and lp[3] is not None:
        R.stroke_label(img, lp[3], pin)
    R.speed_label(img, speed)
    R.caption(img)
    clear = None
    if state == "WAIT_HAND" and not hp:
        lt = tl.hand_left_t(t)
        if lt is not None:
            clear = (max(0.0, t - lt), Params().hand_clear_s)
    elif state == "WAIT_HAND":
        clear = (0.0, Params().hand_clear_s)
    cmd = "-"
    if lp is not None and lp[3] is not None:
        x0, y0, x1, y1 = (v * 1000 for v in lp[3])
        cmd = (f"({x0:+.0f}, {y0:+.0f}) → ({x1:+.0f}, {y1:+.0f}) mm, " +
               tx.ja(f"長さ {lp[2]:.0f} mm", f"length {lp[2]:.0f} mm"))
    elif lp is not None:
        name = R.pushers["list"][lp[1]]["name"]
        cmd = (f"{SIDE_JA.get(name, name)}のヘラ ({name}) {lp[2]:.0f}°" if tx.cjk
               else f"{name} paddle {lp[2]:.0f}°")
    if dets is None or ot is None:
        det = "-"
    elif not dets:
        det = tx.ja(f"なし  (t={ot:.1f}s)", f"none (t={ot:.1f}s)")
    else:
        d = max(dets, key=lambda d: d[4])
        det = f"{d[0]} conf {d[4]:.2f}  ({len(dets)})"
    dp, da = pose_error(L, *pose)
    tg = R.tgt
    err = f"{dp * 1000:.1f} mm / {da:.1f}°   (tol {tg['pos_tol'] * 1000:.0f} mm / {tg['ang_tol']:.0f}°)"
    t_before = sum(x.t_end for x in timelines[:e])
    info = dict(state=state, detail=detail, letter=L, idx=e, pushes=tl.n_pushes(t), cmd=cmd,
                hand_present=hp, hand_visible=hand is not None, clear=clear, det=det, err=err,
                time=f"{t:5.1f} s  " + tx.ja(f"（合計 {t_before + t:.1f} s）", f"(total {t_before + t:.1f} s)"))
    R.panel(img, info)
    R.shelf(img, done + ([L] if kind == "slide" and a == e and b >= 1.0 else []))
    if kind == "ok":
        f = 0.5 + 0.5 * math.cos(b * 3 * math.pi)
        col = (int(60 + 60 * f), int(200 + 40 * f), 110)
        cv2.rectangle(img, (CAM_X - 2, CAM_Y - 2), (CAM_X + CAM_S + 1, CAM_Y + CAM_S + 1), col, 6)
        r = tl.ep["row"]
        tx.add((CAM_X + CAM_S / 2, CAM_Y + 90), "OK", 84, col, "mm")
        tx.add((CAM_X + CAM_S / 2, CAM_Y + 150),
               tx.ja(f"{L} がそろった：{r['pushes']} 回の押しで {r['final_dp_mm']:.1f} mm / {r['final_da_deg']:.1f}°",
                     f"{L} aligned: {r['pushes']} pushes, {r['final_dp_mm']:.1f} mm / {r['final_da_deg']:.1f}°"),
               22, (255, 255, 255), "mm")
    elif kind == "fail":
        r = tl.ep["row"]
        tx.add((CAM_X + CAM_S / 2, CAM_Y + 90), "FAILED", 64, (240, 90, 80), "mm")
        tx.add((CAM_X + CAM_S / 2, CAM_Y + 150), f"{L}: {r['fail_reason']} ({r['pushes']} pushes)", 22,
               (255, 255, 255), "mm")
    elif kind == "slide":
        u = ease(b)
        c0 = R.to_screen([pose[:2]], R.thick)[0]
        c1 = np.array(R.slots[L]) + [0, -4]
        s0 = R.k * R.cam.px_per_m(R.thick)
        c = c0 + (c1 - c0) * u
        c[1] -= 80 * math.sin(math.pi * u)
        sc = s0 + (R.shelf_scale - s0) * u
        th = pose[2] * (1 - u)
        R.draw_shape(img, L, c[0], c[1], sc, th, (240, 244, 250))
        tx.add((CAM_X + CAM_S / 2, CAM_Y + 90), tx.ja("展示棚へ", "to the shelf"), 30, ACCENT, "mm")
    return tx.flush(img)


# ================================================================ output

class FFmpegWriter:
    def __init__(self, path, fps, crf=20):
        if not shutil.which("ffmpeg"):
            raise SystemExit("ffmpeg not found")
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        self.p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
             "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", path],
            stdin=subprocess.PIPE)
        self.n = 0

    def write(self, rgb):
        self.p.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        if self.p.wait() != 0:
            raise RuntimeError("ffmpeg failed")


def pick_sheet_frames(schedule, timelines):
    """Indices of 6 representative frames: hand knocking (no hand: the disturbed
    start), waiting for the hand (no hand: the 2nd letter's start), a paddle /
    the pusher mid-stroke (real time), a fast push, the shelf slide, the finale."""
    want = {}
    zd = 0.01
    for i, (kind, a, b, sp) in enumerate(schedule):
        if kind == "sim":
            tl = timelines[a]
            pose, hand, paddle = tl.world(b)
            pin = tl.pin(b)
            if pin is not None and pin[2] < 0.005:
                paddle = (-1, math.pi)  # lowered pusher: counts as a push in progress
            st, _ = tl.state(b)
            if not tl.ep.get("hand", True):
                if "hand" not in want and a == 0 and b > 0.3:
                    want["hand"] = i
                if "wait" not in want and a == 1 and b > 0.3:
                    want["wait"] = i
            elif "hand" not in want and hand is not None and hand[2] < zd and a == 0:
                want["hand"] = i
            elif "wait" not in want and st == "WAIT_HAND" and not tl.hand_present(b) and a == 0 \
                    and (hand is None or hand[2] > 0.03):
                want["wait"] = i
            if "push" not in want and paddle is not None and sp == 1.0 and a == 0 \
                    and abs(paddle[1]) > math.radians(20):
                want["push"] = i
            if "fast" not in want and paddle is not None and sp > 1.0 and a >= 1 \
                    and abs(paddle[1]) > math.radians(20):
                want["fast"] = i
        elif kind == "slide" and a == len(timelines) - 1 and b >= 0.5 and "slide" not in want:
            want["slide"] = i
        elif kind == "finale" and a >= 0.6 and "finale" not in want:
            want["finale"] = i
    return want


def write_sheet(path, frames):
    tw, th = 640, 360
    keys = list(frames)
    cols = 3
    rows = (len(keys) + cols - 1) // cols
    sheet = np.full((rows * (th + 8) + 8, cols * (tw + 8) + 8, 3), 255, np.uint8)
    for j, k in enumerate(keys):
        r, c = divmod(j, cols)
        sm = cv2.resize(frames[k], (tw, th), interpolation=cv2.INTER_AREA)
        y, x = 8 + r * (th + 8), 8 + c * (tw + 8)
        sheet[y:y + th, x:x + tw] = sm
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    cv2.imwrite(path, sheet[:, :, ::-1])


def make_video(episodes, seed, perception, out, fps=30, speedup=4.0, max_seconds=100.0,
               realtime_pushes=2, max_frames=None, sheet=None, crf=20, log=print, max_speedup=8.0):
    sc = load_scene()
    timelines = [Timeline(e) for e in episodes]
    sp, sched = choose_speedup(timelines, fps, speedup, realtime_pushes, max_seconds, max_speedup)
    if max_frames is not None:
        sched = sched[:max_frames]
    log(f"schedule: {len(sched)} frames = {len(sched) / fps:.1f} s at {fps} fps, speedup x{sp:g} "
        f"after push {realtime_pushes}")
    tx = Text()
    if not tx.cjk:
        log("no CJK font found: English labels only")
    pusher = episodes[0].get("pusher", "paddles") if episodes else "paddles"
    R = Renderer(sc, perception, seed, fps, tx, pusher=pusher,
                 hand=episodes[0].get("hand", True) if episodes else True)
    snow = Snow()
    want = pick_sheet_frames(sched, timelines) if sheet else {}
    grabbed = {}
    wr = FFmpegWriter(out, fps, crf)
    w0 = time.perf_counter()
    try:
        for i, spec in enumerate(sched):
            fr = render_frame(R, spec, timelines, snow, fps)
            wr.write(fr)
            for k, j in want.items():
                if j == i:
                    grabbed[k] = fr
            if i and i % 300 == 0:
                log(f"  frame {i}/{len(sched)} ({time.perf_counter() - w0:.0f} s)")
    finally:
        wr.close()
    if sheet and grabbed:
        order = ["hand", "wait", "push", "fast", "slide", "finale"]
        write_sheet(sheet, {k: grabbed[k] for k in order if k in grabbed})
    return dict(frames=wr.n, seconds=wr.n / fps, speedup=sp, sheet_keys=list(grabbed))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--perception", choices=["truth", "classic", "cnn", "hybrid"], default=None,
                    help="default: hybrid if onnxruntime + the CNN model are available, else classic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--search", type=int, default=40,
                    help="try seeds seed..seed+N-1 until all letters succeed (1: use --seed as is)")
    ap.add_argument("--letters", default=None, help="default: scene sequence (IUYK)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "data", "video", "demo_2d.mp4"))
    ap.add_argument("--sheet", default=None, help="also write a 6-frame contact sheet PNG")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-seconds", type=float, default=100.0)
    ap.add_argument("--speedup", type=float, default=2.0)  # course rule: at most 2x
    ap.add_argument("--max-speedup", type=float, default=2.0,
                    help="never speed up more than this (course rule: x2)")
    ap.add_argument("--pusher", choices=["point", "paddles"], default="point",
                    help="point: Push-T style point pusher (docs/point_pusher.md); paddles: 4 paddles")
    ap.add_argument("--hand", action="store_true",
                    help="the hand knocks each letter (default: start at a random disturbed pose)")
    ap.add_argument("--realtime-pushes", type=int, default=2,
                    help="pushes per letter shown in real time before speeding up")
    ap.add_argument("--max-pushes", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=None, help="stop after N frames (tests)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--crf", type=int, default=20)
    args = ap.parse_args(argv)

    perception = args.perception or default_perception()
    letters = args.letters or load_scene()["sequence"]
    t0 = time.perf_counter()
    eps, seed, ok = run_episodes(letters, args.seed, perception, args.max_pushes, jobs=args.jobs,
                                 search=args.search, pusher=args.pusher, hand=args.hand)
    if not ok:
        print(f"WARNING: no seed in [{args.seed}, {args.seed + args.search}) where every letter "
              f"succeeds; rendering seed {seed} as it happened")
    for e in eps:
        r = e["row"]
        print(f"  {e['letter']}: success={r['success']} {r['fail_reason'] or ''} pushes={r['pushes']} "
              f"sim={r['sim_time_s']}s final={r['final_dp_mm']}mm/{r['final_da_deg']}deg "
              f"wait_hand={r['wait_hand_s']}s disturb={r['disturb_dp_mm']}mm/{r['disturb_da_deg']}deg")
    res = make_video(eps, seed, perception, args.out, args.fps, args.speedup, args.max_seconds,
                     args.realtime_pushes, args.max_frames, args.sheet, args.crf,
                     max_speedup=args.max_speedup)
    print(f"wrote {args.out}: {res['frames']} frames, {res['seconds']:.1f} s, speedup x{res['speedup']:g}, "
          f"seed {seed}, perception {perception}, pusher {args.pusher}, hand {args.hand}, wall {time.perf_counter() - t0:.0f} s")
    if args.sheet:
        print(f"wrote {args.sheet} ({', '.join(res['sheet_keys'])})")
    return dict(res, seed=seed, all_ok=ok, episodes=eps)


if __name__ == "__main__":
    main()
