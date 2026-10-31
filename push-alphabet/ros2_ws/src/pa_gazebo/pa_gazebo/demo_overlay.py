"""Demo video overlay and scripted hand (pure python + OpenCV, no ROS).

Used by pa_gazebo.demo_recorder:

- `image_to_bgr`: sensor_msgs/Image fields -> BGR ndarray (no cv_bridge needed).
- `compose_frame`: overhead camera frame + HUD (detected letter outlines and
  labels, planner state badge, active letter, push count, hand indicator,
  sim-time clock) + a bottom panel with the progress word ("_ _ _ _" filling
  in Y U K I as letters reach the display shelf).
- `ScriptedHand`: decides when and where a fake hand sweeps through the active
  letter, so a demo can be recorded without the webcam (auto_hand).

Text is drawn with OpenCV Hershey fonts (ASCII only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random

import cv2
import numpy as np

from pa_core.letters import letter_polygons

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
PANEL_H = 120

# BGR
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (140, 140, 140)
DARK = (35, 35, 35)
STATE_COLORS = {
    "WAIT_LETTER": (120, 120, 120),
    "IDLE": (60, 160, 60),
    "WAIT_HAND": (0, 140, 255),
    "RECOVER": (200, 120, 0),
    "LOST": (0, 0, 200),
    "FAILED": (0, 0, 200),
    "DONE": (170, 60, 170),
}
STATE_LABELS = {
    "WAIT_LETTER": "WAIT LETTER",
    "IDLE": "IDLE",
    "WAIT_HAND": "WAIT HAND",
    "RECOVER": "RECOVER",
    "LOST": "LOST",
    "FAILED": "FAILED",
    "DONE": "DONE",
}
SIDE_NAMES = ("S", "E", "N", "W")


# ------------------------------------------------------------------ image I/O

def image_to_bgr(encoding: str, height: int, width: int, step: int, data) -> np.ndarray:
    """sensor_msgs/Image fields -> (height, width, 3) uint8 BGR (copy)."""
    enc = (encoding or "").lower()
    try:
        buf = np.frombuffer(data, dtype=np.uint8)   # bytes / array.array('B') / memoryview
    except TypeError:
        buf = np.asarray(data, dtype=np.uint8).ravel()
    channels = {"bgr8": 3, "rgb8": 3, "bgra8": 4, "rgba8": 4, "mono8": 1, "8uc1": 1, "8uc3": 3}
    if enc not in channels:
        raise ValueError(f"unsupported image encoding {encoding!r}")
    c = channels[enc]
    rows = buf[: height * step].reshape(height, step)[:, : width * c]
    img = rows.reshape(height, width, c) if c > 1 else rows.reshape(height, width)
    if enc in ("rgb8",):
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if c == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def even_size(img: np.ndarray) -> np.ndarray:
    """Crop to even width/height (H.264 yuv420p needs even dimensions)."""
    h, w = img.shape[:2]
    return img[: h - h % 2, : w - w % 2]


def side_by_side(left: np.ndarray, right: np.ndarray, gap: int = 0) -> np.ndarray:
    """Horizontal concat; the shorter image is padded at the bottom with black."""
    h = max(left.shape[0], right.shape[0])

    def pad(im):
        if im.shape[0] == h:
            return im
        return cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
    parts = [pad(left)]
    if gap > 0:
        parts.append(np.zeros((h, gap, 3), np.uint8))
    parts.append(pad(right))
    return np.hstack(parts)


# --------------------------------------------------------------------- state

@dataclass
class LetterPose:
    letter: str
    x: float
    y: float
    theta: float
    confidence: float = 1.0


@dataclass
class HudState:
    status: str = ""                 # /planner/status
    active: str | None = None        # /active_letter (None = not received yet, "" = all done)
    word: str = "YUKI"               # display word (progress order)
    done: list = field(default_factory=list)   # finished letters, in finishing order
    pushes_total: int = 0
    pushes_letter: int = 0           # pushes for the current active letter
    last_push: tuple | None = None   # (side, angle_deg)
    pushing: bool = False            # a /push_cmd has not come back on /push_done yet
    last_stroke: tuple | None = None  # point pusher (x0, y0, x1, y1) [m], world frame
    last_stroke_speed: float = 0.0   # [m/s] (0 = scene default)
    hand_present: bool = False
    hand_xy: tuple | None = None
    sim_time: float | None = None    # [s] stamp of the current frame
    start_time: float | None = None  # [s] stamp of the first recorded frame
    letters: list = field(default_factory=list)  # [LetterPose] from /letters
    label: str = ""                  # free text for the panel, e.g. "detector: hybrid CNN" (optional)
    hand_source: str = ""            # "webcam" | "auto" | "" (optional)

    @property
    def complete(self) -> bool:
        return self.active == "" and len(self.done) >= len(self.word) > 0


def progress_slots(word: str, done) -> list:
    """Letters of `word` that are finished, "_" for the rest. Repeated letters fill left to right."""
    remaining = [c.upper() for c in done]
    slots = []
    for c in word.upper():
        if c in remaining:
            remaining.remove(c)
            slots.append(c)
        else:
            slots.append("_")
    return slots


def progress_text(word: str, done) -> str:
    return " ".join(progress_slots(word, done))


def format_clock(t: float | None) -> str:
    """Seconds -> 'm:ss.s' ('--:--.-' when unknown)."""
    if t is None or not math.isfinite(t):
        return "--:--.-"
    t = max(0.0, t)
    m = int(t // 60)
    return f"{m}:{t - 60 * m:04.1f}"


def state_color(status: str) -> tuple:
    return STATE_COLORS.get(status, (90, 90, 90))


# ------------------------------------------------------------------- drawing

def draw_badge(img, text, org, bg, fg=WHITE, scale=0.6, thickness=1, pad=6, font=FONT):
    """Filled rounded-ish box with text; org = top-left. Returns (x1, y1) bottom-right corner."""
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    x0, y0 = int(org[0]), int(org[1])
    x1, y1 = x0 + tw + 2 * pad, y0 + th + base + 2 * pad
    cv2.rectangle(img, (x0, y0), (x1, y1), bg, -1, cv2.LINE_AA)
    cv2.putText(img, text, (x0 + pad, y0 + pad + th), font, scale, fg, thickness, cv2.LINE_AA)
    return x1, y1


def letter_pixel_polys(cam, letter, x, y, theta, z=0.0) -> list:
    """Pixel polygons (int32 Nx2) of the letter's rectangles at height z."""
    return [np.round(cam.world_to_pixel(p, z)).astype(np.int32) for p in letter_polygons(letter, x, y, theta)]


def draw_letters(img, letters, cam, z=0.0, active=None):
    """Outline + heading + label of each detected letter (in place)."""
    for lp in letters:
        name = (lp.letter or "").upper()
        if not name:
            c = cam.world_to_pixel([[lp.x, lp.y]], z)[0]
            cv2.circle(img, (int(c[0]), int(c[1])), 8, (0, 0, 255), 2, cv2.LINE_AA)
            continue
        color = (0, 220, 0) if active and name == active.upper() else (255, 200, 0)
        try:
            polys = letter_pixel_polys(cam, name, lp.x, lp.y, lp.theta, z)
        except KeyError:
            continue
        for poly in polys:
            cv2.polylines(img, [poly], True, color, 2, cv2.LINE_AA)
        c = cam.world_to_pixel([[lp.x, lp.y]], z)[0]
        h = cam.world_to_pixel([[lp.x - 0.03 * math.sin(lp.theta), lp.y + 0.03 * math.cos(lp.theta)]], z)[0]
        cv2.arrowedLine(img, (int(c[0]), int(c[1])), (int(h[0]), int(h[1])), (0, 0, 255), 2,
                        cv2.LINE_AA, tipLength=0.25)
        label = f"{name} {lp.confidence:.2f} {math.degrees(lp.theta):+.0f}deg"
        (tw, _), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        allp = np.vstack(polys)
        ly = allp[:, 1].min() - 10                      # above the outline,
        if ly < 60:                                     # or below it near the top HUD
            ly = allp[:, 1].max() + 22
        org = (int(np.clip(c[0] - tw / 2, 2, img.shape[1] - tw - 2)), int(np.clip(ly, 14, img.shape[0] - 4)))
        cv2.putText(img, label, org, FONT, 0.5, BLACK, 3, cv2.LINE_AA)
        cv2.putText(img, label, org, FONT, 0.5, color, 1, cv2.LINE_AA)
    return img


def draw_hand_marker(img, cam, xy):
    c = cam.world_to_pixel([list(xy)], 0.0)[0]
    p = (int(c[0]), int(c[1]))
    cv2.circle(img, p, 14, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(img, p, (0, 0, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    return img


STROKE_COLOR = (0, 140, 255)       # BGR orange, as the pusher
STROKE_DONE_COLOR = (0, 90, 160)


def draw_stroke(img, cam, stroke, z=0.0, active=True):
    """Point-pusher stroke: circle at the start (pusher footprint), arrow to the end."""
    x0, y0, x1, y1 = stroke
    pts = cam.world_to_pixel([[x0, y0], [x1, y1]], z)
    a = (int(round(pts[0][0])), int(round(pts[0][1])))
    b = (int(round(pts[1][0])), int(round(pts[1][1])))
    col = STROKE_COLOR if active else STROKE_DONE_COLOR
    cv2.circle(img, a, 7, BLACK, 4, cv2.LINE_AA)
    cv2.circle(img, a, 7, col, 2, cv2.LINE_AA)
    if a != b:
        cv2.arrowedLine(img, a, b, BLACK, 5, cv2.LINE_AA, tipLength=0.2)
        cv2.arrowedLine(img, a, b, col, 2, cv2.LINE_AA, tipLength=0.2)
    return img


def draw_progress(panel, word, done, org, scale=1.6, highlight=None):
    """The progress word: finished letters bright, blanks grey. `highlight` = active letter."""
    x, y = int(org[0]), int(org[1])
    slots = progress_slots(word, done)
    for i, s in enumerate(slots):
        if s == "_":
            ch = word[i].upper()
            is_active = highlight and ch == highlight.upper()
            col = (0, 200, 255) if is_active else GREY
            text = "_"
        else:
            col, text = WHITE, s
        (tw, th), _ = cv2.getTextSize("W", FONT_BOLD, scale, 2)
        cv2.putText(panel, text, (x, y), FONT_BOLD, scale, col, 3, cv2.LINE_AA)
        x += tw + int(18 * scale / 1.6)
    return panel


def compose_frame(img, hud: HudState, cam=None, z: float = 0.0, panel_h: int = PANEL_H) -> np.ndarray:
    """Annotated frame: img (BGR, HxW) with HUD on top + a panel_h-pixel bottom panel.

    Output shape: (H + panel_h, W, 3), uint8. `cam` (pa_perception CameraModel
    sized for img) is needed for letter outlines and the hand marker; without it
    only the text HUD is drawn.
    """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    out = np.zeros((h + panel_h, w, 3), np.uint8)
    out[:h] = img
    view = out[:h]

    if cam is not None:
        draw_letters(view, hud.letters, cam, z, hud.active)
        if hud.hand_present and hud.hand_xy is not None:
            draw_hand_marker(view, cam, hud.hand_xy)
        if hud.last_stroke is not None:
            draw_stroke(view, cam, hud.last_stroke, z, active=hud.pushing)

    # top-left: planner state + active letter
    status = hud.status or "--"
    x1, _ = draw_badge(view, STATE_LABELS.get(status, status), (8, 8), state_color(status), scale=0.7,
                       thickness=2)
    if hud.active is None:
        act = "letter: -"
    elif hud.active == "":
        act = "all letters done"
    else:
        act = f"letter: {hud.active}"
    draw_badge(view, act, (x1 + 8, 8), DARK, scale=0.7, thickness=2)

    # top-right: hand indicator
    if hud.hand_present:
        text, bg = "HAND", (0, 0, 220)
    else:
        text, bg = "no hand", (80, 80, 80)
    if hud.hand_source:
        text += f" ({hud.hand_source})"
    (tw, _), _ = cv2.getTextSize(text, FONT, 0.6, 1)
    draw_badge(view, text, (w - tw - 20, 8), bg, scale=0.6)

    if hud.complete:
        msg = f"{hud.word.upper()} complete!"
        (tw, th), _ = cv2.getTextSize(msg, FONT_BOLD, 1.4, 3)
        org = ((w - tw) // 2, h // 2 + th // 2)
        cv2.putText(view, msg, org, FONT_BOLD, 1.4, BLACK, 7, cv2.LINE_AA)
        cv2.putText(view, msg, org, FONT_BOLD, 1.4, (255, 230, 200), 3, cv2.LINE_AA)

    # bottom panel
    panel = out[h:]
    panel[:] = (24, 24, 24)
    cv2.line(panel, (0, 0), (w, 0), (70, 70, 70), 1)
    draw_progress(panel, hud.word, hud.done, (16, int(panel_h * 0.62)), highlight=hud.active or None)
    lines = [
        f"pushes {hud.pushes_total}" + (f"  ({hud.active}: {hud.pushes_letter})" if hud.active else ""),
    ]
    if hud.last_stroke is not None:
        x0, y0, x1, y1 = hud.last_stroke
        length = math.hypot(x1 - x0, y1 - y0)
        spd = f" @ {hud.last_stroke_speed * 100:.0f}cm/s" if hud.last_stroke_speed > 0 else ""
        lines.append(f"last push {length * 100:.1f}cm{spd}" + ("  ..." if hud.pushing else ""))
    elif hud.last_push is not None:
        side, ang = hud.last_push
        name = SIDE_NAMES[side] if 0 <= int(side) < len(SIDE_NAMES) else str(side)
        lines.append(f"last push {name} {ang:.0f}deg" + ("  ..." if hud.pushing else ""))
    elapsed = None
    if hud.sim_time is not None and hud.start_time is not None:
        elapsed = hud.sim_time - hud.start_time
    lines.append(f"sim t {format_clock(hud.sim_time)}  rec {format_clock(elapsed)}")
    if hud.label:
        lines.append(hud.label)
    x = int(w * 0.48)
    for i, line in enumerate(lines[:4]):
        cv2.putText(panel, line, (x, 24 + 26 * i), FONT, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return out


# -------------------------------------------------------------- scripted hand

@dataclass
class KnockConfig:
    knocks_per_letter: int = 1     # sweeps per active letter
    after_pushes: int = 2          # knock once this many pushes of the letter came back
    idle_s: float = 4.0            # ... or when the planner sat in IDLE this long
    start_delay_s: float = 5.0     # nothing before this (let the scene come up)
    settle_s: float = 0.6          # after /push_done before knocking
    approach_s: float = 1.6        # hand present at the start point (hand_actor flies + descends)
    speed: float = 0.10            # sweep speed [m/s]
    back_off: float = 0.10         # start this far from the letter center
    overshoot: float = -0.03       # palm ends this far past the letter center (+-jitter);
                                   # -0.03: the palm edge (half width 0.0375) just passes the center
    jitter: float = 0.01           # random offsets [m]
    limit: float = 0.095           # keep the palm within |x|,|y| <= limit
    hold_s: float = 0.3            # stay at the end point before lifting


def knock_path(cx: float, cy: float, rng: random.Random, cfg: KnockConfig):
    """Start and end palm positions of a sideways sweep through a letter at (cx, cy).

    The hand comes in along x from the side with more room (fingers point +y,
    so a sideways sweep keeps the descent clear of the letter) and shoves the
    letter a few cm; the palm line is a bit below the letter center so the
    shove also rotates it.
    """
    d = 1.0 if cx >= 0.0 else -1.0   # start on the far side of the table center
    y = cy - 0.02 + rng.uniform(-cfg.jitter, cfg.jitter)
    x0 = cx - d * cfg.back_off
    x1 = cx + d * (cfg.overshoot + rng.uniform(-cfg.jitter, cfg.jitter))
    # never shove beyond a little past the table center line
    x1 = d * min(d * x1, 0.02)
    lim = cfg.limit
    clip = lambda v: max(-lim, min(lim, v))  # noqa: E731
    return (clip(x0), clip(y)), (clip(x1), clip(y))


class ScriptedHand:
    """Fake /hand/state stream that knocks the active letter during recovery.

    Feed it events (on_active, on_status, on_push, on_push_done, on_letters)
    and call step(now) at the publish rate; it returns (present, x, y).
    """

    def __init__(self, cfg: KnockConfig | None = None, rng: random.Random | None = None):
        self.cfg = cfg or KnockConfig()
        self.rng = rng or random.Random()
        self.t0 = None
        self.active = ""
        self.knocks = 0
        self.pushes_done = 0
        self.inflight = 0
        self.last_push_done = -1e9
        self.status = ""
        self.status_since = 0.0
        self.letter_xy = None
        self.sweep = None   # (t_start, p0, p1, t_sweep)
        self.total_knocks = 0

    # events
    def on_active(self, letter: str, now: float):
        letter = (letter or "").upper()
        if letter != self.active:
            self.active = letter
            self.knocks = 0
            self.pushes_done = 0
            self.inflight = 0
            self.letter_xy = None

    def on_status(self, status: str, now: float):
        if status != self.status:
            self.status = status
            self.status_since = now

    def on_push(self, now: float):
        self.inflight += 1

    def on_push_done(self, now: float):
        self.inflight = max(0, self.inflight - 1)
        self.pushes_done += 1
        self.last_push_done = now

    def on_letters(self, letters, now: float):
        for lp in letters:
            if self.active and (lp.letter or "").upper() == self.active:
                self.letter_xy = (lp.x, lp.y)
                return

    # decisions
    @property
    def busy(self) -> bool:
        return self.sweep is not None

    def want_knock(self, now: float) -> bool:
        c = self.cfg
        if self.t0 is None:
            self.t0 = now
        if not self.active or self.busy or self.knocks >= c.knocks_per_letter:
            return False
        if now - self.t0 < c.start_delay_s:
            return False
        if self.status == "RECOVER" and self.inflight == 0 and self.pushes_done >= c.after_pushes \
                and now - self.last_push_done >= c.settle_s:
            return True
        return self.status == "IDLE" and now - self.status_since >= c.idle_s

    def start(self, now: float):
        cx, cy = self.letter_xy if self.letter_xy is not None else (0.0, 0.0)
        p0, p1 = knock_path(cx, cy, self.rng, self.cfg)
        dist = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        self.sweep = (now, p0, p1, dist / max(self.cfg.speed, 1e-6))
        self.knocks += 1
        self.total_knocks += 1

    def step(self, now: float):
        """(present, x, y) to publish at time `now`."""
        if self.sweep is None and self.want_knock(now):
            self.start(now)
        if self.sweep is None:
            return False, 0.0, 0.0
        t0, p0, p1, ts = self.sweep
        t = now - t0
        c = self.cfg
        if t < c.approach_s:
            return True, p0[0], p0[1]
        t -= c.approach_s
        if t < ts:
            a = t / ts if ts > 0 else 1.0
            return True, p0[0] + (p1[0] - p0[0]) * a, p0[1] + (p1[1] - p0[1]) * a
        if t < ts + c.hold_s:
            return True, p1[0], p1[1]
        self.sweep = None
        return False, 0.0, 0.0
