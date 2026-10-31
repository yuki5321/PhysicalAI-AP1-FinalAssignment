"""Image -> table mapping and smoothing for hand_bridge.py (pure Python, no mediapipe/cv2).

Calibration: a crop rectangle of the webcam image, in normalized image
coordinates (u right, v down, both 0..1), is mapped onto the table square
[-half_size, half_size]^2 of the world frame in scene.yaml (+x right, +y "up"
of the letters = away from the person). `mirror` flips left/right (use it when
the camera faces you, so moving your hand right moves the sim hand right);
`rotate_deg` (multiple of 90) rotates the image frame if the camera is mounted
sideways.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

DEFAULT_HALF_SIZE = 0.16

# MediaPipe Hands landmark ids: wrist and the four finger MCP joints.
PALM_LANDMARKS = (0, 5, 9, 13, 17)


@dataclass(frozen=True)
class Calibration:
    crop: tuple = (0.0, 0.0, 1.0, 1.0)   # u0, v0, u1, v1 (normalized image coords)
    half_size: float = DEFAULT_HALF_SIZE  # scene.yaml workspace.table_half_size [m]
    mirror: bool = False
    rotate_deg: int = 0                   # 0 / 90 / 180 / 270, counter-clockwise


def image_to_table(u: float, v: float, cal: Calibration):
    """Normalized image point -> (x, y, inside) on the table in meters.

    `inside` is False when the point is outside the crop rectangle; x, y are
    then still extrapolated (callers usually clamp or drop them).
    """
    u0, v0, u1, v1 = cal.crop
    if u1 <= u0 or v1 <= v0:
        raise ValueError(f"bad crop rectangle {cal.crop}")
    s = (u - u0) / (u1 - u0)
    t = (v - v0) / (v1 - v0)
    if cal.mirror:
        s = 1.0 - s
    a = 2.0 * s - 1.0          # -1 (left) .. 1 (right)
    b = 1.0 - 2.0 * t          # -1 (bottom) .. 1 (top)
    r = math.radians(cal.rotate_deg)
    c, sn = round(math.cos(r), 12), round(math.sin(r), 12)
    a, b = c * a - sn * b, sn * a + c * b
    inside = -1.0 <= a <= 1.0 and -1.0 <= b <= 1.0
    return a * cal.half_size, b * cal.half_size, inside


def table_to_image(x: float, y: float, cal: Calibration):
    """Inverse of image_to_table (for drawing the table outline in the preview)."""
    a, b = x / cal.half_size, y / cal.half_size
    r = -math.radians(cal.rotate_deg)
    c, sn = round(math.cos(r), 12), round(math.sin(r), 12)
    a, b = c * a - sn * b, sn * a + c * b
    s, t = (a + 1.0) / 2.0, (1.0 - b) / 2.0
    if cal.mirror:
        s = 1.0 - s
    u0, v0, u1, v1 = cal.crop
    return u0 + s * (u1 - u0), v0 + t * (v1 - v0)


def palm_center(landmarks):
    """Mean of wrist + MCP landmarks. `landmarks` is a sequence of objects with
    .x/.y (MediaPipe) or (x, y) pairs, in normalized image coordinates."""
    pts = []
    for i in PALM_LANDMARKS:
        p = landmarks[i]
        pts.append((p.x, p.y) if hasattr(p, "x") else (p[0], p[1]))
    return sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)


class HandSmoother:
    """Exponential smoothing of the palm position plus presence hysteresis:
    `present` turns on after `on_frames` detections and off after
    `off_frames` misses, so one dropped frame does not release the robot."""

    def __init__(self, alpha: float = 0.4, on_frames: int = 2, off_frames: int = 5):
        self.alpha = alpha
        self.on_frames, self.off_frames = on_frames, off_frames
        self.present = False
        self.x = self.y = 0.0
        self._hits = self._misses = 0

    def update(self, xy):
        """xy: (x, y) of the detected palm, or None. Returns (present, x, y)."""
        if xy is None:
            self._hits = 0
            self._misses += 1
            if self._misses >= self.off_frames:
                self.present = False
            return self.present, self.x, self.y
        self._misses = 0
        self._hits += 1
        if self._hits == 1 and not self.present:
            self.x, self.y = xy          # jump to a new hand instead of sliding in
        else:
            self.x += self.alpha * (xy[0] - self.x)
            self.y += self.alpha * (xy[1] - self.y)
        if self._hits >= self.on_frames:
            self.present = True
        return self.present, self.x, self.y


def load_table_half_size(scene_path: Path | None = None) -> float:
    """workspace.table_half_size from scene.yaml in this repo, if it can be read."""
    path = scene_path or Path(__file__).resolve().parents[1] / "ros2_ws/src/pa_core/pa_core/scene.yaml"
    try:
        import yaml
        return float(yaml.safe_load(path.read_text(encoding="utf-8"))["workspace"]["table_half_size"])
    except Exception:  # noqa: BLE001 - missing file / yaml: fall back to the default
        return DEFAULT_HALF_SIZE
