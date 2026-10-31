"""Pinhole model of the overhead camera (no ROS dependency).

The camera looks straight down at the table center from `camera.height`
(scene.yaml). Image right = world +x, image up = world +y (so image v grows
toward world -y). Pixel coordinates are OpenCV "index" coordinates: the center
of pixel (col=i, row=j) is at (u, v) = (i, j).

A point at height z above the table projects with scale f / (H - z) [px/m],
so the top surface of a letter (z = letter.thickness) appears slightly larger
than the table plane and its position is scaled away from the image center.
"""

import math

import numpy as np


class CameraModel:
    def __init__(self, height, width_px, height_px, hfov_deg, cx=None, cy=None):
        self.height = float(height)
        self.width_px = int(width_px)
        self.height_px = int(height_px)
        self.hfov_deg = float(hfov_deg)
        # Square pixels: f is the same in u and v (Gazebo derives vfov from the aspect).
        self.f = (self.width_px / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        self.cx = (self.width_px - 1) / 2.0 if cx is None else float(cx)
        self.cy = (self.height_px - 1) / 2.0 if cy is None else float(cy)

    @classmethod
    def from_scene(cls, scene=None):
        if scene is None:
            from pa_core.config import load_scene
            scene = load_scene()
        c = scene["camera"]
        return cls(c["height"], c["width"], c["height_px"], c["hfov_deg"])

    def px_per_m(self, z=0.0) -> float:
        """Image scale [px/m] for a horizontal plane at height z above the table."""
        return self.f / (self.height - z)

    def world_to_pixel(self, pts, z=0.0) -> np.ndarray:
        """(N,2) world xy at height z (scalar or (N,)) -> (N,2) pixel (u, v)."""
        pts = np.asarray(pts, dtype=float).reshape(-1, 2)
        s = self.f / (self.height - np.asarray(z, dtype=float))
        u = self.cx + s * pts[:, 0]
        v = self.cy - s * pts[:, 1]
        return np.stack([u, v], axis=1)

    def pixel_to_world(self, uv, z=0.0) -> np.ndarray:
        """(N,2) pixel (u, v) -> (N,2) world xy on the plane at height z."""
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        m = (self.height - np.asarray(z, dtype=float)) / self.f
        x = (uv[:, 0] - self.cx) * m
        y = -(uv[:, 1] - self.cy) * m
        return np.stack([x, y], axis=1)
