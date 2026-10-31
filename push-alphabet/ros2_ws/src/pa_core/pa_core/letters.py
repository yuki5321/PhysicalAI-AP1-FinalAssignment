"""Letter shapes (unions of rectangles), poses and Gazebo SDF generation."""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np

from pa_core.config import load_scene


@dataclass(frozen=True)
class Rect:
    cx: float
    cy: float
    w: float
    h: float
    angle: float  # rad

    def corners(self) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        hw, hh = self.w / 2, self.h / 2
        local = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]])
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.cx, self.cy])

    def contains(self, pts: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        d = pts - np.array([self.cx, self.cy])
        u = d[:, 0] * c + d[:, 1] * s
        v = -d[:, 0] * s + d[:, 1] * c
        return (np.abs(u) <= self.w / 2) & (np.abs(v) <= self.h / 2)


def _union_mask(rects, res=0.00025, span=0.06):
    # Cell centers, symmetric about 0 (so a centered shape has its mean at 0).
    g = np.arange(-span, span, res) + res / 2
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    inside = np.zeros(len(pts), dtype=bool)
    for r in rects:
        inside |= r.contains(pts)
    return pts, inside, res


@lru_cache(maxsize=None)
def letter_rects(name: str) -> tuple:
    """Rectangles of `name` in its body frame (origin at the area centroid)."""
    raw = [Rect(cx, cy, w, h, math.radians(a)) for cx, cy, w, h, a in load_scene()["letters"][name]]
    pts, inside, _ = _union_mask(raw)
    c = pts[inside].mean(axis=0)
    return tuple(Rect(r.cx - c[0], r.cy - c[1], r.w, r.h, r.angle) for r in raw)


@lru_cache(maxsize=None)
def letter_area(name: str) -> float:
    _, inside, res = _union_mask(letter_rects(name))
    return float(inside.sum()) * res * res


def letter_names() -> list:
    return list(load_scene()["letters"].keys())


def transform(points: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return points @ np.array([[c, -s], [s, c]]).T + np.array([x, y])


def letter_polygons(name: str, x=0.0, y=0.0, theta=0.0) -> list:
    """Corner arrays (4x2, meters) of each rectangle at the given pose."""
    return [transform(r.corners(), x, y, theta) for r in letter_rects(name)]


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def angle_error_deg(name: str, theta: float, target: float = 0.0) -> float:
    """Signed angle error in degrees, respecting the letter's rotational symmetry."""
    sym = load_scene()["symmetry"].get(name, 360.0)
    e = (math.degrees(theta - target) + sym / 2) % sym - sym / 2
    return e


def pose_error(name: str, x: float, y: float, theta: float):
    """(position error [m], |angle error| [deg]) w.r.t. the target pose."""
    t = load_scene()["target"]
    return math.hypot(x - t["x"], y - t["y"]), abs(angle_error_deg(name, theta, math.radians(t["theta"])))


def is_aligned(name, x, y, theta) -> bool:
    t = load_scene()["target"]
    dp, da = pose_error(name, x, y, theta)
    return dp < t["pos_tol"] and da < t["ang_tol"]


def is_disturbed(name, x, y, theta) -> bool:
    t = load_scene()["target"]
    dp, da = pose_error(name, x, y, theta)
    return dp > t["disturb_pos"] or da > t["disturb_ang"]


def letter_sdf(name: str, model_name: str | None = None, color=(0.95, 0.95, 0.95)) -> str:
    """SDF of a letter block: one link, one box collision/visual per rectangle."""
    sc = load_scene()["letter"]
    t, m, mu = sc["thickness"], sc["mass"], sc["friction"]
    area = letter_area(name)
    # Inertia of a thin plate approximated from its rectangles.
    izz = 0.0
    for r in letter_rects(name):
        mi = m * (r.w * r.h) / max(area, 1e-9)
        izz += mi * ((r.w ** 2 + r.h ** 2) / 12 + r.cx ** 2 + r.cy ** 2)
    ixx = iyy = izz / 2
    rgba = f"{color[0]} {color[1]} {color[2]} 1"
    parts = []
    for i, r in enumerate(letter_rects(name)):
        pose = f"{r.cx:.5f} {r.cy:.5f} {t / 2:.5f} 0 0 {r.angle:.5f}"
        size = f"{r.w:.5f} {r.h:.5f} {t:.5f}"
        parts.append(f"""
      <collision name="c{i}">
        <pose>{pose}</pose>
        <geometry><box><size>{size}</size></box></geometry>
        <surface>
          <friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction>
          <contact><ode><kp>1e5</kp><kd>10</kd><max_vel>0.1</max_vel><min_depth>0.0005</min_depth></ode></contact>
        </surface>
      </collision>
      <visual name="v{i}">
        <pose>{pose}</pose>
        <geometry><box><size>{size}</size></box></geometry>
        <material><ambient>{rgba}</ambient><diffuse>{rgba}</diffuse><emissive>0.3 0.3 0.3 1</emissive></material>
      </visual>""")
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name or 'letter_' + name}">
    <link name="body">
      <inertial>
        <pose>0 0 {t / 2:.5f} 0 0 0</pose>
        <mass>{m}</mass>
        <inertia><ixx>{ixx:.3e}</ixx><iyy>{iyy:.3e}</iyy><izz>{izz:.3e}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>{''.join(parts)}
    </link>
  </model>
</sdf>
"""
