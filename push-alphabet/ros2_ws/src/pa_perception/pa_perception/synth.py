"""Synthetic overhead-camera images of letters at known poses (no ROS dependency).

Used by the tests and as a cheap data source for the CNN later. Each letter
rectangle is a box of height `letter.thickness`; its image silhouette is the
convex hull of its 8 projected corners (side walls are visible toward the image
center), the top face is drawn on top of it.
"""

import math

import cv2
import numpy as np

from pa_core.config import load_scene
from pa_core.letters import letter_names, letter_polygons
from pa_perception.camera_model import CameraModel

SHIFT = 8  # fixed-point bits for sub-pixel polygon drawing


def _fixed(p: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(p) * (1 << SHIFT)).astype(np.int32)


def fill_polys(img, polys, value, offset=(0.0, 0.0), aa=True):
    """Fill convex pixel polygons into img (in place). offset is subtracted (crop origin)."""
    off = np.asarray(offset, dtype=float)
    lt = cv2.LINE_AA if aa else cv2.LINE_8
    for p in polys:
        cv2.fillConvexPoly(img, _fixed(p - off), value, lineType=lt, shift=SHIFT)
    return img


def box_pixel_polys(cam: CameraModel, corners_xy, z_top, z_bottom=0.0, sides=True):
    """Top-face polygon and (optionally) full silhouette hull of a vertical box."""
    top = cam.world_to_pixel(corners_xy, z_top)
    if not sides:
        return top, None
    bot = cam.world_to_pixel(corners_xy, z_bottom)
    hull = cv2.convexHull(np.vstack([top, bot]).astype(np.float32)).reshape(-1, 2).astype(float)
    return top, hull


def letter_pixel_polys(cam: CameraModel, name, x, y, theta, thickness, sides=True):
    """(top polygons, silhouette hulls or []) of a letter at a world pose, in pixels."""
    tops, hulls = [], []
    for corners in letter_polygons(name, x, y, theta):
        top, hull = box_pixel_polys(cam, corners, thickness, 0.0, sides)
        tops.append(top)
        if hull is not None:
            hulls.append(hull)
    return tops, hulls


def _color(v):
    return tuple(int(c) for c in v) if isinstance(v, (tuple, list)) else (int(v),) * 3


def render_scene(letters, cam=None, scene=None, *, table_value=55, floor_value=140,
                 top_value=235, side_value=175, sides=True, pusher_value=None,
                 noise_sigma=0.0, blur_ksize=0, rng=None):
    """BGR uint8 image of `letters` = [(name, x, y, theta_rad), ...].

    Values may be a gray level or a BGR tuple. `pusher_value` draws the four
    paddles at rest (None: not drawn). Noise is Gaussian (sigma in gray levels),
    blur is a Gaussian kernel of size `blur_ksize` (odd, 0: none).
    """
    scene = scene or load_scene()
    cam = cam or CameraModel.from_scene(scene)
    t = scene["letter"]["thickness"]
    img = np.zeros((cam.height_px, cam.width_px, 3), np.uint8)
    img[:] = _color(floor_value)

    h = scene["workspace"]["table_half_size"]
    table = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    fill_polys(img, [cam.world_to_pixel(table, 0.0)], _color(table_value))

    if pusher_value is not None:
        p = scene["pushers"]
        L, w, zt = p["arm_length"], p["arm_thickness"], p["arm_height"]
        for pu in p["list"]:
            a = math.radians(pu["rest_heading"])
            d = np.array([math.cos(a), math.sin(a)])
            n = np.array([-d[1], d[0]])
            c0 = np.asarray(pu["pivot"], dtype=float)
            corners = np.array([c0 - n * w / 2, c0 + d * L - n * w / 2,
                                c0 + d * L + n * w / 2, c0 + n * w / 2])
            top, _ = box_pixel_polys(cam, corners, zt, 0.0, sides=False)
            fill_polys(img, [top], _color(pusher_value))

    all_tops, all_hulls = [], []
    for name, x, y, th in letters:
        tops, hulls = letter_pixel_polys(cam, name, x, y, th, t, sides)
        all_tops += tops
        all_hulls += hulls
    if sides:
        fill_polys(img, all_hulls, _color(side_value))
    fill_polys(img, all_tops, _color(top_value))

    if blur_ksize and blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)
    if noise_sigma > 0:
        rng = rng if rng is not None else np.random.default_rng()
        noisy = img.astype(np.float32) + rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
        img = np.clip(noisy, 0, 255).astype(np.uint8)
    return img


def random_pose(rng, half_size=None, scene=None):
    """Uniform random (x, y, theta) with the letter centroid inside +-half_size."""
    scene = scene or load_scene()
    if half_size is None:
        half_size = scene["workspace"]["half_size"]
    x, y = rng.uniform(-half_size, half_size, 2)
    return float(x), float(y), float(rng.uniform(-math.pi, math.pi))


def random_scene(rng, n_letters=1, names=None, half_size=None, min_dist=0.115, scene=None):
    """Non-overlapping random letters (rejection sampling on centroid distance)."""
    names = list(names or letter_names())
    out = []
    for _ in range(1000):
        if len(out) == n_letters:
            break
        x, y, th = random_pose(rng, half_size, scene)
        if all(math.hypot(x - o[1], y - o[2]) >= min_dist for o in out):
            out.append((names[rng.integers(len(names))], x, y, th))
    return out
