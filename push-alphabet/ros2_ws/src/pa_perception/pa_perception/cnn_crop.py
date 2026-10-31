"""Crop extraction and target encoding/decoding for the letter CNN (docs/cnn.md).

numpy / OpenCV only (no torch): this module runs in the Docker image
(Python 3.10, numpy<1.25) next to onnxruntime.

Crop: a CROP_WINDOW x CROP_WINDOW px window centered on (u, v) (pixels outside
the image are 0), shrunk 2x to CROP_SIZE x CROP_SIZE, float32 in 0..1, shape
[1, CROP_SIZE, CROP_SIZE]. The window center is exactly (u, v) (sub-pixel): the
image is 2x2 box-averaged and sampled bilinearly at the centers of the 2x2 cells,
which for integer-aligned windows is the plain 2x downscale.

Targets (per crop):
  cls    = LETTERS.index(letter)
  angle  = (cos phi, sin phi), phi = theta * 360 / sym  (I: phi = 2 theta)
  offset = (x, y) - center_world, in cm; center_world is (u, v) mapped to the
           world plane at the letter top height (crop_center_world).
"""

import math

import cv2
import numpy as np

from pa_core.config import load_scene

CROP_WINDOW = 192   # px, window side in the full-resolution image
CROP_SIZE = 96      # px, network input side (CROP_WINDOW / 2)
JITTER_PX = 8       # training-time crop-center jitter (+-px)
LETTERS = tuple(load_scene()["letters"].keys())   # I, U, Y, K (logits order)
SYMMETRY = {n: float(load_scene().get("symmetry", {}).get(n, 360.0)) for n in LETTERS}
LETTER_TOP_Z = float(load_scene()["letter"]["thickness"])

_HALF = CROP_WINDOW / 2.0
_PAD = 2  # extra source pixels around the window (bilinear support)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def extract_crop(gray, u, v):
    """float32 [1, CROP_SIZE, CROP_SIZE] crop of a uint8 gray image around pixel (u, v)."""
    gray = to_gray(gray)
    H, W = gray.shape
    # Source region (integer, zero padded) that covers the window plus a margin.
    x0 = int(math.floor(u - _HALF)) - _PAD
    y0 = int(math.floor(v - _HALF)) - _PAD
    n = CROP_WINDOW + 2 * _PAD + 2
    reg = np.zeros((n, n), np.float32)
    a0, b0 = max(x0, 0), max(y0, 0)
    a1, b1 = min(x0 + n, W), min(y0 + n, H)
    if a1 > a0 and b1 > b0:
        reg[b0 - y0:b1 - y0, a0 - x0:a1 - x0] = gray[b0:b1, a0:a1]
    # 2x2 box average: box[j, i] is centered at region coords (i + 0.5, j + 0.5).
    box = 0.25 * (reg[:-1, :-1] + reg[1:, :-1] + reg[:-1, 1:] + reg[1:, 1:])
    # Output pixel i is centered at image coord u + 2 (i - (CROP_SIZE - 1) / 2).
    c = (CROP_SIZE - 1) / 2.0
    M = np.array([[2.0, 0.0, u - 2.0 * c - x0 - 0.5],
                  [0.0, 2.0, v - 2.0 * c - y0 - 0.5]], np.float64)
    out = cv2.warpAffine(box, M, (CROP_SIZE, CROP_SIZE),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return (out * (1.0 / 255.0)).astype(np.float32)[None]


# ------------------------------------------------------------------ blobs
_DETECTORS = {}


def _detector(detector_params=None):
    from pa_perception.classic import ClassicDetector, DetectorParams
    p = detector_params or DetectorParams()
    key = repr(p)
    if key not in _DETECTORS:
        _DETECTORS[key] = ClassicDetector(params=p)
    return _DETECTORS[key]


def blob_centers(img_bgr, detector_params=None, detector=None, filter_area=True):
    """[(u, v, area_px), ...] of foreground blobs (ClassicDetector.segment_frame).

    The image must be BGR (the "v2" / "hsv" segmenters use color to keep paddles
    and hands out of the letter blobs; a gray image gets the legacy threshold).
    With `filter_area`, only blobs whose area is plausible for some letter
    (the detector's area_min_ratio .. area_max_ratio, lower bound relaxed next
    to an occluder) are kept.
    """
    det = detector or _detector(detector_params)
    seg = det.segment_frame(img_bgr)
    out = []
    for b in seg.blobs:
        if filter_area:
            amin, amax = det.area_bounds(b)
            if not amin <= b.area_px <= amax:
                continue
        out.append((float(b.centroid_px[0]), float(b.centroid_px[1]), int(b.area_px)))
    return out


# --------------------------------------------------------------- geometry
def crop_center_world(u, v, cam, z=LETTER_TOP_Z):
    """World (x, y) of pixel (u, v) on the plane of the letter top surface."""
    x, y = cam.pixel_to_world([[u, v]], z)[0]
    return float(x), float(y)


def letter_center_pixel(x, y, cam, z=LETTER_TOP_Z):
    """Pixel (u, v) of a letter center (x, y) at the letter top height."""
    u, v = cam.world_to_pixel([[x, y]], z)[0]
    return float(u), float(v)


def encode_target(letter, x, y, theta, center_world):
    """(cls, (cos phi, sin phi), (dx_cm, dy_cm)) for a letter pose and crop center."""
    cls = LETTERS.index(letter)
    phi = theta * 360.0 / SYMMETRY[letter]
    dx = (x - center_world[0]) * 100.0
    dy = (y - center_world[1]) * 100.0
    return cls, (math.cos(phi), math.sin(phi)), (dx, dy)


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def decode(logits, angle, offset, center_world):
    """(letter, x, y, theta, conf) from one crop's network outputs.

    logits [4], angle [2] = (cos phi, sin phi) (need not be normalized),
    offset [2] in cm. theta is in (-sym/2, sym/2] (rad).
    """
    p = softmax(np.asarray(logits, dtype=np.float64).reshape(-1))
    k = int(np.argmax(p))
    letter = LETTERS[k]
    c, s = (float(a) for a in np.asarray(angle).reshape(-1)[:2])
    theta = math.atan2(s, c) * SYMMETRY[letter] / 360.0
    off = np.asarray(offset, dtype=np.float64).reshape(-1)
    x = center_world[0] + off[0] / 100.0
    y = center_world[1] + off[1] / 100.0
    return letter, float(x), float(y), float(theta), float(p[k])


def decode_batch(logits, angle, offset, centers_world):
    """decode() over the rows of [N, 4], [N, 2], [N, 2] outputs."""
    return [decode(l, a, o, c) for l, a, o, c in zip(logits, angle, offset, centers_world)]
