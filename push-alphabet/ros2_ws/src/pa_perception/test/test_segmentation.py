"""Tests of pa_perception.segmentation (docs/cnn.md "前段の分割")."""

import math
import time

import cv2
import numpy as np
import pytest

from pa_core.config import load_scene
from pa_core.letters import letter_area
from pa_perception.camera_model import CameraModel
from pa_perception.segmentation import Blob, SegParams, Segmentation, segment_letters
from pa_perception.synth import fill_polys, letter_pixel_polys, render_scene

ORANGE = (20, 120, 235)   # BGR paddle
SKIN = (140, 175, 220)    # BGR hand


@pytest.fixture(scope="module")
def env():
    scene = load_scene()
    cam = CameraModel.from_scene(scene)
    t = scene["letter"]["thickness"]
    return scene, cam, t, cam.px_per_m(t)


def _area_px(env, name):
    return letter_area(name) * env[3] ** 2


def _top_mask(env, name, x, y, th):
    scene, cam, t, _ = env
    tops, _ = letter_pixel_polys(cam, name, x, y, th, t, sides=False)
    m = np.zeros((cam.height_px, cam.width_px), np.uint8)
    return fill_polys(m, tops, 1, aa=False) > 0


def _blob_on(seg, mask):
    """Blob covering most of `mask` and the fraction it covers."""
    ov = np.bincount(seg.labels[mask], minlength=seg.labels.max() + 1)
    ov[0] = 0
    lab = int(np.argmax(ov))
    blob = next((b for b in seg.blobs if b.label == lab), None)
    return blob, ov[lab] / max(int(mask.sum()), 1)


def _box(env, cx, cy, heading, length, width, z):
    """Pixel polygon of a horizontal rectangle (paddle / finger) at height z."""
    scene, cam, _, _ = env
    d = np.array([math.cos(heading), math.sin(heading)])
    n = np.array([-d[1], d[0]])
    c = np.array([cx, cy])
    pts = np.array([c - d * length / 2 - n * width / 2, c + d * length / 2 - n * width / 2,
                    c + d * length / 2 + n * width / 2, c - d * length / 2 + n * width / 2])
    return cam.world_to_pixel(pts, z)


def _illum(img, cam, gain=1.0, gradient=0.0, direction=0.0, vignette=0.0):
    ys, xs = np.mgrid[0:cam.height_px, 0:cam.width_px].astype(np.float32)
    xn = (xs - cam.cx) / (cam.width_px / 2)
    yn = (ys - cam.cy) / (cam.height_px / 2)
    f = gain * (1 + 0.5 * gradient * (math.cos(direction) * xn + math.sin(direction) * yn))
    f *= 1 - 0.5 * vignette * (xn * xn + yn * yn)
    return np.clip(img.astype(np.float32) * f[..., None], 0, 255).astype(np.uint8)


def _noise(img, sigma, seed=0):
    r = np.random.default_rng(seed)
    return np.clip(img + r.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ contract
def test_output_contract(env):
    scene, cam, t, _ = env
    letters = [("U", -0.05, 0.03, 0.4), ("K", 0.05, -0.04, -1.0)]
    img = render_scene(letters, cam, scene, pusher_value=ORANGE, noise_sigma=3, rng=np.random.default_rng(1))
    seg = segment_letters(img)
    assert isinstance(seg, Segmentation)
    H, W = img.shape[:2]
    for m in (seg.fg, seg.occluder):
        assert m.shape == (H, W) and m.dtype == np.uint8
        assert set(np.unique(m)) <= {0, 255}
    assert seg.labels.shape == (H, W) and seg.labels.dtype == np.int32
    assert np.isfinite(seg.threshold)
    assert len(seg.blobs) == 2
    for b in seg.blobs:
        assert isinstance(b, Blob)
        m = seg.labels == b.label
        assert int(m.sum()) == b.area_px
        vs, us = np.nonzero(m)
        assert abs(us.mean() - b.centroid_px[0]) < 1e-6 and abs(vs.mean() - b.centroid_px[1]) < 1e-6
        assert b.bbox == (us.min(), vs.min(), us.max() - us.min() + 1, vs.max() - vs.min() + 1)
        assert b.split_from is None and not b.touches_occluder
        assert np.all(seg.fg[m] == 255)
    # every labelled pixel belongs to a listed blob
    assert set(np.unique(seg.labels)) == {0} | {b.label for b in seg.blobs}
    # the paddles at rest are occluders, not foreground
    assert np.count_nonzero(seg.occluder) > 3000
    assert np.count_nonzero(seg.fg[seg.occluder > 0]) == 0


def test_explicit_scene_and_cam(env):
    scene, cam, _, _ = env
    img = render_scene([("Y", 0.0, 0.0, 0.3)], cam, scene)
    a = segment_letters(img, SegParams(), scene=scene, cam=cam)
    b = segment_letters(img)
    assert len(a.blobs) == len(b.blobs) == 1
    assert a.blobs[0].area_px == b.blobs[0].area_px


# ------------------------------------------------------------------ cases
def test_empty_table_no_blobs(env):
    scene, cam, _, _ = env
    for k, (table, grad, vig) in enumerate([(55, 0.0, 0.0), (95, 0.3, 0.25), (20, 0.2, 0.0)]):
        img = render_scene([], cam, scene, table_value=table, pusher_value=ORANGE)
        img = _noise(_illum(img, cam, 1.0, grad, 0.7, vig), 5, seed=k)
        seg = segment_letters(img)
        assert seg.blobs == [], (table, [b.area_px for b in seg.blobs])


def test_dim_letter_under_vignette_is_found(env):
    """A dim letter in a dark corner next to a bright letter in the bright center."""
    scene, cam, _, _ = env
    letters = [("I", 0.0, 0.0, 0.2), ("Y", 0.095, 0.09, -0.6)]
    img = render_scene(letters, cam, scene, table_value=95, top_value=170, side_value=120)
    # The corner is ~40 % darker than the center: a single global threshold (Otsu over
    # the table) keeps only about a third of the dim letter.
    img = _noise(_illum(img, cam, 1.15, gradient=0.4, direction=2.4, vignette=0.8), 4)
    seg = segment_letters(img)
    for name, x, y, th in letters:
        blob, cover = _blob_on(seg, _top_mask(env, name, x, y, th))
        assert blob is not None and cover > 0.85, name
        assert 0.7 < blob.area_px / _area_px(env, name) < 1.5, (name, blob.area_px)
    assert len(seg.blobs) == 2


def test_letter_touching_paddle(env):
    scene, cam, _, _ = env
    x, y, th = 0.03, -0.02, 0.3
    img = render_scene([("K", x, y, th)], cam, scene, table_value=50)
    # a paddle lying on the end of the letter
    paddle = _box(env, x + 0.05, y - 0.03, 2.5, 0.16, 0.008, 0.03)
    fill_polys(img, [paddle], ORANGE)
    img = _noise(cv2.GaussianBlur(img, (0, 0), 0.8), 3)
    seg = segment_letters(img)
    pm = fill_polys(np.zeros(img.shape[:2], np.uint8), [paddle], 1, aa=False) > 0
    inner = cv2.erode(pm.view(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.mean(seg.occluder[inner] > 0) > 0.98
    assert np.count_nonzero(seg.fg[pm]) == 0
    assert len(seg.blobs) == 1
    b = seg.blobs[0]
    assert b.touches_occluder
    A = _area_px(env, "K")
    assert 0.75 < b.area_px / A < 1.3


def test_hand_next_to_letter(env):
    scene, cam, _, _ = env
    x, y, th = -0.04, 0.02, 1.0
    img = render_scene([("U", x, y, th)], cam, scene, table_value=40)
    palm = _box(env, x - 0.07, y - 0.02, 0.2, 0.09, 0.08, 0.06)
    finger = _box(env, x - 0.02, y - 0.01, 0.3, 0.07, 0.018, 0.06)
    fill_polys(img, [palm, finger], SKIN)
    img = _noise(img, 3)
    seg = segment_letters(img)
    hand = fill_polys(np.zeros(img.shape[:2], np.uint8), [palm, finger], 1, aa=False) > 0
    assert np.count_nonzero(seg.fg[hand]) < 0.01 * hand.sum()
    blob, cover = _blob_on(seg, _top_mask(env, "U", x, y, th) & ~hand)
    assert blob is not None and cover > 0.9 and blob.touches_occluder
    assert blob.area_px < 1.2 * _area_px(env, "U")


def test_paddle_across_letter_is_one_blob(env):
    scene, cam, _, _ = env
    x, y, th = -0.02, 0.01, 0.0
    img = render_scene([("I", x, y, th)], cam, scene, table_value=60)
    paddle = _box(env, x, y + 0.005, 0.2, 0.2, 0.008, 0.03)   # crosses the bar
    fill_polys(img, [paddle], ORANGE)
    seg = segment_letters(img)
    assert len(seg.blobs) == 1
    b = seg.blobs[0]
    assert b.touches_occluder and len(b.merged_from) == 2
    assert np.count_nonzero(seg.labels == b.label) == b.area_px
    assert 0.6 < b.area_px / _area_px(env, "I") < 1.1


def test_separate_letters_on_one_paddle_are_not_merged(env):
    scene, cam, _, _ = env
    letters = [("I", -0.06, 0.0, 0.0), ("Y", 0.06, 0.0, 0.0)]
    img = render_scene(letters, cam, scene, table_value=60)
    fill_polys(img, [_box(env, 0.0, 0.0, 0.0, 0.2, 0.008, 0.03)], ORANGE)
    seg = segment_letters(img)
    assert len(seg.blobs) >= 2
    for name, x, y, th in letters:
        blob, cover = _blob_on(seg, _top_mask(env, name, x, y, th))
        assert blob.area_px < 1.15 * _area_px(env, name)


def _touching_pair(env, a, b, th_a, th_b, direction):
    """Pose of letter b slid along `direction` until its top face just touches letter a."""
    pa = (a, 0.0, 0.0, th_a)
    ma = _top_mask(env, *pa)
    for d in np.arange(0.12, 0.0, -0.0005):
        x, y = d * math.cos(direction), d * math.sin(direction)
        if np.any(ma & _top_mask(env, b, x, y, th_b)):
            return pa, (b, x, y, th_b)
    raise AssertionError("no contact")


@pytest.mark.parametrize("a,b,th_a,th_b,direction", [
    ("I", "I", 0.0, 0.0, math.radians(80)),       # end to end, slightly offset
    ("U", "K", 0.3, -0.5, math.radians(20)),
    ("Y", "I", 0.0, 0.6, math.radians(-60)),
])
def test_two_touching_letters_are_split(env, a, b, th_a, th_b, direction):
    scene, cam, _, _ = env
    la, lb = _touching_pair(env, a, b, th_a, th_b, direction)
    img = render_scene([la, lb], cam, scene, table_value=50, blur_ksize=3)
    img = _noise(img, 2)
    seg = segment_letters(img)
    for name, x, y, th in (la, lb):
        blob, cover = _blob_on(seg, _top_mask(env, name, x, y, th))
        assert blob is not None and cover > 0.8, name
        assert 0.6 < blob.area_px / _area_px(env, name) < 1.5, (name, blob.area_px / _area_px(env, name))
    labs = {_blob_on(seg, _top_mask(env, *l))[0].label for l in (la, lb)}
    assert len(labs) == 2
    assert all(b.split_from is not None for b in seg.blobs)


def test_single_letters_are_not_split(env):
    scene, cam, _, _ = env
    rng = np.random.default_rng(3)
    for name in ["I", "U", "Y", "K"]:
        for _ in range(6):
            x, y = rng.uniform(-0.1, 0.1, 2)
            th = rng.uniform(-math.pi, math.pi)
            img = render_scene([(name, x, y, th)], cam, scene, blur_ksize=3)
            seg = segment_letters(_noise(img, 3))
            assert len(seg.blobs) == 1 and seg.blobs[0].split_from is None, (name, x, y, th)


def test_fixed_threshold(env):
    scene, cam, _, _ = env
    img = render_scene([("U", 0.0, 0.0, 0.0)], cam, scene)
    seg = segment_letters(img, SegParams(threshold=150))
    assert seg.threshold == 150 and len(seg.blobs) == 1


def test_timing(env):
    scene, cam, _, _ = env
    letters = [("U", -0.05, 0.04, 0.4), ("K", 0.05, -0.04, -1.0), ("Y", 0.06, 0.06, 2.0)]
    img = render_scene(letters, cam, scene, pusher_value=ORANGE, noise_sigma=4, blur_ksize=3,
                       rng=np.random.default_rng(0))
    segment_letters(img)
    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        segment_letters(img)
    dt = (time.perf_counter() - t0) / n
    assert dt < 0.05, f"{dt * 1000:.1f} ms per frame"   # target < 8 ms; loose for slow CI
