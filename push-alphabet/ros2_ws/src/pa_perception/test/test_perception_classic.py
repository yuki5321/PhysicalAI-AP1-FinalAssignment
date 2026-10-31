import math

import cv2
import numpy as np
import pytest

from pa_core.letters import angle_error_deg
from pa_perception.camera_model import CameraModel
from pa_perception.classic import ClassicDetector, DetectorParams, draw_detections
from pa_perception.synth import fill_polys, random_pose, random_scene, render_scene

POS_TOL = 0.002   # m
ANG_TOL = 3.0     # deg (symmetry-aware: I is mod 180)


@pytest.fixture(scope="module")
def det():
    return ClassicDetector()


def _match(dets, name, x, y, th):
    near = [d for d in dets if math.hypot(d.x - x, d.y - y) < 0.02]
    assert len(near) == 1, f"{name} at ({x:.3f},{y:.3f}): got {[(d.letter, d.x, d.y) for d in dets]}"
    d = near[0]
    assert d.letter == name
    assert math.hypot(d.x - x, d.y - y) < POS_TOL
    assert abs(angle_error_deg(name, d.theta, th)) < ANG_TOL
    assert 0.0 < d.confidence <= 1.0
    return d


@pytest.mark.parametrize("name", ["I", "U", "Y", "K"])
@pytest.mark.parametrize("sides", [False, True])
def test_single_letter_random_poses(det, name, sides):
    rng = np.random.default_rng(10 * "IUYK".index(name) + int(sides))
    for _ in range(12):
        x, y, th = random_pose(rng)
        img = render_scene([(name, x, y, th)], sides=sides)
        dets = det.detect(img)
        assert len(dets) == 1
        _match(dets, name, x, y, th)


def test_symmetric_I_theta_range(det):
    for th in (-3.0, -1.6, 0.4, 2.0, 3.1):
        d = det.detect(render_scene([("I", 0.02, -0.03, th)]))[0]
        assert -math.pi / 2 - 1e-9 <= d.theta < math.pi / 2


def test_two_letters(det):
    rng = np.random.default_rng(7)
    for _ in range(15):
        sc = random_scene(rng, n_letters=2)
        dets = det.detect(render_scene(sc, sides=True))
        assert len(dets) == 2
        for name, x, y, th in sc:
            _match(dets, name, x, y, th)


def test_noise_and_blur(det):
    rng = np.random.default_rng(11)
    for i in range(16):
        sc = random_scene(rng, n_letters=1 + i % 2)
        img = render_scene(sc, noise_sigma=8.0, blur_ksize=5, rng=rng, table_value=70, top_value=215)
        dets = det.detect(img)
        assert len(dets) == len(sc)
        for name, x, y, th in sc:
            _match(dets, name, x, y, th)


def test_bright_paddles_and_floor_ignored(det):
    img = render_scene([("K", 0.01, 0.02, 0.7)], pusher_value=240, floor_value=250)
    dets = det.detect(img)
    assert [d.letter for d in dets] == ["K"]


def test_empty_table_and_outside_workspace(det):
    rng = np.random.default_rng(3)
    assert det.detect(render_scene([], noise_sigma=6.0, rng=rng)) == []
    assert det.detect(render_scene([("U", 0.0, 0.135, 0.0)])) == []


def test_fixed_threshold_and_gray_input():
    det = ClassicDetector(params=DetectorParams(threshold=150, sides="top"))
    img = cv2.cvtColor(render_scene([("Y", -0.04, 0.03, -2.0)], sides=False), cv2.COLOR_BGR2GRAY)
    dets = det.detect(img)
    _match(dets, "Y", -0.04, 0.03, -2.0)
    assert draw_detections(img, dets, det).shape == (*img.shape, 3)


# ------------------------------------------------- occluders and merged blobs
ORANGE = (20, 100, 220)   # BGR paddle
SKIN = (140, 170, 215)    # BGR hand


def _bar(img, p0, p1, w=0.008, z=0.03, color=ORANGE):
    """Paddle-like bar (top face at height z) drawn over img between world points p0, p1."""
    cam = CameraModel.from_scene()
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * w / 2
    fill_polys(img, [cam.world_to_pixel(np.array([p0 - n, p1 - n, p1 + n, p0 + n]), z)], color)
    return img


@pytest.mark.parametrize("segmenter", ["v2", "hsv"])
def test_letter_under_paddle(segmenter):
    # A bright paddle across the letter: with the legacy gray threshold it merges with the
    # letter blob (pose off by mm); with color segmentation the paddle pixels are unknown.
    det = ClassicDetector(params=DetectorParams(segmenter=segmenter))
    img = render_scene([("K", 0.02, -0.01, 0.6)], pusher_value=ORANGE)
    _bar(img, (-0.08, 0.03), (0.10, -0.02))
    d = _match(det.detect(img), "K", 0.02, -0.01, 0.6)
    assert 0.03 < d.occluded_frac < 0.4 and not d.split
    assert d.iou > 0.9


def test_letter_cut_in_two_by_paddle():
    # A wide occluder over the U's bottom bar leaves its two arms as separate blobs:
    # each alone looks like a partly hidden I. They are joined across the occluder
    # and fitted as one U.
    det = ClassicDetector()
    rng = np.random.default_rng(4)
    x0, y0, yb = -0.01, 0.02, -0.027  # yb: bar center line in the U body frame
    for th in (0.0, 1.0, -2.3):
        img = render_scene([("U", x0, y0, th)], pusher_value=ORANGE, noise_sigma=3.0, rng=rng)
        c, s = math.cos(th), math.sin(th)
        ends = [(x0 + c * bx - s * yb, y0 + s * bx + c * yb) for bx in (-0.08, 0.08)]
        _bar(img, *ends, w=0.02)
        assert any(b.joined_from for b in det.segment_frame(img).blobs)
        dets = det.detect(img)
        assert len(dets) == 1
        d = _match(dets, "U", x0, y0, th)
        assert 0.2 < d.occluded_frac < 0.5


def test_merged_letters_give_two_detections():
    sc = [("I", -0.02, 0.0, 0.0), ("Y", 0.022, 0.0, 0.0)]  # Y's left arm touches the I
    img = render_scene(sc)
    # Legacy threshold: one blob, rejected by the area / shape gates.
    assert ClassicDetector(params=DetectorParams(segmenter="legacy")).detect(img) == []
    # In-blob search of the matcher (the hsv segmenter does not split blobs).
    det = ClassicDetector(params=DetectorParams(segmenter="hsv"))
    dets = det.detect(img)
    assert len(dets) == 2 and all(d.split for d in dets)
    for name, x, y, th in sc:
        _match(dets, name, x, y, th)
    # Default (segmentation splits it, or the matcher does): same letters.
    dets = ClassicDetector().detect(img)
    assert len(dets) == 2
    for name, x, y, th in sc:
        _match(dets, name, x, y, th)


@pytest.mark.parametrize("segmenter", ["legacy", "hsv", "v2"])
def test_paddles_and_hand_alone_not_detected(segmenter):
    det = ClassicDetector(params=DetectorParams(segmenter=segmenter))
    img = render_scene([], pusher_value=ORANGE)
    _bar(img, (-0.08, 0.0), (0.08, 0.02))
    _bar(img, (0.0, -0.05), (0.03, 0.03), w=0.06, z=0.05, color=SKIN)  # hand-like slab
    assert det.detect(img) == []
    # A letter next to them is still found, and nothing else.
    _bar(img, (-0.09, -0.07), (0.02, -0.07))
    img2 = img.copy()
    k = render_scene([("Y", -0.06, 0.06, 2.0)], pusher_value=None)
    m = np.any(k != render_scene([]), axis=2)
    img2[m] = k[m]
    dets = det.detect(img2)
    assert [d.letter for d in dets] == ["Y"]


def test_legacy_segmenter_and_bad_segmenter():
    img = render_scene([("I", 0.03, 0.02, 0.3)])
    seg = ClassicDetector(params=DetectorParams(segmenter="legacy")).segment_frame(img)
    assert seg.source == "legacy" and seg.known is None and len(seg.blobs) == 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert ClassicDetector().segment_frame(gray).source == "legacy"  # gray: no color cues
    with pytest.raises(ValueError):
        ClassicDetector(params=DetectorParams(segmenter="bogus"))
