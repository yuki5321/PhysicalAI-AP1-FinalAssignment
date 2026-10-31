import math

import numpy as np
import pytest

from pa_core.letters import angle_error_deg
from pa_perception.camera_model import CameraModel
from pa_perception.cnn_crop import (CROP_SIZE, LETTERS, blob_centers, crop_center_world, decode,
                                    encode_target, extract_crop, letter_center_pixel)
from pa_perception.synth import render_scene


@pytest.fixture(scope="module")
def cam():
    return CameraModel.from_scene()


def test_letters_order():
    assert LETTERS == ("I", "U", "Y", "K")


def _logits(cls):
    z = np.full(4, -3.0)
    z[cls] = 3.0
    return z


@pytest.mark.parametrize("letter", ["I", "U", "Y", "K"])
def test_encode_decode_round_trip(cam, letter):
    rng = np.random.default_rng(LETTERS.index(letter))
    for _ in range(50):
        x, y = rng.uniform(-0.1, 0.1, 2)
        th = rng.uniform(-math.pi, math.pi)
        u, v = letter_center_pixel(x, y, cam)
        u += rng.uniform(-8, 8)
        v += rng.uniform(-8, 8)
        cw = crop_center_world(u, v, cam)
        cls, ang, off = encode_target(letter, x, y, th, cw)
        assert LETTERS[cls] == letter
        assert abs(math.hypot(*ang) - 1.0) < 1e-9
        assert max(abs(o) for o in off) < 0.5  # 8 px is ~4 mm
        # Unnormalized angle vector must decode the same.
        name, xd, yd, thd, conf = decode(_logits(cls), 3.0 * np.array(ang), off, cw)
        assert name == letter
        assert conf > 0.9
        assert math.hypot(xd - x, yd - y) < 1e-9
        assert abs(angle_error_deg(letter, thd, th)) < 1e-6
        sym = 180.0 if letter == "I" else 360.0
        assert -sym / 2 - 1e-9 < math.degrees(thd) <= sym / 2 + 1e-9


def test_i_symmetry_same_target(cam):
    cw = (0.0, 0.0)
    _, a1, _ = encode_target("I", 0.0, 0.0, 0.3, cw)
    _, a2, _ = encode_target("I", 0.0, 0.0, 0.3 + math.pi, cw)
    assert np.allclose(a1, a2)
    _, b1, _ = encode_target("U", 0.0, 0.0, 0.3, cw)
    _, b2, _ = encode_target("U", 0.0, 0.0, 0.3 + math.pi, cw)
    assert np.allclose(b1, -np.array(b2))
    # I decodes into (-90, 90] deg.
    _, _, _, th, _ = decode(_logits(0), (math.cos(2 * 2.0), math.sin(2 * 2.0)), (0, 0), cw)
    assert abs(th - (2.0 - math.pi)) < 1e-9


def test_crop_center_world_inverts_projection(cam):
    u, v = letter_center_pixel(0.05, -0.03, cam)
    assert np.allclose(crop_center_world(u, v, cam), (0.05, -0.03))


def test_crop_shape_and_values():
    img = np.full((640, 640), 255, np.uint8)
    c = extract_crop(img, 320, 320)
    assert c.shape == (1, CROP_SIZE, CROP_SIZE) and c.dtype == np.float32
    assert np.allclose(c, 1.0)


@pytest.mark.parametrize("u,v", [(0, 0), (639, 0), (0, 639), (639, 639), (-50, 320)])
def test_crop_padding_at_borders(u, v):
    img = np.full((640, 640), 200, np.uint8)
    c = extract_crop(img, u, v)[0]
    inside = c > 0.5
    # Expected inside-image fraction of the 192 px window.
    def frac(p):
        lo, hi = p - 96, p + 96
        return max(0, min(hi, 640) - max(lo, 0)) / 192
    expect = frac(u) * frac(v)
    assert abs(inside.mean() - expect) < 0.02
    assert np.allclose(c[inside & (c > 0.75)], 200 / 255, atol=1e-5)
    assert c.min() == 0.0
    # Corner pixel of the window beyond the image is zero padding.
    if u <= 0:
        assert np.all(c[:, 0] == 0)
    if v <= 0:
        assert np.all(c[0, :] == 0)


def test_crop_is_plain_2x_downscale_when_aligned():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (640, 640), dtype=np.uint8)
    u, v = 300, 250  # window = pixels [u-96, u+96)  -> centered at u - 0.5
    c = extract_crop(img, u - 0.5, v - 0.5)[0]
    win = img[v - 96:v + 96, u - 96:u + 96].astype(np.float32) / 255
    ref = win.reshape(96, 2, 96, 2).mean(axis=(1, 3))
    assert np.allclose(c, ref, atol=1e-4)


def test_crop_centered_on_letter(cam):
    x, y, th = 0.04, -0.03, 0.4
    img = render_scene([("U", x, y, th)], sides=False)
    u, v = letter_center_pixel(x, y, cam)
    c = extract_crop(img, u, v)[0]
    w = np.clip(c - c.min(), 0, None)
    w = np.where(w > 0.5, w, 0)
    jj, ii = np.mgrid[0:CROP_SIZE, 0:CROP_SIZE]
    ci, cj = (w * ii).sum() / w.sum(), (w * jj).sum() / w.sum()
    mid = (CROP_SIZE - 1) / 2
    assert abs(ci - mid) < 0.5 and abs(cj - mid) < 0.5


def test_blob_centers(cam):
    letters = [("I", -0.05, 0.05, 0.2), ("K", 0.05, -0.04, 1.0)]
    img = render_scene(letters)
    blobs = blob_centers(img)
    assert len(blobs) == 2
    for name, x, y, _ in letters:
        u, v = letter_center_pixel(x, y, cam)
        d = min(math.hypot(b[0] - u, b[1] - v) for b in blobs)
        assert d < 8.0
        assert all(b[2] > 1000 for b in blobs)


def test_blob_centers_paddle_not_merged(cam):
    from pa_perception.classic import DetectorParams
    from pa_perception.synth import fill_polys
    x, y = 0.03, -0.02
    img = render_scene([("Y", x, y, 0.4)], pusher_value=(20, 100, 220))
    # paddle touching the letter's side
    poly = np.array([[x - 0.1, y - 0.035], [x + 0.1, y - 0.035], [x + 0.1, y - 0.027], [x - 0.1, y - 0.027]])
    fill_polys(img, [cam.world_to_pixel(poly, 0.03)], (20, 100, 220))
    u, v = letter_center_pixel(x, y, cam)
    blobs = blob_centers(img)
    assert len(blobs) == 1
    dist = math.hypot(blobs[0][0] - u, blobs[0][1] - v)
    assert dist < 20.0  # centroid of the visible part (the paddle hides a strip)
    # legacy gray threshold: the paddles are foreground, the letter blob includes one
    legacy = blob_centers(img, DetectorParams(segmenter="legacy"), filter_area=False)
    big = max(legacy, key=lambda b: b[2])
    assert big[2] > 1.5 * blobs[0][2] and math.hypot(big[0] - u, big[1] - v) > 2 * dist
