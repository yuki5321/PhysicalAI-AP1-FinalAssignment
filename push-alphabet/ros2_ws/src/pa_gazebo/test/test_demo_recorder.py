"""Tests for pa_gazebo.demo_overlay (the ROS-free part of demo_recorder / auto_hand)."""

import random

import cv2
import numpy as np
import pytest

from pa_core.config import load_scene
from pa_gazebo.demo_overlay import (PANEL_H, HudState, KnockConfig, LetterPose, ScriptedHand,
                                    compose_frame, draw_letters, even_size, format_clock,
                                    image_to_bgr, knock_path, letter_pixel_polys, progress_slots,
                                    progress_text, side_by_side, state_color)
from pa_perception.camera_model import CameraModel


@pytest.fixture
def cam():
    return CameraModel.from_scene()


def blank(h=640, w=640, v=90):
    return np.full((h, w, 3), v, np.uint8)


# ---------------------------------------------------------------- image I/O

def test_image_to_bgr_encodings():
    h, w = 4, 5
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = 200  # red
    step = w * 3 + 3   # row padding must be skipped
    data = np.zeros((h, step), np.uint8)
    data[:, : w * 3] = rgb.reshape(h, -1)
    out = image_to_bgr("rgb8", h, w, step, data.tobytes())
    assert out.shape == (h, w, 3) and out[0, 0].tolist() == [0, 0, 200]
    out = image_to_bgr("bgr8", h, w, step, memoryview(data.tobytes()))
    assert out[0, 0].tolist() == [200, 0, 0]
    import array
    mono = array.array("B", [7] * (h * w))
    out = image_to_bgr("mono8", h, w, w, mono)
    assert out.shape == (h, w, 3) and out[2, 3].tolist() == [7, 7, 7]
    out = image_to_bgr("bgr8", h, w, w * 3, list(data[:, : w * 3].ravel()))
    assert out.shape == (h, w, 3)
    with pytest.raises(ValueError):
        image_to_bgr("16UC1", h, w, w * 2, bytes(h * w * 2))


def test_even_size_and_side_by_side():
    assert even_size(blank(641, 639)).shape == (640, 638, 3)
    sbs = side_by_side(blank(100, 50), blank(80, 30), gap=4)
    assert sbs.shape == (100, 84, 3)
    assert sbs[90, 70].tolist() == [0, 0, 0]  # padding under the shorter image


# ---------------------------------------------------------------- text bits

def test_progress_word_fills_in_display_order():
    assert progress_text("YUKI", []) == "_ _ _ _"
    assert progress_text("YUKI", ["I"]) == "_ _ _ I"
    assert progress_text("YUKI", ["I", "U"]) == "_ U _ I"
    assert progress_text("YUKI", list("IUYK")) == "Y U K I"
    assert progress_slots("yuki", ["k"]) == ["_", "_", "K", "_"]
    # repeated letters fill left to right, unknown letters are ignored
    assert progress_text("AA", ["A"]) == "A _"
    assert progress_text("YUKI", ["Z"]) == "_ _ _ _"


def test_format_clock():
    assert format_clock(None) == "--:--.-"
    assert format_clock(float("nan")) == "--:--.-"
    assert format_clock(0.0) == "0:00.0"
    assert format_clock(75.25) == "1:15.2" or format_clock(75.25) == "1:15.3"
    assert format_clock(-3) == "0:00.0"


def test_state_colors_distinct():
    assert state_color("RECOVER") != state_color("WAIT_HAND") != state_color("IDLE")
    assert state_color("SOMETHING_NEW")  # unknown states still get a color


def test_hud_complete():
    hud = HudState(word="YUKI", active="K", done=list("IUY"))
    assert not hud.complete
    hud.done.append("K")
    hud.active = ""
    assert hud.complete


# ------------------------------------------------------------------ drawing

def test_compose_frame_size_and_panel(cam):
    img = blank()
    hud = HudState(status="RECOVER", active="U", done=["I"], pushes_total=7, pushes_letter=3,
                   last_push=(1, 45.0), pushing=True, hand_present=True, hand_xy=(0.02, -0.03),
                   sim_time=83.4, start_time=12.0, label="detector: hybrid", hand_source="scripted",
                   letters=[LetterPose("U", 0.01, 0.02, 0.3, 0.9)])
    out = compose_frame(img, hud, cam, z=0.012)
    assert out.shape == (640 + PANEL_H, 640, 3) and out.dtype == np.uint8
    assert np.any(out[640:] != out[640, 0])          # panel has text
    assert not np.shares_memory(out, img) and np.all(img == 90)   # input untouched
    # state badge in the top-left corner uses the RECOVER color
    assert out[12, 12].tolist() == list(state_color("RECOVER"))


def test_compose_frame_without_camera_and_odd_sizes():
    img = blank(480, 360)
    out = compose_frame(img, HudState(), None, panel_h=100)
    assert out.shape == (580, 360, 3)
    out = compose_frame(np.zeros((300, 400), np.uint8), HudState(active=""), None)
    assert out.shape == (300 + PANEL_H, 400, 3)


def test_complete_banner_drawn(cam):
    img = blank()
    base = compose_frame(img, HudState(active="K", done=list("IUY")), cam)
    done = compose_frame(img, HudState(active="", done=list("IUYK")), cam)
    mid = slice(280, 360)
    assert np.count_nonzero(np.any(done[mid] != base[mid], axis=2)) > 500


def test_letter_outline_drawn_at_projected_pose(cam):
    img = np.zeros((640, 640, 3), np.uint8)
    lp = LetterPose("I", 0.03, -0.02, 0.0, 1.0)
    draw_letters(img, [lp], cam, z=0.012, active="I")
    polys = letter_pixel_polys(cam, "I", lp.x, lp.y, lp.theta, 0.012)
    assert len(polys) == 1
    poly = polys[0]
    # the I is 16 x 80 mm; in pixels at the top surface:
    s = cam.px_per_m(0.012)
    xs, ys = poly[:, 0], poly[:, 1]
    assert xs.max() - xs.min() == pytest.approx(0.016 * s, abs=2)
    assert ys.max() - ys.min() == pytest.approx(0.080 * s, abs=2)
    green = (img[..., 1] > 150) & (img[..., 0] < 50) & (img[..., 2] < 50)
    for u, v in poly:  # every corner has green outline pixels next to it
        win = green[max(v - 3, 0): v + 4, max(u - 3, 0): u + 4]
        assert win.any()
    # center of the outline is right of / below the image center (x > 0, y < 0)
    c = poly.mean(axis=0)
    assert c[0] > cam.cx and c[1] > cam.cy
    # nothing drawn far away from the letter
    assert not green[:100, :100].any()


def test_all_letters_and_unknown_draw(cam):
    img = np.zeros((640, 640, 3), np.uint8)
    lps = [LetterPose(n, -0.05 + 0.03 * i, 0.0, 0.4 * i, 0.8) for i, n in enumerate(load_scene()["letters"])]
    lps.append(LetterPose("", 0.0, 0.07, 0.0, 0.0))
    lps.append(LetterPose("Q", 0.0, -0.07, 0.0, 0.0))  # not a known letter: skipped
    draw_letters(img, lps, cam, z=0.012, active="K")
    assert np.count_nonzero(img) > 1000


def test_frames_write_to_mp4(tmp_path, cam):
    path = str(tmp_path / "t.mp4")
    hud = HudState(status="IDLE", active="I", letters=[LetterPose("I", 0, 0, 0.1, 1.0)])
    frame = even_size(compose_frame(blank(), hud, cam))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (frame.shape[1], frame.shape[0]))
    if not vw.isOpened():
        pytest.skip("OpenCV build without an mp4 writer")
    for _ in range(10):
        vw.write(frame)
    vw.release()
    cap = cv2.VideoCapture(path)
    ok, back = cap.read()
    assert ok and back.shape == frame.shape


# ----------------------------------------------------------- scripted hand

def test_knock_path_stays_on_table_and_crosses_letter():
    cfg = KnockConfig()
    rng = random.Random(0)
    for cx, cy in [(0.0, 0.0), (0.05, 0.02), (-0.06, -0.05), (0.09, 0.09)]:
        p0, p1 = knock_path(cx, cy, rng, cfg)
        for p in (p0, p1):
            assert abs(p[0]) <= cfg.limit + 1e-12 and abs(p[1]) <= cfg.limit + 1e-12
        assert p0[1] == p1[1]                      # sideways sweep
        d = 1.0 if cx >= 0 else -1.0
        assert (p1[0] - p0[0]) * d > 0.03          # moves toward / through the letter
        assert d * p1[0] <= 0.02 + 1e-12           # never shoves far past the center line


def run_script(hand, t0, t1, dt=1 / 15):
    out = []
    t = t0
    while t < t1:
        out.append((t,) + hand.step(t))
        t += dt
    return out


def test_scripted_hand_knocks_once_during_recovery():
    cfg = KnockConfig(knocks_per_letter=1, after_pushes=2, start_delay_s=1.0)
    hand = ScriptedHand(cfg, random.Random(1))
    hand.on_active("I", 0.0)
    hand.on_status("RECOVER", 0.0)
    hand.on_letters([LetterPose("I", 0.02, 0.01, 0.5)], 0.0)
    assert not any(p for _, p, _, _ in run_script(hand, 0.0, 3.0))   # no pushes yet
    hand.on_push(3.0)
    hand.on_push_done(4.0)
    hand.on_push(4.1)
    assert not any(p for _, p, _, _ in run_script(hand, 4.1, 5.0))   # push in flight
    hand.on_push_done(5.0)
    trace = run_script(hand, 5.0, 12.0)
    present = [p for _, p, _, _ in trace]
    assert any(present)
    first = present.index(True)
    last = len(present) - 1 - present[::-1].index(True)
    assert all(present[first: last + 1])            # one contiguous sweep
    assert trace[first][0] - 5.0 >= cfg.settle_s - 1e-6
    xs = [x for _, p, x, _ in trace if p]
    assert xs[-1] > xs[0]                            # letter at x>=0: sweep toward +x
    assert hand.knocks == 1
    # no second knock for the same letter
    hand.on_push(12.0)
    hand.on_push_done(13.0)
    assert not any(p for _, p, _, _ in run_script(hand, 13.0, 20.0))
    # next letter: counters reset
    hand.on_active("U", 20.0)
    assert hand.knocks == 0 and hand.pushes_done == 0


def test_scripted_hand_idle_and_done():
    cfg = KnockConfig(start_delay_s=0.0, idle_s=2.0)
    hand = ScriptedHand(cfg, random.Random(2))
    hand.on_active("K", 0.0)
    hand.on_status("IDLE", 0.0)
    trace = run_script(hand, 0.0, 6.0)
    on = [t for t, p, _, _ in trace if p]
    assert on and on[0] >= 2.0 - 1e-6
    hand.on_active("", 10.0)       # all letters done: never knock
    hand.on_status("DONE", 10.0)
    assert not any(p for _, p, _, _ in run_script(hand, 10.0, 30.0))


def test_point_stroke_overlay(cam):
    hud = HudState(word="YUKI", status="RECOVER", active="I")
    hud.last_stroke = (-0.04, 0.0, 0.02, 0.0)
    hud.last_stroke_speed = 0.05
    hud.pushing = True
    img = blank()
    out = compose_frame(img, hud, cam, 0.012)
    assert out.shape == (640 + PANEL_H, 640, 3)
    # the arrow is drawn in orange along the stroke (image middle row band)
    a = cam.world_to_pixel([[-0.01, 0.0]], 0.012)[0]
    px = out[int(round(a[1])), int(round(a[0]))]
    assert px.tolist() != [90, 90, 90]
    assert out[: 640].astype(int).sum() != img.astype(int).sum()
    # without a camera model only text changes; no crash
    compose_frame(img, hud, None)
