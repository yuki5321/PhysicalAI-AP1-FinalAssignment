"""Fast tests of tools/make_demo_video.py (pytest, < 30 s).

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner:ros2_ws/src/pa_perception \\
        python3 -m pytest -q tools/test_make_demo_video.py
"""

import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import make_demo_video as mdv  # noqa: E402
from pa_core.config import load_scene  # noqa: E402
from pa_core.pushsim import PushSim, sweep_profile  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                  reason="ffmpeg / ffprobe not installed")


@pytest.fixture(scope="module")
def episode():
    # Truth perception keeps it fast; I is the cheapest letter.
    return mdv.simulate_letter(("I", 0, "truth", 30, 4))


def test_recording_does_not_change_the_push():
    """The recording wrapper leaves PushSim.push's result untouched."""
    pose = (0.03, -0.02, 0.4)
    a = PushSim("U")
    a.set_pose(*pose)
    ref = a.push(1, 60.0, settle=0.3)
    be = mdv.RecordingBackend("truth")
    be.reset("U", pose, np.random.default_rng(0))
    be.push(1, 60.0)
    assert np.allclose(be.true_pose(), ref, atol=1e-9)
    t_sw, hold = sweep_profile(60.0)
    assert be.now() == pytest.approx(2 * t_sw + hold + be.settle_s)
    # intermediate samples with the paddle out, at 30 Hz, ending at the final pose
    swept = [p for p in be.phys if p[5] is not None]
    assert len(swept) >= int((2 * t_sw + hold) * 30) - 1
    assert max(p[5][1] for p in swept) == pytest.approx(math.radians(60.0), abs=1e-6)
    assert be.phys[-1][1:4] == pytest.approx(ref)
    ts = [p[0] for p in be.phys]
    assert ts == sorted(ts)


def test_episode_and_timeline(episode):
    r = episode["row"]
    assert r["disturb"] == "hand" and r["hand_violations"] == 0
    tl = mdv.Timeline(episode)
    states = [s[1] for s in tl.states]
    assert states[0] == "IDLE" and "WAIT_HAND" in states
    if r["success"]:
        assert states[-1] == "WAIT_LETTER" and "RECOVER" in states
        assert tl.n_pushes(tl.t_end) == r["pushes"]
    # the hand is in view at some point and is drawn
    assert any(p[4] is not None for p in tl.phys)
    # no push while the hand is present or within hand_clear_s of it leaving
    for t, *_ in tl.pushes:
        assert not tl.hand_present(t)


@needs_ffmpeg
def test_render_short_video(episode, tmp_path):
    out = str(tmp_path / "demo.mp4")
    sheet = str(tmp_path / "sheet.png")
    n = 12
    res = mdv.make_video([episode], 0, "truth", out, fps=30, max_frames=n, sheet=sheet,
                         log=lambda *a: None)
    assert res["frames"] == n
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,nb_read_frames", "-of", "json", out],
        capture_output=True, text=True, check=True)
    st = json.loads(probe.stdout)["streams"][0]
    assert st["codec_name"] == "h264"
    assert (st["width"], st["height"]) == (mdv.W, mdv.H)
    assert int(st["nb_read_frames"]) == n


def test_frames_of_each_kind(episode):
    """One frame of every schedule kind renders at full size."""
    sc = load_scene()
    tls = [mdv.Timeline(episode)]
    sched = mdv.build_schedule(tls, 10, 4.0, 2)
    R = mdv.Renderer(sc, "truth", 0, 10, mdv.Text())
    snow = mdv.Snow()
    seen = {}
    for spec in sched:
        seen.setdefault(spec[0], spec)
    want = {"intro", "sim", "finale"} | ({"ok", "slide"} if episode["row"]["success"] else {"fail"})
    assert want <= set(seen)
    for spec in seen.values():
        fr = mdv.render_frame(R, spec, tls, snow, 10)
        assert fr.shape == (mdv.H, mdv.W, 3) and fr.dtype == np.uint8
    # speed-up applies only after the real-time pushes
    fast = [s for s in sched if s[0] == "sim" and s[3] > 1]
    if fast:
        assert min(s[2] for s in fast) >= tls[0].push_start(3) - 1e-9


# ---------------------------------------------------------------- point pusher

@pytest.fixture(scope="module")
def point_episode():
    # No hand: the letter starts at a random disturbed pose; truth keeps it fast.
    return mdv.simulate_letter(("I", 0, "truth", 30, 4, "point", False))


def test_recording_does_not_change_the_stroke():
    """The recording wrappers leave PointPushSim.push's result untouched."""
    from pa_core.pointpush import PointPushSim, stroke_duration
    pose = (0.03, -0.02, 0.4)
    stroke = (0.03, -0.075, 0.035, -0.02)
    a = PointPushSim("U")
    a.set_pose(*pose)
    ref = a.push(*stroke, settle=0.3)
    be = mdv.RecordingBackend("truth", pusher="point")
    be.reset("U", pose, np.random.default_rng(0))
    be.push_stroke(*stroke)
    assert np.allclose(be.true_pose(), ref, atol=1e-9)
    L = math.hypot(stroke[2] - stroke[0], stroke[3] - stroke[1])
    assert be.now() == pytest.approx(mdv.POINT_OVERHEAD_S + stroke_duration(L) + be.settle_s)
    moving = [p for p in be.phys if p[6] is not None]
    assert len(moving) >= int((be.now() - 0.05) * 30) - 2
    # lowered during the stroke, along the stroke line
    low = [p[6] for p in moving if p[6][2] < 0.005]
    assert low and min(q[1] for q in low) == pytest.approx(stroke[1], abs=0.003)
    assert be.phys[-1][1:4] == pytest.approx(ref)
    assert not hasattr(be.sim, "__dict__") or "_build" not in be.sim.__dict__
    ts = [p[0] for p in be.phys]
    assert ts == sorted(ts)
    # the letter moved during the recorded stroke (not only at the end)
    mid = [p for p in be.phys if p[6] is not None and p[6][2] < 0.005]
    assert any(abs(p[1] - pose[0]) + abs(p[2] - pose[1]) > 1e-4 for p in mid)


def test_point_episode(point_episode, tmp_path):
    r = point_episode["row"]
    assert r["disturb"] == "teleport" and r["hand_violations"] == 0
    tl = mdv.Timeline(point_episode)
    # starts at the disturbed pose, no frame at the target
    assert tl.phys[0][0] == 0.0
    assert tl.phys[0][1:4] == pytest.approx((r["start_x"], r["start_y"], r["start_theta"]), abs=1e-4)
    assert all(p[1] == -1 and p[3] is not None for p in tl.pushes)
    assert not any(p[4] is not None or p[5] is not None for p in tl.phys)  # no hand, no paddle
    sc = load_scene()
    R = mdv.Renderer(sc, "truth", 0, 10, mdv.Text(), pusher="point", hand=False)
    t_mid = tl.pushes[0][0] + 0.6
    assert tl.pin(t_mid) is not None
    fr = mdv.render_frame(R, ("sim", 0, t_mid, 1.0), [tl], mdv.Snow(), 10)
    assert fr.shape == (mdv.H, mdv.W, 3)
    # the pusher is drawn (orange) in the camera image, no paddles at rest
    img = R.camera_image("I", tl.world(t_mid)[0], None, None, noise=0.0, pin=tl.pin(t_mid))
    orange = (img[:, :, 2] > 180) & (img[:, :, 0] < 60)
    assert 20 < orange.sum() < 3000
    # speed-up never above the cap
    sp, _ = mdv.choose_speedup([tl], 10, 2.0, 2, 1.0, max_speedup=2.0)
    assert sp == 2.0
