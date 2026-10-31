"""Dataset format (docs/cnn.md) and the synthetic / Gazebo generator helpers."""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

from pa_core.config import load_scene
from pa_perception.camera_model import CameraModel
from pa_perception.classic import ClassicDetector
from pa_perception.dataset import DatasetWriter, iter_frames, letter_tuples, load_dataset, load_meta

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_GEN = os.path.join(_REPO, "tools", "gen_dataset.py")


@pytest.fixture(scope="module")
def gen():
    if not os.path.exists(_GEN):
        pytest.skip("tools/gen_dataset.py not found")
    spec = importlib.util.spec_from_file_location("gen_dataset", _GEN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_dataset"] = mod  # worker processes pickle functions by module name
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scene_cam():
    scene = load_scene()
    return scene, CameraModel.from_scene(scene)


def test_writer_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (640, 640, 3), dtype=np.uint8) for _ in range(3)]
    letters = [[("U", 0.012, -0.031, 1.234)],
               [{"letter": "I", "x": 0.0, "y": 0.05, "theta": -3.0, "occ": 0.5}, ("K", -0.05, 0.0, 0.1)],
               []]
    with DatasetWriter(tmp_path, {"source": "synth", "seed": 4}) as w:
        for f, l in zip(frames, letters):
            w.add(f, l, "synth")
    meta = load_meta(tmp_path)
    assert meta["count"] == 3 and meta["seed"] == 4
    recs = load_dataset(tmp_path)
    assert [r["frame"] for r in recs] == ["frames/000000.png", "frames/000001.png", "frames/000002.png"]
    with open(tmp_path / "labels.jsonl") as fh:
        first = json.loads(fh.readline())
    assert first == {"frame": "frames/000000.png", "source": "synth",
                     "letters": [{"letter": "U", "x": 0.012, "y": -0.031, "theta": 1.234}]}
    assert recs[1]["letters"][0]["occ"] == 0.5
    assert letter_tuples(recs[1]) == [("I", 0.0, 0.05, -3.0), ("K", -0.05, 0.0, 0.1)]
    for (img, rec), f in zip(iter_frames(tmp_path), frames):
        assert np.array_equal(img, f)  # PNG is lossless


def test_writer_append(tmp_path):
    img = np.zeros((8, 8, 3), np.uint8)
    with DatasetWriter(tmp_path, {"source": "gazebo"}) as w:
        w.add(img, [("I", 0, 0, 0)], "gazebo")
    with DatasetWriter(tmp_path, append=True) as w:
        assert w.add(img, [], "gazebo") == 1
    assert len(load_dataset(tmp_path)) == 2
    assert load_meta(tmp_path)["count"] == 2


def test_writer_rejects_bad_frame(tmp_path):
    with DatasetWriter(tmp_path, {}) as w:
        with pytest.raises(ValueError):
            w.add(np.zeros((8, 8), np.uint8), [], "synth")


def test_synth_deterministic(gen, scene_cam):
    scene, cam = scene_cam
    cfg = gen.DEFAULT_RANDOMIZATION
    a, la = gen.synth_frame(7, 3, cfg, cam, scene)
    b, lb = gen.synth_frame(7, 3, cfg, cam, scene)
    c, _ = gen.synth_frame(8, 3, cfg, cam, scene)
    assert a.shape == (640, 640, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b) and la == lb
    assert not np.array_equal(a, c)


def test_synth_letters_valid(gen, scene_cam):
    scene, cam = scene_cam
    cfg = gen.DEFAULT_RANDOMIZATION
    rng = np.random.default_rng(1)
    counts = set()
    for _ in range(40):
        letters = gen.sample_letters(rng, cfg, cam, scene)
        counts.add(len(letters))
        names = [n for n, *_ in letters]
        assert len(names) == len(set(names))
        shape = (cam.height_px // 2, cam.width_px // 2)
        masks = [gen.footprint_mask(cam, scene, *l, shape) for l in letters]
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                assert not np.any(masks[i] & masks[j])
        for _, x, y, th in letters:
            assert abs(x) < 0.16 and abs(y) < 0.16 and -math.pi <= th <= math.pi
    assert {1, 2, 3} <= counts


def test_synth_labels_match_render(gen, scene_cam):
    """The classic detector, run on clean randomized frames, finds the labelled poses."""
    scene, cam = scene_cam
    cfg = dict(gen.DEFAULT_RANDOMIZATION, p_paddles=0.0, p_hand=0.0, p_outside=0.0, p_empty=0.0,
               p_jpeg=0.0, noise_sigma=[0.0, 2.0])
    det = ClassicDetector(scene)
    n = 0
    for i in range(8):
        img, labels = gen.synth_frame(11, i, cfg, cam, scene)
        dets = det.detect(img)
        for lab in labels:
            assert lab["occ"] == 0.0
            d = min(dets, key=lambda q: math.hypot(q.x - lab["x"], q.y - lab["y"]))
            assert math.hypot(d.x - lab["x"], d.y - lab["y"]) < 0.002
            assert d.letter == lab["letter"]
            sym = math.radians(scene["symmetry"][lab["letter"]])
            da = (d.theta - lab["theta"] + sym / 2) % sym - sym / 2
            assert abs(math.degrees(da)) < 2.0
            n += 1
    assert n >= 8


def test_occlusion_label(gen, scene_cam):
    scene, cam = scene_cam
    params = gen.sample_frame_params(np.random.default_rng(0), gen.DEFAULT_RANDOMIZATION, cam, scene)
    params.update(letters=[("I", 0.0, 0.0, 0.0)], paddles=None, hand=None)
    _, lab = gen.render_frame(params, cam, scene)
    assert lab[0]["occ"] == 0.0
    params["hand"] = {"x": 0.0, "y": 0.0, "yaw": 0.0, "z": 0.02, "scale": 1.0, "fingers": 0,
                      "bgr": [150.0, 180.0, 230.0]}
    _, lab = gen.render_frame(params, cam, scene)
    assert lab[0]["occ"] > 0.9


def test_generate_synth_small(gen, tmp_path):
    gen.generate_synth(str(tmp_path / "a"), 6, seed=5, workers=1, log=None)
    gen.generate_synth(str(tmp_path / "b"), 6, seed=5, workers=2, log=None)
    ra, rb = load_dataset(tmp_path / "a"), load_dataset(tmp_path / "b")
    assert len(ra) == 6 and [r["letters"] for r in ra] == [r["letters"] for r in rb]
    meta = load_meta(tmp_path / "a")
    assert meta["source"] == "synth" and meta["count"] == 6 and "randomization" in meta
    assert meta["camera"]["width"] == 640
    for (ia, _), (ib, _) in zip(iter_frames(tmp_path / "a"), iter_frames(tmp_path / "b")):
        assert np.array_equal(ia, ib)


def test_quaternion_to_yaw(gen):
    for th in np.linspace(-3.1, 3.1, 13):
        q = gen.yaw_to_quaternion(th)
        assert gen.quaternion_to_yaw(*q) == pytest.approx(th, abs=1e-9)
        assert gen.quaternion_tilt(*q) == pytest.approx(0.0, abs=1e-6)
    # roll of 90 deg about x: tilted, yaw 0
    s = math.sqrt(0.5)
    assert gen.quaternion_tilt(s, 0.0, 0.0, s) == pytest.approx(math.pi / 2)
    assert gen.quaternion_to_yaw(s, 0.0, 0.0, s) == pytest.approx(0.0, abs=1e-9)
    # yaw then small roll keeps the yaw
    th, r = 0.7, 0.05
    qz = (0.0, 0.0, math.sin(th / 2), math.cos(th / 2))
    qx = (math.sin(r / 2), 0.0, 0.0, math.cos(r / 2))
    x1, y1, z1, w1 = qz
    x2, y2, z2, w2 = qx
    q = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    assert gen.quaternion_to_yaw(*q) == pytest.approx(th, abs=1e-9)


def test_gazebo_pose_sampling(gen, scene_cam):
    scene, cam = scene_cam
    rng = np.random.default_rng(3)
    for _ in range(30):
        letters = gen.sample_gazebo_scene(rng, gen.DEFAULT_RANDOMIZATION, cam, scene)
        assert 1 <= len(letters) <= 3
    assert gen.poses_settled({"a": (0, 0, 0)}, {"a": (0.0001, 0, 0.001)})
    assert not gen.poses_settled({"a": (0, 0, 0)}, {"a": (0.001, 0, 0)})
    assert not gen.poses_settled({"a": (0, 0, math.pi - 0.001)}, {"a": (0, 0, -math.pi + 0.1)})
    assert gen.poses_settled({"a": (0, 0, math.pi - 0.001)}, {"a": (0, 0, -math.pi + 0.001)})
    assert not gen.poses_settled(None, {"a": (0, 0, 0)})
    for i in range(4):  # parked letters are outside the camera view
        x, y, _ = gen.park_pose(i)
        uv = cam.world_to_pixel([[x, y]], 0.012)[0]
        assert not (0 <= uv[0] < cam.width_px and 0 <= uv[1] < cam.height_px)


def test_image_msg_to_bgr(gen):
    class Msg:
        pass
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    m = Msg()
    m.height, m.width, m.encoding = 4, 5, "rgb8"
    m.step = 16  # padded rows
    m.data = np.pad(img.reshape(4, 15), ((0, 0), (0, 1))).tobytes()
    out = gen.image_msg_to_bgr(m)
    assert np.array_equal(out, img[..., ::-1])
    m.encoding = "bgr8"
    assert np.array_equal(gen.image_msg_to_bgr(m), img)
