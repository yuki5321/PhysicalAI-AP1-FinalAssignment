"""CNNDetector plumbing (tiny ONNX models built on the fly) and, if the trained
model is present, accuracy on synthetic scenes."""

import math
import os
import sys

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")

from pa_core.letters import angle_error_deg  # noqa: E402
from pa_perception import cnn_crop  # noqa: E402
from pa_perception.classic import ClassicDetector, DetectorParams  # noqa: E402
from pa_perception.cnn_detector import CNNDetector, default_model_path, make_detector  # noqa: E402
from pa_perception.camera_model import CameraModel  # noqa: E402
from pa_perception.synth import fill_polys, random_scene, render_scene  # noqa: E402

S = cnn_crop.CROP_SIZE
LETTERS = list(cnn_crop.LETTERS)


# ------------------------------------------------------------ model builders
def _onnx_linear(path, W, b, batch=None):
    """input [N,1,S,S] -> Flatten -> Gemm(W, b) -> split 4/2/2 into logits/angle/offset."""
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        onnx = None
    if onnx is not None:
        n = batch if batch else "N"
        nodes = [
            helper.make_node("Flatten", ["input"], ["flat"], axis=1),
            helper.make_node("Gemm", ["flat", "W", "b"], ["y"]),
            helper.make_node("Split", ["y", "split"], ["logits", "angle", "offset"], axis=1),
        ]
        inits = [numpy_helper.from_array(W.astype(np.float32), "W"),
                 numpy_helper.from_array(b.astype(np.float32), "b"),
                 numpy_helper.from_array(np.array([4, 2, 2], np.int64), "split")]
        g = helper.make_graph(
            nodes, "tiny", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [n, 1, S, S])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [n, 4]),
             helper.make_tensor_value_info("angle", TensorProto.FLOAT, [n, 2]),
             helper.make_tensor_value_info("offset", TensorProto.FLOAT, [n, 2])], inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
        m.ir_version = 8  # loadable by older onnxruntime (Docker: 1.18)
        onnx.save(m, str(path))
        return str(path)
    torch = pytest.importorskip("torch", reason="needs onnx or torch to build a test model")

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(S * S, 8)
            with torch.no_grad():
                self.fc.weight.copy_(torch.from_numpy(W.T.astype(np.float32)))
                self.fc.bias.copy_(torch.from_numpy(b.astype(np.float32)))

        def forward(self, x):
            y = self.fc(x.flatten(1))
            return y[:, :4], y[:, 4:6], y[:, 6:8]

    kw = {} if batch else {"dynamic_axes": {k: {0: "N"} for k in ("input", "logits", "angle", "offset")}}
    try:
        torch.onnx.export(Tiny().eval(), torch.zeros(batch or 2, 1, S, S), str(path),
                          input_names=["input"], output_names=["logits", "angle", "offset"],
                          opset_version=17, dynamo=False, **kw)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"torch.onnx.export unavailable: {e}")
    return str(path)


def constant_model(path, letter, theta, offset_cm=(0.0, 0.0), logit=10.0, batch=None):
    """Zero weights; the bias says `letter` at `theta` with the given offset."""
    sym = cnn_crop.SYMMETRY.get(letter, 360.0)
    phi = theta * 360.0 / sym
    b = np.zeros(8)
    if letter:
        b[LETTERS.index(letter)] = logit
    b[4:6] = (3.0 * math.cos(phi), 3.0 * math.sin(phi))  # unnormalized on purpose
    b[6:8] = offset_cm
    return _onnx_linear(path, np.zeros((S * S, 8)), b, batch)


def random_model(path, seed=0, batch=None):
    rng = np.random.default_rng(seed)
    return _onnx_linear(path, rng.normal(0, 0.01, (S * S, 8)), rng.normal(0, 1, 8), batch)


@pytest.fixture(scope="module")
def classic():
    return ClassicDetector()


# ------------------------------------------------------------------ plumbing
def test_constant_model_decodes_pose(tmp_path, classic):
    x, y, th = 0.031, -0.022, 0.9
    det = CNNDetector(constant_model(tmp_path / "m.onnx", "U", th), classic=classic)
    dets = det.detect(render_scene([("U", x, y, th)], sides=False))
    assert len(dets) == 1
    d = dets[0]
    assert d.letter == "U" and d.model == "cnn"
    assert abs(angle_error_deg("U", d.theta, th)) < 1e-3
    # offset 0: position = blob centroid on the letter top plane (exact for a top-only image)
    assert math.hypot(d.x - x, d.y - y) < 1e-3
    assert d.confidence > 0.99 and 0.8 < d.iou <= 1.0
    assert set(d.coarse_scores) == set(LETTERS)
    assert abs(sum(d.coarse_scores.values()) - 1.0) < 1e-6


def test_offset_and_symmetric_I(tmp_path, classic):
    th = 1.3  # I is 180 deg symmetric: phi = 2 theta
    det = CNNDetector(constant_model(tmp_path / "m.onnx", "I", th, offset_cm=(0.5, -0.3)),
                      classic=classic)
    d = det.detect(render_scene([("I", 0.0, 0.01, th)], sides=False))[0]
    assert d.letter == "I"
    assert -math.pi / 2 - 1e-9 <= d.theta < math.pi / 2
    assert abs(angle_error_deg("I", d.theta, th)) < 1e-3
    assert abs(d.x - 0.005) < 1e-3 and abs(d.y - (0.01 - 0.003)) < 1e-3


def test_hybrid_refines_coarse_cnn_angle(tmp_path, classic):
    x, y, th = -0.04, 0.035, -2.1
    # CNN 4 deg off: the classic IoU refinement must fix the angle; the position
    # is the CNN one (offset 0 -> blob centroid on the letter top plane).
    path = constant_model(tmp_path / "m.onnx", "K", th + math.radians(4.0))
    hyb = CNNDetector(path, refine="classic", classic=classic)
    cnn = CNNDetector(path, classic=classic)
    for sides in (False, True):
        img = render_scene([("K", x, y, th)], sides=sides)
        d, c = hyb.detect(img)[0], cnn.detect(img)[0]
        assert d.letter == "K" and d.model.startswith("cnn+")
        assert abs(angle_error_deg("K", d.theta, th)) < 1.0
        assert abs(angle_error_deg("K", c.theta, th)) > 3.9
        assert (d.x, d.y) == (c.x, c.y)
        assert d.iou > 0.8 and 0 < d.confidence <= 1
    d = hyb.detect(render_scene([("K", x, y, th)], sides=False))[0]
    assert math.hypot(d.x - x, d.y - y) < 1e-3


def test_hybrid_switches_to_runner_up_letter(tmp_path, classic):
    # CNN: Y with p ~ 0.6, K ~ 0.4 on a K blob -> the IoU fit picks K.
    th = 0.7
    b = np.zeros(8)
    for name, v in {"I": -5.0, "U": -5.0, "Y": 1.0, "K": 0.6}.items():
        b[LETTERS.index(name)] = v
    b[4:6] = (math.cos(th), math.sin(th))
    m = _onnx_linear(tmp_path / "m.onnx", np.zeros((S * S, 8)), b)
    img = render_scene([("K", 0.01, 0.02, th)], sides=False)
    assert CNNDetector(m, classic=classic, min_confidence=0.0,
                       params=DetectorParams(min_iou=0.0)).detect(img)[0].letter == "Y"
    d = CNNDetector(m, refine="classic", classic=classic).detect(img)[0]
    assert d.letter == "K" and abs(angle_error_deg("K", d.theta, th)) < 1.0


def test_batching_single_run(tmp_path, classic):
    det = CNNDetector(random_model(tmp_path / "r.onnx"), params=DetectorParams(min_iou=0.0),
                      min_confidence=0.0)
    calls = []
    real_run = det.sess.run

    def run(names, feeds):
        calls.append(feeds["input"].shape)
        return real_run(names, feeds)

    det.sess.run = run
    sc = random_scene(np.random.default_rng(5), n_letters=3)
    dets = det.detect(render_scene(sc, sides=True))
    assert len(calls) == 1 and calls[0] == (3, 1, S, S)
    assert len(dets) == 3
    lg, an, of = det.infer(np.zeros((5, 1, S, S), np.float32))
    assert lg.shape == (5, 4) and an.shape == (5, 2) and of.shape == (5, 2)


def test_fixed_batch_model(tmp_path, classic):
    det = CNNDetector(random_model(tmp_path / "r1.onnx", batch=1), classic=classic)
    assert det.fixed_batch == 1
    lg, an, of = det.infer(np.zeros((3, 1, S, S), np.float32))
    assert lg.shape == (3, 4) and an.shape == (3, 2) and of.shape == (3, 2)


def test_empty_and_outside_workspace(tmp_path, classic):
    det = CNNDetector(constant_model(tmp_path / "m.onnx", "U", 0.0), classic=classic)
    assert det.detect(render_scene([])) == []
    assert det.detect(render_scene([("U", 0.0, 0.135, 0.0)])) == []


def test_shape_gate_rejects_mismatched_blob(tmp_path, classic):
    # The CNN says "U" but the blob is an I: its U silhouette IoU is low -> dropped
    # (the network has no reject class; the IoU gate stands in for it).
    det = CNNDetector(constant_model(tmp_path / "m.onnx", "U", 0.0), classic=classic)
    img = render_scene([("I", 0.02, -0.03, 0.4)], sides=False)
    assert det.detect(img) == []
    d = CNNDetector(det.model_path, params=DetectorParams(min_iou=0.0)).detect(img)
    assert len(d) == 1 and d[0].letter == "U" and d[0].iou < DetectorParams().min_iou


def test_min_confidence_and_report_unknown(tmp_path, classic):
    path = constant_model(tmp_path / "m.onnx", "", 0.0)  # uniform logits: p = 0.25
    img = render_scene([("Y", 0.0, 0.0, 0.3)])
    assert CNNDetector(path, classic=classic).detect(img) == []
    p = DetectorParams(report_unknown=True)
    d = CNNDetector(path, params=p).detect(img)
    assert len(d) == 1 and d[0].letter == "" and d[0].confidence == 0.0


def test_make_detector_fallback(tmp_path, monkeypatch):
    msgs = []
    det, used = make_detector("cnn", model_path=str(tmp_path / "missing.onnx"), warn=msgs.append)
    assert used == "classic" and isinstance(det, ClassicDetector) and msgs
    with pytest.raises(FileNotFoundError):
        make_detector("hybrid", model_path=str(tmp_path / "missing.onnx"), fallback=False)
    with pytest.raises(ValueError):
        make_detector("bogus")
    path = constant_model(tmp_path / "m.onnx", "U", 0.0)
    det, used = make_detector("hybrid", model_path=path)
    assert used == "hybrid" and det.refine == "classic"
    monkeypatch.setitem(sys.modules, "onnxruntime", None)  # import fails
    det, used = make_detector("cnn", model_path=path, warn=msgs.append)
    assert used == "classic" and "onnxruntime" in msgs[-1]


def test_bad_model_outputs_rejected(tmp_path):
    onnx = pytest.importorskip("onnx")
    path = constant_model(tmp_path / "m.onnx", "U", 0.0)
    m = onnx.load(path)
    m.graph.output[2].name = "off"
    m.graph.node[-1].output[2] = "off"
    onnx.save(m, path)
    with pytest.raises(ValueError):
        CNNDetector(path)


# ------------------------------------------------- occluders and merged blobs
ORANGE = (20, 100, 220)   # BGR paddle
SKIN = (140, 170, 215)    # BGR hand


def _bar(img, p0, p1, w=0.008, z=0.03, color=ORANGE):
    cam = CameraModel.from_scene()
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * w / 2
    fill_polys(img, [cam.world_to_pixel(np.array([p0 - n, p1 - n, p1 + n, p0 + n]), z)], color)
    return img


def test_segmentation_gets_bgr(tmp_path, classic):
    det = CNNDetector(constant_model(tmp_path / "m.onnx", "U", 0.3), classic=classic)
    seen = []
    real = classic.segment_frame

    def seg(img):
        seen.append(img.ndim)
        return real(img)

    classic.segment_frame = seg
    try:
        det.detect(render_scene([("U", 0.0, 0.0, 0.3)]))
    finally:
        del classic.segment_frame
    assert seen == [3]


@pytest.mark.parametrize("refine", ["none", "classic"])
def test_letter_under_paddle(tmp_path, classic, refine):
    x, y, th = 0.02, -0.01, 0.6
    img = render_scene([("K", x, y, th)], pusher_value=ORANGE)
    _bar(img, (-0.08, 0.03), (0.10, -0.02))
    det = CNNDetector(constant_model(tmp_path / "m.onnx", "K", th), refine=refine, classic=classic)
    dets = det.detect(img)
    assert len(dets) == 1
    d = dets[0]
    assert d.letter == "K" and 0.03 < d.occluded_frac < 0.4
    assert math.hypot(d.x - x, d.y - y) < 0.004
    assert abs(angle_error_deg("K", d.theta, th)) < 3.0


def test_merged_blob_split(tmp_path):
    # One blob holding an I and a Y (hsv segmenter: no blob splitting in segmentation).
    # The CNN crop sits between the letters; the in-blob search finds both.
    sc = [("I", -0.02, 0.0, 0.0), ("Y", 0.022, 0.0, 0.0)]
    img = render_scene(sc)
    p = DetectorParams(segmenter="hsv")
    for refine in ("none", "classic"):
        det = CNNDetector(constant_model(tmp_path / "m.onnx", "U", 0.0), params=p, refine=refine)
        dets = det.detect(img)
        assert sorted(d.letter for d in dets) == ["I", "Y"] and all(d.split for d in dets)
        for name, x, y, _ in sc:
            assert any(d.letter == name and math.hypot(d.x - x, d.y - y) < 0.002 for d in dets)


@pytest.mark.parametrize("letter", ["I", "K"])
def test_paddles_and_hand_not_detected(tmp_path, classic, letter):
    # The CNN has no reject class (here it always says `letter`): nothing may reach it.
    img = render_scene([], pusher_value=ORANGE)
    _bar(img, (-0.08, 0.0), (0.08, 0.02))
    _bar(img, (0.0, -0.05), (0.03, 0.03), w=0.06, z=0.05, color=SKIN)
    path = constant_model(tmp_path / "m.onnx", letter, 0.0)
    for refine in ("none", "classic"):
        assert CNNDetector(path, refine=refine, classic=classic).detect(img) == []


# ----------------------------------------------------------- trained model
TRAINED = default_model_path()


@pytest.mark.skipif(not os.path.isfile(TRAINED), reason="trained letter_cnn.onnx not present")
@pytest.mark.parametrize("refine", ["none", "classic"])
def test_trained_model_accuracy(refine, classic):
    det = CNNDetector(TRAINED, refine=refine, classic=classic)
    rng = np.random.default_rng(1234)
    pos_tol, ang_tol = (0.004, 6.0) if refine == "none" else (0.002, 3.0)
    n = correct = 0
    for i in range(40):
        sc = random_scene(rng, n_letters=1 + i % 3)
        img = render_scene(sc, sides=bool(i % 2), noise_sigma=4.0 * (i % 3), rng=rng)
        dets = det.detect(img)
        for name, x, y, th in sc:
            near = [d for d in dets if math.hypot(d.x - x, d.y - y) < 0.02]
            assert len(near) == 1, (name, x, y, [(d.letter, d.x, d.y) for d in dets])
            d = near[0]
            n += 1
            if d.letter != name:
                continue
            correct += 1
            assert math.hypot(d.x - x, d.y - y) < pos_tol
            assert abs(angle_error_deg(name, d.theta, th)) < ang_tol
    assert correct / n >= 0.97


@pytest.mark.skipif(not os.path.isfile(TRAINED), reason="trained letter_cnn.onnx not present")
@pytest.mark.parametrize("refine", ["none", "classic"])
def test_trained_model_letter_touching_paddle(refine, classic):
    det = CNNDetector(TRAINED, refine=refine, classic=classic)
    rng = np.random.default_rng(99)
    for i in range(8):
        (name, x, y, th), = random_scene(rng, n_letters=1, half_size=0.06)
        img = render_scene([(name, x, y, th)], pusher_value=ORANGE, sides=bool(i % 2))
        a = rng.uniform(0, math.pi)
        dx, dy = math.cos(a), math.sin(a)
        _bar(img, (x - 0.1 * dx + 0.01 * dy, y - 0.1 * dy - 0.01 * dx),
             (x + 0.1 * dx + 0.01 * dy, y + 0.1 * dy - 0.01 * dx))
        dets = det.detect(img)
        assert len(dets) == 1 and dets[0].letter == name, (name, x, y, th, dets)
        assert math.hypot(dets[0].x - x, dets[0].y - y) < 0.004
