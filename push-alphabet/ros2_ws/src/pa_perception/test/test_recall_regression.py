"""Recall / precision floors of the letter detectors on a fixed synthetic set.

N_FRAMES frames are rendered on the fly with tools/gen_dataset.py (same
randomization as data/synth_*, seed SEED, which neither synth_train (1) nor
synth_val (2) uses) and scored with tools/eval_perception.score (same matching
and counting as the eval tool). Floors are the latest measured results minus a
small margin (docs/perception_eval.md); raise them when detection improves.

Print the current numbers (to update the floors):

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_perception \
        python3 ros2_ws/src/pa_perception/test/test_recall_regression.py
"""

import importlib.util
import os
import sys

import pytest

from pa_core.config import load_scene
from pa_perception.camera_model import CameraModel
from pa_perception.classic import DetectorParams
from pa_perception.cnn_detector import make_detector

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
_TOOLS = os.path.join(_REPO, "tools")

SEED = 7
N_FRAMES = 120
VISIBLE_OCC = 0.1

# Floors = current results minus a margin (docs/perception_eval.md). On this set
# (seed 7, 120 frames: 134 visible / 212 all labels):
#   method   visible recall / precision      all recall
#   c1664f5  classic .9552/.9846, cnn .9478/.9845, hybrid .9552/.9771;  .6698 / .6840 / .7217
#   afaeea7  classic 1.000/.9926, cnn 1.000/1.000, hybrid 1.000/.9926;  .9434 / .9387 / .9387
# subset: "visible" = labels with occ <= 0.1, "all" = every label in the workspace.
FLOORS = {
    # method: {(subset, metric): floor}
    "classic": {("visible", "recall"): 0.97, ("visible", "precision"): 0.97, ("all", "recall"): 0.91},
    "cnn": {("visible", "recall"): 0.97, ("visible", "precision"): 0.97, ("all", "recall"): 0.91},
    "hybrid": {("visible", "recall"): 0.97, ("visible", "precision"): 0.97, ("all", "recall"): 0.91},
}


def _load(name):
    path = os.path.join(_TOOLS, name + ".py")
    if not os.path.exists(path):
        pytest.skip(f"tools/{name}.py not found")
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_CACHE = {}


def synth_items():
    """[(bgr image, [(letter, x, y, theta, occ), ...]), ...] (rendered once per process)."""
    if "items" not in _CACHE:
        gen = _load("gen_dataset")
        scene = load_scene()
        cam = CameraModel.from_scene(scene)
        cfg = dict(gen.DEFAULT_RANDOMIZATION)
        items = []
        for i in range(N_FRAMES):
            img, labels = gen.synth_frame(SEED, i, cfg, cam, scene)
            items.append((img, [(d["letter"], float(d["x"]), float(d["y"]), float(d["theta"]),
                                 float(d.get("occ", 0.0))) for d in labels]))
        _CACHE["items"] = items
    return _CACHE["items"]


def measure(method):
    """{"visible": score dict, "all": score dict} or None if the method is unavailable."""
    if method in _CACHE:
        return _CACHE[method]
    ev = _load("eval_perception")
    scene = load_scene()
    params = DetectorParams()
    try:
        det, used = make_detector(method, scene=scene, params=params, fallback=False)
    except Exception:  # noqa: BLE001 - onnxruntime / model missing
        _CACHE[method] = None
        return None
    res = ev.run_detector(det, synth_items(), warmup=0)
    out = {"all": ev.score(res, scene, margin=params.workspace_margin),
           "visible": ev.score(res, scene, margin=params.workspace_margin, max_occ=VISIBLE_OCC)}
    _CACHE[method] = out
    return out


@pytest.mark.parametrize("method", list(FLOORS))
def test_recall_precision_floor(method):
    if method != "classic":
        pytest.importorskip("onnxruntime")
    r = measure(method)
    if r is None:
        pytest.skip(f"{method} unavailable (model / onnxruntime missing)")
    assert r["visible"]["labels"] >= 100, "synthetic set too small to be meaningful"
    failed = []
    for (subset, metric), floor in FLOORS[method].items():
        v = r[subset][metric]
        if not v >= floor:
            failed.append(f"{subset} {metric} {v:.4f} < floor {floor:.2f}")
    assert not failed, f"{method}: " + "; ".join(failed)


if __name__ == "__main__":
    for m in FLOORS:
        r = measure(m)
        if r is None:
            print(f"{m}: unavailable")
            continue
        print(f"{m}: " + ", ".join(f"{s} recall {r[s]['recall']:.4f} precision {r[s]['precision']:.4f} "
                                   f"(labels {r[s]['labels']}, missed {r[s]['missed']}, false {r[s]['false']})"
                                   for s in ("visible", "all")))
