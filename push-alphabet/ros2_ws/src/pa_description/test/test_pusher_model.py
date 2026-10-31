import colorsys
from pathlib import Path
import xml.etree.ElementTree as ET

from pa_core.config import load_scene
from pa_description import pusher_model

MODELS = Path(__file__).resolve().parents[1] / "models"


def test_pusher_sdf_matches_scene():
    pp = load_scene()["point_pusher"]
    root = ET.fromstring(pusher_model.pusher_sdf())
    assert root.tag == "sdf"
    model = root.find("model")
    assert model.get("name") == "point_pusher"
    link = model.find("link")
    assert link.find("gravity").text == "false"
    assert link.find("kinematic").text == "true"
    for tag in ("collision", "visual"):
        el = link.find(tag)
        cyl = el.find("geometry/cylinder")
        assert abs(float(cyl.find("radius").text) - pp["radius"]) < 1e-6
        assert abs(float(cyl.find("length").text) - pp["height"]) < 1e-6
        # origin = bottom face: the cylinder center sits at height/2
        assert abs(float(el.find("pose").text.split()[2]) - pp["height"] / 2) < 1e-6
    assert float(link.find("inertial/mass").text) > 0


def test_pusher_is_orange_and_caught_by_occluder_mask():
    mat = ET.fromstring(pusher_model.pusher_sdf()).find("model/link/visual/material")
    for tag in ("ambient", "diffuse"):
        r, g, b, a = (float(v) for v in mat.find(tag).text.split())
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        assert 15 / 360 < h < 45 / 360 and s > 0.8 and v > 0.8 and a == 1.0
    assert mat.find("script") is None   # no Gazebo/* script (rendered white in the container)


def test_generated_model_files_are_up_to_date():
    d = MODELS / "point_pusher"
    assert (d / "model.sdf").read_text() == pusher_model.pusher_sdf(), \
        "run: python3 tools/gen_pusher_model.py"
    cfg = ET.fromstring((d / "model.config").read_text())
    assert cfg.find("name").text == "point_pusher" and cfg.find("sdf").text == "model.sdf"
