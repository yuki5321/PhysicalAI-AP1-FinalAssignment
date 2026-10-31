import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from pa_core import letters
from pa_core.config import load_scene

NAMES = list("IUYK")


def test_letter_names():
    assert sorted(letters.letter_names()) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_centroid_at_origin(name):
    pts, inside, _ = letters._union_mask(letters.letter_rects(name))
    c = pts[inside].mean(axis=0)
    assert np.allclose(c, 0.0, atol=1e-4), c


@pytest.mark.parametrize("name", NAMES)
def test_area_reasonable(name):
    sc = load_scene()["letter"]
    a = letters.letter_area(name)
    # At least one full stroke, at most the bounding box of the letter.
    assert sc["stroke"] * sc["height"] * 0.95 <= a <= sc["height"] ** 2


def test_area_of_I_exact():
    assert letters.letter_area("I") == pytest.approx(0.016 * 0.080, rel=0.01)


@pytest.mark.parametrize("name", NAMES)
def test_polygons_follow_pose(name):
    base = letters.letter_polygons(name)
    moved = letters.letter_polygons(name, 0.01, -0.02, math.pi / 2)
    assert len(base) == len(moved) == len(letters.letter_rects(name))
    for b, m in zip(base, moved):
        assert b.shape == (4, 2)
        expect = letters.transform(b, 0.01, -0.02, math.pi / 2)
        assert np.allclose(m, expect)


def test_wrap_deg():
    assert letters.wrap_deg(0) == 0
    assert letters.wrap_deg(190) == pytest.approx(-170)
    assert letters.wrap_deg(-190) == pytest.approx(170)
    assert letters.wrap_deg(720 + 5) == pytest.approx(5)


def test_angle_error_I_is_180_symmetric():
    assert letters.angle_error_deg("I", math.pi) == pytest.approx(0, abs=1e-9)
    assert letters.angle_error_deg("I", math.radians(170)) == pytest.approx(-10)
    assert letters.angle_error_deg("I", math.radians(10)) == pytest.approx(10)
    for deg in range(-360, 361, 15):
        assert -90 <= letters.angle_error_deg("I", math.radians(deg)) < 90


@pytest.mark.parametrize("name", list("UYK"))
def test_angle_error_asymmetric_letters(name):
    assert abs(letters.angle_error_deg(name, math.pi)) == pytest.approx(180)
    assert letters.angle_error_deg(name, math.radians(30)) == pytest.approx(30)
    assert letters.angle_error_deg(name, math.radians(-30)) == pytest.approx(-30)
    assert letters.angle_error_deg(name, math.radians(350)) == pytest.approx(-10)
    # Antisymmetric about the target.
    for deg in (5, 45, 100, 170):
        e1 = letters.angle_error_deg(name, math.radians(deg))
        e2 = letters.angle_error_deg(name, math.radians(-deg))
        assert e1 == pytest.approx(-e2)


def test_angle_error_relative_to_target():
    assert letters.angle_error_deg("U", math.radians(50), math.radians(40)) == pytest.approx(10)


def test_pose_error():
    t = load_scene()["target"]
    dp, da = letters.pose_error("U", t["x"] + 0.003, t["y"] + 0.004, math.radians(t["theta"] - 7))
    assert dp == pytest.approx(0.005)
    assert da == pytest.approx(7)


def test_aligned_and_disturbed():
    t = load_scene()["target"]
    x0, y0, th0 = t["x"], t["y"], math.radians(t["theta"])
    for name in NAMES:
        assert letters.is_aligned(name, x0, y0, th0)
        assert not letters.is_disturbed(name, x0, y0, th0)
        assert letters.is_aligned(name, x0 + 0.5 * t["pos_tol"], y0, th0 + math.radians(0.5 * t["ang_tol"]))
        # Dead band: neither aligned nor disturbed.
        mid = 0.5 * (t["pos_tol"] + t["disturb_pos"])
        assert not letters.is_aligned(name, x0 + mid, y0, th0)
        assert not letters.is_disturbed(name, x0 + mid, y0, th0)
        assert letters.is_disturbed(name, x0, y0 + 1.5 * t["disturb_pos"], th0)
        assert letters.is_disturbed(name, x0, y0, th0 + math.radians(1.5 * t["disturb_ang"]))
    # I upside down is still aligned; U upside down is disturbed.
    assert letters.is_aligned("I", x0, y0, th0 + math.pi)
    assert letters.is_disturbed("U", x0, y0, th0 + math.pi)


@pytest.mark.parametrize("name", NAMES)
def test_letter_sdf_is_well_formed(name):
    root = ET.fromstring(letters.letter_sdf(name))
    assert root.tag == "sdf"
    model = root.find("model")
    assert model.get("name") == f"letter_{name}"
    link = model.find("link")
    n = len(letters.letter_rects(name))
    assert len(link.findall("collision")) == n
    assert len(link.findall("visual")) == n
    mass = float(link.find("inertial/mass").text)
    assert mass == pytest.approx(load_scene()["letter"]["mass"])
    inertia = link.find("inertial/inertia")
    izz = float(inertia.find("izz").text)
    assert izz > 0
    assert float(inertia.find("ixx").text) > 0
    for col in link.findall("collision"):
        size = [float(v) for v in col.find("geometry/box/size").text.split()]
        assert len(size) == 3 and all(v > 0 for v in size)
        assert len(col.find("pose").text.split()) == 6


def test_letter_sdf_custom_name_and_color():
    root = ET.fromstring(letters.letter_sdf("K", model_name="k0", color=(1.0, 0.0, 0.0)))
    assert root.find("model").get("name") == "k0"
    assert root.find("model/link/visual/material/diffuse").text == "1.0 0.0 0.0 1"
