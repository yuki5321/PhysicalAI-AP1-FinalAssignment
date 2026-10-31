import math
import random
import xml.etree.ElementTree as ET

from pa_core.config import load_scene
from pa_core.letters import is_disturbed, letter_polygons
from pa_gazebo import scene_logic as sl


def test_model_names_unique():
    assert sl.model_names("IUYK") == ["letter_I", "letter_U", "letter_Y", "letter_K"]
    assert sl.model_names("IUI") == ["letter_I", "letter_U", "letter_I_1"]


def test_random_disturbed_pose_inside_workspace():
    rng = random.Random(0)
    ws = load_scene()["workspace"]["half_size"]
    for letter in "IUYK":
        for _ in range(20):
            p = sl.random_disturbed_pose(letter, rng)
            assert is_disturbed(letter, p.x, p.y, p.theta)
            for poly in letter_polygons(letter, p.x, p.y, p.theta):
                assert abs(poly).max() <= ws


def test_shelf_slots_follow_display_word_and_are_out_of_view_and_reach():
    scene = load_scene()
    word = scene["display_word"]
    shelf = sl.ShelfAllocator(word)
    xs = {ch: shelf.assign(ch).x for ch in scene["sequence"]}
    assert sorted(word, key=lambda ch: xs[ch]) == list(word)
    # Outside the camera footprint on the table and beyond the walls.
    cam = scene["camera"]
    view_half = cam["height"] * math.tan(math.radians(cam["hfov_deg"]) / 2)
    margin = scene["letter"]["height"] / 2 + 0.01
    assert abs(sl.SHELF_Y) - margin > view_half
    assert abs(sl.SHELF_Y) - sl.SHELF_DEPTH / 2 > 0.21  # beyond the table walls (~0.204)


def test_shelf_extra_slot_for_unexpected_letter():
    shelf = sl.ShelfAllocator("YU")
    a = shelf.assign("Y")
    b = shelf.assign("Y")
    assert b.x > a.x and b.x > shelf.slots[-1][0]


def test_sequencer():
    s = sl.Sequencer("IU")
    assert s.active == "I" and s.active_model == "letter_I"
    assert not s.finish("U")
    assert s.finish("i")
    assert s.active == "U"
    assert s.finish("U")
    assert s.done and s.active == ""


def test_sdf_snippets_are_well_formed():
    for xml in (sl.shelf_sdf("YUKI"), sl.snow_sdf("snow_000"), sl.celebration_light_sdf()):
        ET.fromstring(xml.split("?>", 1)[1])


def test_snow_positions_over_table():
    pts = sl.snow_positions(50, random.Random(1), 0.16)
    assert len(pts) == 50
    assert all(abs(x) <= 0.16 and abs(y) <= 0.16 and 0.15 <= z <= 0.40 for x, y, z in pts)


def test_yaw_quaternion():
    qx, qy, qz, qw = sl.yaw_to_quaternion(math.pi / 2)
    assert (qx, qy) == (0.0, 0.0)
    assert math.isclose(qz, math.sqrt(0.5)) and math.isclose(qw, math.sqrt(0.5))
