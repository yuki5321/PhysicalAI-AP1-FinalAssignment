"""scene.yaml is the single source of truth shared by Gazebo, the 2D model and perception."""

from pa_core.config import load_scene


def test_scene_has_shared_keys():
    sc = load_scene()
    for key in ("workspace", "letter", "pushers", "target", "camera", "letters", "symmetry",
                "sequence", "display_word"):
        assert key in sc, key
    for key in ("x", "y", "theta", "pos_tol", "ang_tol", "disturb_pos", "disturb_ang"):
        assert key in sc["target"], key


def test_scene_is_cached():
    assert load_scene() is load_scene()


def test_letters_and_sequence_consistent():
    sc = load_scene()
    assert set(sc["letters"]) == set("YUKI")
    assert set(sc["sequence"]) <= set(sc["letters"])
    assert set(sc["display_word"]) <= set(sc["letters"])
    assert set(sc["symmetry"]) == set(sc["letters"])


def test_pushers_layout():
    sp = load_scene()["pushers"]
    assert [p["name"] for p in sp["list"]] == ["south", "east", "north", "west"]
    ws = load_scene()["workspace"]["half_size"]
    for p in sp["list"]:
        # Pivots sit just outside the workspace; the arm is long enough to cross it.
        assert max(abs(p["pivot"][0]), abs(p["pivot"][1])) > ws
    assert sp["arm_length"] > ws
    assert 0 < sp["max_angle"] <= 180


def test_tolerances_ordered():
    t = load_scene()["target"]
    # Between "aligned" and "disturbed" there must be a dead band (hysteresis).
    assert t["pos_tol"] < t["disturb_pos"]
    assert t["ang_tol"] < t["disturb_ang"]
