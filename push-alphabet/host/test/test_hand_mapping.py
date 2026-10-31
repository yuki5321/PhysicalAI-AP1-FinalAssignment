"""python3 -m pytest host  (no mediapipe / cv2 needed)"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hand_mapping import (Calibration, HandSmoother, image_to_table, load_table_half_size,  # noqa: E402
                          palm_center, table_to_image)

H = 0.16


def approx(a, b):
    return a == pytest.approx(b, abs=1e-9)


def test_full_frame_corners():
    cal = Calibration(half_size=H)
    x, y, inside = image_to_table(0.5, 0.5, cal)
    assert approx(x, 0) and approx(y, 0) and inside
    x, y, _ = image_to_table(0.0, 0.0, cal)   # image top-left -> world (-H, +H)
    assert approx(x, -H) and approx(y, H)
    x, y, _ = image_to_table(1.0, 1.0, cal)   # image bottom-right -> world (+H, -H)
    assert approx(x, H) and approx(y, -H)


def test_crop_and_outside():
    cal = Calibration(crop=(0.2, 0.1, 0.6, 0.5), half_size=H)
    x, y, inside = image_to_table(0.4, 0.3, cal)
    assert approx(x, 0) and approx(y, 0) and inside
    x, y, inside = image_to_table(0.6, 0.1, cal)
    assert approx(x, H) and approx(y, H) and inside
    _, _, inside = image_to_table(0.1, 0.3, cal)
    assert not inside


def test_mirror_and_rotate():
    x, y, _ = image_to_table(0.0, 0.5, Calibration(half_size=H, mirror=True))
    assert approx(x, H) and approx(y, 0)
    # camera rotated: image right -> world +y
    x, y, _ = image_to_table(1.0, 0.5, Calibration(half_size=H, rotate_deg=90))
    assert approx(x, 0) and approx(y, H)


@pytest.mark.parametrize("cal", [
    Calibration(crop=(0.1, 0.2, 0.7, 0.9), half_size=H),
    Calibration(crop=(0.1, 0.2, 0.7, 0.9), half_size=H, mirror=True, rotate_deg=270),
])
def test_inverse(cal):
    for u, v in [(0.3, 0.4), (0.65, 0.25), (0.1, 0.9)]:
        x, y, _ = image_to_table(u, v, cal)
        uu, vv = table_to_image(x, y, cal)
        assert approx(uu, u) and approx(vv, v)


def test_bad_crop():
    with pytest.raises(ValueError):
        image_to_table(0.5, 0.5, Calibration(crop=(0.5, 0.0, 0.5, 1.0)))


def test_palm_center():
    lm = [(0.0, 0.0)] * 21
    lm = list(lm)
    for i, p in zip((0, 5, 9, 13, 17), [(0.5, 0.9), (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.6)]):
        lm[i] = p
    u, v = palm_center(lm)
    assert approx(u, 0.54) and approx(v, 0.6)


def test_smoother_hysteresis():
    s = HandSmoother(alpha=0.5, on_frames=2, off_frames=3)
    assert s.update((0.1, 0.0))[0] is False
    p, x, _ = s.update((0.1, 0.0))
    assert p is True and approx(x, 0.1)
    p, x, _ = s.update((0.0, 0.0))
    assert approx(x, 0.05)
    assert s.update(None)[0] is True   # one dropped frame keeps the hand
    assert s.update(None)[0] is True
    assert s.update(None)[0] is False


def test_scene_half_size():
    assert load_table_half_size() == pytest.approx(0.16)
