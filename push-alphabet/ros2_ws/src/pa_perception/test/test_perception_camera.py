import math

import numpy as np

from pa_perception.camera_model import CameraModel


def test_axes_and_center():
    cam = CameraModel.from_scene()
    c = cam.world_to_pixel([[0.0, 0.0]])[0]
    assert np.allclose(c, [(cam.width_px - 1) / 2, (cam.height_px - 1) / 2])
    r = cam.world_to_pixel([[0.05, 0.0]])[0]
    u = cam.world_to_pixel([[0.0, 0.05]])[0]
    assert r[0] > c[0] and abs(r[1] - c[1]) < 1e-9      # +x -> image right
    assert u[1] < c[1] and abs(u[0] - c[0]) < 1e-9      # +y -> image up


def test_fov_and_height_scale():
    cam = CameraModel.from_scene()
    # The image edge sees exactly tan(hfov/2) * H at the table plane.
    half = cam.height * math.tan(math.radians(cam.hfov_deg) / 2)
    edge = cam.world_to_pixel([[half, 0.0]])[0]
    assert abs(edge[0] - (cam.cx + cam.width_px / 2)) < 1e-6
    # Raised surfaces look bigger.
    assert cam.px_per_m(0.012) > cam.px_per_m(0.0)


def test_roundtrip():
    cam = CameraModel.from_scene()
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.15, 0.15, (50, 2))
    for z in (0.0, 0.012):
        back = cam.pixel_to_world(cam.world_to_pixel(pts, z), z)
        assert np.allclose(back, pts, atol=1e-12)
