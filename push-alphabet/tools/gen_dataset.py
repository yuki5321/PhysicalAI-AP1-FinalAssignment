#!/usr/bin/env python3
"""Generate CNN training / evaluation data in the docs/cnn.md format.

    # synthetic, domain randomized (no ROS)
    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_perception \\
        python3 tools/gen_dataset.py --source synth --out data/synth_train --frames 6000 --seed 1

    # from Gazebo (needs ROS 2 + a running sim without the scene manager / hand):
    #   ros2 launch pa_gazebo sim.launch.py gui:=false scene_manager:=false hand_actor:=false
    python3 tools/gen_dataset.py --source gazebo --out data/gazebo_test --frames 300 --seed 7

Synth mode renders with pa_perception.synth geometry (same camera model, same
letter polygons) and randomizes everything the CNN should be invariant to:
table / plate / letter / side-wall brightness and tint, side-wall shading per
face from a random light direction, illumination gradient and vignette, table
texture, paddles at random sweep angles (they may cover letters), a skin-colored
hand, blur, Gaussian noise and JPEG artifacts. Labels are exact.

Gazebo mode teleports letter_<X> models to random poses (/set_entity_state,
spawning them first if needed), optionally swings a paddle (/push_cmd), waits
until everything is still, then stores /overhead_camera/image_raw together with
the true poses from /model_states. rclpy is imported only in this mode.
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ("ros2_ws/src/pa_core", "ros2_ws/src/pa_perception"):
    _full = os.path.join(_HERE, "..", _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.insert(0, _full)

from pa_core.config import load_scene  # noqa: E402
from pa_core.letters import letter_names, letter_polygons  # noqa: E402
from pa_perception.camera_model import CameraModel  # noqa: E402
from pa_perception.dataset import DatasetWriter  # noqa: E402
from pa_perception.synth import fill_polys  # noqa: E402

# Paddle geometry constants that live in pa_description.urdf_gen (kept in sync by hand;
# pa_description is not needed on the path for synth generation).
ARM_GAP = 0.0015
HUB_HEIGHT = 0.006

# ------------------------------------------------------------------ settings
# Every per-frame random choice is drawn from these ranges (stored in meta.json).
DEFAULT_RANDOMIZATION = {
    "n_letters_p": [0.5, 0.3, 0.2],       # P(1), P(2), P(3) letters
    "p_empty": 0.02,                      # frames without any letter (negatives)
    "unique_letters": True,               # one physical block per letter
    "p_outside": 0.2,                     # centroid sampled beyond the workspace ...
    "outside_half": [0.10, 0.135],        # ... up to this half size [m]
    "p_near_target": 0.15,                # pose near the target (0, 0, 0)
    "near_target_sigma": [0.01, 0.25],    # m, rad
    "min_gap_px": 0,                      # letters may touch (0 px gap at half res)
    "table_gray": [15, 100],
    "plate_same_p": 0.6,                  # plate (outside table_half_size) ~ table gray
    "plate_gray": [20, 200],
    "tint_sigma": 0.035,                  # per-channel color gain sigma
    "texture_amp": [0, 5],                # low-frequency table texture (gray levels)
    "letter_gray": [165, 255],
    "side_ratio": [0.4, 0.9],             # side-wall / top brightness
    "light_elev_deg": [35, 90],
    "p_paddles": 0.85,                    # paddles drawn at all
    "p_paddle_sweep": 0.35,               # per paddle: swept to a random angle (else rest)
    "paddle_rest_jitter_deg": 3.0,
    "paddle_red": [170, 255],             # orange: G = R * paddle_g, B = R * paddle_b
    "paddle_g": [0.35, 0.6],
    "paddle_b": [0.0, 0.2],
    "p_dark_hub": 0.5,                    # else the hub has the paddle color (Gazebo/Orange)
    "p_hand": 0.08,
    "hand_red": [170, 250],               # skin: BGR = R * (hand_b, hand_g, 1)
    "hand_g": [0.72, 0.88],
    "hand_b": [0.55, 0.75],
    "hand_z": [0.003, 0.08],
    "gain": [0.7, 1.25],
    "gradient": [0.0, 0.3],               # linear illumination gradient across the table
    "vignette": [0.0, 0.25],
    "p_blur": 0.7,
    "blur_sigma": [0.3, 1.4],
    "noise_sigma": [0.0, 6.0],
    "p_jpeg": 0.3,
    "jpeg_quality": [35, 90],
}


def _u(rng, lo_hi):
    return float(rng.uniform(lo_hi[0], lo_hi[1]))


# ------------------------------------------------------------ pose sampling
def sample_pose(rng, cfg, scene):
    """Random letter pose (x, y, theta) following cfg (workspace / outside / near target)."""
    half = scene["workspace"]["half_size"]
    if rng.random() < cfg["p_near_target"]:
        t = scene["target"]
        sp, sa = cfg["near_target_sigma"]
        x, y = rng.normal(0.0, sp, 2) + [t["x"], t["y"]]
        th = math.radians(t["theta"]) + rng.normal(0.0, sa)
    else:
        h = _u(rng, cfg["outside_half"]) if rng.random() < cfg["p_outside"] else half
        x, y = rng.uniform(-h, h, 2)
        th = rng.uniform(-math.pi, math.pi)
    th = (th + math.pi) % (2 * math.pi) - math.pi
    return float(x), float(y), float(th)


def footprint_mask(cam, scene, name, x, y, th, shape, scale=0.5):
    """uint8 mask (at `scale` of the image) of a letter's image silhouette."""
    m = np.zeros(shape, np.uint8)
    t = scene["letter"]["thickness"]
    for corners in letter_polygons(name, x, y, th):
        for z in (0.0, t):
            fill_polys(m, [cam.world_to_pixel(corners, z) * scale], 1, aa=False)
    return m


def sample_letters(rng, cfg, cam, scene, max_tries=60):
    """[(name, x, y, theta), ...] non-overlapping in the image (may touch)."""
    names = list(letter_names())
    if rng.random() < cfg["p_empty"]:
        return []
    p = np.asarray(cfg["n_letters_p"], float)
    n = int(rng.choice(len(p), p=p / p.sum())) + 1
    if cfg["unique_letters"]:
        chosen = [names[i] for i in rng.permutation(len(names))[:n]]
    else:
        chosen = [names[i] for i in rng.integers(len(names), size=n)]
    shape = (cam.height_px // 2, cam.width_px // 2)
    occupied = np.zeros(shape, np.uint8)
    gap = int(cfg["min_gap_px"])
    kernel = np.ones((2 * gap + 1, 2 * gap + 1), np.uint8) if gap > 0 else None
    out = []
    for name in chosen:
        for _ in range(max_tries):
            x, y, th = sample_pose(rng, cfg, scene)
            m = footprint_mask(cam, scene, name, x, y, th, shape)
            test = cv2.dilate(m, kernel) if kernel is not None else m
            if not np.any(test & occupied):
                occupied |= m
                out.append((name, x, y, th))
                break
    return out


def sample_frame_params(rng, cfg, cam, scene):
    """All random choices of one synth frame (JSON-able)."""
    tint = lambda: (1.0 + rng.normal(0, cfg["tint_sigma"], 3)).clip(0.7, 1.3).tolist()  # noqa: E731
    table = _u(rng, cfg["table_gray"])
    plate = table + rng.uniform(-15, 15) if rng.random() < cfg["plate_same_p"] else _u(rng, cfg["plate_gray"])
    letter = _u(rng, cfg["letter_gray"])
    elev = math.radians(_u(rng, cfg["light_elev_deg"]))
    az = rng.uniform(-math.pi, math.pi)
    paddles = None
    if rng.random() < cfg["p_paddles"]:
        red = _u(rng, cfg["paddle_red"])
        paddles = {
            "angles_deg": [float(rng.uniform(0, scene["pushers"]["max_angle"]))
                           if rng.random() < cfg["p_paddle_sweep"]
                           else float(abs(rng.normal(0, cfg["paddle_rest_jitter_deg"])))
                           for _ in scene["pushers"]["list"]],
            "bgr": [red * _u(rng, cfg["paddle_b"]), red * _u(rng, cfg["paddle_g"]), red],
            "side_ratio": float(rng.uniform(0.5, 0.85)),
            "dark_hub": bool(rng.random() < cfg["p_dark_hub"]),
        }
    hand = None
    if rng.random() < cfg["p_hand"]:
        red = _u(rng, cfg["hand_red"])
        hand = {
            "x": float(rng.uniform(-0.14, 0.14)), "y": float(rng.uniform(-0.14, 0.14)),
            "yaw": float(rng.uniform(-math.pi, math.pi)), "z": _u(rng, cfg["hand_z"]),
            "scale": float(rng.uniform(0.8, 1.2)), "fingers": int(rng.integers(0, 6)),
            "bgr": [red * _u(rng, cfg["hand_b"]), red * _u(rng, cfg["hand_g"]), red],
        }
    return {
        "letters": sample_letters(rng, cfg, cam, scene),
        "table": table, "table_tint": tint(),
        "plate": float(np.clip(plate, 5, 230)), "plate_tint": tint(),
        "texture_amp": _u(rng, cfg["texture_amp"]), "texture_seed": int(rng.integers(2 ** 31)),
        "letter": letter, "letter_tint": tint(),
        "side_ratio": _u(rng, cfg["side_ratio"]),
        "light": [math.cos(elev) * math.cos(az), math.cos(elev) * math.sin(az), -math.sin(elev)],
        "paddles": paddles, "hand": hand,
        "gain": _u(rng, cfg["gain"]), "gain_tint": tint(),
        "gradient": _u(rng, cfg["gradient"]), "gradient_dir": float(rng.uniform(-math.pi, math.pi)),
        "vignette": _u(rng, cfg["vignette"]),
        "blur_sigma": _u(rng, cfg["blur_sigma"]) if rng.random() < cfg["p_blur"] else 0.0,
        "noise_sigma": _u(rng, cfg["noise_sigma"]), "noise_seed": int(rng.integers(2 ** 31)),
        "jpeg_quality": int(rng.integers(cfg["jpeg_quality"][0], cfg["jpeg_quality"][1] + 1))
        if rng.random() < cfg["p_jpeg"] else 0,
    }


# ----------------------------------------------------------------- rendering
def _side_quads(cam, corners, z_top, z_bot=0.0):
    """Pixel quads of the 4 vertical faces of a box, with outward 2D normals."""
    top = cam.world_to_pixel(corners, z_top)
    bot = cam.world_to_pixel(corners, z_bot)
    c = corners.mean(axis=0)
    out = []
    for i in range(4):
        j = (i + 1) % 4
        e = corners[j] - corners[i]
        n = np.array([e[1], -e[0]])
        if np.dot(n, (corners[i] + corners[j]) / 2 - c) < 0:
            n = -n
        n /= np.linalg.norm(n) + 1e-12
        # A face is visible from the camera (above the origin) when it faces the origin.
        mid = (corners[i] + corners[j]) / 2
        visible = np.dot(n, -mid) > 0
        out.append((np.array([top[i], top[j], bot[j], bot[i]]), n, visible))
    return top, out


def _lambert(normal3, light):
    d = -np.asarray(light, float)
    d /= np.linalg.norm(d)
    return max(0.0, float(np.dot(normal3, d)))


def _box_layer(img, cam, corners, z_top, z_bot, top_bgr, side_ratio, light, masks=None):
    """Draw a lit vertical box (side faces then top) into img; returns the top polygon."""
    top, quads = _side_quads(cam, corners, z_top, z_bot)
    top_l = _lambert(np.array([0.0, 0.0, 1.0]), light)
    for q, n, vis in quads:
        if not vis:
            continue
        k = side_ratio * (0.55 + 0.45 * _lambert(np.array([n[0], n[1], 0.0]), light)) / max(0.55 + 0.45 * top_l, 1e-3)
        k = min(k, 1.0)
        col = tuple(float(c * k) for c in top_bgr)
        fill_polys(img, [q], col)
        if masks is not None:
            fill_polys(masks, [q], 1, aa=False)
    return top


def _paddle_boxes(scene, angles_deg):
    p = scene["pushers"]
    L, w = p["arm_length"], p["arm_thickness"]
    out = []
    for pu, a_deg in zip(p["list"], angles_deg):
        a = math.radians(pu["rest_heading"] + a_deg)
        d = np.array([math.cos(a), math.sin(a)])
        n = np.array([-d[1], d[0]])
        c0 = np.asarray(pu["pivot"], dtype=float)
        out.append((c0, np.array([c0 - n * w / 2, c0 + d * L - n * w / 2, c0 + d * L + n * w / 2, c0 + n * w / 2])))
    return out


def _hand_polys(hand):
    """World rectangles (palm + fingers) of the hand, Gazebo-hand-like boxes."""
    s = hand["scale"]
    c, sn = math.cos(hand["yaw"]), math.sin(hand["yaw"])
    rot = np.array([[c, -sn], [sn, c]])
    o = np.array([hand["x"], hand["y"]])

    def rect(cx, cy, w, h, ang=0.0):
        ca, sa = math.cos(ang), math.sin(ang)
        loc = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        loc = loc @ np.array([[ca, -sa], [sa, ca]]).T + [cx, cy]
        return (loc * s) @ rot.T + o

    polys = [rect(0.0, 0.0, 0.08, 0.09)]
    finger_x = [-0.03, -0.01, 0.01, 0.03]
    for i in range(min(hand["fingers"], 4)):
        polys.append(rect(finger_x[i], 0.08, 0.018, 0.075))
    if hand["fingers"] >= 5:
        polys.append(rect(-0.055, 0.015, 0.02, 0.06, 0.6))
    polys.append(rect(0.0, -0.08, 0.06, 0.08))  # wrist / forearm toward the camera edge
    return polys


def _texture(shape, amp, seed):
    if amp <= 0:
        return 0.0
    r = np.random.default_rng(seed)
    g = r.normal(0, 1, (8, 8)).astype(np.float32)
    t = cv2.resize(g, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return (t * amp)[..., None]


_GRID = {}


def _grid(cam):
    key = (cam.width_px, cam.height_px, cam.cx, cam.cy)
    if key not in _GRID:
        ys, xs = np.mgrid[0:cam.height_px, 0:cam.width_px].astype(np.float32)
        _GRID[key] = ((xs - cam.cx) / (cam.width_px / 2), (ys - cam.cy) / (cam.height_px / 2))
    return _GRID[key]


def render_frame(params, cam=None, scene=None):
    """Render one randomized frame. Returns (bgr uint8, [letter dicts with `occ`])."""
    scene = scene or load_scene()
    cam = cam or CameraModel.from_scene(scene)
    H, W = cam.height_px, cam.width_px
    img = np.empty((H, W, 3), np.float32)
    img[:] = np.asarray(params["plate_tint"]) * params["plate"]
    h = scene["workspace"]["table_half_size"]
    table = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
    fill_polys(img, [cam.world_to_pixel(table, 0.0)], tuple(np.asarray(params["table_tint"]) * params["table"]))
    img += _texture((H, W), params["texture_amp"], params["texture_seed"])

    light = params["light"]
    t = scene["letter"]["thickness"]
    letter_bgr = np.asarray(params["letter_tint"]) * params["letter"]
    letters = [tuple(x) for x in params["letters"]]
    tops_mask = []
    all_tops = []
    for rects in (letter_polygons(n, x, y, th) for n, x, y, th in letters):
        tops = []
        for corners in rects:
            _, quads = _side_quads(cam, corners, t)
            top_l = _lambert(np.array([0.0, 0.0, 1.0]), light)
            for q, n, vis in quads:
                if vis:
                    k = params["side_ratio"] * (0.55 + 0.45 * _lambert(np.array([n[0], n[1], 0.0]), light))
                    k = min(k / max(0.55 + 0.45 * top_l, 1e-3), 1.0)
                    fill_polys(img, [q], tuple(float(c * k) for c in letter_bgr))
            tops.append(cam.world_to_pixel(corners, t))
        all_tops.append(tops)
    for tops in all_tops:
        fill_polys(img, tops, tuple(float(c) for c in letter_bgr))
        m = np.zeros((H, W), np.uint8)
        fill_polys(m, tops, 1, aa=False)
        tops_mask.append(m)

    occluder = np.zeros((H, W), np.uint8)
    pad = params.get("paddles")
    if pad:
        pz = scene["pushers"]["arm_height"] + ARM_GAP
        bgr = pad["bgr"]
        for pivot, corners in _paddle_boxes(scene, pad["angles_deg"]):
            _box_layer(img, cam, corners, pz, ARM_GAP, bgr, pad["side_ratio"], light, occluder)
            top = cam.world_to_pixel(corners, pz)
            fill_polys(img, [top], tuple(bgr))
            fill_polys(occluder, [top], 1, aa=False)
            # dark hub on top of the pivot
            r = 1.6 * scene["pushers"]["arm_thickness"]
            ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
            hub = pivot + r * np.stack([np.cos(ang), np.sin(ang)], 1)
            hub_col = (20.0, 20.0, 22.0) if pad.get("dark_hub", True) else tuple(bgr)
            fill_polys(img, [cam.world_to_pixel(hub, pz + HUB_HEIGHT)], hub_col)
    hand = params.get("hand")
    if hand:
        zb = hand["z"]
        zt = zb + 0.025
        for corners in _hand_polys(hand):
            _box_layer(img, cam, corners, zt, zb, hand["bgr"], 0.75, light, occluder)
            top = cam.world_to_pixel(corners, zt)
            fill_polys(img, [top], tuple(hand["bgr"]))
            fill_polys(occluder, [top], 1, aa=False)

    # illumination: gain * tint * (1 + gradient) * vignette
    xn, yn = _grid(cam)
    gd = params["gradient_dir"]
    field = np.float32(params["gain"]) * (1.0 + np.float32(params["gradient"] * 0.5 * math.cos(gd)) * xn
                                          + np.float32(params["gradient"] * 0.5 * math.sin(gd)) * yn)
    field *= 1.0 - np.float32(params["vignette"] * 0.5) * (xn * xn + yn * yn)
    for ch, k in enumerate(params["gain_tint"]):
        img[..., ch] *= field * np.float32(k)

    if params["blur_sigma"] > 0:
        img = cv2.GaussianBlur(img, (0, 0), params["blur_sigma"])
    if params["noise_sigma"] > 0:
        r = np.random.default_rng(params["noise_seed"])
        sig = np.float32(params["noise_sigma"])
        noise = r.standard_normal((H, W, 3), dtype=np.float32)
        noise *= sig * np.float32(0.35)
        noise += r.standard_normal((H, W, 1), dtype=np.float32) * sig
        img += noise
    out = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    if params["jpeg_quality"]:
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg_quality"]])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    labels = []
    for (name, x, y, th), m in zip(letters, tops_mask):
        area = int(m.sum())
        occ = float((m & occluder).sum()) / area if area else 1.0
        labels.append({"letter": name, "x": x, "y": y, "theta": th, "occ": round(occ, 4)})
    return out, labels


# ------------------------------------------------------------- synth driver
_W = {}


def _worker_init(seed, cfg, png_compression):
    _W["seed"], _W["cfg"], _W["png"] = seed, cfg, png_compression
    _W["scene"] = load_scene()
    _W["cam"] = CameraModel.from_scene(_W["scene"])
    cv2.setNumThreads(1)


def synth_frame(seed, index, cfg, cam=None, scene=None):
    """Deterministic (image, labels) of frame `index` for a given seed."""
    scene = scene or load_scene()
    cam = cam or CameraModel.from_scene(scene)
    rng = np.random.default_rng([int(seed), int(index)])
    params = sample_frame_params(rng, cfg, cam, scene)
    return render_frame(params, cam, scene)


def _worker(index):
    img, labels = synth_frame(_W["seed"], index, _W["cfg"], _W["cam"], _W["scene"])
    ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, _W["png"]])
    return index, buf.tobytes(), labels


def generate_synth(out, frames, seed, workers=2, cfg=None, png_compression=3, log=print):
    cfg = dict(DEFAULT_RANDOMIZATION, **(cfg or {}))
    scene = load_scene()
    meta = {"source": "synth", "seed": int(seed), "frames": int(frames), "camera": scene["camera"],
            "randomization": cfg, "generator": "tools/gen_dataset.py",
            "label_extra": {"occ": "fraction of the letter top face hidden by a paddle / hand"}}
    t0 = time.time()
    with DatasetWriter(out, meta, png_compression=png_compression) as w:
        if workers <= 1:
            _worker_init(seed, cfg, png_compression)
            results = map(_worker, range(frames))
            pool = None
        else:
            import multiprocessing as mp
            pool = mp.get_context("fork").Pool(workers, _worker_init, (seed, cfg, png_compression))
            results = pool.imap(_worker, range(frames), chunksize=8)
        try:
            for k, (index, png, labels) in enumerate(results):
                w.add(None, labels, "synth", png_bytes=png)
                if log and (k + 1) % 500 == 0:
                    log(f"  {k + 1}/{frames}  {(k + 1) / (time.time() - t0):.1f} frames/s")
        finally:
            if pool is not None:
                pool.close()
                pool.join()
    dt = time.time() - t0
    if log:
        log(f"wrote {frames} frames to {out} in {dt:.1f} s ({frames / max(dt, 1e-9):.1f} frames/s)")
    return frames / max(dt, 1e-9)


# ----------------------------------------------------------- gazebo helpers
def quaternion_to_yaw(x, y, z, w) -> float:
    """Yaw (rotation about +z, rad in [-pi, pi]) of a quaternion (ZYX convention)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_tilt(x, y, z, w) -> float:
    """Angle (rad) between the body +z axis and world +z (0 = lying flat)."""
    zz = 1.0 - 2.0 * (x * x + y * y)
    return math.acos(max(-1.0, min(1.0, zz)))


def yaw_to_quaternion(theta) -> tuple:
    return 0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)


def park_pose(index: int):
    """Out-of-view pose for an unused letter: on the desk behind the table (the desk top
    is 2 cm below the table top, so it drops onto it), outside the camera view."""
    return -0.3 + 0.2 * index, 0.30, 0.0


def sample_gazebo_scene(rng, cfg, cam, scene):
    """Letter poses for one Gazebo frame: like synth but with at least one letter,
    every letter on the table (walls stop them anyway) and no empty frames."""
    c = dict(cfg, p_empty=0.0)
    return sample_letters(rng, c, cam, scene)


def poses_settled(prev: dict, cur: dict, pos_tol=0.0003, ang_tol=0.005) -> bool:
    """True if every model in `cur` moved less than the tolerances since `prev`."""
    if not prev or set(prev) != set(cur):
        return False
    for k, (x, y, th) in cur.items():
        px, py, pth = prev[k]
        dth = (th - pth + math.pi) % (2 * math.pi) - math.pi
        if math.hypot(x - px, y - py) > pos_tol or abs(dth) > ang_tol:
            return False
    return True


def image_msg_to_bgr(msg) -> np.ndarray:
    """sensor_msgs/Image (bgr8 / rgb8 / mono8) -> HxWx3 uint8 BGR without cv_bridge."""
    buf = np.frombuffer(bytes(msg.data), np.uint8)
    enc = msg.encoding.lower()
    ch = 1 if enc in ("mono8", "8uc1") else 3
    img = buf.reshape(msg.height, msg.step)[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if ch == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc == "rgb8":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img.copy()


def generate_gazebo(out, frames, seed, cfg=None, p_push=0.4, p_midswing=0.3, settle_timeout=8.0,
                    view_half=0.17, log=print):
    """Collect `frames` labelled frames from a running Gazebo sim (see module doc)."""
    import rclpy
    from rclpy.node import Node
    from gazebo_msgs.msg import EntityState, ModelStates
    from gazebo_msgs.srv import SetEntityState, SpawnEntity
    from geometry_msgs.msg import Pose
    from sensor_msgs.msg import Image
    from pa_core.letters import letter_sdf
    from pa_interfaces.msg import PushCommand

    cfg = dict(DEFAULT_RANDOMIZATION, **(cfg or {}))
    scene = load_scene()
    cam = CameraModel.from_scene(scene)
    rng = np.random.default_rng(int(seed))
    names = list(letter_names())
    n_sides = len(scene["pushers"]["list"])

    rclpy.init()
    node = Node("pa_dataset_collector")
    state = {"models": None, "models_t": 0.0, "image": None, "image_t": 0.0, "push_done": set()}

    def on_models(msg):
        d = {}
        for n, p in zip(msg.name, msg.pose):
            q = p.orientation
            d[n] = (p.position.x, p.position.y, p.position.z, q.x, q.y, q.z, q.w)
        state["models"], state["models_t"] = d, time.time()

    def on_image(msg):
        state["image"], state["image_t"] = msg, time.time()

    node.create_subscription(ModelStates, "/model_states", on_models, 10)
    node.create_subscription(Image, "/overhead_camera/image_raw", on_image, 2)
    node.create_subscription(PushCommand, "/push_done", lambda m: state["push_done"].add(m.id), 10)
    push_pub = node.create_publisher(PushCommand, "/push_cmd", 10)
    set_cli = node.create_client(SetEntityState, "/set_entity_state")
    spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")

    def spin_for(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, end - time.time())))

    def wait_until(cond, timeout):
        end = time.time() + timeout
        while time.time() < end:
            if cond():
                return True
            rclpy.spin_once(node, timeout_sec=0.05)
        return cond()

    def call(cli, req, timeout=5.0):
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
        return fut.result()

    def make_pose(x, y, z, th):
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        q = yaw_to_quaternion(th)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
        return p

    def set_pose(model, x, y, th, z=0.002):
        req = SetEntityState.Request()
        req.state = EntityState(name=model, pose=make_pose(x, y, z, th), reference_frame="world")
        res = call(set_cli, req)
        return res is not None and res.success

    def letter_poses():
        m = state["models"] or {}
        return {k: (v[0], v[1], quaternion_to_yaw(*v[3:])) for k, v in m.items() if k.startswith("letter_")}

    def wait_settled(timeout):
        prev, end = None, time.time() + timeout
        while time.time() < end:
            spin_for(0.25)
            cur = letter_poses()
            if poses_settled(prev, cur):
                return True
            prev = cur
        return False

    def fresh_image(after, timeout=3.0):
        ok = wait_until(lambda: state["image"] is not None and state["image_t"] > after, timeout)
        return state["image"] if ok else None

    def labels_now(active):
        m = state["models"] or {}
        out = []
        for n in active:
            v = m.get(f"letter_{n}")
            if v is None:
                return None
            if quaternion_tilt(*v[3:]) > 0.1:  # tipped over / standing on an edge
                return None
            x, y = v[0], v[1]
            if abs(x) < view_half and abs(y) < view_half:
                out.append({"letter": n, "x": x, "y": y, "theta": quaternion_to_yaw(*v[3:]),
                            "z": round(v[2], 5)})
        return out

    try:
        log("waiting for /set_entity_state, /model_states and the camera ...")
        if not set_cli.wait_for_service(timeout_sec=60.0):
            raise RuntimeError("/set_entity_state not available (is sim.launch.py running?)")
        if not wait_until(lambda: state["models"] is not None and state["image"] is not None, 60.0):
            raise RuntimeError("no /model_states or /overhead_camera/image_raw")
        for n in names:  # spawn missing letters out of view
            model = f"letter_{n}"
            if model not in state["models"]:
                if not spawn_cli.wait_for_service(timeout_sec=10.0):
                    raise RuntimeError("/spawn_entity not available")
                req = SpawnEntity.Request()
                req.name, req.xml, req.reference_frame = model, letter_sdf(n, model), "world"
                px, py, _ = park_pose(names.index(n))
                req.initial_pose = make_pose(px, py, 0.002, 0.0)
                res = call(spawn_cli, req, 15.0)
                log(f"spawn {model}: {'ok' if res is not None and res.success else 'FAILED'}")
        spin_for(1.0)

        meta = {"source": "gazebo", "seed": int(seed), "frames": int(frames), "camera": scene["camera"],
                "randomization": {"poses": {k: cfg[k] for k in ("n_letters_p", "p_outside", "outside_half",
                                                                 "p_near_target", "near_target_sigma")},
                                  "p_push": p_push, "p_midswing": p_midswing},
                "generator": "tools/gen_dataset.py",
                "label_extra": {"z": "model origin height [m] from /model_states"}}
        push_id = int(rng.integers(1, 2 ** 20))
        t0 = time.time()
        skipped = 0
        with DatasetWriter(out, meta) as w:
            while w.count < frames:
                letters = sample_gazebo_scene(rng, cfg, cam, scene)
                active = [n for n, *_ in letters]
                # Park every letter first so a new pose never overlaps an old one.
                for i, n in enumerate(names):
                    px, py, _ = park_pose(i)
                    set_pose(f"letter_{n}", px, py, 0.0)
                for n, x, y, th in letters:
                    set_pose(f"letter_{n}", x, y, th)
                if not wait_settled(settle_timeout):
                    skipped += 1
                    continue
                if rng.random() < p_push:
                    push_id += 1
                    cmd = PushCommand()
                    cmd.id, cmd.side = push_id, int(rng.integers(n_sides))
                    cmd.angle_deg = float(rng.uniform(10, scene["pushers"]["max_angle"]))
                    push_pub.publish(cmd)
                    if rng.random() < p_midswing:
                        # A frame while the paddle is in the image; letters may be moving
                        # slowly, so pair it with the next /model_states after the image.
                        spin_for(float(rng.uniform(0.3, 1.0)))
                        img_t = time.time()
                        msg = fresh_image(img_t)
                        if msg is not None and wait_until(lambda: state["models_t"] > state["image_t"], 1.0):
                            lab = labels_now(active)
                            if lab and w.count < frames:
                                w.add(image_msg_to_bgr(msg), lab, "gazebo")
                    wait_until(lambda: push_id in state["push_done"], 10.0)
                    if not wait_settled(settle_timeout):
                        skipped += 1
                        continue
                t_req = time.time()
                msg = fresh_image(t_req)
                if msg is None:
                    skipped += 1
                    continue
                before = letter_poses()
                spin_for(0.15)
                if not poses_settled(before, letter_poses()):
                    skipped += 1
                    continue
                lab = labels_now(active)
                if lab is None:
                    skipped += 1
                    continue
                if w.count < frames:
                    w.add(image_msg_to_bgr(msg), lab, "gazebo")
                if log and w.count % 25 == 0:
                    log(f"  {w.count}/{frames} ({w.count / (time.time() - t0):.2f} frames/s, skipped {skipped})")
        log(f"wrote {frames} gazebo frames to {out} in {time.time() - t0:.0f} s (skipped {skipped})")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# ---------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", choices=["synth", "gazebo"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=2, help="synth: worker processes")
    ap.add_argument("--png-compression", type=int, default=3)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=JSON",
                    help="override a randomization setting, e.g. --set p_hand=0.2")
    ap.add_argument("--p-push", type=float, default=0.4, help="gazebo: chance of a paddle swing per frame")
    ap.add_argument("--p-midswing", type=float, default=0.3, help="gazebo: chance of an extra mid-swing frame")
    a = ap.parse_args(argv)
    cfg = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        if k not in DEFAULT_RANDOMIZATION:
            ap.error(f"unknown setting {k!r}")
        cfg[k] = json.loads(v)
    if a.source == "synth":
        generate_synth(a.out, a.frames, a.seed, a.workers, cfg, a.png_compression)
    else:
        generate_gazebo(a.out, a.frames, a.seed, cfg, a.p_push, a.p_midswing)


if __name__ == "__main__":
    main()
