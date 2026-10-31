"""Gazebo backend for tools/eval_recovery.py (contract: docs/recovery_eval.md).

Needs ROS 2 Humble and a running simulation without the scene manager, the
hand actor and the planner node, plus a letter detector:

    ros2 launch pa_gazebo eval.launch.py            # gui:=false method:=classic hand:=true
    python3 tools/eval_recovery.py --backend gazebo --disturb teleport --trials 20 --csv ...

World = Gazebo. True pose = /model_states (model letter_<X>). Place / teleport
= /set_entity_state (the letter is spawned with pa_core.letters.letter_sdf if it
is missing; every other letter_* model is parked out of view on the desk).
Observation = /letters from letter_detector (or /model_states with
perception="truth"). Push = /push_cmd -> /push_done (push_executor), matched by id.

disturb("hand") drives the kinematic `hand` model itself through
/set_entity_state (same motion law as pa_gazebo.hand_actor: hand_logic.step_hand)
instead of publishing /hand/state for hand_actor: the stroke timing is then
deterministic and tied to the sim clock, there is no staleness timeout or
topic latency to tune, nothing else (planner_node) reacts to a fake hand, and
it works whether or not hand_actor runs. eval.launch.py therefore spawns the
hand model but does not start hand_actor.

Everything below the "pure helpers" line is plain Python and unit tested
without ROS (tools/test_recovery_gazebo.py); rclpy and the message packages are
imported only when GazeboBackend is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, "ros2_ws/src/pa_core", "ros2_ws/src/pa_gazebo", "ros2_ws/src/pa_perception"):
    _full = _p if os.path.isabs(_p) else os.path.join(_HERE, "..", _p)
    if os.path.isdir(_full) and _full not in sys.path:
        sys.path.append(_full)

from gen_dataset import park_pose, quaternion_tilt, quaternion_to_yaw, yaw_to_quaternion  # noqa: E402
from pa_core.config import load_scene  # noqa: E402
from pa_core.letters import is_disturbed, letter_polygons  # noqa: E402
from pa_gazebo.hand_logic import HandConfig, step_hand  # noqa: E402
from pa_gazebo.scene_logic import random_disturbed_pose  # noqa: E402

__all__ = ["GazeboBackend", "quaternion_to_yaw", "letter_status", "SettleDetector", "PushTracker",
           "ObservationBuffer", "HandStroke", "sample_hand_stroke", "hand_trajectory",
           "sample_teleport_pose", "letter_radius"]

PERCEPTIONS = ("detector", "truth", "classic", "cnn", "hybrid")

# ================================================================ pure helpers (no ROS)

# Gazebo hand model (pa_description/models/hand): palm box 0.075 x 0.08 centered on
# the link origin, fingers toward body +y. With the fingers trailing the stroke the
# leading edge is the palm's back edge, PALM_HALF_Y ahead of the origin.
PALM_HALF_X = 0.0375
PALM_HALF_Y = 0.04

TIPPED_RAD = 0.35   # body z axis this far from vertical -> standing on an edge / upside down
FELL_Z = -0.01      # model origin below the table top -> fell off (the desk is 2 cm lower)
UNRECOVERABLE = ("off_table", "fell", "tipped")


def letter_status(x: float, y: float, z: float, tilt: float, ws_half: float, table_half: float) -> str:
    """"ok" | "outside" (center left the workspace, still on the table) | "off_table" |
    "fell" (below the table top) | "tipped" (not lying flat)."""
    if z < FELL_Z:
        return "fell"
    if max(abs(x), abs(y)) > table_half:
        return "off_table"
    if tilt > TIPPED_RAD:
        return "tipped"
    if max(abs(x), abs(y)) > ws_half:
        return "outside"
    return "ok"


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class SettleDetector:
    """Settled = the pose stayed within (pos_tol, ang_tol) of a reference pose for
    `window_s` (sample time, e.g. sim seconds). Any larger move restarts the window."""

    def __init__(self, pos_tol: float = 0.0005, ang_tol: float = 0.01, window_s: float = 0.4):
        self.pos_tol, self.ang_tol, self.window_s = pos_tol, ang_tol, window_s
        self.ref = None
        self.ref_t = None
        self.last_t = None

    def reset(self) -> None:
        self.ref = self.ref_t = self.last_t = None

    def update(self, t: float, pose) -> bool:
        """Feed one (x, y, theta) sample at time t; returns settled()."""
        if pose is None:
            self.reset()
            return False
        if self.ref is None or self._moved(pose) or (self.last_t is not None and t < self.last_t):
            self.ref, self.ref_t = tuple(pose), t
        self.last_t = t
        return self.settled()

    def _moved(self, pose) -> bool:
        dx, dy = pose[0] - self.ref[0], pose[1] - self.ref[1]
        return math.hypot(dx, dy) > self.pos_tol or abs(wrap_angle(pose[2] - self.ref[2])) > self.ang_tol

    def settled(self) -> bool:
        return self.ref_t is not None and self.last_t - self.ref_t >= self.window_s - 1e-9


class PushTracker:
    """Push ids and their /push_done echoes. Ids start at a random-ish base so an echo
    from a previous run of the tool can never match."""

    def __init__(self, start: int | None = None):
        self.last_id = int(start if start is not None else (time.time_ns() // 1000) % 2 ** 31)
        self.done = set()

    def next_id(self) -> int:
        self.last_id = (self.last_id + 1) % 2 ** 32
        return self.last_id

    def on_done(self, push_id: int) -> None:
        self.done.add(int(push_id))
        if len(self.done) > 1000:  # keep it bounded
            self.done = {i for i in self.done if (self.last_id - i) % 2 ** 32 < 500}

    def is_done(self, push_id: int) -> bool:
        return int(push_id) in self.done


class ObservationBuffer:
    """Latest perception messages, each with its image stamp and receive time
    (both on the sim clock). `latest_after(t)` returns the newest one whose image
    was taken at or after t (receive time is used when the stamp is 0)."""

    def __init__(self, maxlen: int = 50):
        self.items = []  # (stamp, recv_t, letters)
        self.maxlen = maxlen

    def add(self, stamp: float, recv_t: float, letters: list) -> None:
        self.items.append((float(stamp), float(recv_t), list(letters)))
        if len(self.items) > self.maxlen:
            del self.items[:-self.maxlen]

    @staticmethod
    def _time(item) -> float:
        return item[0] if item[0] > 0.0 else item[1]

    def latest_after(self, t: float):
        for item in reversed(self.items):
            if self._time(item) >= t:
                return item[2]
        return None

    def clear(self) -> None:
        self.items.clear()


def sample_teleport_pose(letter: str, rng) -> tuple:
    """Random pose fully inside the workspace that counts as disturbed (same sampler as
    scene_manager). `rng` is random.Random or numpy Generator (only .uniform is used)."""
    p = random_disturbed_pose(letter, rng)
    return float(p.x), float(p.y), float(p.theta)


def letter_radius(letter: str) -> float:
    """Largest distance of the letter outline from its model origin."""
    return max(float(math.hypot(px, py)) for poly in letter_polygons(letter) for px, py in poly)


@dataclass(frozen=True)
class HandStroke:
    start: tuple   # (x, y) where the hand comes down
    end: tuple     # (x, y) where it lifts off
    yaw: float     # hand model yaw (fingers trailing the motion)
    heading: float  # stroke direction [rad]
    offset: float  # signed miss distance of the letter origin from the stroke line [m]


def sample_hand_stroke(letter: str, pose, rng, limit: float, max_offset: float = 0.02,
                       travel=(0.005, 0.04), margin: float = 0.006, tries: int = 200) -> HandStroke:
    """A straight hand stroke through the letter at `pose`.

    The palm comes down `margin` clear of the letter (bounding circle) behind it,
    slides along a random heading past the letter origin by a random `travel`
    and lifts off. The line misses the origin by up to `max_offset` so the letter
    also rotates. Hand positions stay within |x|,|y| <= limit (like hand_actor);
    headings whose landing point would not fit are re-drawn.
    """
    x0, y0 = float(pose[0]), float(pose[1])
    s0 = letter_radius(letter) + PALM_HALF_Y + margin
    best = None
    for _ in range(tries):
        h = float(rng.uniform(-math.pi, math.pi))
        d = float(rng.uniform(-max_offset, max_offset))
        u = (math.cos(h), math.sin(h))
        n = (-u[1], u[0])
        sx, sy = x0 - s0 * u[0] + d * n[0], y0 - s0 * u[1] + d * n[1]
        over = max(abs(sx), abs(sy)) - limit
        s1 = float(rng.uniform(*travel))
        ex = min(limit, max(-limit, x0 + s1 * u[0] + d * n[0]))
        ey = min(limit, max(-limit, y0 + s1 * u[1] + d * n[1]))
        stroke = HandStroke((sx, sy), (ex, ey), wrap_angle(h + math.pi / 2), h, d)
        if over <= 0.0:
            return stroke
        if best is None or over < best[0]:
            best = (over, stroke)
    over, st = best  # nothing fits (letter near the edge): clip the landing point inward
    sx = min(limit, max(-limit, st.start[0]))
    sy = min(limit, max(-limit, st.start[1]))
    return HandStroke((sx, sy), st.end, st.yaw, st.heading, st.offset)


def hand_trajectory(start, stroke: HandStroke, dt: float, cfg: HandConfig, max_steps: int = 20000) -> list:
    """[(pos, vel)] every dt: fly in at z_up, come down at stroke.start, slide to
    stroke.end, rise and park (hand_logic.step_hand, the hand_actor motion law)."""
    goals = [(stroke.start[0], stroke.start[1], cfg.z_down), (stroke.end[0], stroke.end[1], cfg.z_down),
             tuple(cfg.park)]
    out, pos = [], tuple(float(v) for v in start)
    for goal in goals:
        for _ in range(max_steps):
            if all(abs(a - b) < 1e-9 for a, b in zip(pos, goal)):
                break
            pos, vel = step_hand(pos, goal, dt, cfg)
            out.append((pos, vel))
    return out


def stroke_cfg(limit: float, vmax: float) -> HandConfig:
    """hand_actor defaults, with the stroke speed and the workspace limit."""
    return HandConfig(limit=limit, vmax=vmax)


# ================================================================ ROS backend


class GazeboBackend:
    """Backend over a running Gazebo sim (see module doc). All waits are on the sim
    clock with a wall-clock cap, so a slow (RTF < 1) Gazebo is fine."""

    def __init__(self, perception: str = "detector", settle_s: float = 0.4, settle_timeout_s: float = 15.0,
                 push_timeout_s: float = 30.0, obs_timeout_s: float = 3.0, hand_vmax: float = 0.15,
                 hand_rate_hz: float = 30.0, hand_tries: int = 4, hand_entity: str = "hand",
                 startup_timeout_s: float = 60.0, node_name: str = "pa_eval_recovery", log=print, **unused):
        if perception not in PERCEPTIONS:
            raise ValueError(f"perception: {' | '.join(PERCEPTIONS)}")
        self.log = log or (lambda *a, **k: None)
        if unused:
            self.log(f"[gazebo] ignoring options {sorted(unused)}")
        import rclpy
        from rclpy.callback_groups import ReentrantCallbackGroup
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.parameter import Parameter
        from gazebo_msgs.msg import ModelStates
        from gazebo_msgs.srv import SetEntityState, SpawnEntity
        from pa_interfaces.msg import LetterStateArray, PushCommand
        from rcl_interfaces.srv import SetParameters

        self._rclpy = rclpy
        self.scene = load_scene()
        self.ws_half = float(self.scene["workspace"]["half_size"])
        self.table_half = float(self.scene["workspace"]["table_half_size"])
        self.max_angle = float(self.scene["pushers"]["max_angle"])
        self.n_sides = len(self.scene["pushers"]["list"])
        self.perception = perception
        self.settle_s, self.settle_timeout = float(settle_s), float(settle_timeout_s)
        self.push_timeout, self.obs_timeout = float(push_timeout_s), float(obs_timeout_s)
        self.hand_cfg = stroke_cfg(self.ws_half, float(hand_vmax))
        self.hand_dt = 1.0 / float(hand_rate_hz)
        self.hand_tries = int(hand_tries)
        self.hand_entity = hand_entity
        self.letter = None
        self.last_disturb = {}

        self._cv = threading.Condition()
        self._models = {}           # name -> (x, y, z, qx, qy, qz, qw)
        self._models_t = -1.0       # sim time of the last /model_states
        self._obs = ObservationBuffer()
        self._push = PushTracker()
        self._obs_after = 0.0       # observations must be at least this new (sim time)

        self._own_rclpy = not rclpy.ok()
        if self._own_rclpy:
            rclpy.init()
        self.node = Node(node_name, parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)])
        cbg = ReentrantCallbackGroup()
        n = self.node
        n.create_subscription(ModelStates, "/model_states", self._on_models, 10, callback_group=cbg)
        n.create_subscription(LetterStateArray, "/letters", self._on_letters, 10, callback_group=cbg)
        n.create_subscription(PushCommand, "/push_done", self._on_push_done, 10, callback_group=cbg)
        self._PushCommand = PushCommand
        self._push_pub = n.create_publisher(PushCommand, "/push_cmd", 10)
        self._set_cli = n.create_client(SetEntityState, "/set_entity_state", callback_group=cbg)
        self._spawn_cli = n.create_client(SpawnEntity, "/spawn_entity", callback_group=cbg)
        self._param_cli = n.create_client(SetParameters, "/letter_detector/set_parameters", callback_group=cbg)
        self._SetEntityState, self._SpawnEntity, self._SetParameters = SetEntityState, SpawnEntity, SetParameters

        self._executor = MultiThreadedExecutor(num_threads=4)
        self._executor.add_node(n)
        self._thread = threading.Thread(target=self._spin, name="gazebo_backend_spin", daemon=True)
        self._thread.start()
        try:
            self._startup(float(startup_timeout_s))
        except Exception:
            self.close()
            raise

    # ------------------------------------------------------------ plumbing
    def _spin(self):
        try:
            self._executor.spin()
        except Exception as e:  # noqa: BLE001 - shutdown races
            if self._rclpy.ok():
                self.log(f"[gazebo] executor stopped: {e}")

    def now(self) -> float:
        return self.node.get_clock().now().nanoseconds * 1e-9

    def _on_models(self, msg):
        d = {}
        for name, p in zip(msg.name, msg.pose):
            q = p.orientation
            d[name] = (p.position.x, p.position.y, p.position.z, q.x, q.y, q.z, q.w)
        with self._cv:
            self._models, self._models_t = d, self.now()
            self._cv.notify_all()

    def _on_letters(self, msg):
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        letters = [(s.letter, float(s.x), float(s.y), float(s.theta), float(s.confidence)) for s in msg.letters]
        with self._cv:
            self._obs.add(stamp, self.now(), letters)
            self._cv.notify_all()

    def _on_push_done(self, msg):
        with self._cv:
            self._push.on_done(msg.id)
            self._cv.notify_all()

    def _wait_for(self, cond, wall_timeout: float) -> bool:
        """Wait (wall seconds) until cond() is true; cond runs under the lock."""
        end = time.monotonic() + wall_timeout
        with self._cv:
            while not cond():
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.05))
            return True

    def _call(self, cli, req, timeout: float = 5.0):
        if not cli.wait_for_service(timeout_sec=timeout):
            return None
        fut = cli.call_async(req)
        ev = threading.Event()
        fut.add_done_callback(lambda _f: ev.set())
        if not ev.wait(timeout):
            fut.cancel()
            return None
        return fut.result()

    def _startup(self, timeout: float):
        self.log("[gazebo] waiting for /clock, /model_states and /set_entity_state ...")
        end = time.monotonic() + timeout
        while self.now() <= 0.0:
            if time.monotonic() > end:
                raise RuntimeError("no /clock (is Gazebo running? ros2 launch pa_gazebo eval.launch.py)")
            time.sleep(0.05)
        if not self._wait_for(lambda: bool(self._models), max(1.0, end - time.monotonic())):
            raise RuntimeError("no /model_states (gazebo_ros_state plugin missing?)")
        if not self._set_cli.wait_for_service(timeout_sec=max(1.0, end - time.monotonic())):
            raise RuntimeError("/set_entity_state not available")
        while self._push_pub.get_subscription_count() == 0:
            if time.monotonic() > end:
                raise RuntimeError("nobody subscribes /push_cmd (push_executor not running?)")
            time.sleep(0.1)
        if self.perception != "truth":
            # Wait for the detector to publish something at all (it runs on every frame).
            if not self._wait_for(lambda: bool(self._obs.items), max(1.0, end - time.monotonic())):
                raise RuntimeError("no /letters (letter_detector not running?)")
            if self.perception in ("classic", "cnn", "hybrid"):
                self._set_detector_method(self.perception)
        if self.hand_entity not in self._models_snapshot():
            self.log(f"[gazebo] no '{self.hand_entity}' model: disturb('hand') will spawn it")
        self.log("[gazebo] ready")

    def _set_detector_method(self, method: str):
        from rclpy.parameter import Parameter
        req = self._SetParameters.Request()
        req.parameters = [Parameter("method", Parameter.Type.STRING, method).to_parameter_msg()]
        res = self._call(self._param_cli, req, 10.0)
        ok = res is not None and res.results and res.results[0].successful
        self.log(f"[gazebo] letter_detector method={method}: {'ok' if ok else 'FAILED (check its log)'}")
        if not ok:
            raise RuntimeError(f"could not set letter_detector method={method}")

    def _models_snapshot(self) -> dict:
        with self._cv:
            return dict(self._models)

    def _pose_msg(self, x, y, z, yaw):
        from geometry_msgs.msg import Pose
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = yaw_to_quaternion(yaw)
        return p

    def _set_state(self, name, x, y, z, yaw, vel=(0.0, 0.0, 0.0), timeout=5.0) -> bool:
        from gazebo_msgs.msg import EntityState
        req = self._SetEntityState.Request()
        st = EntityState()
        st.name, st.reference_frame = name, "world"
        st.pose = self._pose_msg(x, y, z, yaw)
        st.twist.linear.x, st.twist.linear.y, st.twist.linear.z = (float(v) for v in vel)
        req.state = st
        res = self._call(self._set_cli, req, timeout)
        return res is not None and res.success

    def _spawn(self, name, xml, x, y, z, yaw=0.0) -> bool:
        req = self._SpawnEntity.Request()
        req.name, req.xml, req.reference_frame = name, xml, "world"
        req.initial_pose = self._pose_msg(x, y, z, yaw)
        res = self._call(self._spawn_cli, req, 20.0)
        ok = res is not None and res.success
        self.log(f"[gazebo] spawn {name}: {'ok' if ok else 'FAILED'}")
        return ok

    def _model_name(self, letter=None) -> str:
        return f"letter_{letter or self.letter}"

    def _wait_models_after(self, t: float, wall_timeout: float = 5.0) -> bool:
        return self._wait_for(lambda: self._models_t >= t, wall_timeout)

    def _wait_settled(self, what: str = "") -> bool:
        """Wait until the letter's true pose is still for settle_s of sim time; then only
        observations taken after this moment count."""
        det = SettleDetector(window_s=self.settle_s)
        end = time.monotonic() + self.settle_timeout
        last_t = -1.0
        ok = False
        while time.monotonic() < end:
            self._wait_for(lambda: self._models_t > last_t, 1.0)
            with self._cv:
                last_t = self._models_t
                m = self._models.get(self._model_name())
            pose = None if m is None else (m[0], m[1], quaternion_to_yaw(*m[3:]))
            if det.update(last_t, pose):
                ok = True
                break
        if not ok:
            self.log(f"[gazebo] {self._model_name()} did not settle within {self.settle_timeout:.0f} s {what}")
        with self._cv:
            self._obs_after = self.now()
        return ok

    def _sleep_sim(self, until_t: float, wall_cap: float = 5.0):
        end = time.monotonic() + wall_cap
        while self.now() < until_t and time.monotonic() < end:
            time.sleep(0.002)

    # ------------------------------------------------------------ contract
    def reset(self, letter: str, pose: tuple, rng=None) -> None:
        from pa_core.letters import letter_sdf
        self.letter = letter
        model = self._model_name()
        models = self._models_snapshot()
        if model not in models:
            self._spawn(model, letter_sdf(letter, model), *park_pose(0)[:2], 0.002)
            self._wait_for(lambda: model in self._models, 10.0)
            models = self._models_snapshot()
        # Park every other letter (and the hand) out of the camera view.
        others = sorted(k for k in models if k.startswith("letter_") and k != model)
        for i, name in enumerate(others):
            px, py, _ = park_pose(i + 1)
            m = models[name]
            if math.hypot(m[0] - px, m[1] - py) > 0.01 or m[2] < FELL_Z:
                self._set_state(name, px, py, 0.002, 0.0)
        hand = models.get(self.hand_entity)
        if hand is not None and math.dist(hand[:3], self.hand_cfg.park) > 0.01:
            self._set_state(self.hand_entity, *self.hand_cfg.park, 0.0)
        self._place(*pose)

    def _place(self, x, y, theta):
        if not self._set_state(self._model_name(), x, y, 0.002, theta):
            raise RuntimeError(f"/set_entity_state {self._model_name()} failed")
        self._wait_models_after(self.now())
        self._wait_settled("after placing")

    def disturb(self, mode: str, rng) -> tuple:
        if self.letter is None:
            raise RuntimeError("disturb() before reset()")
        if mode == "teleport":
            pose = sample_teleport_pose(self.letter, rng)
            self._place(*pose)
            self.last_disturb = {"mode": "teleport", "requested": pose}
            return self.true_pose()
        if mode == "hand":
            return self._disturb_hand(rng)
        raise ValueError(f"disturb mode: teleport | hand (got {mode!r})")

    def _disturb_hand(self, rng) -> tuple:
        start_pose = self.true_pose()
        for k in range(self.hand_tries):
            if k:
                self._place(*start_pose)
            stroke = sample_hand_stroke(self.letter, self.true_pose(), rng, self.hand_cfg.limit)
            self._run_stroke(stroke)
            self._wait_settled("after the hand stroke")
            p = self.true_pose()
            ok = self.letter_status() == "ok" and is_disturbed(self.letter, *p)
            self.last_disturb = {"mode": "hand", "stroke": stroke, "tries": k + 1}
            if ok:
                return p
        self.log(f"[gazebo] {self.hand_tries} hand strokes left {self.letter} undisturbed / off the "
                 "workspace; falling back to a teleport")
        p = self.disturb("teleport", rng)
        self.last_disturb["fallback_from"] = "hand"
        return p

    def _ensure_hand(self):
        if self.hand_entity in self._models_snapshot():
            return
        from ament_index_python.packages import get_package_share_directory
        path = os.path.join(get_package_share_directory("pa_description"), "models", "hand", "model.sdf")
        with open(path) as f:
            xml = f.read()
        if not self._spawn(self.hand_entity, xml, *self.hand_cfg.park):
            raise RuntimeError("could not spawn the hand model")
        self._wait_for(lambda: self.hand_entity in self._models, 10.0)

    def _run_stroke(self, stroke: HandStroke):
        self._ensure_hand()
        m = self._models_snapshot().get(self.hand_entity)
        start = tuple(m[:3]) if m is not None else tuple(self.hand_cfg.park)
        traj = hand_trajectory(start, stroke, self.hand_dt, self.hand_cfg)
        t = self.now()
        for pos, vel in traj:
            t += self.hand_dt
            self._sleep_sim(t)
            self._set_state(self.hand_entity, *pos, stroke.yaw, vel, timeout=2.0)
        # Zero twist at the park pose so the kinematic link stays put.
        self._set_state(self.hand_entity, *self.hand_cfg.park, stroke.yaw)

    def observe(self) -> list:
        """[(letter, x, y, theta, confidence)] from the newest perception output taken
        after the last reset / disturb / push had settled; [] if none arrives in time."""
        if self.perception == "truth":
            after = self._obs_after
            self._wait_models_after(after, self.obs_timeout)
            out = []
            for name, m in self._models_snapshot().items():
                if not name.startswith("letter_"):
                    continue
                tilt = quaternion_tilt(*m[3:])
                if letter_status(m[0], m[1], m[2], tilt, self.ws_half, self.table_half) in ("ok", "outside"):
                    out.append((name[len("letter_"):].split("_")[0], m[0], m[1], quaternion_to_yaw(*m[3:]), 1.0))
            return out
        with self._cv:
            after = self._obs_after
        found = {}

        def ready():
            found["v"] = self._obs.latest_after(after)
            return found["v"] is not None

        if not self._wait_for(ready, self.obs_timeout):
            self.log("[gazebo] no fresh /letters (detector stalled?)")
            return []
        return list(found["v"])

    def push(self, side: int, angle_deg: float) -> None:
        cmd = self._PushCommand()
        with self._cv:
            cmd.id = self._push.next_id()
        cmd.side = int(side) if 0 <= int(side) < 256 else 255  # out of range -> executor echoes at once
        cmd.angle_deg = float(min(max(float(angle_deg), 0.0), self.max_angle))
        cmd.header.stamp = self.node.get_clock().now().to_msg()
        self._push_pub.publish(cmd)
        if not self._wait_for(lambda: self._push.is_done(cmd.id), self.push_timeout):
            self.log(f"[gazebo] push id={cmd.id} side={side}: no /push_done within {self.push_timeout:.0f} s")
        self._wait_settled(f"after push id={cmd.id}")

    def wait(self, dt: float) -> None:
        """Let dt seconds of sim time pass (eval_recovery uses this between observations)."""
        self._sleep_sim(self.now() + float(dt), wall_cap=max(2.0, 20.0 * float(dt)))

    def true_pose(self) -> tuple:
        self._wait_for(lambda: self._model_name() in self._models, 5.0)
        m = self._models_snapshot().get(self._model_name())
        if m is None:
            raise RuntimeError(f"{self._model_name()} not in /model_states")
        return float(m[0]), float(m[1]), float(quaternion_to_yaw(*m[3:]))

    def letter_status(self) -> str:
        """letter_status() of the current letter: ok | outside | off_table | fell | tipped."""
        m = self._models_snapshot().get(self._model_name())
        if m is None:
            return "fell"
        return letter_status(m[0], m[1], m[2], quaternion_tilt(*m[3:]), self.ws_half, self.table_half)

    def out_of_workspace(self) -> bool:
        """True when the letter can not be recovered: it left the table, fell off it or
        tipped over (eval_recovery should end the trial with fail_reason=out_of_workspace).
        A letter that is merely outside the workspace but still lying on the table is not
        reported here; the final true pose decides that case."""
        return self.letter_status() in UNRECOVERABLE

    def close(self) -> None:
        try:
            self._executor.shutdown(timeout_sec=2.0)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if self._own_rclpy and self._rclpy.ok():
            self._rclpy.shutdown()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
