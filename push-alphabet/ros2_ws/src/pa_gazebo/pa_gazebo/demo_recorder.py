"""demo_recorder: records the overhead camera with a HUD to MP4; auto_hand: scripted /hand/state.

demo_recorder
  In:  /overhead_camera/image_raw, /letters, /planner/status, /active_letter
       (transient_local), /letter_done, /hand/state, /push_cmd, /push_done,
       /point_push_cmd, /point_push_done (point pusher: stroke drawn as an arrow)
  Out: <out_dir>/<prefix>_<YYYYmmdd_HHMMSS>.mp4      annotated (HUD + progress panel)
       <out_dir>/<prefix>_<YYYYmmdd_HHMMSS>_raw.mp4  plain camera frames (write_raw)
       Streamed to ffmpeg as fragmented H.264 MP4 (cv2 mp4v if ffmpeg is
       missing), so the file stays playable even if the node is killed.
  Service: ~/set_recording (std_srvs/SetBool): true starts a new pair of files,
       false closes them.
  Frames are written at `fps` in *sim time*: one video frame per 1/fps of
  camera stamp, duplicating frames over gaps, so the video plays at the speed
  of the simulated world even if Gazebo runs slower than real time.

auto_hand
  Publishes /hand/state at `rate_hz`: a scripted sweep through the active
  letter while the planner is recovering it (pa_gazebo.demo_overlay.ScriptedHand),
  otherwise present=false. mode:=auto stays silent while any other node
  (rosbridge <- host/hand_bridge.py) publishes /hand/state; mode:=always never
  yields.
"""

from datetime import datetime
import os
import random
import shutil
import signal
import subprocess

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool

from pa_core.config import load_scene
from pa_gazebo.demo_overlay import (HudState, KnockConfig, LetterPose, ScriptedHand, compose_frame,
                                    even_size, image_to_bgr)
from pa_interfaces.msg import HandState, LetterStateArray, PointPushCommand, PushCommand
from pa_perception.camera_model import CameraModel

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
AUTO_HAND_NODE = "auto_hand"


def default_out_dir() -> str:
    env = os.environ.get("PA_DEMO_DIR")
    if env:
        return env
    repo = "/ws/src/push-alphabet"   # bind mount of the repository (docker/compose.yaml)
    if os.path.isdir(repo):
        return os.path.join(repo, "videos")
    return os.path.expanduser("~/pa_demo")


def stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


class VideoFile:
    """H.264 MP4 streamed through an ffmpeg pipe (falls back to cv2 mp4v).

    The MP4 is fragmented, so whatever was written stays playable even if this
    node is killed (closed terminal, lost noVNC session, SIGKILL): ffmpeg runs in
    its own session and finishes the file as soon as its stdin closes.
    """

    def __init__(self, path: str, fps: float, h264: bool, logger):
        self.path = path
        self.fps = fps
        self.h264 = h264 and shutil.which("ffmpeg") is not None
        self.log = logger
        self.proc = None
        self.writer = None
        self.size = None
        self.frames = 0

    def _open(self, w: int, h: int):
        self.size = (w, h)
        if self.h264:
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{self.fps:g}",
                   "-i", "pipe:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                   "-pix_fmt", "yuv420p", "-g", str(max(1, int(self.fps))),
                   "-movflags", "+frag_keyframe+empty_moov+default_base_moof", self.path]
            try:
                # Own session: the launch's SIGINT/SIGTERM/SIGHUP never reach ffmpeg.
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, start_new_session=True,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except OSError as e:
                self.log.warn(f"ffmpeg failed ({e}); writing mp4v with OpenCV")
                self.h264 = False
        self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self.size)
        if not self.writer.isOpened():
            self.log.error(f"cannot open video writer {self.path}")

    def write(self, frame, n: int = 1):
        frame = even_size(frame)
        if self.size is None:
            self._open(frame.shape[1], frame.shape[0])
        if (frame.shape[1], frame.shape[0]) != self.size:
            frame = cv2.resize(frame, self.size)
        for _ in range(n):
            if self.proc is not None:
                try:
                    self.proc.stdin.write(frame.tobytes())
                except (BrokenPipeError, ValueError):
                    self.log.error(f"ffmpeg stopped; {self.path} ends at frame {self.frames}")
                    self.proc = None
                    return
            elif self.writer is not None:
                self.writer.write(frame)
        self.frames += n

    def close(self):
        dur = self.frames / self.fps
        if self.proc is not None:
            try:
                self.proc.stdin.close()  # ffmpeg finishes the file on EOF
            except (BrokenPipeError, ValueError):
                pass
            self.proc = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.size is not None:
            self.log.info(f"saved {self.path} ({self.frames} frames, {dur:.1f} s)")
            self.size = None


class DemoRecorder(Node):
    def __init__(self):
        super().__init__("demo_recorder")
        scene = load_scene()
        self.declare_parameter("record", True)
        self.declare_parameter("out_dir", "")
        self.declare_parameter("prefix", "demo")
        self.declare_parameter("fps", 15.0)
        self.declare_parameter("h264", True)
        self.declare_parameter("write_raw", True)
        self.declare_parameter("sync_sim_time", True)
        self.declare_parameter("max_dup", 15)
        self.declare_parameter("image_topic", "/overhead_camera/image_raw")
        self.declare_parameter("word", "")
        self.declare_parameter("label", "")
        self.declare_parameter("preview_topic", "/demo/image")

        self.fps = float(self.get_parameter("fps").value)
        self.h264 = bool(self.get_parameter("h264").value)
        self.write_raw = bool(self.get_parameter("write_raw").value)
        self.sync = bool(self.get_parameter("sync_sim_time").value)
        self.max_dup = max(1, int(self.get_parameter("max_dup").value))
        self.out_dir = self.get_parameter("out_dir").value or default_out_dir()
        self.prefix = self.get_parameter("prefix").value or "demo"
        word = self.get_parameter("word").value or scene["display_word"]
        self.z_top = float(scene["letter"]["thickness"])
        self.cam_scene = scene["camera"]
        self.cam = None

        self.hud = HudState(word=word.upper(), label=self.get_parameter("label").value)
        self.finished_flag = False
        self.inflight = set()
        self.files = None
        self.next_t = None
        self.last_hand_check = -1.0

        self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(LetterStateArray, "/letters", self.on_letters, qos_profile_sensor_data)
        self.create_subscription(String, "/planner/status", self.on_status, RELIABLE)
        self.create_subscription(String, "/active_letter", self.on_active, LATCHED)
        self.create_subscription(String, "/letter_done", self.on_letter_done, RELIABLE)
        self.create_subscription(HandState, "/hand/state", self.on_hand, qos_profile_sensor_data)
        self.create_subscription(PushCommand, "/push_cmd", self.on_push, RELIABLE)
        self.create_subscription(PushCommand, "/push_done", self.on_push_done, RELIABLE)
        self.create_subscription(PointPushCommand, "/point_push_cmd", self.on_point_push, RELIABLE)
        self.create_subscription(PointPushCommand, "/point_push_done", self.on_point_push_done, RELIABLE)
        self.preview_pub = self.create_publisher(Image, self.get_parameter("preview_topic").value, 1)
        self.create_service(SetBool, "~/set_recording", self.on_set_recording)

        self.recording = bool(self.get_parameter("record").value)
        self.get_logger().info(
            f"demo_recorder: record={self.recording} out_dir={self.out_dir} fps={self.fps} "
            f"h264={'yes' if self.h264 and shutil.which('ffmpeg') else 'no (mp4v)'}")

    # ------------------------------------------------------------- state
    def on_letters(self, msg):
        self.hud.letters = [LetterPose(s.letter, s.x, s.y, s.theta, s.confidence) for s in msg.letters]

    def on_status(self, msg):
        self.hud.status = msg.data

    def _mark_done(self, letter):
        if letter and not self.finished_flag and len(self.hud.done) < len(self.hud.word):
            self.hud.done.append(letter.upper())
        self.finished_flag = True

    def on_letter_done(self, msg):
        if self.hud.active and msg.data.strip().upper() == self.hud.active.upper():
            self._mark_done(self.hud.active)

    def on_active(self, msg):
        new = msg.data.strip().upper()
        prev = self.hud.active
        if new == prev:
            return
        if prev:  # the previous letter left the table: it was finished
            self._mark_done(prev)
        self.hud.active = new
        self.hud.pushes_letter = 0
        self.finished_flag = False

    def on_hand(self, msg):
        self.hud.hand_present = bool(msg.present)
        self.hud.hand_xy = (msg.x, msg.y) if msg.present else None

    def on_push(self, msg):
        self.hud.pushes_total += 1
        self.hud.pushes_letter += 1
        self.hud.last_push = (int(msg.side), float(msg.angle_deg))
        self.inflight.add(int(msg.id))
        self.hud.pushing = True

    def on_push_done(self, msg):
        self.inflight.discard(int(msg.id))
        self.hud.pushing = bool(self.inflight)

    def on_point_push(self, msg):
        self.hud.pushes_total += 1
        self.hud.pushes_letter += 1
        self.hud.last_stroke = (float(msg.x0), float(msg.y0), float(msg.x1), float(msg.y1))
        self.hud.last_stroke_speed = float(msg.speed)
        self.inflight.add(("point", int(msg.id)))
        self.hud.pushing = True

    def on_point_push_done(self, msg):
        self.inflight.discard(("point", int(msg.id)))
        self.hud.pushing = bool(self.inflight)

    def update_hand_source(self, now):
        if now - self.last_hand_check < 1.0 and self.last_hand_check >= 0:
            return
        self.last_hand_check = now
        try:
            names = [i.node_name for i in self.get_publishers_info_by_topic("/hand/state")]
        except Exception:  # noqa: BLE001 - purely cosmetic
            return
        if any(n != AUTO_HAND_NODE for n in names):
            self.hud.hand_source = "webcam"
        elif names:
            self.hud.hand_source = "scripted"
        else:
            self.hud.hand_source = ""

    # ---------------------------------------------------------- recording
    def on_set_recording(self, req, res):
        if req.data and not self.recording:
            self.recording = True
            res.message = "recording"
        elif not req.data and self.recording:
            self.recording = False
            self.close_files()
            res.message = "stopped"
        else:
            res.message = "unchanged"
        res.success = True
        return res

    def open_files(self):
        os.makedirs(self.out_dir, exist_ok=True)
        stem = os.path.join(self.out_dir, f"{self.prefix}_{datetime.now():%Y%m%d_%H%M%S}")
        log = self.get_logger()
        self.files = {"hud": VideoFile(stem + ".mp4", self.fps, self.h264, log)}
        if self.write_raw:
            self.files["raw"] = VideoFile(stem + "_raw.mp4", self.fps, self.h264, log)
        self.next_t = None
        self.hud.start_time = None
        log.info(f"recording -> {stem}.mp4")

    def close_files(self):
        if self.files:
            for f in self.files.values():
                f.close()
        self.files = None

    def on_image(self, msg: Image):
        try:
            img = image_to_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        except ValueError as e:
            self.get_logger().warn(str(e), throttle_duration_sec=5.0)
            return
        t = stamp_seconds(msg.header.stamp)
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        if self.cam is None or (self.cam.width_px, self.cam.height_px) != (img.shape[1], img.shape[0]):
            c = self.cam_scene
            self.cam = CameraModel(c["height"], img.shape[1], img.shape[0], c["hfov_deg"])
        self.update_hand_source(t)
        self.hud.sim_time = t

        if not self.recording:
            if self.preview_pub.get_subscription_count() > 0:
                self.publish_preview(compose_frame(img, self.hud, self.cam, self.z_top), msg)
            return
        if self.files is None:
            self.open_files()
        if self.hud.start_time is None:
            self.hud.start_time = t
        # number of video frames this camera frame stands for (sim-time sync)
        if not self.sync or self.next_t is None:
            n = 1
            self.next_t = t + 1.0 / self.fps
        else:
            if t + 0.5 / self.fps < self.next_t:
                n = 0                       # camera faster than fps: skip
            else:
                n = min(self.max_dup, 1 + int((t - self.next_t) * self.fps + 0.5))
                self.next_t += n / self.fps
                if t - self.next_t > 1.0:   # long stall (paused sim): resync
                    self.next_t = t + 1.0 / self.fps
        frame = compose_frame(img, self.hud, self.cam, self.z_top)
        if n > 0:
            self.files["hud"].write(frame, n)
            if "raw" in self.files:
                self.files["raw"].write(img, n)
        if self.preview_pub.get_subscription_count() > 0:
            self.publish_preview(frame, msg)

    def publish_preview(self, frame, src: Image):
        out = Image()
        out.header = src.header
        out.height, out.width = frame.shape[:2]
        out.encoding = "bgr8"
        out.step = frame.shape[1] * 3
        out.data = frame.tobytes()
        self.preview_pub.publish(out)

    def destroy_node(self):
        self.close_files()
        super().destroy_node()


class AutoHand(Node):
    def __init__(self):
        super().__init__(AUTO_HAND_NODE)
        d = KnockConfig()
        self.declare_parameter("mode", "auto")   # auto | always
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("seed", -1)
        self.declare_parameter("knocks_per_letter", d.knocks_per_letter)
        self.declare_parameter("after_pushes", d.after_pushes)
        self.declare_parameter("idle_s", d.idle_s)
        self.declare_parameter("start_delay_s", d.start_delay_s)
        self.declare_parameter("speed", d.speed)
        cfg = KnockConfig(
            knocks_per_letter=int(self.get_parameter("knocks_per_letter").value),
            after_pushes=int(self.get_parameter("after_pushes").value),
            idle_s=float(self.get_parameter("idle_s").value),
            start_delay_s=float(self.get_parameter("start_delay_s").value),
            speed=float(self.get_parameter("speed").value),
        )
        seed = int(self.get_parameter("seed").value)
        self.script = ScriptedHand(cfg, random.Random(seed if seed >= 0 else None))
        self.always = str(self.get_parameter("mode").value).lower() == "always"
        self.yielded = False
        self.was_present = False

        self.pub = self.create_publisher(HandState, "/hand/state", 10)
        self.create_subscription(String, "/active_letter",
                                 lambda m: self.script.on_active(m.data.strip(), self.now()), LATCHED)
        self.create_subscription(String, "/planner/status",
                                 lambda m: self.script.on_status(m.data, self.now()), RELIABLE)
        self.create_subscription(PushCommand, "/push_cmd", lambda m: self.script.on_push(self.now()), RELIABLE)
        self.create_subscription(PushCommand, "/push_done", lambda m: self.script.on_push_done(self.now()),
                                 RELIABLE)
        self.create_subscription(PointPushCommand, "/point_push_cmd", lambda m: self.script.on_push(self.now()),
                                 RELIABLE)
        self.create_subscription(PointPushCommand, "/point_push_done",
                                 lambda m: self.script.on_push_done(self.now()), RELIABLE)
        self.create_subscription(LetterStateArray, "/letters", self.on_letters, qos_profile_sensor_data)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self.tick)
        self.get_logger().info(f"auto_hand: mode={'always' if self.always else 'auto'} {cfg}")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_letters(self, msg):
        self.script.on_letters([LetterPose(s.letter, s.x, s.y, s.theta, s.confidence) for s in msg.letters],
                               self.now())

    def tick(self):
        now = self.now()
        if not self.always:
            others = self.count_publishers("/hand/state") - 1
            if others > 0:
                if not self.yielded:
                    self.get_logger().info("another /hand/state publisher (webcam bridge): scripted hand paused")
                    self.yielded = True
                    self.script.sweep = None
                return
            if self.yielded:
                self.get_logger().info("no other /hand/state publisher: scripted hand active")
                self.yielded = False
        present, x, y = self.script.step(now)
        if present != self.was_present:
            self.was_present = present
            self.get_logger().info(f"scripted hand {'down' if present else 'up'} "
                                   f"(knock {self.script.total_knocks}, letter {self.script.active!r})")
        msg = HandState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.present, msg.x, msg.y = bool(present), float(x), float(y)
        self.pub.publish(msg)


def _sigterm(signum, frame):
    raise KeyboardInterrupt


def _spin(node_cls, args=None):
    rclpy.init(args=args)
    signal.signal(signal.SIGTERM, _sigterm)  # launch escalates to SIGTERM: still close the files
    signal.signal(signal.SIGHUP, _sigterm)   # terminal / noVNC session closed
    node = node_cls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    _spin(DemoRecorder, args)


def auto_hand_main(args=None):
    _spin(AutoHand, args)


if __name__ == "__main__":
    main()
