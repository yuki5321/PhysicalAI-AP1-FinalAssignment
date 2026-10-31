"""ROS 2 node: overhead camera image -> /letters (pa_interfaces/LetterStateArray).

Parameters (all can be changed at runtime with `ros2 param set`):
  image_topic        /overhead_camera/image_raw
  letters_topic      /letters
  debug_topic        /perception/debug_image (published only while subscribed)
  publish_debug      true
  frame_id           world
  threshold          -1 (auto: Otsu over the table, >= median + min_contrast) or 0..255
  min_contrast, max_threshold, open_ksize, workspace_margin, sides ("auto"|"top"|"extruded"),
  min_iou, report_unknown, area_min_ratio, area_max_ratio, coarse_step_deg,
  fine_step_deg, refine_iters  -- see pa_perception.classic.DetectorParams
  method             classic | cnn | hybrid (default classic). cnn / hybrid need
                     onnxruntime and the ONNX model; if either is missing the node
                     warns and falls back to classic (see pa_perception.cnn_detector)
  model_path         "" = <share>/pa_perception/models/letter_cnn.onnx
  min_confidence     0.5 (cnn: drop detections whose softmax max is lower)
  cnn_threads        0 (onnxruntime intra-op threads, 0 = its default)
"""

import dataclasses
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rcl_interfaces.msg import SetParametersResult
from sensor_msgs.msg import Image

from pa_interfaces.msg import LetterState, LetterStateArray
from pa_perception.classic import SEGMENTERS, DetectorParams, draw_detections
from pa_perception.cnn_detector import METHODS, CNNDetector, make_detector

CNN_PARAMS = ("method", "model_path", "min_confidence", "cnn_threads")


class LetterDetectorNode(Node):
    def __init__(self):
        super().__init__("letter_detector")
        self.declare_parameter("image_topic", "/overhead_camera/image_raw")
        self.declare_parameter("letters_topic", "/letters")
        self.declare_parameter("debug_topic", "/perception/debug_image")
        self.declare_parameter("publish_debug", True)
        self.declare_parameter("frame_id", "world")
        defaults = DetectorParams()
        for f in dataclasses.fields(DetectorParams):
            self.declare_parameter(f.name, getattr(defaults, f.name))

        self.declare_parameter("method", "classic")
        self.declare_parameter("model_path", "")
        self.declare_parameter("min_confidence", 0.5)
        self.declare_parameter("cnn_threads", 0)

        try:
            self.det = self._build(self._read_params(), {})
        except ValueError as e:
            self.get_logger().warn(f"{e}; using classic")
            self.det = self._build(self._read_params(), {"method": "classic"})
        self.add_on_set_parameters_callback(self._on_params)
        self.bridge = CvBridge()
        self.frame_id = self.get_parameter("frame_id").value
        self.publish_debug = self.get_parameter("publish_debug").value

        self.pub = self.create_publisher(
            LetterStateArray, self.get_parameter("letters_topic").value, 10)
        self.pub_dbg = self.create_publisher(Image, self.get_parameter("debug_topic").value, 1)
        self.sub = self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._on_image, qos_profile_sensor_data)
        self._times = []
        self._size_warned = False
        self.get_logger().info(
            f"letter_detector ready: {self.get_parameter('image_topic').value} -> "
            f"{self.get_parameter('letters_topic').value}")

    def _build(self, params, overrides):
        """Detector for the current (or overridden) method / model parameters."""
        def get(name):
            return overrides[name] if name in overrides else self.get_parameter(name).value

        method = str(get("method"))
        if method not in METHODS:
            raise ValueError(f"method: {' | '.join(METHODS)}")
        det, used = make_detector(
            method, model_path=str(get("model_path")) or None, params=params,
            min_confidence=float(get("min_confidence")), threads=int(get("cnn_threads")),
            warn=self.get_logger().warn)
        if used != "classic":
            self.get_logger().info(f"method {used}: model {det.model_path}")
        else:
            self.get_logger().info("method classic")
        return det

    def _read_params(self, overrides=None):
        vals = {}
        for f in dataclasses.fields(DetectorParams):
            v = self.get_parameter(f.name).value
            if overrides and f.name in overrides:
                v = overrides[f.name]
            vals[f.name] = type(getattr(DetectorParams(), f.name))(v)
        return DetectorParams(**vals)

    def _on_params(self, params):
        overrides = {p.name: p.value for p in params}
        names = {f.name for f in dataclasses.fields(DetectorParams)}
        try:
            if "frame_id" in overrides:
                self.frame_id = overrides["frame_id"]
            if "publish_debug" in overrides:
                self.publish_debug = bool(overrides["publish_debug"])
            if names & overrides.keys():
                new = self._read_params(overrides)
                if new.sides not in ("auto", "top", "extruded"):
                    return SetParametersResult(successful=False, reason="sides: auto|top|extruded")
                if new.segmenter not in SEGMENTERS:
                    return SetParametersResult(successful=False, reason="segmenter: " + "|".join(SEGMENTERS))
            else:
                new = self.det.p
            if set(CNN_PARAMS) & overrides.keys():
                self.det = self._build(new, overrides)
            elif names & overrides.keys():
                if new.coarse_step_deg != self.det.p.coarse_step_deg or new.open_ksize != self.det.p.open_ksize:
                    self.det = self._build(new, overrides)  # rebuilds templates / kernel
                elif isinstance(self.det, CNNDetector):
                    self.det.params = new
                else:
                    self.det.p = new
        except (TypeError, ValueError) as e:
            return SetParametersResult(successful=False, reason=str(e))
        return SetParametersResult(successful=True)

    def _on_image(self, msg: Image):
        t0 = time.perf_counter()
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:  # noqa: BLE001 - bad encoding etc.
            self.get_logger().error(f"cv_bridge: {e}", throttle_duration_sec=5.0)
            return
        cam = self.det.cam
        if img.shape[1] != cam.width_px or img.shape[0] != cam.height_px:
            if not self._size_warned:
                self.get_logger().warn(
                    f"image {img.shape[1]}x{img.shape[0]} != scene.yaml camera "
                    f"{cam.width_px}x{cam.height_px}; resizing (poses assume scene.yaml optics)")
                self._size_warned = True
            img = cv2.resize(img, (cam.width_px, cam.height_px), interpolation=cv2.INTER_AREA)

        dets = self.det.detect(img)
        out = LetterStateArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id
        for d in dets:
            s = LetterState()
            s.letter = d.letter
            s.x, s.y, s.theta = d.x, d.y, d.theta
            s.confidence = float(d.confidence)
            out.letters.append(s)
        self.pub.publish(out)

        if self.publish_debug and self.pub_dbg.get_subscription_count() > 0:
            dbg = self.bridge.cv2_to_imgmsg(draw_detections(img, dets, self.det), encoding="bgr8")
            dbg.header = msg.header
            self.pub_dbg.publish(dbg)

        self._times.append(time.perf_counter() - t0)
        if len(self._times) >= 100:
            ms = 1000.0 * sum(self._times) / len(self._times)
            self.get_logger().info(f"{len(dets)} letters, {ms:.1f} ms/frame (avg of 100)")
            self._times.clear()


def main(args=None):
    rclpy.init(args=args)
    node = LetterDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
