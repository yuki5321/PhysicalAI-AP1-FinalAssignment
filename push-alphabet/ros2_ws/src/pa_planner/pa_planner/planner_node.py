"""ROS 2 wrapper around `PlannerStateMachine` (topics per docs/interfaces.md).

In:  /letters (pa_interfaces/LetterStateArray), /active_letter (std_msgs/String,
     transient_local), /hand/state (pa_interfaces/HandState),
     pusher=paddles: /push_done (pa_interfaces/PushCommand)
     pusher=point:   /point_push_done (pa_interfaces/PointPushCommand)
Out: pusher=paddles: /push_cmd (pa_interfaces/PushCommand)
     pusher=point:   /point_push_cmd (pa_interfaces/PointPushCommand)
     /letter_done (std_msgs/String),
     /planner/status (std_msgs/String, state name; re-sent at 1 Hz)

Parameter `pusher` ("paddles" | "point"; "" = scene.yaml `pusher_mode`) picks the
pusher (docs/point_pusher.md); `point_speed` [m/s] goes into PointPushCommand.speed
(0 = scene.yaml point_pusher.speed).

`PushPlanner.plan` takes hundreds of ms, so it runs on a single worker thread;
its result is queued and fed back to the state machine from the timer callback,
so the state machine itself only ever runs on the executor thread.
"""

from concurrent.futures import ThreadPoolExecutor
import math
import queue

import rclpy
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy, HistoryPolicy,
                       qos_profile_sensor_data)
from std_msgs.msg import String

from pa_interfaces.msg import HandState, LetterStateArray, PointPushCommand, PushCommand
from pa_planner.state_machine import (PUSHERS, LetterDone, LetterObs, Log, Params, PlannerPool,
                                      PlannerStateMachine, PlanRequest, PushCmd, Status,
                                      default_pusher, feed_plan)


class PlannerNode(Node):
    def __init__(self):
        super().__init__("planner_node")
        defaults = Params()
        for name, value in vars(defaults).items():
            self.declare_parameter(name, value)
        self.declare_parameter("tick_hz", 20.0)
        self.declare_parameter("pusher", "")        # "" -> scene.yaml pusher_mode
        self.declare_parameter("point_speed", 0.0)  # 0 -> scene.yaml point_pusher.speed
        params = Params(**{n: self.get_parameter(n).value for n in vars(defaults)})
        self.pusher = str(self.get_parameter("pusher").value or "").strip() or default_pusher()
        if self.pusher not in PUSHERS:
            raise ValueError(f"parameter pusher must be one of {PUSHERS}, got {self.pusher!r}")
        self.point_speed = float(self.get_parameter("point_speed").value)
        self.get_logger().info(f"pusher: {self.pusher}; params: {params}")

        self.sm = PlannerStateMachine(params)
        self.pool = PlannerPool(self.pusher)
        self.worker = ThreadPoolExecutor(max_workers=1)
        self.results = queue.Queue()
        self.status = self.sm.state

        reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             history=HistoryPolicy.KEEP_LAST)

        if self.pusher == "point":
            self.pub_push = self.create_publisher(PointPushCommand, "/point_push_cmd", reliable)
        else:
            self.pub_push = self.create_publisher(PushCommand, "/push_cmd", reliable)
        self.pub_done = self.create_publisher(String, "/letter_done", reliable)
        self.pub_status = self.create_publisher(String, "/planner/status", reliable)

        # Best-effort subscriptions match both best-effort and reliable publishers.
        self.create_subscription(LetterStateArray, "/letters", self.on_letters, qos_profile_sensor_data)
        self.create_subscription(HandState, "/hand/state", self.on_hand, qos_profile_sensor_data)
        self.create_subscription(String, "/active_letter", self.on_active, latched)
        if self.pusher == "point":
            self.create_subscription(PointPushCommand, "/point_push_done", self.on_push_done, reliable)
        else:
            self.create_subscription(PushCommand, "/push_done", self.on_push_done, reliable)

        self.create_timer(1.0 / float(self.get_parameter("tick_hz").value), self.on_tick)
        self.create_timer(1.0, self.publish_status)
        self.publish_status()

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ callbacks

    def on_letters(self, msg):
        obs = [LetterObs(s.letter, s.x, s.y, s.theta, s.confidence) for s in msg.letters]
        self.run(self.sm.on_letters(obs, self.now()))

    def on_hand(self, msg):
        self.run(self.sm.on_hand(msg.present, self.now()))

    def on_active(self, msg):
        if msg.data:
            self.worker.submit(self.pool.get, msg.data)  # warm up the planner
        self.run(self.sm.on_active_letter(msg.data, self.now()))

    def on_push_done(self, msg):
        self.run(self.sm.on_push_done(msg.id, self.now()))

    def on_tick(self):
        while True:
            try:
                req, plan, err = self.results.get_nowait()
            except queue.Empty:
                break
            if err is not None:
                self.get_logger().error(f"planning failed: {err!r}")
                continue
            if hasattr(plan, "x0"):
                what = f"stroke ({plan.x0:.3f}, {plan.y0:.3f})->({plan.x1:.3f}, {plan.y1:.3f})"
            else:
                what = f"side {plan.side} angle {plan.angle_deg:.0f}"
            pred = getattr(plan, "predicted", None)
            pred = (f", predicted ({pred[0]:.3f}, {pred[1]:.3f}, {math.degrees(pred[2]):.0f} deg)"
                    if pred is not None else "")
            self.get_logger().info(
                f"plan #{req.req_id} {req.letter}: {what} "
                f"(cost {plan.cost:.2f}{', explore' if req.explore else ''}{pred})")
            self.run(feed_plan(self.sm, req, plan, self.now()))
        self.run(self.sm.tick(self.now()))

    # ------------------------------------------------------------ actions

    def _plan_job(self, req):
        try:
            self.results.put((req, self.pool.plan(req), None))
        except Exception as e:  # noqa: BLE001 - report and keep the node alive
            self.results.put((req, None, e))

    def run(self, actions):
        for a in actions:
            if isinstance(a, PlanRequest):
                self.worker.submit(self._plan_job, a)
            elif isinstance(a, PushCmd) and a.stroke is not None:
                if self.pusher != "point":
                    self.get_logger().error(f"push #{a.id}: stroke command but pusher={self.pusher}")
                    continue
                m = PointPushCommand()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "world"
                m.id = a.id
                m.x0, m.y0, m.x1, m.y1 = (float(v) for v in a.stroke)
                m.speed = self.point_speed
                self.pub_push.publish(m)
                x0, y0, x1, y1 = a.stroke
                self.get_logger().info(
                    f"push #{a.id}: stroke ({x0:.3f}, {y0:.3f})->({x1:.3f}, {y1:.3f})")
            elif isinstance(a, PushCmd):
                if self.pusher != "paddles":
                    self.get_logger().error(f"push #{a.id}: paddle command but pusher={self.pusher}")
                    continue
                m = PushCommand()
                m.header.stamp = self.get_clock().now().to_msg()
                m.id, m.side, m.angle_deg = a.id, a.side, float(a.angle_deg)
                self.pub_push.publish(m)
                self.get_logger().info(f"push #{a.id}: side {a.side} -> {a.angle_deg:.0f} deg")
            elif isinstance(a, LetterDone):
                self.pub_done.publish(String(data=a.letter))
            elif isinstance(a, Status):
                if a.state != self.status:
                    self.status = a.state
                    self.publish_status()
                self.get_logger().debug(f"state {a.state} {a.detail}")
            elif isinstance(a, Log):
                # rclpy forbids changing severity at one call site, so keep two calls.
                if a.level == "warn":
                    self.get_logger().warn(a.msg)
                else:
                    self.get_logger().info(a.msg)

    def publish_status(self):
        self.pub_status.publish(String(data=self.status))

    def destroy_node(self):
        self.worker.shutdown(wait=False, cancel_futures=True)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
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
