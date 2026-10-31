"""point_push_executor: /point_push_cmd -> point pusher stroke in Gazebo -> /point_push_done.

Commands (pa_interfaces/PointPushCommand) are queued and executed one at a
time. The kinematic 'point_pusher' model (pa_description.pusher_model,
spawned at the park pose if it is not in the world yet) is moved with
/set_entity_state on sim time: park -> above start -> lower -> straight stroke
at `speed` -> lift -> park (pa_gazebo.point_stroke). Each tick writes the pose
on the path at the current sim time and the twist toward the pose one tick
later, so ODE moves the body smoothly between ticks and pushes the letter.
The command is echoed on /point_push_done once the pusher is parked and the
letter had `settle_s` to come to rest.
"""

from collections import deque
import math

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from pa_description.pusher_model import MODEL_NAME, pusher_sdf
from pa_gazebo.point_stroke import PointPusherConfig, plan_stroke, valid_command
from pa_interfaces.msg import PointPushCommand

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST)


class PointPushExecutor(Node):
    def __init__(self):
        super().__init__("point_push_executor")
        d = PointPusherConfig.from_scene()
        self.declare_parameter("entity", MODEL_NAME)
        self.declare_parameter("rate_hz", 100.0)
        self.declare_parameter("settle_s", 0.5)   # as pa_core.pointpush push(settle=0.5)
        self.declare_parameter("travel_speed", d.travel_speed)
        self.declare_parameter("vertical_speed", d.vertical_speed)
        self.declare_parameter("z_down", d.z_down)
        self.declare_parameter("z_travel", d.z_travel)
        self.declare_parameter("spawn", True)
        self.cfg = PointPusherConfig.from_scene(
            travel_speed=float(self.get_parameter("travel_speed").value),
            vertical_speed=float(self.get_parameter("vertical_speed").value),
            z_down=float(self.get_parameter("z_down").value),
            z_travel=float(self.get_parameter("z_travel").value),
        )
        self.entity = self.get_parameter("entity").value
        self.settle_s = float(self.get_parameter("settle_s").value)
        self.period = 1.0 / float(self.get_parameter("rate_hz").value)

        self.queue = deque()
        self.active = None          # (msg, plan, t_start)
        self.pos = tuple(self.cfg.park)
        self.pending = None         # SetEntityState future
        self.parked_written = False
        self.h = self.period        # measured sim-time tick interval (EMA)
        self.last_tick = None
        self.ready = not bool(self.get_parameter("spawn").value)
        self.spawn_future = None

        self.state_cli = self.create_client(SetEntityState, "/set_entity_state")
        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.done_pub = self.create_publisher(PointPushCommand, "/point_push_done", RELIABLE)
        self.create_subscription(PointPushCommand, "/point_push_cmd", self.on_cmd, RELIABLE)
        self.create_timer(self.period, self.tick)
        self.get_logger().info(
            f"ready: entity={self.entity} park={self.cfg.park} stroke speed={self.cfg.speed} m/s "
            f"travel={self.cfg.travel_speed} m/s z_down={self.cfg.z_down} z_travel={self.cfg.z_travel}")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------ commands
    def on_cmd(self, msg: PointPushCommand):
        if not valid_command(msg.x0, msg.y0, msg.x1, msg.y1, msg.speed):
            self.get_logger().error(f"point push id={msg.id}: non-finite command; echoing as done")
            self.done_pub.publish(msg)
            return
        self.queue.append(msg)
        self.get_logger().info(
            f"queued point push id={msg.id} ({msg.x0:.3f},{msg.y0:.3f}) -> ({msg.x1:.3f},{msg.y1:.3f}) "
            f"speed={msg.speed:.3f}")

    # --------------------------------------------------------------- spawn
    def try_spawn(self):
        if self.spawn_future is None:
            if not self.spawn_cli.service_is_ready():
                self.get_logger().info("waiting for /spawn_entity", throttle_duration_sec=5.0)
                return
            req = SpawnEntity.Request()
            req.name = self.entity
            req.xml = pusher_sdf(name=self.entity)
            req.reference_frame = "world"
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = self.cfg.park
            pose.orientation.w = 1.0
            req.initial_pose = pose
            self.spawn_future = self.spawn_cli.call_async(req)
            return
        if not self.spawn_future.done():
            return
        res = self.spawn_future.result()
        if res is not None and res.success:
            self.get_logger().info(f"spawned '{self.entity}' at park {self.cfg.park}")
        else:
            # Typically "entity already exists": use the model that is there.
            msg = res.status_message if res is not None else "no response"
            self.get_logger().info(f"spawn '{self.entity}': {msg} (using the existing model)")
        self.ready = True

    # ---------------------------------------------------------------- tick
    def tick(self):
        now = self.now()
        if self.last_tick is not None and now > self.last_tick:
            dt = now - self.last_tick
            self.h = min(0.2, max(0.005, 0.7 * self.h + 0.3 * dt))
        if self.last_tick is not None and now < self.last_tick - 1.0:
            self.get_logger().warn("sim time jumped back; restarting the current stroke")
            if self.active is not None:
                self.active = (self.active[0], self.active[1], now)
        self.last_tick = now

        if not self.ready:
            self.try_spawn()
            return
        if not self.state_cli.service_is_ready():
            self.get_logger().info("waiting for /set_entity_state", throttle_duration_sec=5.0)
            return
        if self.pending is not None and not self.pending.done():
            return

        if self.active is None and self.queue:
            msg = self.queue.popleft()
            plan = plan_stroke(self.pos, msg.x0, msg.y0, msg.x1, msg.y1, msg.speed, self.cfg)
            self.active = (msg, plan, now)
            length = math.hypot(plan.end[0] - plan.start[0], plan.end[1] - plan.start[1])
            note = "  (clamped)" if plan.clamped else ""
            self.get_logger().info(
                f"stroke id={msg.id} {length * 100:.1f} cm at {plan.speed * 100:.1f} cm/s, "
                f"total {plan.total_time:.2f} s{note}")
            if length > self.cfg.max_stroke + self.cfg.approach + 0.01:
                self.get_logger().warn(
                    f"stroke id={msg.id} is {length:.3f} m, longer than max_stroke {self.cfg.max_stroke} m")

        if self.active is None:
            if not self.parked_written:
                self.write(tuple(self.cfg.park), (0.0, 0.0, 0.0))
                self.parked_written = True
            return

        msg, plan, t0 = self.active
        t = now - t0
        if t >= plan.total_time:
            if not self.parked_written:
                self.write(plan.final, (0.0, 0.0, 0.0))
                self.parked_written = True
            if t >= plan.total_time + self.settle_s:
                self.done_pub.publish(msg)
                self.get_logger().info(f"done point push id={msg.id}")
                self.active = None
            return
        self.parked_written = False
        self.write(plan.pose_at(t), plan.velocity_at(t, self.h))

    def write(self, pos, vel):
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.entity
        st.reference_frame = "world"
        st.pose.position.x, st.pose.position.y, st.pose.position.z = (float(v) for v in pos)
        st.pose.orientation.w = 1.0
        st.twist.linear.x, st.twist.linear.y, st.twist.linear.z = (float(v) for v in vel)
        req.state = st
        self.pos = tuple(pos)
        self.pending = self.state_cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = PointPushExecutor()
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
