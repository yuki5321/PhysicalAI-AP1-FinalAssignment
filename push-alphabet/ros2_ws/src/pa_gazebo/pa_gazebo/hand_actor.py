"""hand_actor: moves the kinematic "hand" model in Gazebo from /hand/state.

While /hand/state.present is true the hand is lowered onto the table at the
palm position (world frame) and drags through the letters; when it is false
(or no message for `stale_s`) the hand rises and parks outside the camera
view. Poses and velocities are written with /set_entity_state.
"""

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
import rclpy
from rclpy.node import Node

from pa_core.config import load_scene
from pa_interfaces.msg import HandState
from pa_gazebo.hand_logic import HandConfig, HandTracker, step_hand


class HandActor(Node):
    def __init__(self):
        super().__init__("hand_actor")
        ws = load_scene()["workspace"]
        d = HandConfig(limit=ws["half_size"])
        self.declare_parameter("entity", "hand")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("vmax", d.vmax)
        self.declare_parameter("z_down", d.z_down)
        self.declare_parameter("z_up", d.z_up)
        self.declare_parameter("park", list(d.park))
        self.declare_parameter("stale_s", d.stale_s)
        self.declare_parameter("limit", d.limit)
        self.cfg = HandConfig(
            limit=float(self.get_parameter("limit").value),
            z_down=float(self.get_parameter("z_down").value),
            z_up=float(self.get_parameter("z_up").value),
            park=tuple(float(v) for v in self.get_parameter("park").value),
            vmax=float(self.get_parameter("vmax").value),
            stale_s=float(self.get_parameter("stale_s").value),
        )
        self.entity = self.get_parameter("entity").value
        self.dt = 1.0 / float(self.get_parameter("rate_hz").value)
        self.tracker = HandTracker(self.cfg)
        self.pos = tuple(self.cfg.park)
        self.pending = None
        self.wrote_still = False
        self.last_present = None

        self.cli = self.create_client(SetEntityState, "/set_entity_state")
        self.create_subscription(HandState, "/hand/state", self.on_hand, 10)
        self.create_timer(self.dt, self.tick)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_hand(self, msg: HandState):
        self.tracker.update(msg.present, msg.x, msg.y, self.now())
        if msg.present != self.last_present:
            self.last_present = msg.present
            self.get_logger().info("hand present" if msg.present else "hand gone")

    def tick(self):
        if not self.cli.service_is_ready() or (self.pending is not None and not self.pending.done()):
            return
        target = self.tracker.target(self.now())
        nxt, vel = step_hand(self.pos, target, self.dt, self.cfg)
        still = nxt == self.pos
        if still and self.wrote_still:
            return  # holding still and a zero twist was already written
        self.wrote_still = still
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.entity
        st.reference_frame = "world"
        st.pose.position.x, st.pose.position.y, st.pose.position.z = nxt
        st.pose.orientation.w = 1.0
        st.twist.linear.x, st.twist.linear.y, st.twist.linear.z = vel
        req.state = st
        self.pos = nxt
        self.pending = self.cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = HandActor()
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
