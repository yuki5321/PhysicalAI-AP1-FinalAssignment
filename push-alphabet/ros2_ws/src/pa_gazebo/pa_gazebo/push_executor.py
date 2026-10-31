"""push_executor: /push_cmd -> paddle sweep in Gazebo -> /push_done.

Commands are queued and executed one at a time. Each sweep is sent to the
ros2_control JointTrajectoryController (`paddle_controller`) as a densely sampled
trajectory (out at constant speed, hold, back) whose timing is pa_core.pushsim.sweep_profile.
The command is echoed on /push_done once the paddle is back at rest and the
letter had `settle_s` to come to a stop.
"""

from builtin_interfaces.msg import Duration
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from pa_description.urdf_gen import joint_names
from pa_interfaces.msg import PushCommand
from pa_gazebo.sweep import PushQueue


def _duration(t: float) -> Duration:
    ns = int(round(max(t, 0.0) * 1e9))
    return Duration(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class PushExecutor(Node):
    def __init__(self):
        super().__init__("push_executor")
        self.declare_parameter("settle_s", 0.3)
        self.declare_parameter("rest_tol_deg", 3.0)
        self.declare_parameter("rest_timeout_s", 2.0)
        self.declare_parameter("trajectory_topic", "/paddle_controller/joint_trajectory")
        self.declare_parameter("tick_hz", 50.0)

        self.joints = joint_names()
        self.queue = PushQueue(
            len(self.joints),
            settle_s=self.get_parameter("settle_s").value,
            rest_tol_deg=self.get_parameter("rest_tol_deg").value,
            rest_timeout=self.get_parameter("rest_timeout_s").value,
        )
        self.positions = {}

        self.traj_pub = self.create_publisher(JointTrajectory, self.get_parameter("trajectory_topic").value, 10)
        self.done_pub = self.create_publisher(PushCommand, "/push_done", 10)
        self.create_subscription(PushCommand, "/push_cmd", self.on_cmd, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_states, 10)
        self.create_timer(1.0 / self.get_parameter("tick_hz").value, self.tick)
        self.get_logger().info(f"ready, joints: {self.joints}")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_joint_states(self, msg: JointState):
        for n, p in zip(msg.name, msg.position):
            self.positions[n] = p

    def joint_angle(self, side: int):
        return self.positions.get(self.joints[side])

    def on_cmd(self, msg: PushCommand):
        if not self.queue.valid(msg):
            self.get_logger().error(f"push id={msg.id}: invalid side {msg.side}; echoing as done without moving")
            self.done_pub.publish(msg)
            return
        self.queue.push(msg)
        self.get_logger().info(f"queued push id={msg.id} side={msg.side} angle={msg.angle_deg:.1f}")

    def tick(self):
        started, finished = self.queue.tick(self.now(), self.joint_angle)
        if finished is not None:
            self.done_pub.publish(finished)
            self.get_logger().info(f"done push id={finished.id}")
        if started is not None:
            cmd, side, plan = started
            self.send(side, plan)
            self.get_logger().info(
                f"sweep id={cmd.id} {self.joints[side]} -> {plan.angle_deg:.1f} deg, {plan.total_time:.2f} s")

    def send(self, side, plan):
        traj = JointTrajectory()
        traj.joint_names = list(self.joints)  # stamp 0 = start now
        for t, q, qd in plan.waypoints():
            pt = JointTrajectoryPoint()
            pt.positions = [q if i == side else 0.0 for i in range(len(self.joints))]
            pt.velocities = [qd if i == side else 0.0 for i in range(len(self.joints))]
            pt.time_from_start = _duration(t)
            traj.points.append(pt)
        self.traj_pub.publish(traj)


def main(args=None):
    rclpy.init(args=args)
    node = PushExecutor()
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
