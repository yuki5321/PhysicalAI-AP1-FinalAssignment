"""scene_manager: plays the human's role around the robot.

- Spawns the letters one at a time in scene.yaml `sequence` order, each at a
  random "disturbed" pose inside the workspace, and publishes it on
  /active_letter (transient_local).
- On /letter_done moves that letter onto the display shelf (slot given by
  `display_word`), waits `spawn_delay_s` and spawns the next one.
- After the last letter publishes "" on /active_letter and lets it snow:
  small white spheres fall on the table, the sun turns cold blue and a spot
  light comes on.

Uses the gazebo_ros services /spawn_entity, /set_entity_state and
/set_light_properties (factory, state and properties plugins in the world).
"""

import random

from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState, SetLightProperties, SpawnEntity
from geometry_msgs.msg import Pose
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from pa_core.config import load_scene
from pa_core.letters import letter_sdf
from pa_gazebo import scene_logic as sl


def make_pose(x, y, z, theta=0.0) -> Pose:
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    qx, qy, qz, qw = sl.yaw_to_quaternion(theta)
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = qx, qy, qz, qw
    return p


class SceneManager(Node):
    def __init__(self):
        super().__init__("scene_manager")
        scene = load_scene()
        self.declare_parameter("seed", -1)
        self.declare_parameter("sequence", "")
        self.declare_parameter("display_word", "")
        self.declare_parameter("spawn_delay_s", 1.0)
        self.declare_parameter("spawn_z", 0.002)
        self.declare_parameter("snow_count", 80)
        self.declare_parameter("snow_per_tick", 2)
        self.declare_parameter("sun_name", "sun")

        seed = int(self.get_parameter("seed").value)
        self.rng = random.Random(seed if seed >= 0 else None)
        sequence = self.get_parameter("sequence").value or scene["sequence"]
        word = self.get_parameter("display_word").value or scene["display_word"]
        self.seq = sl.Sequencer(sequence)
        self.shelf = sl.ShelfAllocator(word)
        self.word = word
        self.table_half = scene["workspace"]["table_half_size"]
        self.spawn_delay = float(self.get_parameter("spawn_delay_s").value)
        self.spawn_z = float(self.get_parameter("spawn_z").value)
        self.snow_total = int(self.get_parameter("snow_count").value)
        self.snow_per_tick = int(self.get_parameter("snow_per_tick").value)

        self.spawn_cli = self.create_client(SpawnEntity, "/spawn_entity")
        self.state_cli = self.create_client(SetEntityState, "/set_entity_state")
        self.light_cli = self.create_client(SetLightProperties, "/set_light_properties")

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.active_pub = self.create_publisher(String, "/active_letter", qos)
        self.create_subscription(String, "/letter_done", self.on_letter_done, 10)

        self.stage = "wait_services"
        self.next_spawn_at = None
        self.snow_left = []
        self.snow_index = 0
        self.create_timer(0.1, self.tick)
        self.get_logger().info(f"sequence={sequence!r} display_word={word!r}")

    # ---------------------------------------------------------------- helpers
    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def spawn(self, name, xml, pose, done=None):
        req = SpawnEntity.Request()
        req.name, req.xml, req.initial_pose, req.reference_frame = name, xml, pose, "world"
        fut = self.spawn_cli.call_async(req)

        def cb(f):
            res = f.result()
            ok = res is not None and res.success
            if not ok:
                self.get_logger().warn(f"spawn {name}: {res.status_message if res else 'no response'}")
            if done:
                done(ok)
        fut.add_done_callback(cb)

    def set_state(self, name, pose, done=None):
        req = SetEntityState.Request()
        req.state = EntityState(name=name, pose=pose, reference_frame="world")
        fut = self.state_cli.call_async(req)

        def cb(f):
            res = f.result()
            ok = res is not None and res.success
            if not ok:
                self.get_logger().warn(f"set_entity_state {name} failed")
            if done:
                done(ok)
        fut.add_done_callback(cb)

    def publish_active(self, letter: str):
        self.active_pub.publish(String(data=letter))
        self.get_logger().info(f"active letter: {letter!r}")

    # ------------------------------------------------------------------- flow
    def tick(self):
        if self.stage == "wait_services":
            if self.spawn_cli.service_is_ready() and self.state_cli.service_is_ready():
                self.stage = "starting"
                self.spawn("display_shelf", sl.shelf_sdf(self.word), make_pose(0, sl.SHELF_Y, 0))
                self.spawn_active()
        elif self.stage == "waiting_next" and self.now() >= self.next_spawn_at:
            self.spawn_active()
        elif self.stage == "snow":
            for _ in range(self.snow_per_tick):
                if not self.snow_left:
                    self.stage = "finished"
                    break
                x, y, z = self.snow_left.pop()
                name = f"snow_{self.snow_index:03d}"
                self.snow_index += 1
                self.spawn(name, sl.snow_sdf(name), make_pose(x, y, z))

    def spawn_active(self):
        self.stage = "spawning"
        letter, name = self.seq.active, self.seq.active_model
        p = sl.random_disturbed_pose(letter, self.rng)
        pose = make_pose(p.x, p.y, self.spawn_z, p.theta)
        self.get_logger().info(f"spawning {name} at x={p.x:.3f} y={p.y:.3f} theta={p.theta:.2f}")

        def placed(ok):
            self.stage = "active"
            self.publish_active(letter)

        def spawned(ok):
            if ok:
                placed(True)
            else:  # most likely it already exists (relaunch): just move it there
                self.set_state(name, pose, placed)
        self.spawn(name, letter_sdf(letter, name), pose, spawned)

    def on_letter_done(self, msg: String):
        letter = msg.data.strip().upper()
        if self.stage != "active":
            self.get_logger().warn(f"/letter_done {letter!r} ignored in stage {self.stage}")
            return
        model = self.seq.active_model
        if not self.seq.finish(letter):
            self.get_logger().warn(f"/letter_done {letter!r} but active letter is {self.seq.active!r}")
            return
        slot = self.shelf.assign(letter)
        self.stage = "moving"
        self.get_logger().info(f"{model} done -> shelf x={slot.x:.2f}")

        def moved(ok):
            if self.seq.done:
                self.publish_active("")
                self.start_snow()
            else:
                self.next_spawn_at = self.now() + self.spawn_delay
                self.stage = "waiting_next"
        self.set_state(model, make_pose(slot.x, slot.y, self.spawn_z, slot.theta), moved)

    def start_snow(self):
        self.get_logger().info("all letters done: snow!")
        self.snow_left = sl.snow_positions(self.snow_total, self.rng, self.table_half)
        self.stage = "snow"
        self.change_light()
        self.spawn("celebration_light", sl.celebration_light_sdf(), make_pose(0, 0, 0.6))

    def change_light(self):
        if not self.light_cli.service_is_ready():
            self.get_logger().warn("/set_light_properties not available; skipping sun color change")
            return
        req = SetLightProperties.Request()
        req.light_name = self.get_parameter("sun_name").value
        # Field set differs a little between gazebo_msgs versions; set what exists.
        values = {
            "cast_shadows": False,
            "attenuation_constant": 0.9,
            "attenuation_linear": 0.01,
            "attenuation_quadratic": 0.001,
        }
        for k, v in values.items():
            if hasattr(req, k):
                setattr(req, k, v)
        if hasattr(req, "diffuse"):
            req.diffuse.r, req.diffuse.g, req.diffuse.b, req.diffuse.a = 0.45, 0.55, 0.85, 1.0
        if hasattr(req, "specular"):
            req.specular.r, req.specular.g, req.specular.b, req.specular.a = 0.1, 0.1, 0.15, 1.0
        if hasattr(req, "direction"):
            req.direction.x, req.direction.y, req.direction.z = -0.3, 0.2, -0.9
        if hasattr(req, "pose"):
            req.pose = make_pose(0, 0, 10)
        self.light_cli.call_async(req)


def main(args=None):
    rclpy.init(args=args)
    node = SceneManager()
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
