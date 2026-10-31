"""Letter sequencing, random start poses, display shelf and snow (pure python, no ROS).

The display shelf sits in front of the table (world -y), outside the camera
view and out of every paddle's reach, so finished letters never show up in
perception. Its top surface is level with the table top (z = 0).
"""

from dataclasses import dataclass
import math
import random

import numpy as np

from pa_core.config import load_scene
from pa_core.letters import is_disturbed, letter_polygons

SHELF_Y = -0.32            # shelf center line [m]
SHELF_SPACING = 0.10       # distance between letter centers on the shelf
SHELF_DEPTH = 0.13
SHELF_THICKNESS = 0.02
SPAWN_MARGIN = 0.005       # keep spawned letters this far inside the workspace
SNOW_RADIUS = 0.004


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    theta: float


def model_names(sequence: str) -> list:
    """Gazebo model name for each entry of `sequence` (suffix only for repeated letters)."""
    names, seen = [], {}
    for ch in sequence:
        k = seen.get(ch, 0)
        names.append(f"letter_{ch}" if k == 0 else f"letter_{ch}_{k}")
        seen[ch] = k + 1
    return names


def inside_workspace(letter: str, pose: Pose2D, margin: float = SPAWN_MARGIN, scene=None) -> bool:
    scene = scene or load_scene()
    lim = scene["workspace"]["half_size"] - margin
    pts = np.vstack(letter_polygons(letter, pose.x, pose.y, pose.theta))
    return bool(np.all(np.abs(pts) <= lim))


def random_disturbed_pose(letter: str, rng: random.Random, margin: float = SPAWN_MARGIN,
                          max_tries: int = 2000, scene=None) -> Pose2D:
    """Uniform random pose fully inside the workspace that counts as "disturbed"."""
    scene = scene or load_scene()
    lim = scene["workspace"]["half_size"] - margin
    for _ in range(max_tries):
        p = Pose2D(rng.uniform(-lim, lim), rng.uniform(-lim, lim), rng.uniform(-math.pi, math.pi))
        if inside_workspace(letter, p, margin, scene) and is_disturbed(letter, p.x, p.y, p.theta):
            return p
    raise RuntimeError(f"no disturbed start pose found for {letter!r}")


def shelf_slots(display_word: str) -> list:
    """(x, y) of each display_word position, left to right, centered on x = 0."""
    n = len(display_word)
    return [((i - (n - 1) / 2) * SHELF_SPACING, SHELF_Y) for i in range(n)]


def shelf_size(display_word: str) -> tuple:
    return (max(1, len(display_word)) * SHELF_SPACING + 0.04, SHELF_DEPTH, SHELF_THICKNESS)


class ShelfAllocator:
    """Assigns each finished letter to its slot in display_word (first free occurrence).
    Letters not in display_word get extra slots to the right."""

    def __init__(self, display_word: str):
        self.word = display_word
        self.slots = shelf_slots(display_word)
        self.used = set()
        self.extra = 0

    def assign(self, letter: str) -> Pose2D:
        for i, ch in enumerate(self.word):
            if ch == letter and i not in self.used:
                self.used.add(i)
                x, y = self.slots[i]
                return Pose2D(x, y, 0.0)
        self.extra += 1
        n = len(self.word)
        return Pose2D((n - 1) / 2 * SHELF_SPACING + self.extra * SHELF_SPACING, SHELF_Y, 0.0)


class Sequencer:
    """Which letter is active. `finish(letter)` returns True if it matched the active one."""

    def __init__(self, sequence: str):
        self.sequence = sequence
        self.names = model_names(sequence)
        self.index = 0

    @property
    def done(self) -> bool:
        return self.index >= len(self.sequence)

    @property
    def active(self) -> str:
        return "" if self.done else self.sequence[self.index]

    @property
    def active_model(self) -> str:
        return "" if self.done else self.names[self.index]

    def finish(self, letter: str) -> bool:
        if self.done or letter.strip().upper() != self.active:
            return False
        self.index += 1
        return True


def snow_positions(n: int, rng: random.Random, half: float, z_min: float = 0.15, z_max: float = 0.40) -> list:
    return [(rng.uniform(-half, half), rng.uniform(-half, half), rng.uniform(z_min, z_max)) for _ in range(n)]


def yaw_to_quaternion(theta: float) -> tuple:
    """(x, y, z, w) for a rotation about +z."""
    return 0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)


def shelf_sdf(display_word: str, name: str = "display_shelf") -> str:
    """Static shelf whose model origin is the center of its top surface; spawn at (0, SHELF_Y, 0)."""
    sx, sy, sz = shelf_size(display_word)
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <static>true</static>
    <link name="shelf">
      <collision name="c">
        <pose>0 0 {-sz / 2:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision>
      <visual name="v">
        <pose>0 0 {-sz / 2:.4f} 0 0 0</pose>
        <geometry><box><size>{sx:.4f} {sy:.4f} {sz:.4f}</size></box></geometry>
        <material><ambient>0.35 0.22 0.12 1</ambient><diffuse>0.45 0.3 0.18 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""


def snow_sdf(name: str, radius: float = SNOW_RADIUS) -> str:
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{name}">
    <link name="flake">
      <inertial><mass>0.0005</mass>
        <inertia><ixx>3e-9</ixx><iyy>3e-9</iyy><izz>3e-9</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
      </inertial>
      <velocity_decay><linear>0.02</linear><angular>0.5</angular></velocity_decay>
      <collision name="c">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <surface>
          <friction><ode><mu>1.0</mu><mu2>1.0</mu2></ode></friction>
          <bounce><restitution_coefficient>0</restitution_coefficient></bounce>
        </surface>
      </collision>
      <visual name="v">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
        <material><ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><emissive>0.6 0.6 0.65 1</emissive></material>
      </visual>
    </link>
  </model>
</sdf>
"""


def celebration_light_sdf(name: str = "celebration_light") -> str:
    """Spot light pointing down; spawn it with an initial pose about 0.6 m above the table."""
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <light name="{name}" type="spot">
    <cast_shadows>false</cast_shadows>
    <diffuse>0.55 0.7 1.0 1</diffuse>
    <specular>0.2 0.2 0.3 1</specular>
    <attenuation><range>3</range><constant>0.4</constant><linear>0.05</linear><quadratic>0.0</quadratic></attenuation>
    <direction>0 0 -1</direction>
    <spot><inner_angle>0.5</inner_angle><outer_angle>0.9</outer_angle><falloff>1.0</falloff></spot>
  </light>
</sdf>
"""
