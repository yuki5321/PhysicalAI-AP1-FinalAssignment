"""Where the simulated hand should be, and how it gets there (pure python, no ROS).

/hand/state gives the palm center in world coordinates. While a hand is
present the kinematic hand model is lowered onto the table at that point (its
collision boxes are thicker than the letters, so it shoves them around). When
the hand goes away it rises first and then flies to a parking pose outside the
camera view. Motion is speed limited so the physics engine sees a smooth push.
"""

from dataclasses import dataclass
import math


@dataclass
class HandConfig:
    limit: float = 0.10              # clip palm target to |x|,|y| <= limit (workspace half size)
    z_down: float = 0.003            # bottom of the hand above the table when present
    z_up: float = 0.06               # travel height when entering / leaving
    park: tuple = (0.0, 0.42, 0.12)  # out of camera view, away from the display shelf
    vmax: float = 0.35               # m/s
    stale_s: float = 1.0             # treat the hand as absent if no message for this long
    xy_tol: float = 0.004            # "above the target" when closer than this


def clip(v: float, lim: float) -> float:
    return max(-lim, min(lim, v))


def hand_target(present: bool, x: float, y: float, cfg: HandConfig) -> tuple:
    if not present or not (math.isfinite(x) and math.isfinite(y)):
        return tuple(cfg.park)
    return clip(x, cfg.limit), clip(y, cfg.limit), cfg.z_down


def _move_toward(cur, goal, max_d):
    d = [g - c for g, c in zip(goal, cur)]
    n = math.sqrt(sum(v * v for v in d))
    if n <= max_d or n < 1e-12:
        return tuple(goal)
    k = max_d / n
    return tuple(c + v * k for c, v in zip(cur, d))


def _is_down(p, cfg: HandConfig) -> bool:
    return p[2] < cfg.z_up - 1e-4 and abs(p[0]) <= cfg.limit + 0.05 and abs(p[1]) <= cfg.limit + 0.05


def step_hand(cur: tuple, target: tuple, dt: float, cfg: HandConfig) -> tuple:
    """Next hand position after dt seconds, plus the velocity used (for the physics twist).

    - target on the table and the hand already down: slide straight there (this is the push).
    - target on the table, hand not down: fly at z_up (or higher) above the target, then descend.
    - target is the park pose: rise to z_up first if down, then fly there.
    """
    max_d = cfg.vmax * max(dt, 0.0)
    going_down = target[2] <= cfg.z_down + 1e-9
    if going_down:
        if _is_down(cur, cfg):
            goal = target
        else:
            dxy = math.hypot(target[0] - cur[0], target[1] - cur[1])
            if dxy > cfg.xy_tol:
                goal = (target[0], target[1], max(cur[2], cfg.z_up))
            else:
                goal = target
    else:
        if cur[2] < cfg.z_up - 1e-4:
            goal = (cur[0], cur[1], cfg.z_up)
        else:
            goal = target
    nxt = _move_toward(cur, goal, max_d)
    vel = tuple((n - c) / dt for n, c in zip(nxt, cur)) if dt > 0 else (0.0, 0.0, 0.0)
    return nxt, vel


class HandTracker:
    """Holds the latest /hand/state and decides presence with a staleness timeout."""

    def __init__(self, cfg: HandConfig):
        self.cfg = cfg
        self.present = False
        self.x = 0.0
        self.y = 0.0
        self.stamp = None

    def update(self, present: bool, x: float, y: float, now: float) -> None:
        self.present, self.x, self.y, self.stamp = bool(present), float(x), float(y), now

    def target(self, now: float) -> tuple:
        fresh = self.stamp is not None and (now - self.stamp) <= self.cfg.stale_s
        return hand_target(self.present and fresh, self.x, self.y, self.cfg)
