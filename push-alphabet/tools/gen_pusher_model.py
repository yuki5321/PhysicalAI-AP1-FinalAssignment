#!/usr/bin/env python3
"""Write the Gazebo model directory of the point pusher from scene.yaml.

    python3 tools/gen_pusher_model.py
    # -> ros2_ws/src/pa_description/models/point_pusher/{model.config,model.sdf}

Re-run after changing `point_pusher` in pa_core/scene.yaml. point_push_executor
spawns the pusher from pa_description.pusher_model at runtime, so this copy is
for manual use (Gazebo insert panel, spawn_entity.py -file ...).
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "pa_core"))
sys.path.insert(0, str(ROOT / "ros2_ws" / "src" / "pa_description"))

from pa_description.pusher_model import MODEL_NAME, model_config, pusher_sdf  # noqa: E402

DEFAULT_OUT = ROOT / "ros2_ws" / "src" / "pa_description" / "models"


def write_model(out_dir: Path) -> Path:
    d = out_dir / MODEL_NAME
    d.mkdir(parents=True, exist_ok=True)
    (d / "model.config").write_text(model_config(), encoding="utf-8")
    (d / "model.sdf").write_text(pusher_sdf(), encoding="utf-8")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    print(write_model(args.out))


if __name__ == "__main__":
    main()
