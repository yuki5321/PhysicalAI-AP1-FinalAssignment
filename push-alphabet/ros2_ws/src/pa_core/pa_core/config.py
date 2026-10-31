"""Load the shared scene description (scene.yaml)."""

from functools import lru_cache
from importlib import resources

import yaml


@lru_cache(maxsize=1)
def load_scene() -> dict:
    text = resources.files("pa_core").joinpath("scene.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)
