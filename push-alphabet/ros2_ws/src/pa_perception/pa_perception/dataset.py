"""CNN training / evaluation datasets on disk (no torch / ROS dependency).

Format (docs/cnn.md):

    <root>/
      meta.json          # generation settings (source, seed, count, camera, randomization)
      labels.jsonl       # one line per frame
      frames/000000.png  # 640x640 BGR

A labels.jsonl line:

    {"frame": "frames/000000.png", "source": "synth",
     "letters": [{"letter": "U", "x": 0.012, "y": -0.031, "theta": 1.234}]}

`theta` is the raw yaw (rad), not reduced by the letter symmetry. Letter dicts
may carry extra keys (the synth generator adds `occ`, the fraction of the
letter's top face hidden by a paddle / hand); readers must ignore unknown keys.
"""

import json
import os

import cv2
import numpy as np

META = "meta.json"
LABELS = "labels.jsonl"
FRAMES = "frames"


def frame_relpath(index: int) -> str:
    return f"{FRAMES}/{index:06d}.png"


def _letter_dict(item) -> dict:
    """(name, x, y, theta) or a dict with letter/x/y/theta -> a JSON-ready dict."""
    if isinstance(item, dict):
        d = dict(item)
        d["letter"] = str(d["letter"])
        for k in ("x", "y", "theta"):
            d[k] = float(d[k])
    else:
        name, x, y, th = item[:4]
        d = {"letter": str(name), "x": float(x), "y": float(y), "theta": float(th)}
    for k in ("x", "y", "theta"):
        d[k] = round(d[k], 6)
    return d


class DatasetWriter:
    """Writes frames + labels in the docs/cnn.md format.

    `meta` is stored as meta.json (the writer adds `count` on close). Frames are
    numbered in call order. Use as a context manager or call close().
    By default an existing labels.jsonl is overwritten (`append=True` continues it).
    """

    def __init__(self, root, meta=None, append=False, png_compression=3):
        self.root = str(root)
        self.meta = dict(meta or {})
        self.png_params = [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)]
        os.makedirs(os.path.join(self.root, FRAMES), exist_ok=True)
        self.count = 0
        if append and os.path.exists(os.path.join(self.root, LABELS)):
            self.count = len(load_dataset(self.root, check_files=False))
            old = os.path.join(self.root, META)
            if os.path.exists(old) and not self.meta:
                with open(old, encoding="utf-8") as f:
                    self.meta = json.load(f)
        self._labels = open(os.path.join(self.root, LABELS), "a" if append else "w", encoding="utf-8")
        self._write_meta()

    def _write_meta(self):
        meta = dict(self.meta)
        meta["count"] = self.count
        tmp = os.path.join(self.root, META + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        os.replace(tmp, os.path.join(self.root, META))

    def add(self, frame_bgr, letters, source, png_bytes=None) -> int:
        """Save one frame and its labels; returns the frame index.

        `frame_bgr`: HxWx3 uint8 (or None when `png_bytes`, an already encoded
        PNG, is given). `letters`: iterable of (name, x, y, theta) or dicts.
        """
        index = self.count
        rel = frame_relpath(index)
        path = os.path.join(self.root, rel)
        if png_bytes is not None:
            with open(path, "wb") as f:
                f.write(png_bytes)
        else:
            img = np.asarray(frame_bgr)
            if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] != 3:
                raise ValueError(f"frame must be HxWx3 uint8 BGR, got {img.dtype} {img.shape}")
            if not cv2.imwrite(path, img, self.png_params):
                raise IOError(f"could not write {path}")
        rec = {"frame": rel, "source": str(source), "letters": [_letter_dict(it) for it in letters]}
        self._labels.write(json.dumps(rec) + "\n")
        self.count += 1
        return index

    def flush(self):
        self._labels.flush()
        self._write_meta()

    def close(self):
        if self._labels is not None and not self._labels.closed:
            self._labels.close()
            self._write_meta()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def load_meta(root) -> dict:
    path = os.path.join(str(root), META)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(root, check_files=True) -> list:
    """All records of a dataset: dicts with `frame` (relative path), `path`
    (absolute path), `source`, `letters` (list of dicts letter/x/y/theta[/...])."""
    root = os.path.abspath(str(root))
    out = []
    with open(os.path.join(root, LABELS), encoding="utf-8") as f:
        for n, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["path"] = os.path.join(root, rec["frame"])
            if check_files and not os.path.exists(rec["path"]):
                raise FileNotFoundError(f"{LABELS}:{n + 1}: missing {rec['path']}")
            out.append(rec)
    return out


def read_frame(record) -> np.ndarray:
    img = cv2.imread(record["path"], cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {record['path']}")
    return img


def iter_frames(root, records=None):
    """Yield (bgr_image, record) for every frame (or the given records)."""
    for rec in (records if records is not None else load_dataset(root)):
        yield read_frame(rec), rec


def letter_tuples(record) -> list:
    """[(name, x, y, theta), ...] of a record (the synth.render_scene input form)."""
    return [(d["letter"], d["x"], d["y"], d["theta"]) for d in record["letters"]]
