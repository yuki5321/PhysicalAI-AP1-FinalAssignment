"""Quick test for tools/assemble_video.py (needs ffmpeg/ffprobe; ~10-20 s).

    python3 -m pytest -q tools/test_assemble_video.py

Synthetic clips (numpy frames piped to ffmpeg) -> card + 4x sped-up clip with a
picture-in-picture wipe + caption -> checks duration, size, the yellow "x4" label
box (pixel check) and the wipe border.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble_video as av  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                                reason="ffmpeg not installed")


def make_clip(path, seconds, w, h, color, fps=30):
    """Moving bar on a flat background, written as H.264."""
    n = int(seconds * fps)
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
                          "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for i in range(n):
        f = np.empty((h, w, 3), np.uint8)
        f[:] = color
        x = int((i / max(n - 1, 1)) * (w - 20))
        f[:, x:x + 20] = 255 - np.array(color, np.uint8)
        p.stdin.write(f.tobytes())
    p.stdin.close()
    assert p.wait() == 0


def frame_at(path, t):
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t}", "-i", str(path), "-frames:v", "1",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(av.H, av.W, 3)


def test_assemble_speedup_pip(tmp_path):
    main = tmp_path / "main.mp4"
    cam = tmp_path / "cam.mp4"
    make_clip(main, 8.0, 640, 360, (30, 90, 160))     # 16:9, upscaled to 1280x720
    make_clip(cam, 8.0, 320, 240, (40, 140, 60))      # 4:3 wipe
    edit = {
        "base_dir": str(tmp_path),
        "fade": 0,
        "lang": "en",
        "segments": [
            {"card": "text", "big": "TEST", "duration": 1.0},
            {"clip": "main.mp4", "in": 0, "out": 4, "speed": 2, "caption": "caption",
             "pip": {"path": "cam.mp4", "corner": "bl", "width": 320, "label": "cam"}},
        ],
    }
    out, dur = av.assemble(edit, tmp_path / "out.mp4", audio="silent", preset="ultrafast")
    assert out.is_file()
    assert abs(dur - 3.0) < 0.15, dur                    # 1 s card + 4 s / 2
    assert av.probe_size(out) == (av.W, av.H)
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                              str(out)], capture_output=True, text=True).stdout.split()
    assert streams[0] == "h264" and "aac" in streams

    card = frame_at(out, 0.5)
    sped = frame_at(out, 2.0)
    # Speed label: yellow box in the top-right corner, only on the sped-up segment.
    region = (slice(40, 66), slice(av.W - 50, av.W - 36))   # right margin of the label box
    yellow = np.array(av.LABEL_BG, float)
    got = np.median(sped[region].reshape(-1, 3), axis=0)
    assert np.abs(got - yellow).max() < 25, got
    assert np.abs(np.median(card[region].reshape(-1, 3), axis=0) - yellow).max() > 60
    # The label contains dark text pixels (the "x2" glyphs), not just a flat box.
    label_box = sped[28:80, av.W - 200:av.W - 32]
    assert (label_box.mean(axis=2) < 80).sum() > 50
    # PIP: green-ish wipe content inside the bottom-left corner, white border around it.
    pip = sped[av.H - 24 - 120: av.H - 24 - 60, 24 + 40: 24 + 120]
    m = pip.reshape(-1, 3).mean(0)
    assert m[1] > m[2] + 30 and m[1] > m[0] + 30, m
    main_area = sped[300:400, 600:700].reshape(-1, 3).mean(0)
    assert main_area[2] > main_area[1]  # main clip is blue


def test_speed_above_course_limit_is_rejected(tmp_path):
    make_clip(tmp_path / "main.mp4", 2.0, 320, 180, (30, 90, 160))
    edit = {"base_dir": str(tmp_path), "segments": [{"clip": "main.mp4", "speed": 4}]}
    with pytest.raises(ValueError, match="course limit"):
        av.assemble(edit, tmp_path / "out.mp4", dry_run=True)


def test_dry_run_warns_on_length(tmp_path, capsys):
    edit = {"base_dir": str(tmp_path), "segments": [{"card": "title", "duration": 5}]}
    _, total = av.assemble(edit, tmp_path / "x.mp4", dry_run=True)
    assert total == 5
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        av.assemble(edit, tmp_path / "x.mp4", dry_run=True, strict=True)


def test_speed_label_text():
    assert av.fmt_speed(4) == "×4"
    assert av.fmt_speed(2.5) == "×2.5"
    assert av.speed_label_text(8, "ja") == "×8 倍速"
