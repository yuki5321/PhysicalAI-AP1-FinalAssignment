#!/usr/bin/env python3
"""Assemble the submission video from an edit list (YAML or JSON).

    python3 tools/assemble_video.py EDIT.yaml [-o OUT.mp4] [--silent | --bgm FILE] [--dry-run]

Output: 1280x720, 30 fps, H.264 (yuv420p, +faststart), metadata stripped. Without
--silent/--bgm the MP4 has no audio track; --silent adds a silent AAC track (some
uploaders prefer one); --bgm loops/trims a music file and fades it out.

Edit list (paths are relative to ``base_dir``, default: the current directory):

    output: data/video/PAI_final.mp4      # optional, -o overrides
    base_dir: .                           # optional
    lang: auto                            # card / label language: auto | ja | en
    fade: 0.3                             # default fade-in/out from black per segment [s] (0 = cut)
    bgm: music.mp3                        # optional (same as --bgm); bgm_volume: 0.3
    segments:
      - card: title                       # tools/video_cards.py card: title|diagram|ai|results|future|text
        duration: 5
        course: "..."                     # extra args go to the card renderer
      - card: text
        big: "手で崩す → 手を引く → ロボが直す"
        duration: 3
      - image: some.png                   # any still image, letterboxed
        duration: 3
      - clip: data/video/gazebo_demo.mp4
        in: 12.0                          # source seconds (default 0; negative = from the end)
        out: 40.0                         # default: end of file; or `dur:` (source seconds)
        speed: 2                          # >1 = faster (course limit: 2); a "×2" label is added automatically
        speed_label: true                 # false to hide, or a string to replace the text
        crop: [x, y, w, h]                # optional crop of the source (pixels) before scaling
        caption: "I を直す"               # bottom caption for the whole segment, or a list:
        captions:                         #   - {text: "...", start: 0, end: 3}  (output seconds)
          - {text: "手で崩す", start: 0, end: 2.5}
        pip:                              # picture-in-picture wipe (e.g. the hand webcam)
          path: data/video/handcam.mp4
          in: 3.5                         # source seconds that line up with the main clip's `in`
          speed: 2                        # default: same as the main clip
          corner: br                      # tl | tr | bl | br
          width: 360                      # px (height follows the aspect ratio)
          margin: 24
          radius: 18
          border: 4                       # px, white; border_color: [255,255,255]
          label: "手元カメラ"             # small tag above the wipe (optional)
        fade: 0                           # per-segment override

Each segment is rendered to an intermediate MP4 with ffmpeg (all overlays are PNGs
drawn with PIL, so Japanese text works without ffmpeg drawtext fonts), then they are
concatenated losslessly. The total duration is printed; outside 60-180 s it warns
(--strict makes that an error).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import video_cards as vc  # noqa: E402

W, H, FPS = 1280, 720, 30
LABEL_BG = vc.ACCENT          # speed label box colour (the test checks for it)
LABEL_FG = (20, 20, 20)
MAX_SPEED = 2.0               # the course allows at most 2x playback
MIN_S, MAX_S = 60.0, 180.0


# --------------------------------------------------------------------------- helpers

def run(cmd, verbose=False):
    if verbose:
        print(" ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-4000:])
        raise RuntimeError(f"command failed: {cmd[0]} ... (exit {r.returncode})")
    return r.stdout


def probe_duration(path) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
               "default=noprint_wrappers=1:nokey=1", path])
    return float(out.strip())


def probe_size(path) -> tuple[int, int]:
    out = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
               "-of", "csv=p=0", path])
    w, h = out.strip().split(",")[:2]
    return int(w), int(h)


def fmt_speed(speed: float) -> str:
    s = f"{speed:.2f}".rstrip("0").rstrip(".")
    return f"×{s}"


def speed_label_text(speed: float, lang: str) -> str:
    return f"{fmt_speed(speed)} 倍速" if lang == "ja" else f"{fmt_speed(speed)} speed"


def load_edit_list(path) -> dict:
    txt = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith(".json"):
        return json.loads(txt)
    import yaml
    return yaml.safe_load(txt)


def clip_range(seg: dict, base: Path) -> tuple[float, float]:
    """(in, out) source seconds; negative values count from the end of the file."""
    src = base / seg["clip"]
    t_in = float(seg.get("in", 0))
    t_out = seg.get("out")
    length = None
    if t_in < 0 or t_out is None and "dur" not in seg or (t_out is not None and float(t_out) < 0):
        length = probe_duration(src)
    if t_in < 0:
        t_in = max(0.0, length + t_in)
    if t_out is not None:
        t_out = float(t_out)
        if t_out < 0:
            t_out = length + t_out
    elif "dur" in seg:
        t_out = t_in + float(seg["dur"])
    else:
        t_out = length
    if t_out - t_in <= 0:
        raise ValueError(f"clip {seg['clip']}: out ({t_out}) <= in ({t_in})")
    return t_in, t_out


# --------------------------------------------------------------------------- overlays (PIL)

def overlay_png(path, lang, label: str | None = None, caption: str | None = None,
                cap_span: tuple[int, int] = (0, W)) -> Path:
    """Full-frame RGBA PNG with an optional speed label (top-right) and caption (bottom)."""
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if label:
        f = vc.font(34, lang)
        tw, th = vc.text_size(d, label, f)
        x1, y0 = W - 28, 24
        x0, y1 = x1 - tw - 36, y0 + 58
        d.rounded_rectangle([x0, y0, x1, y1], radius=12, fill=LABEL_BG + (255,))
        d.text(((x0 + x1) / 2, (y0 + y1) / 2), label, font=f, fill=LABEL_FG + (255,), anchor="mm")
    if caption:
        x0, x1 = cap_span
        f = vc.fit_font(d, caption, x1 - x0 - 100, 36, lang)
        tw, th = vc.text_size(d, caption, f)
        cx, y1 = (x0 + x1) / 2, H - 36
        d.rounded_rectangle([cx - tw / 2 - 24, y1 - th - 30, cx + tw / 2 + 24, y1], radius=12,
                            fill=(0, 0, 0, 170))
        d.text((cx, y1 - 15), caption, font=f, fill=(255, 255, 255, 255), anchor="md")
    img.save(path)
    return Path(path)


def pip_pngs(tmp: Path, idx: int, w: int, h: int, radius: int, border: int, color, label, lang):
    """(mask, frame) PNGs: rounded-rect alpha mask (w x h) and the border+label overlay."""
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=255)
    mpath = tmp / f"pipmask_{idx}.png"
    mask.save(mpath)
    pad = border + 2
    top = 0
    if label:
        f = vc.font(22, lang)
        tmpd = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        lw, lh = vc.text_size(tmpd, label, f)
        top = lh + 16
    fr = Image.new("RGBA", (w + 2 * pad, h + 2 * pad + top), (0, 0, 0, 0))
    d = ImageDraw.Draw(fr)
    if border > 0:
        d.rounded_rectangle([pad - border, top + pad - border, pad + w + border - 1, top + pad + h + border - 1],
                            radius=radius + border, outline=tuple(color) + (255,), width=border)
    if label:
        d.rounded_rectangle([pad - border, 0, pad - border + lw + 20, top - 4], radius=8, fill=(0, 0, 0, 180))
        d.text((pad - border + 10, (top - 4) / 2), label, font=f, fill=(255, 255, 255, 255), anchor="lm")
    fpath = tmp / f"pipframe_{idx}.png"
    fr.save(fpath)
    return mpath, fpath, pad, top


# --------------------------------------------------------------------------- segments

FIT = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,"
       f"setsar=1")


def _fade(dur, fade):
    if not fade or fade <= 0 or dur <= 2 * fade:
        return ""
    return f",fade=t=in:st=0:d={fade},fade=t=out:st={dur - fade:.3f}:d={fade}"


def _captions(seg) -> list[dict]:
    caps = list(seg.get("captions") or [])
    if seg.get("caption"):
        caps.insert(0, {"text": seg["caption"]})
    return caps


def render_segment(seg: dict, idx: int, tmp: Path, ctx: dict) -> tuple[Path, float]:
    lang, base, verbose = ctx["lang"], ctx["base"], ctx["verbose"]
    fade = seg.get("fade", ctx["fade"])
    out = tmp / f"seg_{idx:03d}.mp4"
    inputs: list = []
    filters: list[str] = []

    if "card" in seg or "image" in seg:
        dur = float(seg.get("duration", 4))
        if "card" in seg:
            opts = {k: v for k, v in seg.items() if k not in ("card", "duration", "fade", "caption", "captions", "lang")}
            img = vc.render_card(seg["card"], lang, **opts)
            still = tmp / f"card_{idx:03d}.png"
            img.save(still)
        else:
            still = base / seg["image"]
        inputs += ["-loop", "1", "-framerate", FPS, "-t", f"{dur:.3f}", "-i", still]
        filters.append(f"[0:v]{FIT},fps={FPS},format=yuv420p[v0]")
        speed, label = 1.0, None
    elif "clip" in seg:
        src = base / seg["clip"]
        if not src.is_file():
            raise FileNotFoundError(src)
        t_in, t_out = clip_range(seg, base)
        src_len = t_out - t_in
        speed = float(seg.get("speed", 1))
        if speed > MAX_SPEED + 1e-9:
            raise ValueError(f"{seg['clip']}: speed {speed} exceeds the course limit of {MAX_SPEED}x")
        dur = src_len / speed
        inputs += ["-ss", f"{t_in:.3f}", "-t", f"{src_len:.3f}", "-i", src]
        crop = ""
        if seg.get("crop"):
            x, y, cw, ch = seg["crop"]
            crop = f"crop={cw}:{ch}:{x}:{y},"
        filters.append(f"[0:v]{crop}setpts=(PTS-STARTPTS)/{speed},fps={FPS},{FIT},format=yuv420p[v0]")
        sl = seg.get("speed_label", True)
        label = None
        if abs(speed - 1) > 1e-6 and sl is not False:
            label = sl if isinstance(sl, str) else speed_label_text(speed, lang)
    else:
        raise ValueError(f"segment {idx}: needs card, image or clip: {seg}")

    cur = "v0"
    n_in = 1
    cap_span = (0, W)
    # ---- picture in picture
    pip = seg.get("pip")
    if pip and "clip" in seg:
        psrc = base / pip["path"]
        pspeed = float(pip.get("speed", speed))
        p_in = float(pip.get("in", 0))
        pw0, ph0 = probe_size(psrc)
        if pip.get("crop"):
            pw0, ph0 = pip["crop"][2], pip["crop"][3]
        pw = int(pip.get("width", 360)) // 2 * 2
        ph = int(round(pw * ph0 / pw0)) // 2 * 2
        radius, border = int(pip.get("radius", 18)), int(pip.get("border", 4))
        mpath, fpath, pad, top = pip_pngs(tmp, idx, pw, ph, radius, border, pip.get("border_color", (255, 255, 255)),
                                          pip.get("label"), lang)
        margin = int(pip.get("margin", 24))
        corner = pip.get("corner", "br")
        x = margin + pad if corner[1] == "l" else W - margin - pad - pw
        y = margin + pad + top if corner[0] == "t" else H - margin - pad - ph
        if label and corner == "tr":
            y += 70  # keep clear of the speed label
        if corner[0] == "b":  # keep bottom captions beside the wipe
            cap_span = (0, x - pad - 8) if corner[1] == "r" else (x + pw + pad + 8, W)
        inputs += ["-ss", f"{p_in:.3f}", "-t", f"{dur * pspeed:.3f}", "-i", psrc,
                   "-loop", "1", "-framerate", FPS, "-t", f"{dur:.3f}", "-i", mpath,
                   "-loop", "1", "-framerate", FPS, "-t", f"{dur:.3f}", "-i", fpath]
        pc = ""
        if pip.get("crop"):
            cx, cy, cw, ch = pip["crop"]
            pc = f"crop={cw}:{ch}:{cx}:{cy},"
        filters.append(f"[{n_in}:v]{pc}setpts=(PTS-STARTPTS)/{pspeed},fps={FPS},scale={pw}:{ph},format=rgba[pv]")
        filters.append(f"[{n_in + 1}:v]format=gray,scale={pw}:{ph}[pm]")
        filters.append("[pv][pm]alphamerge[pa]")
        filters.append(f"[{cur}][pa]overlay={x}:{y}:eof_action=pass:format=auto[v1]")
        filters.append(f"[v1][{n_in + 2}:v]overlay={x - pad}:{y - pad - top}:format=auto[v2]")
        cur = "v2"
        n_in += 3

    # ---- speed label (whole segment) + captions (timed)
    caps = _captions(seg)
    if label:
        lp = overlay_png(tmp / f"label_{idx:03d}.png", lang, label=label)
        inputs += ["-loop", "1", "-framerate", FPS, "-t", f"{dur:.3f}", "-i", lp]
        filters.append(f"[{cur}][{n_in}:v]overlay=0:0:format=auto[vl]")
        cur, n_in = "vl", n_in + 1
    for k, c in enumerate(caps):
        cp = overlay_png(tmp / f"cap_{idx:03d}_{k}.png", lang, caption=c["text"], cap_span=cap_span)
        inputs += ["-loop", "1", "-framerate", FPS, "-t", f"{dur:.3f}", "-i", cp]
        en = ""
        if "start" in c or "end" in c:
            en = f":enable='between(t,{float(c.get('start', 0))},{float(c.get('end', dur))})'"
        filters.append(f"[{cur}][{n_in}:v]overlay=0:0:format=auto{en}[vc{k}]")
        cur, n_in = f"vc{k}", n_in + 1

    filters.append(f"[{cur}]format=yuv420p{_fade(dur, fade)}[vout]")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", ";".join(filters),
           "-map", "[vout]", "-an", "-t", f"{dur:.3f}", "-r", FPS, "-c:v", "libx264", "-preset", ctx["preset"],
           "-crf", ctx["crf"], "-pix_fmt", "yuv420p", "-video_track_timescale", "15360", "-map_metadata", "-1", out]
    run(cmd, verbose)
    return out, dur


def concat(parts: list[Path], out: Path, tmp: Path, audio: str | None, bgm: Path | None, bgm_volume: float,
           total: float, verbose=False):
    lst = tmp / "concat.txt"
    lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts), encoding="utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    base = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    tail = ["-map_metadata", "-1", "-map_chapters", "-1", "-movflags", "+faststart", out]
    if bgm:
        fo = max(0.0, total - 2.0)
        cmd = base + ["-stream_loop", "-1", "-i", bgm, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                      "-af", f"volume={bgm_volume},afade=t=in:d=1,afade=t=out:st={fo:.2f}:d=2",
                      "-c:a", "aac", "-b:a", "160k", "-t", f"{total:.3f}"] + tail
    elif audio == "silent":
        cmd = base + ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a",
                      "-c:v", "copy", "-c:a", "aac", "-b:a", "96k", "-t", f"{total:.3f}"] + tail
    else:
        cmd = base + ["-map", "0:v", "-c:v", "copy", "-an"] + tail
    run(cmd, verbose)


def assemble(edit: dict, out: Path | None = None, audio: str | None = None, bgm: str | None = None,
             strict=False, dry_run=False, keep_tmp=False, verbose=False, preset="medium", crf=20) -> tuple[Path, float]:
    base = Path(edit.get("base_dir", ".")).resolve()
    lang = vc.resolve_lang(edit.get("lang", "auto"))
    if out is not None:          # -o / argument: relative to the current directory
        out = Path(out).resolve()
    else:                        # edit list `output`: relative to base_dir
        out = base / edit.get("output", "data/video/PAI_final.mp4")
    bgm = bgm or edit.get("bgm")
    bgm_path = (base / bgm) if bgm else None
    if bgm_path and not bgm_path.is_file():
        raise FileNotFoundError(bgm_path)
    audio = audio or edit.get("audio")
    ctx = dict(lang=lang, base=base, fade=float(edit.get("fade", 0.3)), verbose=verbose, preset=preset,
               crf=str(crf))
    segs = edit["segments"]

    # planned durations first (cheap), so a bad length is reported before rendering
    plan = []
    for i, s in enumerate(segs):
        if "clip" in s:
            if (base / s["clip"]).is_file():
                t_in, t_out = clip_range(s, base)
            else:
                t_in = float(s.get("in", 0))
                t_out = float(s["out"]) if t_in >= 0 and float(s.get("out", -1)) >= 0 else float("nan")
                print(f"  MISSING: {base / s['clip']}")
            if s.get("pip") and not (base / s["pip"]["path"]).is_file():
                print(f"  MISSING (pip): {base / s['pip']['path']}")
                t_out = float("nan")
            if float(s.get("speed", 1)) > MAX_SPEED + 1e-9:
                raise ValueError(f"{s['clip']}: speed {s['speed']} exceeds the course limit of {MAX_SPEED}x")
            d = (t_out - t_in) / float(s.get("speed", 1))
            what = f"clip {s['clip']} [{t_in:g}-{t_out:g}s]" + (f" {fmt_speed(float(s['speed']))}"
                                                                if float(s.get("speed", 1)) != 1 else "")
            if s.get("pip"):
                what += f" + pip {s['pip']['path']}"
        else:
            d = float(s.get("duration", 4))
            what = f"card {s['card']}" if "card" in s else f"image {s.get('image')}"
        plan.append(d)
        print(f"  {i:2d}  {d:6.2f}s  {what}")
    total = sum(plan)
    print(f"planned total: {total:.2f} s  (lang={lang}, font={vc.find_font(lang == 'ja') or 'default'})")
    msg = None
    if math.isnan(total):
        msg = "a clip is missing; cannot compute the total"
    elif total < MIN_S:
        msg = f"total {total:.1f} s is shorter than {MIN_S:.0f} s (course: 1-3 min)"
    elif total > MAX_S:
        msg = f"total {total:.1f} s is longer than {MAX_S:.0f} s (course: 1-3 min)"
    if msg:
        print("WARNING:", msg)
        if strict:
            raise SystemExit(f"--strict: {msg}")
    if dry_run:
        return out, total
    if math.isnan(total):
        raise SystemExit("missing input file(s), see MISSING above")

    tmp = Path(tempfile.mkdtemp(prefix="pa_video_"))
    try:
        parts, real = [], 0.0
        for i, s in enumerate(segs):
            p, d = render_segment(s, i, tmp, ctx)
            parts.append(p)
            real += d
            print(f"  rendered {i:2d}/{len(segs) - 1}")
        concat(parts, out, tmp, audio, bgm_path, float(edit.get("bgm_volume", 0.3)), real, verbose)
    finally:
        if keep_tmp:
            print("intermediate files:", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    final = probe_duration(out)
    print(f"wrote {out}  duration {final:.2f} s  size {os.path.getsize(out) / 1e6:.1f} MB")
    if final < MIN_S or final > MAX_S:
        print(f"WARNING: final duration {final:.1f} s is outside {MIN_S:.0f}-{MAX_S:.0f} s")
    return out, final

# --------------------------------------------------------------------------- example edit lists
# Same text as docs/video.md. `--write-example a|b PATH` writes one to edit.

EXAMPLE_A = """\
# (A) Gazebo version. Replace the XXXX file names and the in/out times
# (open the MP4 in a player and note the seconds where each letter starts).
output: data/video/PAI_final.mp4
lang: ja
fade: 0.3
segments:
  # 1. opening: YUKI complete + snow (the last seconds of the recorder MP4)
  - clip: videos/demo_XXXX.mp4
    in: -8                     # negative = seconds from the end
    caption: "YUKI 完成！ 崩されても自分で直す"
  - card: title
    duration: 5
  - card: diagram
    duration: 9
  - card: ai
    duration: 8
  # 2. Gazebo 3D view (screen capture), short
  - clip: videos/screen_XXXX.mp4
    in: 5
    out: 15
    caption: "Gazebo（ROS 2 Humble）: 台・4本のヘラ・真上カメラ"
  # 3. letter I in real time, with the hand-camera wipe (hands only!)
  - clip: videos/demo_XXXX.mp4
    in: 0
    out: 30
    captions:
      - {text: "手で崩す（MediaPipe Hands）", start: 0, end: 8}
      - {text: "手がある間は待つ → 手を引くと直し始める", start: 8, end: 16}
      - {text: "押す → 見る → 押す… でそろえる", start: 16, end: 30}
    pip:
      path: videos/handcam_XXXX.mp4
      in: 0                    # handcam second that matches demo second `in` above
      corner: br
      width: 360
      label: "手元カメラ（手だけ）"
  # 4. U, Y, K faster (a "x2 倍速" label is added automatically)
  - clip: videos/demo_XXXX.mp4
    in: 30
    out: 70
    speed: 2
    caption: "U"
    pip: {path: videos/handcam_XXXX.mp4, in: 30, corner: br, width: 300, label: "手元カメラ"}
  - clip: videos/demo_XXXX.mp4
    in: 70
    out: 150
    speed: 2
    caption: "Y・K"
  - card: results
    duration: 9
  - card: future
    duration: 6
"""

EXAMPLE_B = """\
# (B) fallback built only from the 2D demo (tools/make_demo_video.py) + cards.
# demo_2d.mp4 already shows its own "x2" labels on sped-up parts, so it is used
# at speed 1 here (do not speed it up again, or the labels would be wrong).
output: data/video/PAI_final_fallback.mp4
lang: ja
fade: 0.3
segments:
  - clip: data/video/demo_2d.mp4
    in: -7                     # opening: YUKI complete + snow (last 7 s; the clip shows "YUKI 完成！" itself)
  - card: title
    duration: 5
  - card: diagram
    duration: 9
  - card: ai
    duration: 8
  - card: text
    big: "手で崩す → 手を引く → ロボが直す"
    heading: "2D 押しモデル（pymunk）でのシミュレーション。倍速部分は右上に表示"
    duration: 4
  - clip: data/video/demo_2d.mp4
  - card: results
    duration: 9
  - card: future
    duration: 6
"""
EXAMPLES = {"a": EXAMPLE_A, "b": EXAMPLE_B}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("edit_list", nargs="?", help="YAML or JSON edit list")
    ap.add_argument("--write-example", nargs=2, metavar=("A|B", "PATH"),
                    help="write example edit list A (Gazebo) or B (2D fallback) to PATH and exit")
    ap.add_argument("-o", "--output", help="output MP4 (overrides the edit list)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--silent", action="store_true", help="add a silent AAC audio track")
    g.add_argument("--bgm", help="background music file (looped / trimmed, faded out)")
    ap.add_argument("--strict", action="store_true", help="fail if the total is outside 60-180 s")
    ap.add_argument("--dry-run", action="store_true", help="only print the segment plan and total")
    ap.add_argument("--keep-tmp", action="store_true")
    ap.add_argument("--preset", default="medium", help="x264 preset (ultrafast for quick previews)")
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    if a.write_example:
        which, path = a.write_example
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(EXAMPLES[which.lower()], encoding="utf-8")
        print("wrote", path)
        return 0
    if not a.edit_list:
        ap.error("edit_list is required")
    edit = load_edit_list(a.edit_list)
    assemble(edit, a.output, "silent" if a.silent else None, a.bgm, a.strict, a.dry_run, a.keep_tmp, a.verbose,
             a.preset, a.crf)
    return 0


if __name__ == "__main__":
    sys.exit(main())
