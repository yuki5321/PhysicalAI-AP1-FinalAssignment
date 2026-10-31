#!/usr/bin/env python3
"""Title / diagram / explanation / result cards for the submission video.

Every card is a 1280x720 PIL image drawn with plain shapes and text (no external
assets). ``tools/assemble_video.py`` turns them into clips; this script can also
write them as PNG (to check) or as short MP4 clips (to drop into another editor).

    python3 tools/video_cards.py --out data/video/cards            # PNG of every card
    python3 tools/video_cards.py --out data/video/cards --mp4 4    # + 4 s MP4 each
    python3 tools/video_cards.py --list

Cards: title, diagram, ai, results, future, text (generic heading + lines).

Japanese text needs a CJK font. ``find_font`` looks at $PA_FONT, data/fonts/,
then common Linux / Windows / macOS locations; without one, cards fall back to
English (``lang="auto"``). No personal names appear on any card: the course name
line is a placeholder that can be overridden with ``--course``.

Recovery numbers on the results card come from data/recovery/**/*.csv
(eval_recovery.py output, seed 0 only; smoke runs skipped). CSVs under
data/recovery/point/ are point-pusher runs, the others paddle runs; the card
shows the point pusher rows first and labels each row with its pusher. If no
CSV is found, wide tables (I/U/Y/K columns with "k/n" cells) in
docs/recovery_results.md / docs/recovery_hand.md are used. They are labelled
as 2D-model (pymunk) numbers unless the source says ``backend=gazebo``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
W, H = 1280, 720

# Palette (dark "snowy night" theme).
BG = (18, 28, 48)
BG2 = (28, 42, 70)
FG = (240, 244, 250)
MUTED = (160, 176, 200)
ACCENT = (255, 196, 0)       # same yellow as the speed label
BLUE = (90, 160, 255)
GREEN = (90, 210, 140)
ORANGE = (255, 140, 60)
PINK = (240, 110, 150)

LETTERS = ["I", "U", "Y", "K"]

# --------------------------------------------------------------------------- fonts

_CJK_CANDIDATES = [
    # repo-local (gitignored) download
    *sorted(glob.glob(str(REPO / "data" / "fonts" / "*.otf"))),
    *sorted(glob.glob(str(REPO / "data" / "fonts" / "*.tt[fc]"))),
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    # Windows
    "C:/Windows/Fonts/YuGothB.ttc",
    "C:/Windows/Fonts/meiryob.ttc",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    # macOS
    "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]
_LATIN_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial.ttf",
]


def _supports_cjk(path: str) -> bool:
    try:
        f = ImageFont.truetype(path, 32)
        # A font without the glyph returns a tofu box of the same width for both.
        return f.getbbox("雪")[2] > 0 and f.getmask("雪").getbbox() != f.getmask("\u0378").getbbox()
    except Exception:
        return False


_FONT_CACHE: dict = {}


def find_font(cjk: bool = True) -> str | None:
    """Path of a usable font file (CJK-capable when ``cjk``), or None."""
    key = ("path", cjk)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    env = os.environ.get("PA_FONT")
    cands = ([env] if env else []) + (_CJK_CANDIDATES if cjk else _LATIN_CANDIDATES + _CJK_CANDIDATES)
    found = None
    for c in cands:
        if c and os.path.isfile(c) and (not cjk or _supports_cjk(c)):
            found = c
            break
    if found is None and cjk:
        try:  # fontconfig, if present
            out = subprocess.run(["fc-list", ":lang=ja", "file"], capture_output=True, text=True,
                                 timeout=5).stdout
            for line in out.splitlines():
                p = line.split(":")[0].strip()
                if p and os.path.isfile(p) and _supports_cjk(p):
                    found = p
                    break
        except Exception:
            pass
    _FONT_CACHE[key] = found
    return found


def has_cjk() -> bool:
    return find_font(cjk=True) is not None


def resolve_lang(lang: str = "auto") -> str:
    if lang == "auto":
        return "ja" if has_cjk() else "en"
    return lang


def font(size: int, lang: str = "ja") -> ImageFont.ImageFont:
    key = (size, lang)
    if key not in _FONT_CACHE:
        path = find_font(cjk=True) if lang == "ja" else (find_font(cjk=False) or find_font(cjk=True))
        _FONT_CACHE[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default(size)
    return _FONT_CACHE[key]


def text_size(draw: ImageDraw.ImageDraw, s: str, f) -> tuple[int, int]:
    b = draw.textbbox((0, 0), s, font=f)
    return b[2] - b[0], b[3] - b[1]


def draw_text(draw, xy, s, f, fill=FG, anchor="la", **kw):
    draw.text(xy, s, font=f, fill=fill, anchor=anchor, **kw)


def fit_font(draw, s: str, max_w: int, size: int, lang: str, min_size: int = 14):
    while size > min_size:
        f = font(size, lang)
        if text_size(draw, s, f)[0] <= max_w:
            return f
        size -= 2
    return font(min_size, lang)


# --------------------------------------------------------------------------- drawing helpers

def _canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    # vertical gradient
    for y in range(H):
        t = y / H
        c = tuple(int(BG[i] * (1 - t) + BG2[i] * t) for i in range(3))
        d.line([(0, y), (W, y)], fill=c)
    return img, d


def _snow(d: ImageDraw.ImageDraw, n: int = 70, seed: int = 3, keep_out=()):
    """Snow dots, skipping the ``keep_out`` rectangles (text areas)."""
    import random
    rng = random.Random(seed)
    for _ in range(n):
        x, y, r = rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1.5, 4.5)
        if any(a - 8 <= x <= c + 8 and b - 8 <= y <= e + 8 for a, b, c, e in keep_out):
            continue
        a = int(rng.uniform(90, 200))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(a, a + 20 if a < 235 else 255, 255))


def _heading(d, s: str, lang: str, y: int = 40, size: int = 44):
    f = fit_font(d, s, W - 120, size, lang)
    draw_text(d, (60, y), s, f, fill=FG)
    d.rectangle([60, y + size + 14, 60 + 120, y + size + 19], fill=ACCENT)


def box(d, xy, title: str, sub: list[str] | str = (), color=BLUE, lang="ja", title_size=26, sub_size=19):
    x0, y0, x1, y1 = xy
    d.rounded_rectangle(xy, radius=16, fill=(32, 48, 80), outline=color, width=4)
    if isinstance(sub, str):
        sub = [sub] if sub else []
    ft = fit_font(d, title, x1 - x0 - 24, title_size, lang)
    fs = [fit_font(d, s, x1 - x0 - 24, sub_size, lang) for s in sub]
    th = text_size(d, "Ag漢", ft)[1]
    sh = [text_size(d, "Ag漢", f)[1] for f in fs]
    total = th + sum(h + 8 for h in sh)
    y = (y0 + y1) / 2 - total / 2
    draw_text(d, ((x0 + x1) / 2, y), title, ft, fill=FG, anchor="ma")
    y += th + 8
    for s, f, h in zip(sub, fs, sh):
        draw_text(d, ((x0 + x1) / 2, y), s, f, fill=MUTED, anchor="ma")
        y += h + 8


def arrow(d, pts, color=FG, width=5, head=16, label: str | None = None, lang="ja", label_pos=None,
          label_size=18):
    """Poly-line arrow; the head is at the last point."""
    import math
    d.line(pts, fill=color, width=width, joint="curve")
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    ang = math.atan2(yb - ya, xb - xa)
    p1 = (xb - head * math.cos(ang - 0.45), yb - head * math.sin(ang - 0.45))
    p2 = (xb - head * math.cos(ang + 0.45), yb - head * math.sin(ang + 0.45))
    d.polygon([(xb, yb), p1, p2], fill=color)
    if label:
        f = font(label_size, lang)
        if label_pos is None:
            label_pos = ((xa + xb) / 2, (ya + yb) / 2 - 6)
        tw, th = text_size(d, label, f)
        lx, ly = label_pos
        d.rounded_rectangle([lx - tw / 2 - 6, ly - th - 8, lx + tw / 2 + 6, ly + 4], radius=6, fill=BG)
        draw_text(d, (lx, ly), label, f, fill=color, anchor="md")


def _bullets(d, x, y, items, lang, size=24, gap=14, color=FG, max_w=1100):
    for it in items:
        f = fit_font(d, it, max_w - 30, size, lang)
        d.ellipse([x, y + size * 0.35, x + 10, y + size * 0.35 + 10], fill=ACCENT)
        draw_text(d, (x + 24, y), it, f, fill=color)
        y += text_size(d, "Ag漢", f)[1] + gap
    return y


def _footer(d, s: str, lang: str):
    f = font(18, lang)
    draw_text(d, (W - 40, H - 28), s, f, fill=MUTED, anchor="rd")


def _letters_row(d, cx, cy, size=90, gap=26):
    """Big Y U K I glyphs drawn like blocks."""
    f = font(size, "en")
    word = "YUKI"
    total = len(word) * size + (len(word) - 1) * gap
    x = cx - total / 2
    for ch in word:
        d.rounded_rectangle([x, cy - size / 2 - 12, x + size, cy + size / 2 + 12], radius=14,
                            fill=(235, 242, 255), outline=(180, 200, 230), width=3)
        draw_text(d, (x + size / 2, cy), ch, f, fill=BG, anchor="mm")
        x += size + gap


# --------------------------------------------------------------------------- cards

T = {
    "title": {
        "ja": ("Push-Alphabet", "崩されても自分で直す雪の文字ロボ",
               "Physical AI 応用講座 最終課題（シミュレーション版: ROS 2 Humble + Gazebo）"),
        "en": ("Push-Alphabet", "A snow-letter robot that fixes itself after you knock it over",
               "Physical AI course final project (simulation: ROS 2 Humble + Gazebo)"),
    },
}


def card_title(lang="ja", course: str | None = None, subtitle: str | None = None, **_):
    img, d = _canvas()
    _snow(d, 110, keep_out=[(100, 90, 1180, 290), (300, 330, 980, 490), (60, 560, 1220, 620)])
    name, sub, crs = T["title"][lang]
    if course:
        crs = course
    if subtitle:
        sub = subtitle
    draw_text(d, (W / 2, 150), name, font(84, "en"), fill=FG, anchor="mm")
    draw_text(d, (W / 2, 250), sub, fit_font(d, sub, W - 140, 46, lang), fill=ACCENT, anchor="mm")
    _letters_row(d, W / 2, 410)
    draw_text(d, (W / 2, 590), crs, fit_font(d, crs, W - 140, 26, lang), fill=MUTED, anchor="mm")
    return img


def card_diagram(lang="ja", **_):
    img, d = _canvas()
    ja = lang == "ja"
    _heading(d, "システム構成" if ja else "System overview", lang, y=28, size=40)
    # Row 1: robot loop
    y0, y1 = 130, 250
    cols = [(40, 270), (330, 620), (680, 960), (1020, 1240)]
    box(d, (*(cols[0][0], y0), *(cols[0][1], y1)), "真上カメラ" if ja else "Top camera",
        ["640×640 画像" if ja else "640×640 image"], BLUE, lang)
    box(d, (cols[1][0], y0, cols[1][1], y1), "知覚 (AI 1)" if ja else "Perception (AI 1)",
        ["classic / CNN / hybrid", "文字の種類・位置・向き" if ja else "letter, position, angle"], GREEN, lang)
    box(d, (cols[2][0], y0, cols[2][1], y1), "プランナー" if ja else "Planner",
        ["状態機械 + 2D 押しモデル" if ja else "state machine + 2D push",
         "押し方の候補を先読み" if ja else "model lookahead (MPC)"],
        ACCENT, lang)
    box(d, (cols[3][0], y0, cols[3][1], y1), "押し棒（点）" if ja else "Point pusher",
        ["降ろして直線で押す" if ja else "lower, push straight", "/point_push_cmd"], ORANGE, lang)
    for a, b in [(0, 1), (1, 2), (2, 3)]:
        arrow(d, [(cols[a][1] + 4, (y0 + y1) / 2), (cols[b][0] - 6, (y0 + y1) / 2)])
    # Row 2: world
    wy0, wy1 = 330, 430
    d.rounded_rectangle([40, wy0, 1240, wy1], radius=18, fill=(40, 60, 96), outline=(200, 215, 240), width=3)
    wt = "Gazebo の台と文字ブロック  Y · U · K · I" if ja else "Gazebo table and letter blocks  Y · U · K · I"
    draw_text(d, (640, (wy0 + wy1) / 2), wt, fit_font(d, wt, 900, 30, lang), anchor="mm")
    arrow(d, [(1130, y1 + 4), (1130, wy0 - 6)], color=ORANGE, label="押す" if ja else "push", lang=lang,
          label_pos=(1185, (y1 + wy0) / 2 + 10))
    arrow(d, [(155, wy0 - 4), (155, y1 + 6)], color=BLUE, label="撮る" if ja else "image", lang=lang,
          label_pos=(210, (y1 + wy0) / 2 + 10))
    # Row 3: hand loop
    hy0, hy1 = 520, 640
    hcols = [(40, 290), (350, 640), (700, 980)]
    box(d, (hcols[0][0], hy0, hcols[0][1], hy1), "Web カメラ" if ja else "Webcam",
        ["手元だけを映す" if ja else "hands only, no face"], BLUE, lang)
    box(d, (hcols[1][0], hy0, hcols[1][1], hy1), "MediaPipe Hands",
        ["(AI 2) " + ("手の有無・掌の位置" if ja else "hand + palm position")], GREEN, lang)
    box(d, (hcols[2][0], hy0, hcols[2][1], hy1), "シミュ内の手" if ja else "Hand in sim",
        ["rosbridge → /hand/state"], PINK, lang)
    arrow(d, [(hcols[0][1] + 4, (hy0 + hy1) / 2), (hcols[1][0] - 6, (hy0 + hy1) / 2)])
    arrow(d, [(hcols[1][1] + 4, (hy0 + hy1) / 2), (hcols[2][0] - 6, (hy0 + hy1) / 2)])
    arrow(d, [(840, hy0 - 4), (840, wy1 + 6)], color=PINK, label="崩す" if ja else "knock", lang=lang,
          label_pos=(895, (wy1 + hy0) / 2 + 10))
    # planner listens to the hand state: side bracket on the right
    note = "手がある間は待つ → 手が消えたら直す" if ja else "waits while a hand is present, then repairs"
    lines = ["プランナーは", "手がある間は待ち、", "消えたら直す"] if ja else ["planner waits", "while a hand is", "there, then repairs"]
    f = fit_font(d, max(lines, key=len), 235, 24, lang)
    yy = hy0 + 4
    for ln in lines:
        draw_text(d, (1010, yy), ln, f, fill=ACCENT)
        yy += 36
    lab = "ROS 2 Humble · Gazebo Classic 11 · Docker" + ("（ホストで手検出）" if ja else " (hand detection on host)")
    _footer(d, lab, lang)
    return img


def card_ai(lang="ja", **_):
    img, d = _canvas()
    ja = lang == "ja"
    _heading(d, "AI の使いどころ（推論）" if ja else "Where AI is used (inference)", lang, y=28, size=40)
    colw = 560
    for i, (title, color, items) in enumerate([
        ("AI 1: CNN で文字の種類と向き" if ja else "AI 1: CNN letter + pose", GREEN, [
            "真上カメラ画像 → 文字ごとに 96×96 で切り出し" if ja else "top image → 96×96 crop per letter",
            "小型 CNN が 文字種 (I/U/Y/K)・向き・中心のずれを出力" if ja else "small CNN: class, angle (cos,sin), offset",
            "合成画像を自動生成して学習（ラベル付けは不要）" if ja else "trained on auto-generated synthetic images",
            "hybrid: CNN + 古典 CV の向き合わせで角度誤差 0.46°" if ja else "hybrid: CNN + classic refine, 0.46° (p95)",
            "ONNX で推論（コンテナの CPU）" if ja else "ONNX inference on CPU in the container",
        ]),
        ("AI 2: MediaPipe Hands で手" if ja else "AI 2: MediaPipe Hands", PINK, [
            "Web カメラで自分の手を検出（手元だけを映す）" if ja else "detects my hand on the webcam (hands only)",
            "掌の位置を台の座標に写像 → シミュ内の手が文字を崩す" if ja else "palm → table coords → sim hand knocks letters",
            "手がある間はロボは待ち、消えて 1 秒後に直し始める" if ja else "robot waits while the hand is there",
            "映像は保存せず、メモリ上で処理するだけ" if ja else "frames are processed in memory, not stored",
        ]),
    ]):
        x = 60 + i * (colw + 40)
        d.rounded_rectangle([x, 130, x + colw, 600], radius=18, fill=(32, 48, 80), outline=color, width=4)
        draw_text(d, (x + 24, 150), title, fit_font(d, title, colw - 48, 30, lang), fill=color)
        _bullets(d, x + 24, 220, items, lang, size=24, gap=30, max_w=colw - 30)
    note = ("判断: 文字の辺から押し棒の直線の押し方の候補を作り、2D 押しモデル（pymunk）で先読みして選ぶ"
            if ja else "Planning: point-pusher strokes from the letter edges, lookahead with a 2D push model")
    draw_text(d, (W / 2, 650), note, fit_font(d, note, W - 120, 22, lang), fill=MUTED, anchor="mm")
    return img


# ----- results data

def _perception_rows():
    """[(method, recall, pos95, ang95, ms)] from README's table, else hard-coded."""
    rows = []
    try:
        txt = (REPO / "README.md").read_text(encoding="utf-8")
        for line in txt.splitlines():
            m = re.match(r"\|\s*(classic|cnn|hybrid)[^|]*\|\s*([\d.]+%)\s*\|\s*([\d.]+%)\s*\|\s*([\d.]+ mm)\s*\|"
                         r"\s*([\d.]+°)\s*\|\s*(\d+ ms)", line)
            if m:
                rows.append(m.groups())
    except OSError:
        pass
    if len(rows) < 3:
        rows = [("classic", "99.7%", "99.9%", "0.61 mm", "0.90°", "28 ms"),
                ("cnn", "99.7%", "100%", "0.90 mm", "4.16°", "19 ms"),
                ("hybrid", "99.6%", "100%", "0.90 mm", "0.46°", "30 ms")]
    return rows


def _recovery_from_docs():
    """Rows [(label, {letter: 'k/n'}, backend)] from docs/recovery_*.md tables, or []."""
    out = []
    for name in ("recovery_results.md", "recovery_hand.md"):
        p = REPO / "docs" / name
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8")
        backend = "gazebo" if re.search(r"backend[ =:]*gazebo", txt, re.I) and "sim2d" not in txt else "sim2d"
        hdr = None
        for line in txt.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else None
            if not cells:
                hdr = None
                continue
            if hdr is None:
                hdr = [c.lower() for c in cells]
                continue
            if set("".join(cells)) <= set("-: "):
                continue
            row = dict(zip(hdr, cells))
            # wide table: columns I/U/Y/K with "k/n"
            if all(l.lower() in row for l in LETTERS):
                vals = {l: re.search(r"\d+\s*/\s*\d+", row[l.lower()]) for l in LETTERS}
                if all(vals.values()):
                    label = " / ".join(c for c in cells[: hdr.index("i")] if c)
                    out.append((label or name, {l: vals[l].group(0).replace(" ", "") for l in LETTERS}, backend))
        if out:
            return out
    return out


def _recovery_from_csv():
    groups: dict = {}
    files = [f for f in glob.glob(str(REPO / "data" / "recovery" / "**" / "*.csv"), recursive=True)
             if "smoke" not in f.lower() and "test" not in Path(f).name.lower()]
    for f in sorted(files, key=os.path.getmtime):
        try:
            rows = list(csv.DictReader(open(f, encoding="utf-8")))
        except Exception:
            continue
        per = defaultdict(lambda: [0, 0])
        keys = set()
        # Runs where the hand comes back mid-recovery are a different experiment.
        rehand = any(r.get("rehand_at_push") for r in rows) or "rehand" in Path(f).name.lower()
        pusher = "point" if "point" in Path(f).parent.name.lower() else r_pusher_default(rows)
        for r in rows:
            try:
                # One seed per table row, so seeds are never mixed in a row.
                if r.get("seed", "0") != "0":
                    continue
                k = (r.get("pusher") or pusher, r["backend"], r["perception"],
                     r["disturb"] + ("+rehand" if rehand else ""))
                per[(k, r["letter"])][0] += int(r["success"])
                per[(k, r["letter"])][1] += 1
                keys.add(k)
            except (KeyError, ValueError):
                continue
        for k in keys:  # newer files override older ones
            groups[k] = {l: f"{per[(k, l)][0]}/{per[(k, l)][1]}" for l in LETTERS if per[(k, l)][1]}
    order = {"truth": 0, "classic": 1, "cnn": 2, "hybrid": 3}
    out = []
    dist_order = {"teleport": 0, "hand": 1}
    pusher_order = {"point": 0, "paddles": 1}
    for (pusher, backend, perc, dist), v in sorted(
            groups.items(), key=lambda kv: (pusher_order.get(kv[0][0], 2), kv[0][1],
                                            order.get(kv[0][2], 9), dist_order.get(kv[0][3], 2))):
        if "+rehand" in dist:
            continue  # the hand coming back mid-recovery: another experiment
        if pusher == "paddles" and perc not in ("truth", "hybrid"):
            continue  # keep the card short: truth and the default perception
        out.append((f"{pusher}|{perc} / {dist}", v, backend))
    return out


def r_pusher_default(rows) -> str:
    """Pusher of a CSV without a pusher column (paddles unless the rows say so)."""
    return next((r["pusher"] for r in rows if r.get("pusher")), "paddles")


PUSHER_LABEL = {"ja": {"point": "押し棒", "paddles": "ヘラ"}, "en": {"point": "point", "paddles": "paddles"}}
PERC_LABEL = {"ja": {"truth": "真値"}, "en": {}}


def row_label(label: str, lang: str) -> str:
    """'point|truth / teleport' -> '押し棒 · 真値 / teleport'."""
    if "|" not in label:
        return label
    pusher, rest = label.split("|", 1)
    perc, _, dist = rest.partition(" / ")
    perc = PERC_LABEL.get(lang, {}).get(perc, perc)
    return f"{PUSHER_LABEL.get(lang, {}).get(pusher, pusher)} · {perc} / {dist}"


def recovery_rows():
    return _recovery_from_csv() or _recovery_from_docs()


def card_results(lang="ja", **_):
    img, d = _canvas()
    ja = lang == "ja"
    _heading(d, "結果" if ja else "Results", lang, y=28, size=40)
    # perception table
    f_h, f_c = font(20, lang), font(20, "en")
    t1 = "知覚（合成画像 1000 枚・遮蔽 10% 以下の 1110 文字）" if ja else "Perception (1000 synthetic frames, 1110 visible letters)"
    draw_text(d, (60, 110), t1, fit_font(d, t1, 1160, 24, lang), fill=ACCENT)
    hdr = ["方式", "検出率", "適合率", "位置誤差95%", "角度誤差95%", "1フレーム"] if ja else \
        ["method", "recall", "precision", "pos p95", "angle p95", "time/frame"]
    xs = [70, 300, 470, 640, 840, 1040]
    y = 150
    for x, h in zip(xs, hdr):
        draw_text(d, (x, y), h, f_h, fill=MUTED)
    y += 34
    for r in _perception_rows():
        d.line([(60, y - 6), (1220, y - 6)], fill=(70, 90, 130), width=1)
        for x, c in zip(xs, r):
            draw_text(d, (x, y), c, f_c, fill=FG if r[0] != "hybrid" else GREEN)
        y += 34
    # recovery table
    rows = recovery_rows()
    backends = {b for _, _, b in rows}
    model = "Gazebo" if backends == {"gazebo"} else ("2D 押しモデル (pymunk)" if ja else "2D push model (pymunk)")
    y += 26
    t2 = (f"崩し → 復帰の成功数（{model} での評価、seed 0、各 20 回、30 回押す以内）" if ja
          else f"Recovery after a disturbance — successes ({model}, seed 0, 20 each, ≤30 pushes)")
    draw_text(d, (60, y), t2, fit_font(d, t2, 1160, 24, lang), fill=ACCENT)
    y += 40
    xs2 = [70, 520, 680, 840, 1000]
    for x, h in zip(xs2, ["押す道具 · 知覚 / 崩し方" if ja else "pusher · perception / disturbance"] + LETTERS):
        draw_text(d, (x, y), h, f_h if x == 70 else font(22, "en"), fill=MUTED)
    y += 34
    if not rows:
        draw_text(d, (70, y), "（評価結果なし）" if ja else "(no results yet)", f_h, fill=MUTED)
    for label, vals, backend in rows[:7]:
        d.line([(60, y - 6), (1220, y - 6)], fill=(70, 90, 130), width=1)
        lab = row_label(label, lang) + ("" if backend == "sim2d" or backends == {backend} else f" [{backend}]")
        draw_text(d, (70, y), lab, fit_font(d, lab, 440, 20, lang),
                  fill=ACCENT if label.startswith("point|") else FG)
        for x, l in zip(xs2[1:], LETTERS):
            v = vals.get(l, "–")
            col = FG
            m = re.match(r"(\d+)/(\d+)", v)
            if m and int(m.group(2)):
                rate = int(m.group(1)) / int(m.group(2))
                goal = 0.9 if l in "IU" else 0.7
                col = GREEN if rate >= goal else ORANGE
            draw_text(d, (x, y), v, f_c, fill=col)
        y += 32
    note = ("※ 復帰の数値は 2D 押しモデル（pymunk）での評価で、Gazebo 上の数値ではありません。緑 = 目標達成（I・U 90%、Y・K 70%）"
            if ja and model != "Gazebo" else
            ("緑 = 目標達成（I・U 90%、Y・K 70%）" if ja else
             ("* recovery numbers are from the 2D model, not Gazebo. " if model != "Gazebo" else "")
             + "green = goal met (I,U 90%; Y,K 70%)"))
    draw_text(d, (60, H - 40), note, fit_font(d, note, 1160, 18, lang), fill=MUTED)
    return img


def card_future(lang="ja", **_):
    img, d = _canvas()
    _snow(d, 60, seed=9, keep_out=[(40, 20, 1240, 470), (40, 610, 400, 680), (900, 420, 1180, 700)])
    ja = lang == "ja"
    _heading(d, "次は実機へ" if ja else "Next: the real robot", lang, y=28, size=44)
    items = ([
        "XY ステージ（またはアーム）の先に押し棒（ピン）を付け、同じ直線の押しを実行",
        "文字はスチレンボードを切り出し（シミュと同じ寸法を scene.yaml から）",
        "Gazebo → USB カメラ、point_push_executor → XY ステージのドライバに差し替え",
        "/point_push_cmd と知覚・プランナー・手検出のノードはそのまま流用",
        "CNN は Gazebo・実写の画像で追加学習（合成画像との見え方の差を埋める）",
    ] if ja else [
        "An XY stage (or an arm) with a pusher pin runs the same straight strokes",
        "Letters cut from styrene board (same dimensions, from scene.yaml)",
        "Swap Gazebo → USB camera, point_push_executor → XY stage driver",
        "/point_push_cmd, perception, planner and hand nodes are reused as they are",
        "Fine-tune the CNN on Gazebo and real images (close the perception gap)",
    ])
    _bullets(d, 80, 140, items, lang, size=28, gap=26, max_w=1100)
    # tiny sketch: table + XY gantry + pusher pin
    cx, cy, s = 1040, 560, 80
    d.rectangle([cx - s, cy - s, cx + s, cy + s], outline=(200, 215, 240), width=3)
    d.rectangle([cx - s - 14, cy - s - 14, cx - s - 6, cy + s + 14], fill=(120, 140, 170))   # rails
    d.rectangle([cx + s + 6, cy - s - 14, cx + s + 14, cy + s + 14], fill=(120, 140, 170))
    gy = cy - 30
    d.rectangle([cx - s - 18, gy - 6, cx + s + 18, gy + 6], fill=(170, 186, 210))           # gantry
    px = cx + 34
    d.rounded_rectangle([px - 12, gy - 12, px + 12, gy + 12], radius=4, fill=(90, 104, 130))
    d.ellipse([px - 7, gy - 7, px + 7, gy + 7], fill=ORANGE)                                 # pin
    draw_text(d, (cx - 10, cy + 34), "YUKI", font(30, "en"), fill=FG, anchor="mm")
    draw_text(d, (80, 640), "To be continued…", font(30, "en"), fill=ACCENT)
    return img


def card_text(lang="ja", heading: str = "", lines: list[str] | None = None, big: str | None = None, **_):
    """Generic card: heading + bullet lines, or one big centred line."""
    img, d = _canvas()
    if big:
        draw_text(d, (W / 2, H / 2), big, fit_font(d, big, W - 120, 64, lang), fill=FG, anchor="mm")
        if heading:
            draw_text(d, (W / 2, H / 2 + 80), heading, fit_font(d, heading, W - 120, 30, lang), fill=ACCENT,
                      anchor="mm")
        return img
    if heading:
        _heading(d, heading, lang, y=40)
    _bullets(d, 80, 150, lines or [], lang, size=30, gap=24)
    return img


CARDS = {
    "title": card_title,
    "diagram": card_diagram,
    "ai": card_ai,
    "results": card_results,
    "future": card_future,
    "text": card_text,
}


def render_card(name: str, lang: str = "auto", **opts) -> Image.Image:
    if name not in CARDS:
        raise ValueError(f"unknown card {name!r}; one of {sorted(CARDS)}")
    return CARDS[name](lang=resolve_lang(lang), **opts)


def card_clip(img: Image.Image, out: str | Path, duration: float, fps: int = 30) -> Path:
    """Write a still image as an H.264 clip (yuv420p, no audio)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    img.save(png)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-framerate", str(fps), "-i", str(png),
                    "-t", f"{duration:.3f}", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                    "-pix_fmt", "yuv420p", "-r", str(fps), "-map_metadata", "-1", str(out)], check=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/video/cards", help="output directory")
    ap.add_argument("--cards", default="title,diagram,ai,results,future", help="comma list")
    ap.add_argument("--lang", default="auto", choices=["auto", "ja", "en"])
    ap.add_argument("--course", default=None, help="course line on the title card (no personal names)")
    ap.add_argument("--mp4", type=float, default=0, metavar="SEC", help="also write SEC-second MP4 clips")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        print("cards:", ", ".join(CARDS))
        print("font (CJK):", find_font(True) or "none -> English cards")
        print("recovery rows:", recovery_rows())
        return 0
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"lang={resolve_lang(a.lang)} font={find_font(True) or find_font(False)}")
    for name in a.cards.split(","):
        img = render_card(name.strip(), a.lang, course=a.course)
        if a.mp4 > 0:
            p = card_clip(img, out / f"{name}.mp4", a.mp4)
        else:
            p = out / f"{name}.png"
            img.save(p)
        print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
