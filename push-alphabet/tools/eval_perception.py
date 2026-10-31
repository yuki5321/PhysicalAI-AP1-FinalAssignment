#!/usr/bin/env python3
"""Compare letter detectors (classic / cnn / hybrid) on a labelled dataset.

Dataset format: docs/cnn.md (meta.json, labels.jsonl, frames/*.png).

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_perception \
        python3 tools/eval_perception.py --data data/synth_val \
        --methods classic,cnn,hybrid --csv out.csv

Per method: classification accuracy, missed / false detections, position error
(mean / p95 / max, mm), symmetry-aware angle error (mean / p95 / max, deg) and
processing time per frame. Detections are matched to labels greedily by
nearest position (gate `--match-dist`). Labels outside the reported workspace
(|x| or |y| > workspace.half_size + workspace_margin) are not counted as
missed, and detections matched to them are not counted as false.

Each method is reported twice: subset "all" (every label) and subset
"occ<=0.1" (only labels whose top face is hidden by the paddle / hand for at
most `--occ-max` of its area, label key `occ`; the more occluded labels are
then ignored like out-of-workspace ones). Detection runs once per frame.

Extras: `--per-letter` prints recall / missed / wrong-class / false counts per
true letter (I, U, Y, K); `--json PATH` (or `--json` alone for stdout) writes
a machine-readable summary (all metrics, per-letter breakdown, confusion) for
scripts and regression checks (see docs/perception_eval.md).
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _src_root(argv):
    """`--src ROOT` (checked before the imports): load pa_core / pa_perception from
    another checkout (e.g. a baseline worktree); default: this repository."""
    for k, a in enumerate(argv):
        if a == "--src" and k + 1 < len(argv):
            return os.path.abspath(argv[k + 1])
        if a.startswith("--src="):
            return os.path.abspath(a.split("=", 1)[1])
    return os.path.join(_HERE, "..")


_SRC = _src_root(sys.argv[1:])
for _p in ("ros2_ws/src/pa_core", "ros2_ws/src/pa_perception"):
    _abs = os.path.join(_SRC, _p)
    if os.path.isdir(_abs) and _abs not in sys.path:
        sys.path.insert(0, _abs)

from pa_core.config import load_scene  # noqa: E402
from pa_core.letters import angle_error_deg  # noqa: E402
from pa_perception.classic import DetectorParams  # noqa: E402
from pa_perception.cnn_detector import make_detector  # noqa: E402
from pa_perception.dataset import load_dataset  # noqa: E402


# ------------------------------------------------------------------ dataset
def read_dataset(root):
    """[(image path, [(letter, x, y, theta, occ), ...]), ...] via pa_perception.dataset.

    occ = fraction of the top face hidden by the paddle / hand (0 when unlabelled).
    """
    return [(rec["path"],
             [(d["letter"], float(d["x"]), float(d["y"]), float(d["theta"]),
               float(d.get("occ", 0.0))) for d in rec["letters"]])
            for rec in load_dataset(root)]


# ------------------------------------------------------------------ metrics
def match(dets, labels, gate):
    """Greedy nearest-position matching: [(det idx, label idx, dist)]."""
    pairs = []
    for i, d in enumerate(dets):
        for j, (_, x, y, _, _) in enumerate(labels):
            dist = math.hypot(d.x - x, d.y - y)
            if dist < gate:
                pairs.append((dist, i, j))
    pairs.sort()
    used_d, used_l, out = set(), set(), []
    for dist, i, j in pairs:
        if i in used_d or j in used_l:
            continue
        used_d.add(i)
        used_l.add(j)
        out.append((i, j, dist))
    return out


def _stats(v):
    if not len(v):
        return float("nan"), float("nan"), float("nan")
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(np.percentile(a, 95)), float(a.max())


def run_detector(det, items, warmup=2):
    """[(dets, labels, seconds), ...]: one detect() per frame (images read outside the timer)."""
    out = []
    for k, (frame, labels) in enumerate(items):
        img = cv2.imread(frame, cv2.IMREAD_COLOR) if isinstance(frame, str) else frame
        if img is None:
            raise FileNotFoundError(frame)
        if k == 0:
            for _ in range(warmup):
                det.detect(img)
        t0 = time.perf_counter()
        dets = det.detect(img)
        out.append((dets, labels, time.perf_counter() - t0))
    return out


def score(results, scene, gate=0.03, margin=0.02, max_occ=1.0):
    """Metrics over run_detector() output.

    Counted labels: inside |x|,|y| <= workspace.half_size + margin and occ <= max_occ.
    Labels not counted are neither missed nor do detections matched to them count
    as false; unmatched detections are false positives.
    """
    lim = scene["workspace"]["half_size"] + margin
    n_lab = n_match = n_correct = n_missed = n_false = n_unknown = 0
    pos_mm, ang_deg, times = [], [], []
    confusion = {}
    per = {}

    def _pl(name):
        return per.setdefault(name, {"labels": 0, "correct": 0, "missed": 0, "wrong": 0,
                                     "unknown": 0, "false": 0})
    for dets, labels, sec in results:
        times.append(sec)
        inside = [abs(x) <= lim and abs(y) <= lim and occ <= max_occ for _, x, y, _, occ in labels]
        pairs = match(dets, labels, gate)
        matched_d = {i for i, _, _ in pairs}
        matched_l = {j for _, j, _ in pairs}
        n_lab += sum(inside)
        n_missed += sum(1 for j, ins in enumerate(inside) if ins and j not in matched_l)
        n_false += sum(1 for i in range(len(dets)) if i not in matched_d)
        for j, ins in enumerate(inside):
            if ins:
                _pl(labels[j][0])["labels"] += 1
                if j not in matched_l:
                    _pl(labels[j][0])["missed"] += 1
        for i in range(len(dets)):
            if i not in matched_d:
                _pl(dets[i].letter or "?")["false"] += 1
        for i, j, dist in pairs:
            if not inside[j]:
                continue
            d, (name, _, _, th, _) = dets[i], labels[j]
            n_match += 1
            confusion[(name, d.letter)] = confusion.get((name, d.letter), 0) + 1
            if not d.letter:
                n_unknown += 1
                _pl(name)["unknown"] += 1
                continue
            if d.letter != name:
                _pl(name)["wrong"] += 1
                continue
            n_correct += 1
            _pl(name)["correct"] += 1
            pos_mm.append(1000.0 * dist)
            ang_deg.append(abs(angle_error_deg(name, d.theta, th)))
    pm, pp, px = _stats(pos_mm)
    am, ap, ax = _stats(ang_deg)
    for v in per.values():
        v["recall"] = v["correct"] / v["labels"] if v["labels"] else float("nan")
    return {
        "frames": len(results), "labels": n_lab, "matched": n_match,
        "acc": n_correct / n_match if n_match else float("nan"),
        "recall": n_correct / n_lab if n_lab else float("nan"),
        "missed": n_missed, "false": n_false, "unknown": n_unknown,
        "pos_mean_mm": pm, "pos_p95_mm": pp, "pos_max_mm": px,
        "ang_mean_deg": am, "ang_p95_deg": ap, "ang_max_deg": ax,
        "ms_per_frame": 1000.0 * float(np.mean(times)) if times else float("nan"),
        "precision": n_correct / (n_match + n_false) if (n_match + n_false) else float("nan"),
        "confusion": confusion,
        "per_letter": per,
    }


def evaluate(det, items, scene, gate=0.03, margin=0.02, warmup=2, max_occ=1.0):
    return score(run_detector(det, items, warmup), scene, gate, margin, max_occ)


COLS = [("method", "{:<7}"), ("subset", "{:<8}"), ("frames", "{:>6}"), ("labels", "{:>6}"),
        ("acc", "{:>6.4f}"), ("recall", "{:>6.4f}"), ("missed", "{:>6}"), ("false", "{:>5}"),
        ("pos_mean_mm", "{:>8.2f}"), ("pos_p95_mm", "{:>8.2f}"), ("pos_max_mm", "{:>8.2f}"),
        ("ang_mean_deg", "{:>8.2f}"), ("ang_p95_deg", "{:>8.2f}"), ("ang_max_deg", "{:>8.2f}"),
        ("ms_per_frame", "{:>8.1f}")]
HEAD = {"pos_mean_mm": "pos_mean", "pos_p95_mm": "pos_p95", "pos_max_mm": "pos_max",
        "ang_mean_deg": "ang_mean", "ang_p95_deg": "ang_p95", "ang_max_deg": "ang_max",
        "ms_per_frame": "ms/frame"}


def format_table(rows):
    cells = [[HEAD.get(c, c) for c, _ in COLS]]
    cells += [[fmt.format(r[c]) for c, fmt in COLS] for r in rows]
    w = [max(len(row[k]) for row in cells) for k in range(len(COLS))]
    left = {"method", "subset"}
    lines = [" | ".join(v.ljust(w[k]) if COLS[k][0] in left else v.rjust(w[k])
                        for k, v in enumerate(row)) for row in cells]
    lines.insert(1, "-" * len(lines[0]))
    lines.append("(pos in mm, angle in deg (symmetry-aware); acc = correct / matched, "
                 "recall = correct / labels; missed / false = unmatched labels / detections)")
    return "\n".join(lines)


def format_markdown(rows):
    out = ["| " + " | ".join(HEAD.get(c, c) for c, _ in COLS) + " |",
           "|" + "|".join("---" if c in ("method", "subset") else "--:" for c, _ in COLS) + "|"]
    out += ["| " + " | ".join(fmt.format(r[c]).strip() for c, fmt in COLS) + " |" for r in rows]
    return "\n".join(out)


LETTER_ORDER = ("I", "U", "Y", "K")


def format_per_letter(rows, markdown=False):
    """Per true letter: labels, recall, missed, wrong class (incl. unknown), false (by predicted)."""
    hdr = ["method", "subset", "letter", "labels", "recall", "missed", "wrong", "false"]
    body = []
    for r in rows:
        per = r.get("per_letter", {})
        keys = [k for k in LETTER_ORDER if k in per] + sorted(k for k in per if k not in LETTER_ORDER)
        for k in keys:
            v = per[k]
            body.append([r["method"], r["subset"], k, str(v["labels"]),
                         "nan" if not v["labels"] else f"{v['recall']:.4f}", str(v["missed"]),
                         str(v["wrong"] + v["unknown"]), str(v["false"])])
    if markdown:
        out = ["| " + " | ".join(hdr) + " |", "|---|---|---|" + "--:|" * (len(hdr) - 3)]
        return "\n".join(out + ["| " + " | ".join(b) + " |" for b in body])
    w = [max(len(x) for x in col) for col in zip(hdr, *body)]
    lines = [" | ".join(v.ljust(w[k]) if k < 3 else v.rjust(w[k]) for k, v in enumerate(row))
             for row in [hdr] + body]
    lines.insert(1, "-" * len(lines[0]))
    return "\n".join(lines)


def _jsonable(v):
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, dict):
        return {(k if isinstance(k, str) else "->".join(map(str, k))): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _jsonable(float(v))
    return v


def json_summary(rows, args=None, extra=None):
    """JSON-ready dict: {"args": ..., "rows": [metrics per method/subset]}."""
    out = {"rows": [_jsonable(r) for r in rows]}
    if args is not None:
        out["args"] = {k: v for k, v in vars(args).items()}
    if extra:
        out.update(_jsonable(extra))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset root (labels.jsonl, frames/)")
    ap.add_argument("--methods", default="classic,cnn,hybrid")
    ap.add_argument("--model", default=None, help="ONNX model (default: pa_perception models/letter_cnn.onnx)")
    ap.add_argument("--csv", nargs="?", const="-", default=None,
                    help="also write the table as CSV to this file (no value: to stdout)")
    ap.add_argument("--markdown", action="store_true", help="also print a Markdown table")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N frames")
    ap.add_argument("--match-dist", type=float, default=0.03, help="matching gate [m]")
    ap.add_argument("--occ-max", type=float, default=0.1,
                    help="second subset: only labels with occ <= this (top face mostly visible)")
    ap.add_argument("--confusion", action="store_true", help="print the confusion counts")
    ap.add_argument("--src", default=None,
                    help="repository root to import pa_core / pa_perception from (default: this one)")
    ap.add_argument("--per-letter", action="store_true",
                    help="also print recall / missed / wrong / false per letter (I, U, Y, K)")
    ap.add_argument("--json", nargs="?", const="-", default=None,
                    help="write a JSON summary (metrics, per-letter, confusion) to this file "
                         "(no value: to stdout)")
    args = ap.parse_args(argv)

    scene = load_scene()
    items = read_dataset(args.data)
    if args.limit > 0:
        items = items[:args.limit]
    n_occ = sum(1 for _, l in items for t in l if t[4] > args.occ_max)
    print(f"[eval] {args.data}: {len(items)} frames, "
          f"{sum(len(l) for _, l in items)} labelled letters ({n_occ} with occ > {args.occ_max:g})")
    params = DetectorParams()
    subsets = [("all", 1.0), (f"occ<={args.occ_max:g}", args.occ_max)]
    rows = []
    for m in [s.strip() for s in args.methods.split(",") if s.strip()]:
        try:
            det, used = make_detector(m, model_path=args.model, scene=scene, params=params,
                                      fallback=False)
        except Exception as e:  # noqa: BLE001 - e.g. model or onnxruntime missing
            print(f"[eval] skip {m}: {e}", file=sys.stderr)
            continue
        results = run_detector(det, items)
        for name, occ in subsets:
            r = score(results, scene, gate=args.match_dist, margin=params.workspace_margin,
                      max_occ=occ)
            r["method"], r["subset"] = used, name
            rows.append(r)
            if args.confusion:
                print(f"  {used} {name} confusion (true, predicted): "
                      f"{dict(sorted(r['confusion'].items()))}")
        print(f"[eval] {m}: done ({rows[-1]['ms_per_frame']:.1f} ms/frame)", file=sys.stderr)
    if not rows:
        print("[eval] nothing evaluated", file=sys.stderr)
        return 1
    print(format_table(rows))
    if args.per_letter:
        print(format_per_letter(rows))
    if args.markdown:
        print(format_markdown(rows))
        if args.per_letter:
            print(format_per_letter(rows, markdown=True))
    if args.json:
        summary = json_summary(rows, args, {"dataset": os.path.abspath(args.data), "frames": len(items)})
        txt = json.dumps(summary, indent=1, sort_keys=True)
        if args.json == "-":
            print(txt)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
            with open(args.json, "w", encoding="utf-8") as f:
                f.write(txt + "\n")
            print(f"[eval] wrote {args.json}")
    if args.csv:
        hdr = [c for c, _ in COLS] + ["matched", "unknown", "precision"]
        body = [[r[c] for c in hdr] for r in rows]
        if args.csv == "-":
            w = csv.writer(sys.stdout)
            w.writerow(hdr)
            w.writerows(body)
        else:
            os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
            with open(args.csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(hdr)
                w.writerows(body)
            print(f"[eval] wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
