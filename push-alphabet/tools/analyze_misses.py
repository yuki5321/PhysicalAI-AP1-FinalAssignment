#!/usr/bin/env python3
"""Why are letters missed? Classify every recall loss and false positive by cause.

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_perception \
        python3 tools/analyze_misses.py --data data/synth_val --methods classic,cnn,hybrid \
        --out /tmp/misses            # optional: contact sheets + per-item CSV + JSON

The detectors are run unchanged (same matching / counting as
tools/eval_perception.py). The cause of each loss is recomputed outside the
detectors from the foreground segmentation, the ground-truth silhouettes and a
simple HSV occluder mask, so no detector instrumentation is needed:

Recall losses (a counted label that is not detected with the right letter),
first rule that applies wins:

  wrong_class      matched (within --match-dist) but with another letter or ""
  merged_letter    its foreground blob (the one holding most of its visible top face) also
                   covers >= 15 % of another letter's top face (occluder pixels excluded)
  merged_occluder  its blob touches the orange-paddle / skin mask and has >= 10 % of a
                   letter area of pixels outside the letter silhouette
  mislocalized     a detection came from its blob but landed outside the gate
  no_foreground    < 15 % of its top face is non-occluder foreground (nothing at the true centre)
  fragmented       its visible top face is split over several blobs (largest < 80 %), e.g. a
                   paddle lying across it
  area_small       blob area / letter area < params.area_min_ratio
  area_large       blob area / letter area > params.area_max_ratio
  outside_ws       blob centroid beyond the workspace check
  shape_rejected   plausible blob, but no detection (IoU / confidence below the gates)

`heavy_occ` replaces mislocalized / no_foreground / area_* / shape_rejected when
the label's occ > 0.3 (the mechanical cause is still in the second table).

False positives (unmatched detections):

  duplicate        within 2 gates of a label that another detection already matched
  mislocalized     within 2 gates of an unmatched label (pose off; its label is a miss too)
  occluder         its blob touches the occluder mask
  background       anything else (plate, hub, texture)

Segmentation used for the causes (`--seg`): `auto` = pa_perception.segmentation
.segment_letters if that module exists (split blobs), else
ClassicDetector.segment + connected components; `classic` forces the latter.
"""

import argparse
import csv
import json
import math
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def _src_root(argv):
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from pa_core.config import load_scene  # noqa: E402
from pa_perception.classic import ClassicDetector, DetectorParams  # noqa: E402
from pa_perception.cnn_detector import make_detector  # noqa: E402
from pa_perception.synth import fill_polys, letter_pixel_polys  # noqa: E402
import eval_perception as ev  # noqa: E402

MISS_CATS = ["wrong_class", "merged_letter", "merged_occluder", "mislocalized", "heavy_occ",
             "no_foreground", "fragmented", "area_small", "area_large", "outside_ws",
             "shape_rejected"]
MECH_CATS = [c for c in MISS_CATS if c != "heavy_occ"]
FP_CATS = ["duplicate", "mislocalized", "occluder", "background"]
OCC_BUCKETS = [("occ<=0.1", 0.0, 0.1), ("0.1<occ<=0.3", 0.1, 0.3), ("occ>0.3", 0.3, 1.01)]
HEAVY_OCC = 0.3


# ------------------------------------------------------------------ masks
def occluder_mask(img):
    """uint8 0/255: orange paddle / skin-coloured pixels (saturated, warm hue, not dark)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    warm = (h <= 30) | (h >= 172)
    m = (warm & (s >= 60) & (v >= 70)).astype(np.uint8) * 255
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


class Segmenter:
    """(fg uint8, labels int32) per image, the way the detectors would see it."""

    def __init__(self, mode, scene, params, cam):
        self.classic = ClassicDetector(scene=scene, params=params, cam=cam)
        self.fn, self.name = None, "classic"
        if mode in ("auto", "module"):
            try:
                from pa_perception.segmentation import segment_letters
                self.fn, self.name = segment_letters, "segmentation.segment_letters"
                self.scene, self.cam, self.params = scene, cam, params
            except Exception as e:  # noqa: BLE001 - module absent (baseline)
                if mode == "module":
                    raise
                _ = e

    def __call__(self, img):
        if self.fn is not None:
            try:
                seg = self.fn(img, scene=self.scene, cam=self.cam)
            except TypeError:
                seg = self.fn(img)
            fg = np.asarray(seg.fg, np.uint8)
            lab = getattr(seg, "labels", None)
            if lab is None:
                _, lab = cv2.connectedComponents(fg, connectivity=8)
            return fg, np.asarray(lab, np.int32)
        fg, _ = self.classic.segment(img)
        _, lab = cv2.connectedComponents(fg, connectivity=8)
        return fg, lab.astype(np.int32)


# ------------------------------------------------------------------ analysis
class Analyzer:
    def __init__(self, scene, params, seg="auto", gate=0.03, max_occ=1.0):
        self.scene, self.p, self.gate = scene, params, gate
        self.classic = ClassicDetector(scene=scene, params=params)
        self.cam, self.t = self.classic.cam, self.classic.t
        self.area_px = self.classic.area_px
        self.seg = Segmenter(seg, scene, params, self.cam)
        self.lim = scene["workspace"]["half_size"] + params.workspace_margin
        self.max_occ = max_occ
        self._k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _label_masks(self, shape, lab):
        name, x, y, th = lab[:4]
        tops, hulls = letter_pixel_polys(self.cam, name, x, y, th, self.t, sides=True)
        top = np.zeros(shape, np.uint8)
        fill_polys(top, tops, 1, aa=False)
        sil = np.zeros(shape, np.uint8)
        fill_polys(sil, hulls + tops, 1, aa=False)
        sil = cv2.dilate(sil, self._k5, iterations=1)
        return top.astype(bool), sil.astype(bool)

    def frame(self, img, dets, labels):
        """(miss records, fp records, n_counted, n_ok) of one frame."""
        H, W = img.shape[:2]
        fg, cc = self.seg(img)
        occm = occluder_mask(img)
        occ_dil = cv2.dilate(occm, self._k5, iterations=2) > 0
        inside = [abs(x) <= self.lim and abs(y) <= self.lim and occ <= self.max_occ
                  for _, x, y, _, occ in labels]
        pairs = ev.match(dets, labels, self.gate)
        d2l = {i: j for i, j, _ in pairs}
        l2d = {j: i for i, j, _ in pairs}
        masks = [self._label_masks((H, W), lab) for lab in labels]
        all_sil = np.zeros((H, W), bool)
        for _, s in masks:
            all_sil |= s

        def det_blob(d):
            c = getattr(d, "centroid_px", None)
            if c is None:
                return 0
            u, v = int(round(c[0])), int(round(c[1]))
            return int(cc[v, u]) if 0 <= u < W and 0 <= v < H else 0

        unmatched_d = [i for i in range(len(dets)) if i not in d2l]
        unmatched_blobs = {det_blob(dets[i]) for i in unmatched_d} - {0}
        misses, fps, n_cnt, n_ok = [], [], 0, 0
        for j, lab in enumerate(labels):
            if not inside[j]:
                continue
            n_cnt += 1
            name, x, y, th, occ = lab
            if j in l2d and dets[l2d[j]].letter == name:
                n_ok += 1
                continue
            top, sil = masks[j]
            top_area = int(top.sum())
            info = {"letter": name, "x": x, "y": y, "theta": th, "occ": occ}
            blob_area = 0
            if j in l2d:
                mech = "wrong_class"
                info["pred"] = dets[l2d[j]].letter
            else:
                fg_in = cc[top & (fg > 0) & (occm == 0)]
                fg_in = fg_in[fg_in > 0]
                if fg_in.size < 0.15 * max(top_area, 1):
                    mech = "no_foreground"
                else:
                    hist = np.bincount(fg_in)
                    b = int(hist.argmax())
                    frag = hist[b] < 0.8 * fg_in.size
                    bm = cc == b
                    blob_area = int(bm.sum())
                    ratio = blob_area / self.area_px[name]
                    bm_vis = bm & (occm == 0)
                    merged = any((bm_vis & masks[k][0]).sum() >= 0.15 * max(masks[k][0].sum(), 1)
                                 for k in range(len(labels)) if k != j)
                    extra = int((bm & ~all_sil).sum())
                    m = cv2.moments(bm.astype(np.uint8), binaryImage=True)
                    cu, cv_ = m["m10"] / m["m00"], m["m01"] / m["m00"]
                    xw, yw = self.cam.pixel_to_world([[cu, cv_]], self.t)[0]
                    if merged:
                        mech = "merged_letter"
                    elif (bm & occ_dil).any() and extra >= 0.1 * self.area_px[name]:
                        mech = "merged_occluder"
                    elif b in unmatched_blobs:
                        mech = "mislocalized"
                    elif frag:
                        mech = "fragmented"
                    elif ratio < self.p.area_min_ratio:
                        mech = "area_small"
                    elif ratio > self.p.area_max_ratio:
                        mech = "area_large"
                    elif abs(xw) > self.lim + 0.01 or abs(yw) > self.lim + 0.01:
                        mech = "outside_ws"
                    else:
                        mech = "shape_rejected"
                    info["area_ratio"] = round(ratio, 3)
            cat = mech
            if occ > HEAVY_OCC and mech in ("mislocalized", "no_foreground", "fragmented",
                                            "area_small", "area_large", "shape_rejected"):
                cat = "heavy_occ"
            info.update(cat=cat, mech=mech, blob_area=blob_area,
                        center_px=tuple(self.cam.world_to_pixel([[x, y]], self.t)[0]))
            misses.append(info)
        matched_l = set(l2d)
        for i in unmatched_d:
            d = dets[i]
            near = [(math.hypot(d.x - lab[1], d.y - lab[2]), k) for k, lab in enumerate(labels)]
            near = [nk for nk in near if nk[0] < 2 * self.gate]
            b = det_blob(d)
            if near:
                k = min(near)[1]
                cat = "duplicate" if k in matched_l else "mislocalized"
            elif (b and ((cc == b) & occ_dil).any()) or (not b and self._occ_at(occ_dil, d)):
                cat = "occluder"
            else:
                cat = "background"
            c = getattr(d, "centroid_px", None)
            if c is None:
                c = tuple(self.cam.world_to_pixel([[d.x, d.y]], self.t)[0])
            fps.append({"cat": cat, "pred": d.letter, "x": d.x, "y": d.y, "center_px": tuple(c),
                        "iou": float(getattr(d, "iou", float("nan")))})
        return misses, fps, n_cnt, n_ok

    def _occ_at(self, occ_dil, d):
        u, v = self.cam.world_to_pixel([[d.x, d.y]], self.t)[0]
        u, v = int(round(u)), int(round(v))
        H, W = occ_dil.shape
        return bool(occ_dil[max(v - 10, 0):min(v + 11, H), max(u - 10, 0):min(u + 11, W)].any())


# ------------------------------------------------------------- contact sheet
def _tile(img, center, cc_fg, occm, labels_px, dets, caption, half=72, out=160):
    H, W = img.shape[:2]
    u, v = int(round(center[0])), int(round(center[1]))
    vis = img.copy()
    tint = vis.copy()
    tint[occm > 0] = (0, 0, 255)
    vis = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)
    cnts, _ = cv2.findContours(cc_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(vis, cnts, -1, (0, 255, 255), 1)
    for polys in labels_px:
        for p in polys:
            cv2.polylines(vis, [np.round(p).astype(np.int32)], True, (0, 220, 0), 1)
    for d in dets:
        c = getattr(d, "centroid_px", None)
        if c is not None:
            cv2.circle(vis, (int(c[0]), int(c[1])), 4, (255, 0, 255), 1)
    pad = cv2.copyMakeBorder(vis, half, half, half, half, cv2.BORDER_CONSTANT, value=0)
    crop = pad[v:v + 2 * half, u:u + 2 * half]
    crop = cv2.resize(crop, (out, out), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(crop, (0, 0), (out - 1, 13), (0, 0, 0), -1)
    cv2.putText(crop, caption[:28], (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1,
                cv2.LINE_AA)
    return crop


def contact_sheet(rows, path, cols=8):
    """rows: [(category, [tiles])] -> one PNG with a header line per category."""
    blocks = []
    tw = 160
    for cat, tiles in rows:
        if not tiles:
            continue
        hdr = np.zeros((18, cols * tw, 3), np.uint8)
        cv2.putText(hdr, cat, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        blocks.append(hdr)
        for r in range(0, len(tiles), cols):
            row = tiles[r:r + cols]
            row += [np.zeros_like(tiles[0])] * (cols - len(row))
            blocks.append(np.hstack(row))
    if blocks:
        cv2.imwrite(path, np.vstack(blocks))


# ------------------------------------------------------------------ driver
def analyze_method(det, items, an, examples=0, rng_seed=0):
    """Counts and records of one detector over items; optional example tiles."""
    miss, fps, results = [], [], []
    import time
    for k, (frame, labels) in enumerate(items):
        img = cv2.imread(frame, cv2.IMREAD_COLOR) if isinstance(frame, str) else frame
        t0 = time.perf_counter()
        dets = det.detect(img)
        results.append((dets, labels, time.perf_counter() - t0))
        m, f, _, _ = an.frame(img, dets, labels)
        for r in m + f:
            r["frame"] = k if not isinstance(frame, str) else os.path.basename(frame)
            r["_k"] = k
        miss += m
        fps += f
    return miss, fps, results


def tiles_for(records, cats, items, an, det_cache, per_cat, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for cat in cats:
        rs = [r for r in records if r["cat"] == cat]
        if not rs:
            continue
        pick = rng.choice(len(rs), size=min(per_cat, len(rs)), replace=False)
        tiles = []
        for p in sorted(pick):
            r = rs[p]
            frame, labels = items[r["_k"]]
            img = cv2.imread(frame, cv2.IMREAD_COLOR) if isinstance(frame, str) else frame
            fg, _ = an.seg(img)
            lp = [letter_pixel_polys(an.cam, *lab[:4], an.t, sides=False)[0] for lab in labels]
            cap = (f"{r['frame']} {r.get('letter', '')}->{r.get('pred', '-') or '?'} "
                   f"occ{r.get('occ', 0):.2f}")
            tiles.append(_tile(img, r["center_px"], fg, occluder_mask(img), lp, det_cache[r["_k"]][0],
                               cap.replace(".png", "")))
        out.append((cat, tiles))
    return out


def count(records, cats, max_occ=1.0, key="cat"):
    c = {k: 0 for k in cats}
    for r in records:
        if r.get("occ", 0.0) <= max_occ:
            c[r[key]] = c.get(r[key], 0) + 1
    return c


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|---" + "|--:" * (len(header) - 1) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--methods", default="classic,cnn,hybrid")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match-dist", type=float, default=0.03)
    ap.add_argument("--occ-max", type=float, default=0.1, help="'visible' subset: occ <= this")
    ap.add_argument("--seg", choices=["auto", "classic", "module"], default="auto")
    ap.add_argument("--src", default=None, help="repository root to import pa_core / pa_perception from")
    ap.add_argument("--out", default=None, help="directory for contact sheets, records CSV, summary JSON")
    ap.add_argument("--examples", type=int, default=8, help="contact-sheet tiles per category")
    args = ap.parse_args(argv)

    scene = load_scene()
    params = DetectorParams()
    items = ev.read_dataset(args.data)
    if args.limit > 0:
        items = items[:args.limit]
    an = Analyzer(scene, params, seg=args.seg, gate=args.match_dist)
    print(f"[misses] {args.data}: {len(items)} frames; causes from {an.seg.name}; src {_SRC}",
          file=sys.stderr)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    summary = {"dataset": os.path.abspath(args.data), "frames": len(items), "seg": an.seg.name,
               "methods": {}}
    methods = []
    for m in [s.strip() for s in args.methods.split(",") if s.strip()]:
        try:
            det, used = make_detector(m, model_path=args.model, scene=scene, params=params,
                                      fallback=False)
        except Exception as e:  # noqa: BLE001
            print(f"[misses] skip {m}: {e}", file=sys.stderr)
            continue
        miss, fps, results = analyze_method(det, items, an)
        s_all = ev.score(results, scene, gate=args.match_dist, margin=params.workspace_margin)
        s_vis = ev.score(results, scene, gate=args.match_dist, margin=params.workspace_margin,
                         max_occ=args.occ_max)
        summary["methods"][used] = {
            "all": {"labels": s_all["labels"], "recall": s_all["recall"], "false": s_all["false"],
                    "miss": count(miss, MISS_CATS), "mech": count(miss, MECH_CATS, key="mech"),
                    "fp": count(fps, FP_CATS)},
            "visible": {"labels": s_vis["labels"], "recall": s_vis["recall"],
                        "miss": count(miss, MISS_CATS, args.occ_max),
                        "mech": count(miss, MECH_CATS, args.occ_max, key="mech")},
            "mech_by_occ": {b: count([r for r in miss if lo < r["occ"] <= hi or (lo == 0 and r["occ"] == 0)],
                                     MECH_CATS, key="mech") for b, lo, hi in OCC_BUCKETS},
        }
        methods.append(used)
        print(f"[misses] {used}: {len(miss)} recall losses, {len(fps)} false positives", file=sys.stderr)
        if args.out:
            with open(os.path.join(args.out, f"{used}_records.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["kind", "frame", "cat", "mech", "letter", "pred", "occ", "x", "y",
                            "area_ratio", "blob_area"])
                for r in miss:
                    w.writerow(["miss", r["frame"], r["cat"], r["mech"], r["letter"], r.get("pred", ""),
                                r["occ"], r["x"], r["y"], r.get("area_ratio", ""), r["blob_area"]])
                for r in fps:
                    w.writerow(["fp", r["frame"], r["cat"], "", "", r["pred"], "", r["x"], r["y"], "", ""])
            if args.examples > 0:
                rows = tiles_for(miss, MISS_CATS, items, an, results, args.examples)
                rows += [("FP " + c, t) for c, t in tiles_for(fps, FP_CATS, items, an, results,
                                                              args.examples)]
                contact_sheet(rows, os.path.join(args.out, f"{used}_misses.png"))
    if not methods:
        return 1

    def f(v):
        return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}"
    S = summary["methods"]
    lines = ["### Recall losses by cause (counted labels; all / occ<=%g)" % args.occ_max, ""]
    hdr = ["cause"] + [f"{m} all" for m in methods] + [f"{m} vis" for m in methods]
    rows = [["labels"] + [S[m]["all"]["labels"] for m in methods] + [S[m]["visible"]["labels"] for m in methods],
            ["recall"] + [f(S[m]["all"]["recall"]) for m in methods] + [f(S[m]["visible"]["recall"]) for m in methods]]
    rows += [[c] + [S[m]["all"]["miss"][c] for m in methods] + [S[m]["visible"]["miss"][c] for m in methods]
             for c in MISS_CATS]
    rows.append(["**total lost**"] + [sum(S[m]["all"]["miss"].values()) for m in methods]
                + [sum(S[m]["visible"]["miss"].values()) for m in methods])
    lines.append(md_table(hdr, rows))
    lines += ["", "### Mechanical cause by occlusion bucket (all labels; heavy_occ not applied)", ""]
    hdr = ["cause"] + [f"{m} {b}" for m in methods for b, _, _ in OCC_BUCKETS]
    rows = [[c] + [S[m]["mech_by_occ"][b][c] for m in methods for b, _, _ in OCC_BUCKETS]
            for c in MECH_CATS]
    lines.append(md_table(hdr, rows))
    lines += ["", "### False positives by cause (all labels)", ""]
    rows = [["false (eval)"] + [S[m]["all"]["false"] for m in methods]]
    rows += [[c] + [S[m]["all"]["fp"][c] for m in methods] for c in FP_CATS]
    lines.append(md_table(["cause"] + methods, rows))
    txt = "\n".join(lines)
    print(txt)
    if args.out:
        with open(os.path.join(args.out, "summary.json"), "w") as fh:
            json.dump(ev._jsonable(summary), fh, indent=1, sort_keys=True)
        with open(os.path.join(args.out, "tables.md"), "w") as fh:
            fh.write(txt + "\n")
        print(f"[misses] wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
