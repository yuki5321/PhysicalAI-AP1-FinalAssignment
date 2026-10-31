"""Letter segmentation shared by the classic and CNN detectors (docs/cnn.md, "前段の分割").

numpy / OpenCV only (Python 3.10, numpy < 1.25). Pipeline per BGR frame:

1. Occluders (half resolution): orange paddles and skin-colored hands are
   bright, saturated and red-to-yellow (R >= G >= B, (R - B) / R large). Those
   pixels form `occluder` and are never foreground, so a paddle or hand touching
   a letter no longer merges with it.
2. Background: the table brightness is a smooth quadratic surface (illumination
   gradient, vignette) fitted to the table pixels of the plate ROI on an 8x
   downscaled image; letters, occluders and dark hubs are rejected iteratively
   as outliers.
3. Threshold: illumination is multiplicative, so a letter is a fixed multiple
   K of the local table brightness. K is estimated from the bright candidate
   pixels; the threshold surface is bg * (1 + alpha (K - 1)), never closer than
   `min_contrast` gray levels to the table (no letters: bg + fallback_contrast).
   Dim letters in a dark corner are found although a single global threshold
   misses them.
4. Blobs: 8-connected components of the foreground inside the table ROI. A
   blob large enough to hold two letters is eroded (distance transform) until
   it falls apart into letter-sized cores, which are grown back with a
   watershed (two letters touching at a corner or a short edge). Blob
   `touches_occluder` is set when its rim is within `touch_px` of an occluder
   (the matcher treats occluder pixels as unseen).
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from pa_core.config import load_scene
from pa_core.letters import letter_area, letter_names, letter_polygons
from pa_perception.camera_model import CameraModel
from pa_perception.synth import fill_polys

_CCL = getattr(cv2, "CCL_GRANA", getattr(cv2, "CCL_DEFAULT", -1))


@dataclass
class SegParams:
    # occluders (orange paddles, skin); evaluated at half resolution
    occ_min_sat: float = 0.25        # (R - B) / R
    occ_min_chroma: int = 25         # R - B [gray levels]
    occ_min_red: int = 45            # darker pixels are never occluders (hubs, shadows)
    occ_yellow_tol: int = 12         # G may exceed R by this much (yellowish skin, JPEG)
    occ_open: int = 0                # opening at half res (speckle removal; 0/1: off)
    occ_close: int = 3               # closing at half res (gaps inside paddles / hands)
    occ_guard_px: int = 1            # fg pixels this close to an occluder are dropped
    # background (table) model
    bg_down: int = 8                 # downscale factor of the fit
    bg_iters: int = 3
    bg_hi: float = 12.0              # residual above this (gray levels) = not table
    bg_lo_mad: float = 4.0           # residual below -bg_lo_mad * MAD = not table
    roi_erode_px: int = 3            # ignore the plate edge in the fit
    # threshold
    threshold: int = -1              # >= 0: fixed global gray threshold (no background model)
    min_contrast: float = 25.0       # threshold >= bg + min_contrast
    fallback_contrast: float = 40.0  # no letter-like pixels: threshold = bg + this
    level_pct: float = 90.0          # letter level K = this percentile of gray / bg
    level_alpha: float = 0.45        # threshold = bg * (1 + alpha (K - 1))
    max_threshold: float = 245.0
    open_ksize: int = 3              # opening of the foreground mask (0/1: off)
    # blobs
    min_area_ratio: float = 0.12     # blobs below this * smallest letter area are dropped
    max_area_ratio: float = 2.5      # ... above this * largest letter area are dropped
    split: bool = True
    split_area_ratio: float = 1.25   # try splitting above this * smallest letter area ...
    split_occ_area_ratio: float = 1.1  # ... or, next to an occluder, above this * largest
    split_max_erode: float = 0.4     # max erosion radius (* stroke width)
    split_min_part: float = 0.35     # each part >= this * smallest letter area
    touch_px: int = 2                # rim distance for touches_occluder
    merge: bool = True               # join fragments of one letter cut apart by an occluder
    merge_gap: float = 1.5           # max gap between fragments (* stroke width)
    merge_bridge_occ: float = 0.5    # min occluder fraction of the gap between them
    merge_max_area: float = 1.05     # merged area <= this * largest letter area
    merge_max_diam: float = 1.1      # merged bbox diagonal <= this * largest letter diameter
    keep_background: bool = False    # store the full-resolution background model


@dataclass
class Blob:
    label: int                       # value in Segmentation.labels
    centroid_px: tuple               # (u, v)
    area_px: int
    bbox: tuple                      # (u0, v0, w, h)
    touches_occluder: bool
    split_from: object = None        # label of the merged component it was split from, else None
    merged_from: tuple = ()          # labels of fragments joined into this blob across an occluder


@dataclass
class Segmentation:
    fg: np.ndarray                   # uint8 0/255 letter candidate pixels
    occluder: np.ndarray             # uint8 0/255 paddles / hands
    blobs: list
    labels: np.ndarray               # int32, 0 = background
    threshold: float                 # gray threshold at the image center
    background: object = None        # float32 (H, W) table model if params.keep_background
    letter_level: float = float("nan")  # estimated letter / table brightness ratio K
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------- geometry
class _Geometry:
    def __init__(self, scene, cam, p: SegParams):
        self.H, self.W = cam.height_px, cam.width_px
        self.Hh, self.Wh = (self.H + 1) // 2, (self.W + 1) // 2
        t = float(scene["letter"]["thickness"])
        ppm = cam.px_per_m(t)
        areas = [letter_area(n) * ppm ** 2 for n in letter_names()]
        self.a_min, self.a_max = min(areas), max(areas)
        self.stroke_px = float(scene["letter"].get("stroke", 0.016)) * ppm
        diam = 0.0
        for n in letter_names():
            c = np.vstack(letter_polygons(n))
            diam = max(diam, float(np.max(np.linalg.norm(c[:, None] - c[None], axis=-1))))
        side = 0.5 * np.hypot(cam.width_px, cam.height_px) * t / (cam.height - t)
        self.diam_px = diam * ppm + side
        h = scene["workspace"]["table_half_size"]
        table = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        self.roi = np.zeros((self.H, self.W), np.uint8)
        fill_polys(self.roi, [cam.world_to_pixel(table, 0.0)], 255, aa=False)
        self.roi_h = cv2.resize(self.roi, (self.Wh, self.Hh), interpolation=cv2.INTER_NEAREST)
        self.cu = min(max(int(round(cam.cx)), 0), self.W - 1)
        self.cv = min(max(int(round(cam.cy)), 0), self.H - 1)
        d = max(1, int(p.bg_down))
        self.hs, self.ws = max(2, -(-self.H // d)), max(2, -(-self.W // d))
        roi_e = self.roi
        if p.roi_erode_px > 0:
            k = 2 * p.roi_erode_px + 1
            roi_e = cv2.erode(self.roi, np.ones((k, k), np.uint8))
        self.roi_small = cv2.resize(roi_e, (self.ws, self.hs), interpolation=cv2.INTER_AREA) == 255
        ys, xs = np.mgrid[0:self.hs, 0:self.ws].astype(np.float64)
        sx, sy = self.W / self.ws, self.H / self.hs
        xn = ((xs + 0.5) * sx - 0.5 - cam.cx) / (self.W / 2)
        yn = ((ys + 0.5) * sy - 0.5 - cam.cy) / (self.H / 2)
        self.basis = np.stack([np.ones_like(xn), xn, yn, xn * xn, xn * yn, yn * yn], -1).reshape(-1, 6)
        self._disk = {}

    def kernel(self, k):
        k = int(k)
        if k not in self._disk:
            self._disk[k] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return self._disk[k]


_GEOM = {}


def _geometry(scene, cam, p):
    key = (id(scene), cam.f, cam.cx, cam.cy, cam.width_px, cam.height_px, cam.height,
           p.bg_down, p.roi_erode_px)
    g = _GEOM.get(key)
    if g is None:
        if len(_GEOM) > 16:
            _GEOM.clear()
        g = _GEOM[key] = _Geometry(scene, cam, p)
    return g


_DEFAULT = {}


def _defaults():
    if not _DEFAULT:
        _DEFAULT["scene"] = load_scene()
        _DEFAULT["cam"] = CameraModel.from_scene(_DEFAULT["scene"])
    return _DEFAULT["scene"], _DEFAULT["cam"]


def _u8(a):
    return np.clip(np.rint(a), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- stages
def occluder_mask(img_bgr, p: SegParams = None, roi=None):
    """uint8 0/255 mask (same size as img_bgr) of saturated red..yellow pixels (paddles, skin)."""
    p = p or SegParams()
    b, g, r = cv2.split(img_bgr)
    chroma = cv2.subtract(r, b)                                   # R - B, >= 0
    m = cv2.compare(chroma, cv2.convertScaleAbs(r, alpha=p.occ_min_sat), cv2.CMP_GE)
    m = cv2.bitwise_and(m, cv2.compare(chroma, p.occ_min_chroma, cv2.CMP_GE))
    m = cv2.bitwise_and(m, cv2.compare(r, p.occ_min_red, cv2.CMP_GE))
    m = cv2.bitwise_and(m, cv2.compare(g, cv2.add(r, p.occ_yellow_tol), cv2.CMP_LE))
    m = cv2.bitwise_and(m, cv2.compare(g, b, cv2.CMP_GE))
    if p.occ_open and p.occ_open > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((p.occ_open, p.occ_open), np.uint8))
    if p.occ_close and p.occ_close > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((p.occ_close, p.occ_close), np.uint8))
    if roi is not None:
        m = cv2.bitwise_and(m, roi)
    return m


def fit_background(gray_h, occ_h, geo: _Geometry, p: SegParams):
    """(hs, ws) float32 table brightness model, or None if no table is visible."""
    gs = cv2.resize(gray_h, (geo.ws, geo.hs), interpolation=cv2.INTER_AREA).astype(np.float32)
    os_ = cv2.resize(occ_h, (geo.ws, geo.hs), interpolation=cv2.INTER_AREA)
    valid = geo.roi_small & (os_ == 0)
    vals = gs[valid]
    if vals.size < 50:
        return None
    sel = valid & (gs <= float(np.median(vals)) + 2 * p.bg_hi)
    gflat = gs.reshape(-1).astype(np.float64)
    coef = None
    for _ in range(max(1, p.bg_iters)):
        idx = np.flatnonzero(sel.reshape(-1))
        if idx.size < 30:
            break
        A = geo.basis[idx]
        try:
            coef = np.linalg.solve(A.T @ A + 1e-6 * np.eye(6), A.T @ gflat[idx])
        except np.linalg.LinAlgError:
            break
        res = gs - (geo.basis @ coef).reshape(geo.hs, geo.ws)
        mad = float(np.median(np.abs(res[sel]))) * 1.4826 + 1.0
        sel = valid & (res <= p.bg_hi) & (res >= -p.bg_lo_mad * mad)
    if coef is None:
        return None
    model = (geo.basis @ coef).reshape(geo.hs, geo.ws).astype(np.float32)
    np.maximum(model, 4.0, out=model)
    return model


def _threshold_small(gray_h, bg_s, occ_h, geo, p):
    """(hs, ws) float32 threshold model and the letter level K (nan if no letter pixels)."""
    bg_h = cv2.resize(bg_s, (geo.Wh, geo.Hh), interpolation=cv2.INTER_LINEAR)
    gh = gray_h.astype(np.float32)
    cand = cv2.compare(gh, bg_h + np.float32(p.min_contrast), cv2.CMP_GT)
    cand = cv2.subtract(cv2.bitwise_and(cand, geo.roi_h), occ_h)
    vs, us = np.nonzero(cand)
    if vs.size >= 0.3 * geo.a_min / 4:
        K = float(np.percentile(gh[vs, us] / bg_h[vs, us], p.level_pct))
        thr = np.maximum(bg_s * np.float32(1.0 + p.level_alpha * (K - 1.0)), bg_s + np.float32(p.min_contrast))
    else:
        K = float("nan")
        thr = bg_s + np.float32(p.fallback_contrast)
    return np.minimum(thr, np.float32(p.max_threshold)), K


def _split_blob(m8, geo, p):
    """Split a uint8 0/255 blob crop into letter-sized parts: int32 part labels (1..k) or None."""
    min_part = p.split_min_part * geo.a_min
    rmax = max(2, int(p.split_max_erode * geo.stroke_px))
    dist = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    for r in range(2, rmax + 1, 2):
        n, lab = cv2.connectedComponents(cv2.compare(dist, float(r), cv2.CMP_GT), connectivity=8)
        if n <= 1:
            return None
        if n == 2:
            continue
        areas = np.bincount(lab.ravel(), minlength=n)
        big = [i for i in range(1, n) if areas[i] >= 0.5 * min_part]
        if len(big) < 2:
            continue
        markers = np.zeros(m8.shape, np.int32)
        for k, i in enumerate(big):
            markers[lab == i] = k + 1
        bgid = len(big) + 1
        markers[m8 == 0] = bgid
        cv2.watershed(cv2.merge([_u8(255 - dist * 8), m8, m8]), markers)
        markers[(markers == bgid) | (markers < 0)] = 0
        parts = np.bincount(markers.ravel(), minlength=bgid + 1)[1:bgid]
        if np.count_nonzero(parts >= min_part) < 2:
            continue
        for k in np.flatnonzero(parts < min_part):
            markers[markers == k + 1] = 0
        return markers
    return None


def _diameter(mask):
    """Largest distance [px] between two pixels of a bool mask (convex hull vertices)."""
    cs = cv2.findContours(mask.view(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    if not cs:
        return 0.0
    h = cv2.convexHull(np.concatenate(cs)).reshape(-1, 2).astype(np.float32)
    d = h[:, None, :] - h[None, :, :]
    return float(np.sqrt((d * d).sum(-1).max()))


def _merge_fragments(blobs, labels, occ, geo, p):
    """Join blobs that touch an occluder and are separated only by it (one letter cut by a paddle)."""
    gap_max = p.merge_gap * geo.stroke_px
    amax = p.merge_max_area * geo.a_max
    dmax = p.merge_max_diam * geo.diam_px
    cand = [b for b in blobs if b.touches_occluder and b.area_px < amax]
    pairs = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            a, b = cand[i], cand[j]
            u0 = min(a.bbox[0], b.bbox[0])
            v0 = min(a.bbox[1], b.bbox[1])
            u1 = max(a.bbox[0] + a.bbox[2], b.bbox[0] + b.bbox[2])
            v1 = max(a.bbox[1] + a.bbox[3], b.bbox[1] + b.bbox[3])
            if a.area_px + b.area_px > amax or max(u1 - u0, v1 - v0) > dmax:
                continue
            # bbox gap lower bound
            gx = max(a.bbox[0] - (b.bbox[0] + b.bbox[2]), b.bbox[0] - (a.bbox[0] + a.bbox[2]), 0)
            gy = max(a.bbox[1] - (b.bbox[1] + b.bbox[3]), b.bbox[1] - (a.bbox[1] + a.bbox[3]), 0)
            if np.hypot(gx, gy) > gap_max:
                continue
            lab = labels[v0:v1, u0:u1]
            ma, mb = lab == a.label, lab == b.label
            da = cv2.distanceTransform(np.where(ma, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
            db = cv2.distanceTransform(np.where(mb, 0, 255).astype(np.uint8), cv2.DIST_L2, 3)
            gap = float(da[mb].min())
            if gap > gap_max:
                continue
            # the pixels between the two fragments must be mostly occluder
            bridge = (da + db <= gap + 2.0) & ~ma & ~mb
            nb = int(np.count_nonzero(bridge))
            if nb and np.count_nonzero(occ[v0:v1, u0:u1][bridge]) < p.merge_bridge_occ * nb:
                continue
            if _diameter(ma | mb) > dmax:
                continue
            pairs.append((gap, a.label, b.label))
    if not pairs:
        return blobs
    by = {b.label: b for b in blobs}
    parent = {}

    def root(x):
        while x in parent:
            x = parent[x]
        return x

    for gap, la, lb in sorted(pairs):
        ra, rb = root(la), root(lb)
        if ra == rb:
            continue
        A, B = by[ra], by[rb]
        u0 = min(A.bbox[0], B.bbox[0])
        v0 = min(A.bbox[1], B.bbox[1])
        u1 = max(A.bbox[0] + A.bbox[2], B.bbox[0] + B.bbox[2])
        v1 = max(A.bbox[1] + A.bbox[3], B.bbox[1] + B.bbox[3])
        area = A.area_px + B.area_px
        if area > amax or max(u1 - u0, v1 - v0) > dmax:
            continue
        if A.merged_from or B.merged_from:   # chains: check the whole group again
            lab = labels[v0:v1, u0:u1]
            if _diameter((lab == A.label) | (lab == B.label)) > dmax:
                continue
        keep, drop = (A, B) if A.area_px >= B.area_px else (B, A)
        sub = labels[drop.bbox[1]:drop.bbox[1] + drop.bbox[3], drop.bbox[0]:drop.bbox[0] + drop.bbox[2]]
        sub[sub == drop.label] = keep.label
        cu = (A.centroid_px[0] * A.area_px + B.centroid_px[0] * B.area_px) / area
        cv = (A.centroid_px[1] * A.area_px + B.centroid_px[1] * B.area_px) / area
        by[keep.label] = Blob(keep.label, (cu, cv), area, (u0, v0, u1 - u0, v1 - v0), True,
                              keep.split_from, tuple(keep.merged_from or (keep.label,))
                              + tuple(drop.merged_from or (drop.label,)))
        del by[drop.label]
        parent[drop.label] = keep.label
    return [by[b.label] for b in blobs if b.label in by]


# ------------------------------------------------------------------ main
def segment_letters(img_bgr, params: SegParams = None, scene=None, cam=None) -> Segmentation:
    """Letter blobs of a BGR frame (see the module doc and docs/cnn.md "前段の分割")."""
    p = params or SegParams()
    if scene is None or cam is None:
        ds, dc = _defaults()
        scene = scene or ds
        cam = cam or (dc if scene is ds else CameraModel.from_scene(scene))
    geo = _geometry(scene, cam, p)
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    if img_bgr.shape[:2] != (geo.H, geo.W):
        raise ValueError(f"image {img_bgr.shape[:2]} does not match the camera {(geo.H, geo.W)}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_h = cv2.resize(img_bgr, (geo.Wh, geo.Hh), interpolation=cv2.INTER_AREA)
    gray_h = cv2.cvtColor(img_h, cv2.COLOR_BGR2GRAY)

    # 1. occluders
    occ_h = occluder_mask(img_h, p, geo.roi_h)
    occ = cv2.compare(cv2.resize(occ_h, (geo.W, geo.H), interpolation=cv2.INTER_LINEAR), 127, cv2.CMP_GT)

    # 2-3. background model and threshold surface
    bg_s = fit_background(gray_h, occ_h, geo, p) if p.threshold < 0 else None
    K = float("nan")
    if bg_s is None:
        thr_c = float(p.threshold if p.threshold >= 0 else 128)
        fg = cv2.compare(gray, thr_c, cv2.CMP_GT)
    else:
        thr_s, K = _threshold_small(gray_h, bg_s, occ_h, geo, p)
        thr = cv2.resize(_u8(thr_s), (geo.W, geo.H), interpolation=cv2.INTER_LINEAR)
        fg = cv2.compare(gray, thr, cv2.CMP_GT)
        thr_c = float(thr[geo.cv, geo.cu])
    fg = cv2.bitwise_and(fg, geo.roi)
    occ_g = occ
    if p.occ_guard_px > 0:
        occ_g = cv2.dilate(occ, geo.kernel(2 * p.occ_guard_px + 1))
    fg = cv2.subtract(fg, occ_g)
    if p.open_ksize and p.open_ksize > 1:
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, geo.kernel(p.open_ksize))

    # 4. blobs
    n, labels, stats, cents = cv2.connectedComponentsWithStatsWithAlgorithm(fg, 8, cv2.CV_32S, _CCL)
    amin = p.min_area_ratio * geo.a_min
    amax = p.max_area_ratio * geo.a_max
    occ_t = cv2.dilate(occ, geo.kernel(2 * (max(1, p.touch_px) + p.occ_guard_px) + 1))
    blobs = []
    next_label = n
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        u0, v0, w, h = (int(v) for v in stats[i, :4])
        sl = (slice(v0, v0 + h), slice(u0, u0 + w))
        sub = labels[sl]
        mine = sub == i
        if area < amin:
            sub[mine] = 0
            continue
        touch = bool(np.any(occ_t[sl][mine]))
        parts = None
        # A partly occluded letter can have necks where the occluder cuts in: next to an
        # occluder only blobs too large for one letter are split.
        if p.split and area >= (p.split_occ_area_ratio * geo.a_max if touch else p.split_area_ratio * geo.a_min):
            parts = _split_blob(mine.view(np.uint8) * np.uint8(255), geo, p)
        if parts is None:
            if area > amax:
                sub[mine] = 0
                continue
            blobs.append(Blob(i, (float(cents[i, 0]), float(cents[i, 1])), area, (u0, v0, w, h), touch))
            continue
        sub[mine] = 0
        for k in range(1, int(parts.max()) + 1):
            pm = parts == k
            vs, us = np.nonzero(pm)
            if us.size < amin:
                continue
            sub[pm] = next_label
            bbox = (u0 + int(us.min()), v0 + int(vs.min()),
                    int(us.max() - us.min() + 1), int(vs.max() - vs.min() + 1))
            touch = bool(np.any(occ_t[sl][pm]))
            blobs.append(Blob(next_label, (u0 + float(us.mean()), v0 + float(vs.mean())), int(us.size),
                              bbox, touch, i))
            next_label += 1

    if p.merge and sum(b.touches_occluder for b in blobs) >= 2:
        blobs = _merge_fragments(blobs, labels, occ, geo, p)

    bg = None
    if p.keep_background and bg_s is not None:
        bg = cv2.resize(bg_s, (geo.W, geo.H), interpolation=cv2.INTER_LINEAR)
    return Segmentation(fg, occ, blobs, labels, thr_c, bg, K, {"background_small": bg_s})
