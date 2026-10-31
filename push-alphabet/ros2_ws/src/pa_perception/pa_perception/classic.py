"""Classical letter detector: segment -> blobs -> template match -> pose refine.

No ROS dependency. Pipeline per frame:

1. Segmentation (`DetectorParams.segmenter`, see `segment_frame`):
   - "legacy": grayscale, restrict to the table plate (ROI), threshold. The
     automatic threshold is Otsu over the ROI, but never below
     `median + min_contrast` (the table dominates the ROI, so its median is the
     table brightness). Bright paddles / hands are foreground too.
   - "hsv": the legacy threshold minus a local HSV occluder mask (saturated
     bright pixels: orange paddles, skin), so a paddle lying on a letter no
     longer merges with it.
   - "v2": pa_perception.segmentation.segment_letters (docs/cnn.md "前段の分割");
     falls back to "hsv" while that module is not importable.
   Gray input images always use "legacy".
2. Connected components; keep blobs whose area is plausible for some letter
   (pa_core.letters.letter_area, projected at the letter top surface). Next to an
   occluder the lower bound drops by the number of occluder pixels near the blob
   (down to `occ_area_min_ratio`).
3. Coarse classification + orientation: the blob, cropped around its centroid
   and downsampled 4x, is compared with precomputed templates of every letter
   at every `coarse_step_deg` rotation (one matrix product, soft IoU).
4. Refinement at full resolution: theta by IoU scan with parabolic sub-step;
   x, y from the blob centroid. The image silhouette can include the side walls
   of the block (visible toward the image center); with `sides="auto"` both a
   "top face only" and an "extruded box" model are fitted (the latter corrects
   x, y by centroid matching) and the better IoU wins.

Occlusion: occluder pixels (grown by `occ_dilate`, not foreground) are
"unknown". They are left out of every IoU (a model pixel under a paddle is
neither a hit nor a miss) and of the model centroid used for the x, y
correction; a fit must keep at least `min_visible` of its top face visible.
`Detection.occluded_frac` is the hidden fraction of the fitted top face.

Merged blobs (`split_merged`): a blob too large for one letter, or a large blob
whose single-letter fit fails `min_iou`, is searched for letters inside it:
templates (8x downsampled, every 4 deg) are scored at every position of the
blob, the best candidates are refined at full resolution against the blob
pixels near the model (`split_min_iou`), and a found letter is removed from the
blob before the next search. A fit is rejected when the blob continues around
it (`split_max_ring`: fraction of a ring 3..10 px outside the model covered by
the blob), so letter-shaped parts of larger bright objects do not match.

Poses are in the scene.yaml world frame; the letter origin is its area centroid.
"""
from dataclasses import dataclass, field
import math
import warnings

import cv2
import numpy as np

from pa_core.config import load_scene
from pa_core.letters import letter_area, letter_names, letter_polygons
from pa_perception.camera_model import CameraModel
from pa_perception.synth import fill_polys, letter_pixel_polys

DOWN = 4  # downsampling factor of the coarse templates
SPLIT_DOWN = 8         # downsampling factor of the merged-blob search templates
SPLIT_STEP_DEG = 4.0   # angle step of the merged-blob search templates
SEGMENTERS = ("legacy", "hsv", "v2")


@dataclass
class DetectorParams:
    threshold: int = -1            # >= 0: fixed gray threshold; < 0: automatic (Otsu)
    min_contrast: int = 40         # auto: threshold >= median(table) + min_contrast
    max_threshold: int = 230       # auto: upper clamp
    open_ksize: int = 3            # morphological opening (0/1: off)
    workspace_margin: float = 0.02  # report letters with |x|,|y| < half_size + margin
    sides: str = "auto"            # "auto" | "top" | "extruded" silhouette model
    min_iou: float = 0.6           # below: not a letter (or "" if report_unknown)
    report_unknown: bool = False
    area_min_ratio: float = 0.6    # blob area / letter area plausibility range
    area_max_ratio: float = 1.6
    coarse_step_deg: float = 2.0
    fine_step_deg: float = 0.5
    refine_iters: int = 3
    # --- segmentation / occlusion (see module docstring)
    segmenter: str = "v2"          # "legacy" | "hsv" | "v2"
    occ_sat: int = 60              # hsv: occluder = HSV saturation >= occ_sat ...
    occ_min_value: int = 60        # ... and value >= occ_min_value
    occ_dilate: int = 2            # px: occluder growth (cut from fg; "unknown" for the IoU)
    occ_area_min_ratio: float = 0.25  # area lower bound next to an occluder (x letter area)
    min_visible: float = 0.4       # min visible fraction of the fitted top face
    min_visible_I: float = 0.75    # same for I (a bare bar: any letter's stroke looks like a partial I)
    join_fragments_px: int = 32    # join occluder-touching blobs closer than 2x this across it
    # --- merged blobs
    split_merged: bool = True
    split_min_iou: float = 0.7     # local IoU of a letter found inside a merged blob
    split_max_ring: float = 0.15   # max blob coverage of the ring around such a letter
    split_max_letters: int = 3
    split_max_rest: float = 0.3    # max unexplained blob area after a split (x smallest letter area)
    # --- cnn / hybrid
    cnn_occluded_fallback: bool = True  # occluder-touching blob rejected by the CNN gates: classic fit


@dataclass
class Detection:
    letter: str
    x: float
    y: float
    theta: float          # rad, wrapped to the letter symmetry: [-sym/2, sym/2)
    confidence: float
    iou: float
    model: str            # "top" or "extruded"
    centroid_px: tuple
    area_px: int
    coarse_scores: dict = field(default_factory=dict)
    occluded_frac: float = 0.0  # fraction of the fitted top face under an occluder
    split: bool = False         # found inside a merged (multi-letter) blob


@dataclass
class Blob:
    label: int                 # index in FrameSegmentation.labels
    centroid_px: tuple         # (u, v)
    area_px: int
    bbox: tuple                # (x, y, w, h) px
    touches_occluder: bool = False
    occ_near: int = 0          # unknown (occluder) pixels near the blob
    joined_from: tuple = None  # labels of the fragments joined across an occluder


@dataclass
class FrameSegmentation:
    fg: np.ndarray             # uint8 0/255 letter candidate pixels
    labels: np.ndarray         # int32 label image (0 = background)
    blobs: list                # [Blob]
    threshold: int
    occluder: np.ndarray = None  # uint8 0/255 or None
    known: np.ndarray = None     # uint8 255 = observed, 0 = unknown (grown occluder); None: all known
    source: str = "legacy"


def _otsu(hist: np.ndarray) -> int:
    h = hist.astype(np.float64)
    total = h.sum()
    if total <= 0:
        return 128
    lv = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(h)
    m0 = np.cumsum(h * lv)
    w1 = total - w0
    mt = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mt * w0 - m0 * total) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = 0
    return int(np.argmax(between))


def wrap_sym(theta: float, sym_deg: float) -> float:
    s = math.radians(sym_deg)
    return (theta + s / 2) % s - s / 2


def _disk(r):
    r = int(r)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _grow(mask, r):
    return cv2.dilate(mask, _disk(r)) if r > 0 else mask


_SEGMENT_LETTERS = []


def _segment_letters_fn():
    """pa_perception.segmentation.segment_letters, or None while it is not importable."""
    if not _SEGMENT_LETTERS:
        try:
            from pa_perception.segmentation import segment_letters
        except Exception as e:  # noqa: BLE001 - module missing / broken
            warnings.warn(f"pa_perception.segmentation unavailable ({type(e).__name__}: {e}); "
                          f"segmenter 'v2' falls back to 'hsv'")
            segment_letters = None
        _SEGMENT_LETTERS.append(segment_letters)
    return _SEGMENT_LETTERS[0]


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class ClassicDetector:
    def __init__(self, scene=None, params: DetectorParams = None, cam: CameraModel = None):
        self.scene = scene or load_scene()
        self.p = params or DetectorParams()
        if self.p.segmenter not in SEGMENTERS:
            raise ValueError(f"segmenter must be one of {SEGMENTERS}, got {self.p.segmenter!r}")
        self.cam = cam or CameraModel.from_scene(self.scene)
        self.t = float(self.scene["letter"]["thickness"])
        self.ppm = self.cam.px_per_m(self.t)
        self.names = letter_names()
        self.sym = {n: float(self.scene.get("symmetry", {}).get(n, 360.0)) for n in self.names}
        self.area_px = {n: letter_area(n) * self.ppm ** 2 for n in self.names}
        self._corners = {n: np.stack(letter_polygons(n)) for n in self.names}  # (k, 4, 2) body frame

        # Table ROI (table plate at z = 0).
        h = self.scene["workspace"]["table_half_size"]
        table = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        self.roi = np.zeros((self.cam.height_px, self.cam.width_px), np.uint8)
        fill_polys(self.roi, [self.cam.world_to_pixel(table, 0.0)], 255, aa=False)

        # Crop size: largest letter radius + side-wall extent at the image corner.
        r_m = max(np.linalg.norm(np.vstack(letter_polygons(n)), axis=1).max() for n in self.names)
        corner_r = math.hypot(self.cam.width_px, self.cam.height_px) / 2
        side_px = corner_r * self.t / (self.cam.height - self.t) + 4
        s = int(math.ceil(2 * (r_m * self.ppm + side_px)))
        self.S = s + (-s) % (2 * DOWN)  # also a multiple of SPLIT_DOWN
        self._build_templates()
        self._T8 = None
        k = self.p.open_ksize
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)) if k and k > 1 else None
        # "near the blob" for the occluded-area relaxation: about one stroke width.
        self._near_px = int(round(0.016 * self.ppm))

    # ------------------------------------------------------------------ setup
    def _body_px(self, name, theta):
        """Top-face polygons of a letter at the crop center (body coordinates)."""
        c = self.S // 2
        out = []
        for poly in letter_polygons(name, 0.0, 0.0, theta):
            out.append(np.stack([c + self.ppm * poly[:, 0], c - self.ppm * poly[:, 1]], axis=1))
        return out

    def _templates(self, down, step_deg):
        angles = np.arange(0.0, 360.0, step_deg)
        d = self.S // down
        rows, letters, angs = [], [], []
        for n in self.names:
            for a in angles:
                img = np.zeros((self.S, self.S), np.uint8)
                fill_polys(img, self._body_px(n, math.radians(a)), 255)
                small = cv2.resize(img, (d, d), interpolation=cv2.INTER_AREA)
                rows.append(small.astype(np.float32).ravel() / 255.0)
                letters.append(n)
                angs.append(math.radians(a))
        return np.stack(rows), np.array(letters), np.array(angs)

    def _build_templates(self):
        self.T, self.tmpl_letter, self.tmpl_angle = self._templates(DOWN, self.p.coarse_step_deg)
        self.T_sum = self.T.sum(axis=1)

    # ------------------------------------------------------------- segmenting
    def segment(self, img):
        """Binary foreground mask (uint8 0/255) and the threshold used (legacy: gray threshold)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if self.p.threshold >= 0:
            thr = int(self.p.threshold)
        else:
            hist = cv2.calcHist([gray], [0], self.roi, [256], [0, 256]).ravel()
            cdf = np.cumsum(hist)
            median = int(np.searchsorted(cdf, cdf[-1] / 2))
            thr = max(_otsu(hist), median + self.p.min_contrast)
            thr = min(thr, self.p.max_threshold)
        _, mask = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_and(mask, self.roi)
        if self._kernel is not None:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return mask, thr

    def occluder_hsv(self, img_bgr):
        """uint8 0/255: saturated, bright pixels (orange paddles, skin) inside the table ROI."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        occ = cv2.inRange(hsv, (0, int(self.p.occ_sat), int(self.p.occ_min_value)), (180, 255, 255))
        return cv2.bitwise_and(occ, self.roi)

    def segment_frame(self, img):
        """FrameSegmentation of a BGR (or gray: always legacy) image, per params.segmenter."""
        mode = self.p.segmenter
        if mode not in SEGMENTERS:
            raise ValueError(f"segmenter must be one of {SEGMENTERS}, got {mode!r}")
        color = img.ndim == 3 and img.shape[2] == 3
        if mode == "v2" and color:
            fn = _segment_letters_fn()
            if fn is not None:
                s = fn(img, scene=self.scene, cam=self.cam)
                blobs = [Blob(int(_get(b, "label")), tuple(float(c) for c in _get(b, "centroid_px")),
                              int(_get(b, "area_px")), tuple(int(c) for c in _get(b, "bbox")))
                         for b in _get(s, "blobs")]
                return self._finish(_get(s, "fg"), np.asarray(_get(s, "labels"), np.int32), blobs,
                                    int(_get(s, "threshold", -1)), _get(s, "occluder"), "v2")
            mode = "hsv"
        fg, thr = self.segment(img)
        occ = None
        if mode != "legacy" and color:
            occ = self.occluder_hsv(img)
            fg = cv2.bitwise_and(fg, cv2.bitwise_not(_grow(occ, self.p.occ_dilate)))
            if self._kernel is not None:
                fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self._kernel)
        else:
            mode = "legacy"
        n, labels, stats, cents = cv2.connectedComponentsWithStats(fg, connectivity=8)
        blobs = [Blob(i, (float(cents[i, 0]), float(cents[i, 1])), int(stats[i, cv2.CC_STAT_AREA]),
                      tuple(int(v) for v in stats[i, :4])) for i in range(1, n)]
        return self._finish(fg, labels, blobs, thr, occ, mode)

    def _finish(self, fg, labels, blobs, thr, occ, source):
        """Unknown mask (grown occluder minus foreground), per-blob occluder contact, and
        fragments of one letter cut apart by an occluder joined into one blob."""
        known = None
        if occ is not None and cv2.countNonZero(occ):
            unk = cv2.bitwise_and(_grow(occ, self.p.occ_dilate), cv2.bitwise_not(fg))
            if cv2.countNonZero(unk):
                known = cv2.bitwise_not(unk)
                ys, xs = np.nonzero(cv2.bitwise_and(cv2.dilate(unk, _disk(1)), fg))
                hit = set(np.unique(labels[ys, xs]).tolist())
                touched = [b for b in blobs if b.label in hit]
                for b in touched:
                    b.touches_occluder = True
                if self.p.join_fragments_px > 0 and len(touched) > 1:
                    labels, blobs = self._join_fragments(labels, blobs, touched, unk)
                H, W = labels.shape
                r = self._near_px
                for b in blobs:
                    if b.touches_occluder:
                        x, y, w, h = b.bbox
                        sl = unk[max(y - r, 0):min(y + h + r, H), max(x - r, 0):min(x + w + r, W)]
                        b.occ_near = int(cv2.countNonZero(sl))
        return FrameSegmentation(fg, labels, blobs, int(thr), occ, known, source)

    def _join_fragments(self, labels, blobs, touched, unk):
        """Blobs touching the occluder whose pixels come within 2 * join_fragments_px of
        each other through occluder pixels become one blob (new label)."""
        r = self.p.join_fragments_px
        H, W = labels.shape
        x0 = max(min(b.bbox[0] for b in touched) - r, 0)
        y0 = max(min(b.bbox[1] for b in touched) - r, 0)
        x1 = min(max(b.bbox[0] + b.bbox[2] for b in touched) + r, W)
        y1 = min(max(b.bbox[1] + b.bbox[3] for b in touched) + r, H)
        lab = labels[y0:y1, x0:x1]
        lut = np.zeros(int(labels.max()) + 1, np.uint8)
        for b in touched:
            lut[b.label] = 255
        tmask = lut[lab]
        sq = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * r + 1, 2 * r + 1))  # separable: fast
        reach = cv2.bitwise_and(cv2.dilate(tmask, sq), unk[y0:y1, x0:x1])
        _, groups = cv2.connectedComponents(cv2.bitwise_or(tmask, reach), connectivity=8)
        members = {}
        for b in touched:
            bx, by, bw, bh = b.bbox
            ys, xs = np.nonzero(labels[by:by + bh, bx:bx + bw] == b.label)
            g = int(groups[ys[0] + by - y0, xs[0] + bx - x0])
            members.setdefault(g, []).append(b)
        multi = [m for m in members.values() if len(m) > 1]
        if not multi:
            return labels, blobs
        labels = labels.copy()
        nxt = int(labels.max()) + 1
        gone = set()
        out = []
        for m in multi:
            area = sum(b.area_px for b in m)
            cu = sum(b.centroid_px[0] * b.area_px for b in m) / area
            cv = sum(b.centroid_px[1] * b.area_px for b in m) / area
            x0 = min(b.bbox[0] for b in m)
            y0 = min(b.bbox[1] for b in m)
            x1 = max(b.bbox[0] + b.bbox[2] for b in m)
            y1 = max(b.bbox[1] + b.bbox[3] for b in m)
            sub = labels[y0:y1, x0:x1]
            sub[np.isin(sub, [b.label for b in m])] = nxt
            out.append(Blob(nxt, (cu, cv), area, (x0, y0, x1 - x0, y1 - y0), True, 0,
                            tuple(b.label for b in m)))
            gone.update(b.label for b in m)
            nxt += 1
        return labels, [b for b in blobs if b.label not in gone] + out

    # -------------------------------------------------------------- matching
    def _crop_arr(self, arr, ic, jc, fill=0):
        """S x S window of a 2-D array whose top-left pixel is (ic - S/2, jc - S/2)."""
        S, half = self.S, self.S // 2
        u0, v0 = ic - half, jc - half
        H, W = arr.shape
        out = np.full((S, S), fill, arr.dtype)
        a0, b0 = max(u0, 0), max(v0, 0)
        a1, b1 = min(u0 + S, W), min(v0 + S, H)
        if a1 > a0 and b1 > b0:
            out[b0 - v0:b1 - v0, a0 - u0:a1 - u0] = arr[b0:b1, a0:a1]
        return out

    def _crop(self, labels, idx, ic, jc):
        S, half = self.S, self.S // 2
        u0, v0 = ic - half, jc - half
        H, W = labels.shape
        out = np.zeros((S, S), bool)
        a0, b0 = max(u0, 0), max(v0, 0)
        a1, b1 = min(u0 + S, W), min(v0 + S, H)
        if a1 > a0 and b1 > b0:
            out[b0 - v0:b1 - v0, a0 - u0:a1 - u0] = labels[b0:b1, a0:a1] == idx
        return out, (u0, v0)

    def observation(self, seg, blob, ic=None, jc=None):
        """(obs, origin) for blob matching: obs = (uint8 mask, pixel count, known crop or None)."""
        if ic is None:
            ic, jc = int(round(blob.centroid_px[0])), int(round(blob.centroid_px[1]))
        ob, origin = self._crop(seg.labels, blob.label, ic, jc)
        ob_u8 = ob.view(np.uint8) * np.uint8(255)
        known = None
        if seg.known is not None and blob.touches_occluder:
            known = self._crop_arr(seg.known, ic, jc, fill=255)
        return (ob_u8, cv2.countNonZero(ob_u8), known), origin

    def _project(self, name, x, y, th, z):
        """(k, 4, 2) pixel corners of the letter rectangles at height z."""
        c, sn = math.cos(th), math.sin(th)
        b = self._corners[name]
        X = x + c * b[..., 0] - sn * b[..., 1]
        Y = y + sn * b[..., 0] + c * b[..., 1]
        s = self.cam.f / (self.cam.height - z)
        return np.stack([self.cam.cx + s * X, self.cam.cy - s * Y], axis=-1)

    def _model_mask(self, name, x, y, th, model, origin, aa=False):
        """uint8 S x S mask of the letter silhouette in crop coordinates."""
        off = np.asarray(origin, dtype=float)
        top = self._project(name, x, y, th, self.t) - off
        if model == "extruded":
            bot = self._project(name, x, y, th, 0.0) - off
            polys = [cv2.convexHull(np.concatenate([a, b]).astype(np.float32)).reshape(-1, 2)
                     for a, b in zip(top, bot)]
        else:
            polys = list(top)
        m = np.zeros((self.S, self.S), np.uint8)
        return fill_polys(m, polys, 255, aa=aa)

    def _iou(self, obs, name, x, y, th, model, origin):
        """obs: (uint8 mask, pixel count[, known mask]). Unknown model pixels are left out."""
        m = self._model_mask(name, x, y, th, model, origin)
        inter = cv2.countNonZero(cv2.bitwise_and(obs[0], m))
        known = obs[2] if len(obs) > 2 else None
        mc = cv2.countNonZero(m if known is None else cv2.bitwise_and(m, known))
        union = obs[1] + mc - inter
        return inter / union if union else 0.0

    def _visible(self, obs, name, x, y, th, origin):
        """Visible fraction of the top face (1 without an unknown mask)."""
        known = obs[2] if len(obs) > 2 else None
        if known is None:
            return 1.0
        m = self._model_mask(name, x, y, th, "top", origin)
        n = cv2.countNonZero(m)
        return cv2.countNonZero(cv2.bitwise_and(m, known)) / n if n else 0.0

    def _refine(self, obs, origin, name, x, y, th, model, obs_c, centroid=None, span0=None):
        """Theta IoU scan (+ x, y by centroid matching for extruded / occluded / forced)."""
        p = self.p
        known = obs[2] if len(obs) > 2 else None
        if centroid is None:
            centroid = model == "extruded" or known is not None
        for it in range(p.refine_iters):
            step_deg = p.fine_step_deg
            if it == 0:
                span = 0.75 * p.coarse_step_deg if span0 is None else span0
                if span0 is not None:  # wide first scan: coarser steps, finer ones follow
                    step_deg = max(p.fine_step_deg, span0 / 4.0)
            else:
                span = p.fine_step_deg
            offs = np.arange(-span, span + 1e-9, step_deg)
            ious = np.array([self._iou(obs, name, x, y, th + math.radians(o), model, origin)
                             for o in offs])
            k = int(np.argmax(ious))
            step = 0.0
            if 0 < k < len(offs) - 1:
                a, b, c = ious[k - 1], ious[k], ious[k + 1]
                den = a - 2 * b + c
                if den < 0:
                    step = float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
            th = th + math.radians(offs[k] + step * step_deg)
            if centroid:
                m = self._model_mask(name, x, y, th, model, origin, aa=True)
                if known is not None:
                    m = cv2.bitwise_and(m, known)
                mo = cv2.moments(m)
                if mo["m00"] > 0:
                    du = obs_c[0] - (mo["m10"] / mo["m00"] + origin[0])
                    dv = obs_c[1] - (mo["m01"] / mo["m00"] + origin[1])
                    x += du / self.ppm
                    y -= dv / self.ppm
            elif it >= 1:
                break
        return x, y, th, self._iou(obs, name, x, y, th, model, origin)

    # ------------------------------------------------------------ blob gates
    def area_bounds(self, blob):
        """(amin, amax) px of a plausible single-letter blob (relaxed next to an occluder)."""
        p = self.p
        a_lo = min(self.area_px.values())
        amin = p.area_min_ratio * a_lo
        if blob.touches_occluder:
            amin = max(p.occ_area_min_ratio * a_lo, amin - blob.occ_near)
        return amin, p.area_max_ratio * max(self.area_px.values())

    def _area_ok(self, name, blob):
        p = self.p
        a = self.area_px[name]
        if blob.area_px > p.area_max_ratio * a:
            return False
        if blob.touches_occluder:
            return (blob.area_px >= p.occ_area_min_ratio * a
                    and blob.area_px + blob.occ_near >= p.area_min_ratio * a)
        return blob.area_px >= p.area_min_ratio * a

    def _min_visible(self, name):
        return self.p.min_visible_I if name == "I" else self.p.min_visible

    def can_split(self, blob, seg=None):
        """Worth searching for several letters in this blob (too big, or failed as one letter).

        Not with the legacy segmentation: there, large blobs are mostly paddles / hands
        merged with letters, and letter-shaped parts of them would be found.
        """
        if seg is not None and seg.source == "legacy":
            return False
        return self.p.split_merged and blob.area_px >= 1.25 * min(self.area_px.values())

    def _lim(self):
        return self.scene["workspace"]["half_size"] + self.p.workspace_margin

    # ---------------------------------------------------------------- detect
    def detect(self, img):
        """List of Detection for the letters in a BGR (or gray) image."""
        p = self.p
        seg = self.segment_frame(img)
        lim = self._lim()
        out = []
        for b in seg.blobs:
            amin, amax = self.area_bounds(b)
            if b.area_px < amin:
                continue
            det = None
            if b.area_px <= amax:
                cu, cv = b.centroid_px
                x0, y0 = self.cam.pixel_to_world([[cu, cv]], self.t)[0]
                if abs(x0) > lim + 0.01 or abs(y0) > lim + 0.01:
                    continue
                det, ok = self._detect_blob(seg, b, x0, y0)
                if ok:
                    out.append(det)
                    continue
            if self.can_split(b, seg):
                found = [d for d in self.split_blob(seg, b) if abs(d.x) <= lim and abs(d.y) <= lim]
                if found:
                    out += found
                    continue
            if det is not None and p.report_unknown and abs(det.x) <= lim and abs(det.y) <= lim:
                det.letter, det.confidence = "", 0.0
                det.theta = wrap_sym(det.theta, 360.0)
                out.append(det)
        return out

    def _detect_blob(self, seg, b, x0, y0):
        """(Detection, ok) of the single-letter fit of a blob; (None, False) if nothing fits.

        Seeds: templates at the blob centroid (coarse_step_deg); for a blob touching an
        occluder, whose centroid is not the letter's, the best placement of each letter
        found by the position search (`_candidates`) instead.
        """
        p = self.p
        cu, cv = b.centroid_px
        obs, origin = self.observation(seg, b)
        seeds = []  # (name, coarse score, angle, x, y, obs, origin)
        if obs[2] is None:
            d = self.S // DOWN
            small = cv2.resize(obs[0], (d, d), interpolation=cv2.INTER_AREA).astype(np.float32).ravel() / 255.0
            dot = np.einsum("ij,j->i", self.T, small)  # not BLAS: avoids thread contention
            score = dot / (self.T_sum + small.sum() - dot + 1e-9)
            for name in self.names:
                if not self._area_ok(name, b):
                    continue
                sel = np.flatnonzero(self.tmpl_letter == name)
                k = sel[int(np.argmax(score[sel]))]
                seeds.append((name, float(score[k]), float(self.tmpl_angle[k]), x0, y0, obs, origin))
            span0 = None
        else:
            rem, known, (rx0, ry0) = self._region(seg, b)
            for sc, name, ang, u, v in self._candidates(rem, known, per_letter=True):
                if not self._area_ok(name, b):
                    continue
                xs, ys = self.cam.pixel_to_world([[u + rx0, v + ry0]], self.t)[0]
                ob, org = self.observation(seg, b, int(round(u + rx0)), int(round(v + ry0)))
                seeds.append((name, sc, ang, xs, ys, ob, org))
            span0 = SPLIT_STEP_DEG
        if not seeds:
            return None, False
        seeds.sort(key=lambda sd: -sd[1])
        best = {sd[0]: sd[1] for sd in seeds}
        top_score = seeds[0][1]
        models = ["top", "extruded"] if p.sides == "auto" else [p.sides]
        fits = []
        for name, sc, ang, xs, ys, ob, org in seeds:
            if sc < top_score - (0.1 if span0 is None else 0.06):
                break
            for model in models:
                fits.append((name, model) + self._refine(ob, org, name, xs, ys, ang, model, (cu, cv),
                                                         span0=span0) + (ob, org))
        name, model, x, y, th, iou, ob, org = max(fits, key=lambda f: f[5])
        others = [f[5] for f in fits if f[0] != name] + [sc for nm, sc in best.items() if nm != name]
        margin = iou - max(others) if others else 1.0
        vis = self._visible(ob, name, x, y, th, org)
        conf = float(np.clip(iou, 0, 1) * np.clip(0.5 + 5.0 * margin, 0, 1) * math.sqrt(vis))
        lim = self._lim()
        ok = iou >= p.min_iou and vis >= self._min_visible(name) and abs(x) <= lim and abs(y) <= lim
        th = wrap_sym(th, self.sym.get(name, 360.0))
        return Detection(name, float(x), float(y), float(th), conf, float(iou), model, (cu, cv),
                         b.area_px, best, 1.0 - vis), ok

    # ---------------------------------------------------------- merged blobs
    def _split_templates(self):
        if self._T8 is None:
            T, letters, angles = self._templates(SPLIT_DOWN, SPLIT_STEP_DEG)
            self._T8 = (np.ascontiguousarray(T.T), T.sum(axis=1), letters, angles)
        return self._T8

    def _candidates(self, rem, known=None, max_cands=4, per_letter=False):
        """[(score, name, angle, u, v)] best template placements in a region mask (region px).

        Soft IoU of the 8x downsampled templates at every position near the mask;
        unknown pixels (known == 0) are left out of the template sum. per_letter:
        the best placement of each letter; else the `max_cands` best overall
        (non-maximum suppressed per letter).
        """
        T8t, T8sum, letters, angles = self._split_templates()
        D, d = SPLIT_DOWN, self.S // SPLIT_DOWN
        H, W = rem.shape
        small = cv2.resize(rem.view(np.uint8) * np.uint8(255), (W // D, H // D),
                           interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if small.shape[0] < d or small.shape[1] < d:
            return []
        win = np.lib.stride_tricks.sliding_window_view(small, (d, d))  # (ny, nx, d, d)
        near = cv2.dilate((small > 0.25).view(np.uint8), _disk(4))
        ny, nx = win.shape[:2]
        a, b = np.nonzero(near[d // 2:d // 2 + ny, d // 2:d // 2 + nx])
        if not len(a):
            return []
        P = win[a, b].reshape(len(a), d * d)
        dot = cv2.gemm(np.ascontiguousarray(P), T8t, 1.0, None, 0.0)  # not numpy BLAS (threads)
        if known is not None:
            ks = cv2.resize(known, (W // D, H // D), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            K = np.lib.stride_tricks.sliding_window_view(ks, (d, d))[a, b].reshape(len(a), d * d)
            tsum = cv2.gemm(np.ascontiguousarray(K), T8t, 1.0, None, 0.0)
        else:
            tsum = T8sum[None, :]
        score = dot / (tsum + P.sum(axis=1)[:, None] - dot + 1e-9)
        out = []
        if per_letter:
            for name in self.names:
                cols = np.flatnonzero(letters == name)
                sub = score[:, cols]
                i, j = np.unravel_index(int(np.argmax(sub)), sub.shape)
                out.append((float(sub[i, j]), name, float(angles[cols[j]]),
                            D * b[i] + self.S / 2.0, D * a[i] + self.S / 2.0))
            return sorted(out, key=lambda c: -c[0])
        order = np.argsort(-score, axis=None)[:400]
        for flat in order:
            i, j = divmod(int(flat), score.shape[1])
            u = D * b[i] + self.S / 2.0
            v = D * a[i] + self.S / 2.0
            name = letters[j]
            if any(n == name and math.hypot(u - uu, v - vv) < 3 * D for _, n, _, uu, vv in out):
                continue
            out.append((float(score[i, j]), str(name), float(angles[j]), u, v))
            if len(out) >= max_cands:
                break
        return out

    def _region(self, seg, b):
        """(blob mask, known or None, (rx0, ry0)) over the blob bbox + S/2 margin,
        sized to a multiple of SPLIT_DOWN (zero / known padded)."""
        S, D = self.S, SPLIT_DOWN
        bx, by, bw, bh = b.bbox
        rx0, ry0 = bx - S // 2, by - S // 2
        W = bw + S + (-(bw + S)) % D
        H = bh + S + (-(bh + S)) % D
        Hi, Wi = seg.labels.shape
        rem = np.zeros((H, W), bool)
        known = None if (seg.known is None or not b.touches_occluder) else np.full((H, W), 255, np.uint8)
        a0, b0, a1, b1 = max(rx0, 0), max(ry0, 0), min(rx0 + W, Wi), min(ry0 + H, Hi)
        if a1 > a0 and b1 > b0:
            rem[b0 - ry0:b1 - ry0, a0 - rx0:a1 - rx0] = seg.labels[b0:b1, a0:a1] == b.label
            if known is not None:
                known[b0 - ry0:b1 - ry0, a0 - rx0:a1 - rx0] = seg.known[b0:b1, a0:a1]
        return rem, known, (rx0, ry0)

    def _fit_in_region(self, rem, known, rorigin, name, ang, u, v, models=None):
        """Best (iou, model, x, y, th, vis, area, mask, origin) of one letter seeded at region px (u, v)."""
        p = self.p
        ic, jc = int(round(u)), int(round(v))
        obs_win = self._crop_arr(rem.view(np.uint8) * np.uint8(255), ic, jc)
        kn_win = None if known is None else self._crop_arr(known, ic, jc, fill=255)
        origin = (ic - self.S // 2 + rorigin[0], jc - self.S // 2 + rorigin[1])
        x0, y0 = self.cam.pixel_to_world([[u + rorigin[0], v + rorigin[1]]], self.t)[0]
        models = models or (["top", "extruded"] if p.sides == "auto" else [p.sides])
        best = None
        for model in models:
            x, y, th = x0, y0, ang
            for k in range(2):
                m = self._model_mask(name, x, y, th, model, origin)
                loc = cv2.bitwise_and(obs_win, _grow(m, 8 if k == 0 else 5))
                mo = cv2.moments(loc, binaryImage=True)
                if mo["m00"] <= 0:
                    break
                obs = (loc, cv2.countNonZero(loc), kn_win)
                c = (mo["m10"] / mo["m00"] + origin[0], mo["m01"] / mo["m00"] + origin[1])
                x, y, th, _ = self._refine(obs, origin, name, x, y, th, model, c, centroid=True,
                                           span0=SPLIT_STEP_DEG if k == 0 else None)
            m = self._model_mask(name, x, y, th, model, origin)
            loc = cv2.bitwise_and(obs_win, _grow(m, 5))
            obs = (loc, cv2.countNonZero(loc), kn_win)
            iou = self._iou(obs, name, x, y, th, model, origin)
            ring = cv2.bitwise_and(_grow(m, 10), cv2.bitwise_not(_grow(m, 3)))
            if kn_win is not None:
                ring = cv2.bitwise_and(ring, kn_win)
            nr = cv2.countNonZero(ring)
            ring_frac = cv2.countNonZero(cv2.bitwise_and(ring, obs_win)) / nr if nr else 1.0
            vis = self._visible(obs, name, x, y, th, origin)
            if ring_frac > p.split_max_ring or vis < self._min_visible(name):
                iou = 0.0
            if best is None or iou > best[0]:
                best = (iou, model, x, y, th, vis, obs[1], m, origin)
        return best

    def split_fits(self, seg, b):
        """Letters inside a merged blob: [(name, model, x, y, th, iou, conf, vis, area, (u, v))]."""
        p = self.p
        rem, known, (rx0, ry0) = self._region(seg, b)
        H, W = rem.shape
        a_min = p.occ_area_min_ratio * min(self.area_px.values())
        out = []
        for _ in range(p.split_max_letters):
            if np.count_nonzero(rem) < a_min:
                break
            # Every candidate with the first silhouette model, then the best one with the other.
            models = ["top", "extruded"] if p.sides == "auto" else [p.sides]
            fits, seeds = [], {}
            for sc, name, ang, u, v in self._candidates(rem, known):
                f = self._fit_in_region(rem, known, (rx0, ry0), name, ang, u, v, models[:1])
                if f is not None:
                    fits.append((name,) + f)
                    seeds[id(fits[-1])] = (name, ang, u, v)
            if not fits:
                break
            best = max(fits, key=lambda f: f[1])
            if len(models) > 1 and best[1] > 0:
                name, ang, u, v = seeds[id(best)]
                f = self._fit_in_region(rem, known, (rx0, ry0), name, ang, u, v, models[1:])
                if f is not None and f[0] > best[1]:
                    best = (name,) + f
            name, iou, model, x, y, th, vis, area, m, origin = best
            if iou < p.split_min_iou:
                break
            others = [f[1] for f in fits if f[0] != name]
            margin = iou - max(others) if others else 1.0
            conf = float(np.clip(iou, 0, 1) * np.clip(0.5 + 5.0 * margin, 0, 1) * math.sqrt(vis))
            uv = tuple(float(c) for c in self.cam.world_to_pixel([[x, y]], self.t)[0])
            out.append((name, model, x, y, th, iou, conf, vis, area, uv))
            # Remove the letter (and a small margin) from the blob before the next search.
            grown = _grow(m, 3) > 0
            ox, oy = origin[0] - rx0, origin[1] - ry0
            ys, xs = np.nonzero(grown)
            ys, xs = ys + oy, xs + ox
            keep = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            rem[ys[keep], xs[keep]] = False
            if self._kernel is not None:
                rem = cv2.morphologyEx(rem.view(np.uint8), cv2.MORPH_OPEN, self._kernel).astype(bool)
        # A real merge is explained by its letters; a large rest means the "letters" are
        # parts of something else (e.g. one stroke of a partly hidden letter fitted as I).
        if out and np.count_nonzero(rem) > p.split_max_rest * min(self.area_px.values()):
            return []
        return out

    def split_blob(self, seg, b):
        """Detections for the letters found inside a merged blob."""
        out = []
        for name, model, x, y, th, iou, conf, vis, area, uv in self.split_fits(seg, b):
            th = wrap_sym(th, self.sym.get(name, 360.0))
            out.append(Detection(name, float(x), float(y), float(th), conf, float(iou), model, uv,
                                 int(area), {}, 1.0 - vis, True))
        return out


def draw_detections(img, dets, detector: ClassicDetector):
    """Copy of img with the fitted letter outlines, heading and labels drawn."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cam = detector.cam
    for d in dets:
        if not d.letter:
            cv2.circle(out, (int(d.centroid_px[0]), int(d.centroid_px[1])), 6, (0, 0, 255), 2)
            continue
        tops, _ = letter_pixel_polys(cam, d.letter, d.x, d.y, d.theta, detector.t, sides=False)
        for poly in tops:
            cv2.polylines(out, [np.round(poly).astype(np.int32)], True, (0, 200, 0), 2)
        c = cam.world_to_pixel([[d.x, d.y]], detector.t)[0]
        h = cam.world_to_pixel([[d.x - 0.03 * math.sin(d.theta), d.y + 0.03 * math.cos(d.theta)]],
                               detector.t)[0]
        cv2.arrowedLine(out, tuple(int(v) for v in c), tuple(int(v) for v in h), (0, 0, 255), 2,
                        tipLength=0.25)
        label = f"{d.letter} {d.confidence:.2f} ({d.x * 1000:.0f},{d.y * 1000:.0f}) {math.degrees(d.theta):.0f}deg"
        cv2.putText(out, label, (int(c[0]) - 60, int(c[1]) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 80, 0), 1, cv2.LINE_AA)
    return out
