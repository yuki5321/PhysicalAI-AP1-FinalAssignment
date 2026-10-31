"""CNN letter detector (docs/cnn.md): blobs -> 96x96 crops -> ONNX CNN -> poses.

No ROS and no torch dependency (onnxruntime + OpenCV + numpy). Pipeline:

1. Segmentation and blob filtering exactly as ClassicDetector
   (`ClassicDetector.segment_frame` on the BGR image: params.segmenter, area
   plausibility relaxed next to occluders, workspace check on the blob centroid).
2. One crop per blob around its centroid (pa_perception.cnn_crop.extract_crop);
   all crops of a frame go through the network in ONE onnxruntime run.
3. Decode (cnn_crop.decode): letter = argmax, confidence = softmax max,
   theta from (cos phi, sin phi), position = crop center (world, letter-top
   height) + offset.
4. refine="classic" (method "hybrid"): the CNN letter / pose seeds the classic
   detector's IoU refinement (`ClassicDetector._refine`: theta scan against the
   top / extruded silhouettes) instead of its template matching. The fit picks
   the letter (CNN top-1, plus the runner-up if its prob >= alt_prob) and the
   angle; the position stays the CNN one (more accurate on synth_val).

Occlusion: the IoU gates leave occluder pixels out (ClassicDetector._iou) and
a letter must keep `min_visible` (I: `min_visible_I`) of its top face visible.
Merged blobs (too large for one letter, or a large blob whose single-letter
result fails the gates) go to the classic in-blob search
(`ClassicDetector.split_blob`, Detection.split = True, classic pose).

Filtering mirrors ClassicDetector: letters whose final position is outside
|x|,|y| < workspace.half_size + workspace_margin are dropped; detections below
`min_confidence` or `params.min_iou` (silhouette IoU at the CNN pose) for cnn,
or below `params.min_iou` after refinement for hybrid, are dropped, or reported with
letter "" and confidence 0 when `report_unknown` is set.

`make_detector(method, ...)` builds classic / cnn / hybrid detectors and falls
back to classic (with a warning) when onnxruntime or the model is missing.
"""

import dataclasses
import math
import os

import numpy as np

from pa_core.config import load_scene
from pa_perception import cnn_crop
from pa_perception.classic import ClassicDetector, Detection, DetectorParams, wrap_sym

MODEL_NAME = "letter_cnn.onnx"
METHODS = ("classic", "cnn", "hybrid")


def default_model_path():
    """Installed share/pa_perception/models/letter_cnn.onnx, else the source tree copy."""
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory("pa_perception"), "models", MODEL_NAME)
        if os.path.exists(p):
            return p
    except Exception:  # noqa: BLE001 - not in a ROS environment / package not installed
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "models", MODEL_NAME))


class CNNDetector:
    """detect(img_bgr) -> list[Detection], same Detection as ClassicDetector.

    Detection fields: `iou` is the IoU of the fitted silhouette with the blob
    (informative for "cnn", the refinement score for "hybrid"); `model` is
    "cnn" for pure CNN output, "cnn+top" / "cnn+extruded" after refinement;
    `coarse_scores` holds the CNN softmax per letter.
    """

    def __init__(self, model_path=None, scene=None, params: DetectorParams = None,
                 refine="none", min_confidence=0.5, refine_window_deg=6.0,
                 alt_prob=0.15, threads=0, classic: ClassicDetector = None):
        if refine not in ("none", "classic"):
            raise ValueError(f"refine must be 'none' or 'classic', got {refine!r}")
        import onnxruntime as ort  # ImportError -> caller falls back

        self.scene = scene or load_scene()
        self.p = params or DetectorParams()
        self.refine = refine
        self.min_confidence = float(min_confidence)
        self.refine_window_deg = float(refine_window_deg)
        self.alt_prob = float(alt_prob)
        self.classic = classic or ClassicDetector(scene=self.scene, params=self.p)
        self.cam = self.classic.cam
        self.t = self.classic.t
        self.model_path = model_path or default_model_path()
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"CNN model not found: {self.model_path}")

        so = ort.SessionOptions()
        # No busy-waiting worker threads after a run: they would compete with the
        # (multi-threaded) OpenCV segmentation of the next frame.
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(self.model_path, sess_options=so,
                                         providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        b = inp.shape[0] if inp.shape else None
        self.fixed_batch = b if isinstance(b, int) and b > 0 else None
        outs = [o.name for o in self.sess.get_outputs()]
        missing = [n for n in ("logits", "angle", "offset") if n not in outs]
        if missing:
            raise ValueError(f"{self.model_path}: missing outputs {missing} (has {outs})")
        self.letters = list(cnn_crop.LETTERS)

    # The classic detector shares the params object; keep them in sync.
    @property
    def params(self):
        return self.p

    @params.setter
    def params(self, p):
        self.p = p
        self.classic.p = p

    # ---------------------------------------------------------------- network
    def infer(self, crops):
        """crops float32 [N, 1, 96, 96] -> (logits [N,4], angle [N,2], offset [N,2])."""
        x = np.ascontiguousarray(crops, dtype=np.float32)
        names = ["logits", "angle", "offset"]
        if self.fixed_batch is None or self.fixed_batch == len(x):
            return [np.asarray(o) for o in self.sess.run(names, {self.input_name: x})]
        parts = [self.sess.run(names, {self.input_name: x[i:i + 1]}) for i in range(len(x))]
        return [np.concatenate([p[k] for p in parts]) for k in range(3)]

    # ------------------------------------------------------------- refinement
    def _silhouette_iou(self, obs, origin, name, x, y, th):
        """Best (iou, model) of the top / extruded silhouettes at a fixed pose."""
        models = ["top", "extruded"] if self.p.sides == "auto" else [self.p.sides]
        return max((self.classic._iou(obs, name, x, y, th, m, origin), m) for m in models)

    def _refine(self, obs, origin, cands, x, y, obs_c):
        """Classic IoU refinement seeded by the CNN pose: (name, model, th, iou, margin).

        Every candidate letter / silhouette model starts from the CNN position and
        angle. Only the letter, angle and IoU are taken from the fit: on
        synth_val the CNN offset is a better position than the classic
        centroid-based correction (seeding with the blob centroid, or taking the
        refined x, y, gave ~2x the position error and lower recall).
        """
        p = self.p
        # ClassicDetector._refine scans +-0.75 * coarse_step_deg first: widen it to the window.
        rp = dataclasses.replace(p, coarse_step_deg=self.refine_window_deg / 0.75)
        models = ["top", "extruded"] if p.sides == "auto" else [p.sides]
        fits = []
        saved = self.classic.p
        self.classic.p = rp
        try:
            for name, th in cands:
                for m in models:
                    _, _, th_r, iou = self.classic._refine(obs, origin, name, x, y, th, m, obs_c)
                    fits.append((name, m, th_r, iou))
        finally:
            self.classic.p = saved
        best = max(fits, key=lambda f: f[-1])
        others = [f[-1] for f in fits if f[0] != best[0]]
        margin = best[-1] - max(others) if others else 1.0
        return best + (margin,)

    # ---------------------------------------------------------------- detect
    def detect(self, img):
        p, det = self.p, self.classic
        seg = det.segment_frame(img)  # BGR in: the segmenter uses color (occluders)
        gray = cnn_crop.to_gray(img)
        lim = self.scene["workspace"]["half_size"] + p.workspace_margin
        blobs, merged = [], []
        for b in seg.blobs:
            amin, amax = det.area_bounds(b)
            if b.area_px < amin:
                continue
            if b.area_px > amax:
                if det.can_split(b, seg):
                    merged.append(b)
                continue
            cu, cv = b.centroid_px
            xw, yw = cnn_crop.crop_center_world(cu, cv, self.cam, self.t)
            if abs(xw) > lim + 0.01 or abs(yw) > lim + 0.01:
                continue
            blobs.append((b, cu, cv, (xw, yw)))

        out, unknown, retry = [], [], []
        if blobs:
            crops = np.stack([cnn_crop.extract_crop(gray, cu, cv) for _, cu, cv, _ in blobs])
            logits, angle, offset = self.infer(crops)
            probs = cnn_crop.softmax(logits)
        for k, (b, cu, cv, cw) in enumerate(blobs):
            name, x, y, th, conf = cnn_crop.decode(logits[k], angle[k], offset[k], cw)
            scores = {nm: float(probs[k, j]) for j, nm in enumerate(self.letters)}
            obs, origin = det.observation(seg, b)
            if self.refine == "classic":
                # Candidates: the CNN letter plus any runner-up with a non-negligible prob.
                phi = math.atan2(float(angle[k][1]), float(angle[k][0]))
                cands = []
                for j in np.argsort(-probs[k])[:2]:
                    nm = self.letters[int(j)]
                    if nm != name and probs[k, j] < self.alt_prob:
                        continue
                    cands.append((nm, phi * cnn_crop.SYMMETRY[nm] / 360.0))
                nm, model, th, iou, margin = self._refine(obs, origin, cands, x, y, (cu, cv))
                if nm != name:
                    margin = min(margin, 0.1)
                name = nm
                conf = float(np.clip(iou, 0, 1) * scores[name] * np.clip(0.5 + 5.0 * margin, 0, 1))
                model = "cnn+" + model
                ok = iou >= p.min_iou
            else:
                # Shape check at the CNN pose: the network has no "not a letter"
                # class, so paddles / hands / walls that pass the area filter
                # would otherwise come out as confident letters.
                iou, _ = self._silhouette_iou(obs, origin, name, x, y, th)
                model = "cnn"
                ok = conf >= self.min_confidence and iou >= p.min_iou
            # Occluded letters: enough of the top face must be visible (unknown pixels
            # are left out of the IoU, so a small fragment would fit anything).
            vis = det._visible(obs, name, x, y, th, origin)
            ok = ok and vis >= det._min_visible(name)
            conf = float(conf * math.sqrt(vis))
            if abs(x) > lim or abs(y) > lim:
                continue
            th = wrap_sym(th, det.sym.get(name, 360.0))
            d = Detection(name, float(x), float(y), float(th), float(conf), float(iou),
                          model, (cu, cv), b.area_px, scores, 1.0 - vis)
            if not ok and b.touches_occluder and p.cnn_occluded_fallback:
                # Partly hidden letter: the crop is centred on the visible part only and the
                # CNN pose is often off; the occlusion-aware classic fit decides instead.
                xw, yw = cw
                dc, okc = det._detect_blob(seg, b, xw, yw)
                if okc:
                    d, ok = dc, True
            if ok:
                out.append(d)
            else:
                if det.can_split(b, seg):
                    retry.append(b)
                unknown.append((b, d))

        # Several letters in one blob: the classic in-blob search (the CNN crop of a
        # merged blob is centred on neither letter). Detection.split marks them. A
        # letter-sized blob the CNN rejected counts as merged only if the search
        # finds two or more letters in it (a single letter stays the CNN's call).
        split_ok = set()
        for b in merged + retry:
            found = [d for d in det.split_blob(seg, b) if abs(d.x) <= lim and abs(d.y) <= lim]
            if len(found) >= (2 if b in retry else 1):
                split_ok.add(b.label)
                out += found
        if p.report_unknown:
            for b, d in unknown:
                if b.label in split_ok:
                    continue
                d.letter, d.confidence = "", 0.0
                d.theta = wrap_sym(d.theta, 360.0)
                out.append(d)
        return out


def make_detector(method="classic", model_path=None, scene=None, params=None,
                  min_confidence=0.5, threads=0, fallback=True, warn=None):
    """(detector, method actually used). method: classic | cnn | hybrid.

    With `fallback`, a missing onnxruntime / model (or an unusable model) gives
    the classic detector and a call to warn(message) (default: print).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    scene = scene or load_scene()
    params = params or DetectorParams()
    if method == "classic":
        return ClassicDetector(scene=scene, params=params), "classic"
    try:
        det = CNNDetector(model_path=model_path, scene=scene, params=params,
                          refine="classic" if method == "hybrid" else "none",
                          min_confidence=min_confidence, threads=threads)
        return det, method
    except Exception as e:  # noqa: BLE001 - ImportError, FileNotFoundError, ort errors
        if not fallback:
            raise
        (warn or print)(f"method '{method}' unavailable ({type(e).__name__}: {e}); "
                        f"falling back to classic")
        return ClassicDetector(scene=scene, params=params), "classic"
