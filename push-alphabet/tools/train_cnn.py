"""Train / export the letter CNN (docs/cnn.md).

Dataset format: <root>/{meta.json, labels.jsonl, frames/*.png} (docs/cnn.md).
Per labelled letter the training set holds crops around the TRUE letter center
(projected at the letter top height) jittered by +-8 px (`--jitters` copies),
plus one crop around the matching blob centroid (cnn_crop.blob_centers), which
is what inference sees. Validation uses the blob-centroid crops (plus the
un-jittered true-center crops, reported separately).

Examples (from the repo root):

  export PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_perception
  # small self-rendered dataset (development only)
  python tools/train_cnn.py --make-dev-data /tmp/dev_train --frames 500 --seed 1
  # train (checkpoints -> data/models/), then export ONNX + model card
  python tools/train_cnn.py --train data/synth_train --val data/synth_val --minutes 35
  python tools/train_cnn.py --export --ckpt data/models/letter_cnn_best.pt
"""

import argparse
import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
for p in ("ros2_ws/src/pa_core", "ros2_ws/src/pa_perception"):
    if str(REPO / p) not in sys.path:
        sys.path.insert(0, str(REPO / p))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from pa_perception import cnn_crop as cc  # noqa: E402
from pa_perception.camera_model import CameraModel  # noqa: E402

ONNX_PATH = REPO / "ros2_ws/src/pa_perception/models/letter_cnn.onnx"
CARD_PATH = ONNX_PATH.with_suffix(".json")
CKPT_DIR = REPO / "data/models"
OCC_MAX = 0.1  # "unoccluded" metrics: letters with occ <= this
SYM = np.array([cc.SYMMETRY[n] for n in cc.LETTERS])

# ------------------------------------------------------------------ dev data


def make_dev_dataset(root, frames, seed):
    """Render a small randomized dataset in the docs/cnn.md format (pa_perception.synth)."""
    from pa_core.config import load_scene
    from pa_perception.synth import random_scene, render_scene
    scene = load_scene()
    cam = CameraModel.from_scene(scene)
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "frames").mkdir(parents=True, exist_ok=True)
    half = scene["workspace"]["half_size"]
    with open(root / "labels.jsonl", "w") as f:
        for i in range(frames):
            n = int(rng.integers(1, 5))
            letters = random_scene(rng, n, half_size=half + 0.01, scene=scene)
            tint = rng.uniform(0.85, 1.15, 3)
            table = float(rng.uniform(15, 95))
            top = float(rng.uniform(max(table + 70, 150), 255))
            side = float(rng.uniform(0.5, 0.95) * top)
            img = render_scene(
                letters, cam, scene,
                table_value=tuple(np.clip(table * tint, 0, 255)),
                floor_value=tuple(np.clip(rng.uniform(60, 220) * tint, 0, 255)),
                top_value=tuple(np.clip(top * tint, 0, 255)),
                side_value=tuple(np.clip(side * tint, 0, 255)),
                sides=bool(rng.random() < 0.85),
                pusher_value=None if rng.random() < 0.3 else float(rng.uniform(60, 200)),
                noise_sigma=float(rng.uniform(0, 6)),
                blur_ksize=int(rng.choice([0, 0, 3, 5])), rng=rng)
            name = f"frames/{i:06d}.png"
            cv2.imwrite(str(root / name), img)
            f.write(json.dumps({"frame": name, "source": "synth", "letters": [
                {"letter": l, "x": x, "y": y, "theta": th} for l, x, y, th in letters]}) + "\n")
    meta = {"source": "synth", "seed": seed, "frames": frames, "camera": scene["camera"],
            "generator": "tools/train_cnn.py --make-dev-data"}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {frames} frames to {root}")


# ------------------------------------------------------------------ loading


def read_labels(root, limit=None):
    rows = []
    with open(Path(root) / "labels.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


_W = {}


def _winit():
    cv2.setNumThreads(1)
    _W["cam"] = CameraModel.from_scene()
    _W["det"] = cc._detector()


def _frame_crops(args):
    """Crops + targets of one frame. kinds: 0 true-center jittered, 1 blob centroid, 2 true center."""
    root, row, jitters, seed = args
    cam, det = _W["cam"], _W["det"]
    img = cv2.imread(str(Path(root) / row["frame"]), cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cc.to_gray(img)
    rng = np.random.default_rng(seed)
    labs = [l for l in row["letters"] if l["letter"] in cc.LETTERS]
    crops, tg = [], []

    def add(lab, u, v, kind):
        cw = cc.crop_center_world(u, v, cam)
        cls, ang, off = cc.encode_target(lab["letter"], lab["x"], lab["y"], lab["theta"], cw)
        crops.append(np.round(cc.extract_crop(gray, u, v)[0] * 255).astype(np.uint8))
        tg.append([cls, ang[0], ang[1], off[0], off[1], kind, lab["x"], lab["y"], lab["theta"],
                   cw[0], cw[1], float(lab.get("occ", 0.0))])

    pix = [cc.letter_center_pixel(l["x"], l["y"], cam) for l in labs]
    for lab, (u, v) in zip(labs, pix):
        if jitters == 0:
            add(lab, u, v, 2)
        for _ in range(jitters):
            du, dv = rng.uniform(-cc.JITTER_PX, cc.JITTER_PX, 2)
            add(lab, u + du, v + dv, 0)
    if pix:
        P = np.array(pix)
        for u, v, _ in cc.blob_centers(img, detector=det):
            d = np.hypot(P[:, 0] - u, P[:, 1] - v)
            k = int(np.argmin(d))
            # Unique, close match only (merged blobs of touching letters are skipped).
            if d[k] < 20.0 and (len(d) == 1 or np.partition(d, 1)[1] > 60.0):
                add(labs[k], u, v, 1)
    if not crops:
        return np.zeros((0, cc.CROP_SIZE, cc.CROP_SIZE), np.uint8), np.zeros((0, 12))
    return np.stack(crops), np.array(tg, dtype=np.float64)


def build_crops(root, jitters, seed, limit=None, workers=2):
    rows = read_labels(root, limit)
    args = [(str(root), r, jitters, seed * 1000003 + i) for i, r in enumerate(rows)]
    t0 = time.time()
    X, T = [], []
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers, initializer=_winit) as pool:
            for res in pool.imap(_frame_crops, args, chunksize=16):
                if res is not None:
                    X.append(res[0])
                    T.append(res[1])
    else:
        _winit()
        for a in args:
            res = _frame_crops(a)
            if res is not None:
                X.append(res[0])
                T.append(res[1])
    X, T = np.concatenate(X), np.concatenate(T)
    print(f"{root}: {len(rows)} frames -> {len(X)} crops "
          f"({int((T[:, 5] == 1).sum())} blob) in {time.time() - t0:.0f}s", flush=True)
    return X, T


# ------------------------------------------------------------------ metrics


def metrics(logits, angle, offset, T):
    """accuracy, position error (mm) and symmetry-aware angle error (deg)."""
    cls = T[:, 0].astype(int)
    pred = logits.argmax(1)
    acc = float((pred == cls).mean()) if len(cls) else float("nan")
    x = T[:, 9] + offset[:, 0] / 100.0
    y = T[:, 10] + offset[:, 1] / 100.0
    pos = np.hypot(x - T[:, 6], y - T[:, 7]) * 1000
    sym = np.radians(SYM[cls])  # true class symmetry
    th = np.arctan2(angle[:, 1], angle[:, 0]) * sym / (2 * np.pi)
    d = (th - T[:, 8] + sym / 2) % sym - sym / 2
    ang = np.degrees(np.abs(d))
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else float("nan")  # noqa: E731
    return {"n": int(len(cls)), "acc": acc,
            "pos_mm_mean": float(pos.mean()), "pos_mm_p95": q(pos, 95), "pos_mm_max": float(pos.max()),
            "ang_deg_mean": float(ang.mean()), "ang_deg_p95": q(ang, 95), "ang_deg_max": float(ang.max())}


def fmt(m):
    return (f"acc {m['acc'] * 100:.2f}% pos {m['pos_mm_mean']:.2f}/{m['pos_mm_p95']:.2f}mm "
            f"ang {m['ang_deg_mean']:.2f}/{m['ang_deg_p95']:.2f}deg (n={m['n']})")


# ------------------------------------------------------------------ training


def augment(x, gen):
    """Photometric augmentation of a [B,1,96,96] batch in 0..1 (torch)."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    r = lambda lo, hi: lo + (hi - lo) * torch.rand(B, 1, 1, 1, generator=gen)  # noqa: E731
    # Contrast about the crop mean, brightness offset, gamma.
    m = x.mean(dim=(2, 3), keepdim=True)
    x = (x - m) * r(0.6, 1.4) + m + r(-0.15, 0.15)
    x = x.clamp(0, 1) ** r(0.7, 1.4)
    # Blur (half the batch) with a random sigma.
    sel = torch.rand(B, generator=gen) < 0.5
    if sel.any():
        sig = float(0.3 + 1.0 * torch.rand(1, generator=gen))
        k = torch.arange(-2, 3, dtype=torch.float32)
        g = torch.exp(-k ** 2 / (2 * sig ** 2))
        g = g / g.sum()
        xb = x[sel]
        xb = F.conv2d(F.pad(xb, (2, 2, 0, 0), mode="replicate"), g.view(1, 1, 1, 5))
        xb = F.conv2d(F.pad(xb, (0, 0, 2, 2), mode="replicate"), g.view(1, 1, 5, 1))
        x = x.clone()
        x[sel] = xb
    # Gaussian noise.
    x = x + torch.randn(x.shape, generator=gen) * r(0.0, 0.05)
    return x.clamp(0, 1)


def predict(model, X, bs=512):
    import torch
    model.eval()
    outs = [[], [], []]
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.from_numpy(X[i:i + bs]).float().div_(255).unsqueeze(1)
            for o, t in zip(outs, model(xb)):
                o.append(t.numpy())
    return [np.concatenate(o) if o else np.zeros((0, 2)) for o in outs]


def evaluate(model, Xv, Tv):
    lg, an, of = predict(model, Xv)
    res = {}
    for kind, name in ((1, "blob"), (2, "center")):
        s = Tv[:, 5] == kind
        if s.any():
            res[name] = metrics(lg[s], an[s], of[s], Tv[s])
        s &= Tv[:, 11] <= OCC_MAX
        if s.any() and (Tv[:, 11] > OCC_MAX).any():
            res[name + "_unoccluded"] = metrics(lg[s], an[s], of[s], Tv[s])
    return res


def train(a):
    import torch
    import torch.nn.functional as F
    from pa_perception.cnn_model import LetterCNN, n_params
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    Xt, Tt = build_crops(a.train, a.jitters, a.seed, a.limit, a.workers)
    Xv, Tv = build_crops(a.val, 0, a.seed + 1, a.val_limit, a.workers)
    model = LetterCNN()
    print(f"LetterCNN: {n_params(model)} params", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, len(Xt) // a.batch)
    # Time-budgeted: estimate total steps after the first epoch, cosine decay over them.
    total_steps = a.epochs * steps_per_epoch
    gen = torch.Generator().manual_seed(a.seed)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best, history, step, t0 = None, [], 0, time.time()
    Tt_t = torch.from_numpy(Tt).float()
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        tl = np.zeros(4)
        for b in range(steps_per_epoch):
            idx = perm[b * a.batch:(b + 1) * a.batch]
            xb = torch.from_numpy(Xt[idx.numpy()]).float().div_(255).unsqueeze(1)
            xb = augment(xb, gen)
            tb = Tt_t[idx]
            lr = a.lr * 0.5 * (1 + math.cos(math.pi * min(step / max(total_steps, 1), 1.0)))
            for g in opt.param_groups:
                g["lr"] = lr
            lg, an, of = model(xb)
            l_cls = F.cross_entropy(lg, tb[:, 0].long(), label_smoothing=0.02)
            l_ang = F.mse_loss(an, tb[:, 1:3])
            l_off = F.smooth_l1_loss(of, tb[:, 3:5], beta=0.1)
            loss = l_cls + a.w_angle * l_ang + a.w_offset * l_off
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            tl += [loss.item(), l_cls.item(), l_ang.item(), l_off.item()]
        tl /= steps_per_epoch
        elapsed = time.time() - t0
        if ep == 0 and a.minutes:
            # Fit the schedule into the time budget.
            fit = int(a.minutes * 60 / elapsed)
            total_steps = min(a.epochs, max(fit, 1)) * steps_per_epoch
            a.epochs = min(a.epochs, max(fit, 1))
            print(f"epoch time {elapsed:.0f}s -> {a.epochs} epochs", flush=True)
        ev = evaluate(model, Xv, Tv)
        rec = {"epoch": ep + 1, "time_s": round(elapsed), "lr": lr,
               "train_loss": [round(float(v), 4) for v in tl], **ev}
        history.append(rec)
        vb = ev.get("blob") or ev.get("center")
        print(f"ep {ep + 1:3d} {elapsed:6.0f}s loss {tl[0]:.4f} (cls {tl[1]:.4f} ang {tl[2]:.4f} "
              f"off {tl[3]:.4f}) | val blob {fmt(vb)}"
              + (f" | center {fmt(ev['center'])}" if "center" in ev else "")
              + (f" | blob occ<={OCC_MAX} {fmt(ev['blob_unoccluded'])}" if "blob_unoccluded" in ev else ""),
              flush=True)
        # Selection score: errors normalized by the alignment tolerances (8 mm, 6 deg).
        score = (1 - vb["acc"]) * 10 + vb["pos_mm_p95"] / 8 + vb["ang_deg_p95"] / 6
        state = {"model": model.state_dict(), "epoch": ep + 1, "val": ev, "history": history,
                 "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
                 "n_train_crops": int(len(Xt)), "n_val_crops": int(len(Xv))}
        torch.save(state, CKPT_DIR / "letter_cnn_last.pt")
        if best is None or score < best:
            best = score
            torch.save(state, CKPT_DIR / "letter_cnn_best.pt")
        if ep + 1 >= a.epochs:
            break
    (CKPT_DIR / "letter_cnn_history.json").write_text(json.dumps(history, indent=1))
    print(f"done in {time.time() - t0:.0f}s, best -> {CKPT_DIR / 'letter_cnn_best.pt'}")


# ------------------------------------------------------------------ export


def export(a):
    import onnxruntime as ort
    import torch
    from pa_perception.cnn_model import LetterCNN, export_onnx, n_params
    torch.set_num_threads(a.threads)
    state = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = LetterCNN()
    model.load_state_dict(state["model"])
    model.eval()
    out = Path(a.onnx)
    out.parent.mkdir(parents=True, exist_ok=True)
    export_onnx(model, out)
    try:  # fold everything into a single file (no external data)
        import onnx
        m = onnx.load(str(out))
        onnx.checker.check_model(m)
        onnx.save(m, str(out))
    except ImportError:
        pass
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out), so, providers=["CPUExecutionProvider"])
    names = ([i.name for i in sess.get_inputs()], [o.name for o in sess.get_outputs()])
    assert names == (["input"], ["logits", "angle", "offset"]), names
    rng = np.random.default_rng(0)
    x = rng.random((7, 1, cc.CROP_SIZE, cc.CROP_SIZE), dtype=np.float32)
    r = sess.run(None, {"input": x})
    with torch.no_grad():
        rt = [t.numpy() for t in model(torch.from_numpy(x))]
    diff = max(float(np.abs(u - w).max()) for u, w in zip(r, rt))
    print(f"onnxruntime vs torch max abs diff: {diff:.2e}")
    assert diff < 1e-3
    x1 = x[:1]
    for _ in range(20):
        sess.run(None, {"input": x1})
    t = time.perf_counter()
    for _ in range(200):
        sess.run(None, {"input": x1})
    ms1 = (time.perf_counter() - t) / 200 * 1000
    x4 = x[:4]
    t = time.perf_counter()
    for _ in range(100):
        sess.run(None, {"input": x4})
    ms4 = (time.perf_counter() - t) / 100 * 1000
    size = os.path.getsize(out)
    print(f"{out}: {size / 1e6:.2f} MB, {ms1:.2f} ms/crop (batch 1), {ms4:.2f} ms/batch of 4 "
          "(onnxruntime CPU, 1 thread)")
    val = state.get("val", {})
    ev_extra = None
    if a.val:
        Xv, Tv = build_crops(a.val, 0, 12345, a.val_limit, a.workers)
        lg, an, of = [], [], []
        for i in range(0, len(Xv), 256):
            o = sess.run(None, {"input": Xv[i:i + 256].astype(np.float32)[:, None] / 255})
            lg.append(o[0]); an.append(o[1]); of.append(o[2])
        lg, an, of = map(np.concatenate, (lg, an, of))
        ev_extra = {}
        for kk, k in ((1, "blob"), (2, "center")):
            s = Tv[:, 5] == kk
            if s.any():
                ev_extra[k] = metrics(lg[s], an[s], of[s], Tv[s])
            s &= Tv[:, 11] <= OCC_MAX
            if s.any() and (Tv[:, 11] > OCC_MAX).any():
                ev_extra[k + "_unoccluded"] = metrics(lg[s], an[s], of[s], Tv[s])
        for k, m in ev_extra.items():
            print(f"onnx val {k}: {fmt(m)}")
    args = state.get("args", {})
    card = {
        "name": "letter_cnn",
        "date": datetime.date.today().isoformat(),
        "contract": "docs/cnn.md",
        "file": "letter_cnn.onnx",
        "opset": 17,
        "input": {"name": "input", "shape": ["N", 1, cc.CROP_SIZE, cc.CROP_SIZE], "dtype": "float32",
                  "crop": f"{cc.CROP_WINDOW}px window around the blob centroid, 2x downscale, 0..1"},
        "outputs": {"logits": ["N", 4], "angle": ["N", 2], "offset": ["N", 2]},
        "letters": list(cc.LETTERS),
        "symmetry_deg": dict(cc.SYMMETRY),
        "params": n_params(model),
        "onnx_bytes": size,
        "ort_ms_per_crop_batch1": round(ms1, 3),
        "ort_ms_batch4": round(ms4, 3),
        "train_data": args.get("train"),
        "val_data": args.get("val"),
        "data_note": a.data_note,
        "n_train_crops": state.get("n_train_crops"),
        "n_val_crops": state.get("n_val_crops"),
        "epochs": state.get("epoch"),
        "train_args": args,
        "val_metrics_at_checkpoint": val,
        "val_metrics_onnx": ev_extra,
        "metric_units": {"pos": "mm", "ang": "deg (symmetry-aware)", "acc": "fraction",
                         "blob": "crops at blob centroids (inference path)",
                         "center": "crops at true letter centers"},
        "history": [{k: h[k] for k in h if k != "lr"} for h in state.get("history", [])],
    }
    Path(a.card).write_text(json.dumps(card, indent=1))
    print(f"model card -> {a.card}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--make-dev-data", metavar="DIR")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--train", type=Path, default=REPO / "data/synth_train")
    ap.add_argument("--val", type=Path, default=REPO / "data/synth_val")
    ap.add_argument("--limit", type=int, default=None, help="max train frames")
    ap.add_argument("--val-limit", type=int, default=None)
    ap.add_argument("--jitters", type=int, default=3, help="jittered true-center crops per letter")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--minutes", type=float, default=35.0, help="time budget (0: run all epochs)")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-angle", type=float, default=1.0)
    ap.add_argument("--w-offset", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2, help="processes for crop extraction")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--ckpt", type=Path, default=CKPT_DIR / "letter_cnn_best.pt")
    ap.add_argument("--onnx", type=Path, default=ONNX_PATH)
    ap.add_argument("--card", type=Path, default=CARD_PATH)
    ap.add_argument("--data-note", default="")
    a = ap.parse_args()
    if a.make_dev_data:
        make_dev_dataset(a.make_dev_data, a.frames, a.seed)
    elif a.export:
        export(a)
    else:
        train(a)


if __name__ == "__main__":
    main()
