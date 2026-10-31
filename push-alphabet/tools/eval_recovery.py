"""Disturb -> recover success rate per letter (contract: docs/recovery_eval.md).

Each trial: place the letter at the target, disturb it (`--disturb`), then let
the production state machine (`pa_planner.state_machine.PlannerStateMachine`)
recover it, with observations from the backend and the backend's clock.
Success is judged on the TRUE pose (`is_aligned`).

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner:ros2_ws/src/pa_perception \\
        python3 tools/eval_recovery.py --backend sim2d --perception truth --trials 20 \\
        --csv data/recovery/truth.csv

`--pusher point` (sim2d): the point pusher of docs/point_pusher.md instead of
the 4 paddles: the world is `PointPushSim`, the planner `PointPushPlanner`
(`PlannerPool(pusher="point")`), and the state machine's stroke commands
(`PushCmd.stroke`) go to `backend.push_stroke(x0, y0, x1, y1)`.

Trials are independent and seeded from (seed, letter, trial), so results do
not depend on `--jobs` or on the order trials run in.

Hand flow (`--disturb hand` on a backend that records the stroke, i.e. sim2d):
the backend's stroke timeline is replayed into the state machine at the
backend's clock: /hand/state-like `on_hand(True/False)` at 30 Hz and the
(possibly hand-occluded) observations at 10 Hz. A push issued while the hand is
present, or less than `hand_clear_s` after it left, is a violation
(`hand_violations`, fail_reason `pushed_with_hand`). `wait_hand_s` is the time
from the hand leaving to the state machine entering RECOVER (summed over the
strokes of the trial). `--rehand-prob p`: with probability p the hand comes back
after push k (k = 1..3): it leaves its park pose during that push or the settle
time after it (it is still in the air when the push ends), so the robot has to
pause and resume.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
from multiprocessing import Pool
import os
import random
import sys
import time

import numpy as np

from pa_core.config import load_scene
from pa_core.letters import is_aligned, pose_error
from pa_planner.state_machine import (FAILED, LOST, RECOVER, LetterDone, LetterObs, Params,
                                      PlannerPool, PlannerStateMachine, PushCmd, Status,
                                      run_plan_sync)

FIELDS = ["trial", "backend", "perception", "letter", "seed", "disturb", "start_x", "start_y",
          "start_theta", "success", "fail_reason", "pushes", "sim_time_s", "wall_time_s",
          "final_dp_mm", "final_da_deg"]
# Optional trailing columns (after the contract ones; "" where not applicable):
#   disturb_tries    disturbance attempts (hand: strokes until disturbed; > hand_tries = teleport fallback)
#   disturb_dp_mm    position error right after the disturbance
#   disturb_da_deg   angle error right after the disturbance
#   wait_hand_s      hand gone -> RECOVER, summed over the trial's strokes (hand flow only)
#   rehand_at_push   the hand came back during / right after this push (--rehand-prob)
#   hand_violations  pushes issued while the hand was present or < hand_clear_s after it left
EXTRA_FIELDS = ["disturb_tries", "disturb_dp_mm", "disturb_da_deg", "wait_hand_s",
                "rehand_at_push", "hand_violations"]
CONTRACT_FIELDS = list(FIELDS)
FIELDS = FIELDS + EXTRA_FIELDS
FAIL_REASONS = ["max_pushes", "failed", "timeout", "lost", "out_of_workspace", "false_done",
                "pushed_with_hand"]
TARGETS = {"I": 0.9, "U": 0.9, "Y": 0.7, "K": 0.7}

CAMERA_DT = 0.1    # observation / tick period [s] while waiting (10 Hz camera)
LOST_GIVEUP_S = 5.0  # LOST this long (after the state machine's own timeout) -> lost


# ---------------------------------------------------------------- backends

def make_backend(name, perception="truth", obs_noise=(0.0, 0.0), **kw):
    """Backend instance. `name` sim2d | gazebo, or an object (tests)."""
    if not isinstance(name, str):
        return name
    if name == "sim2d":
        from recovery_sim2d import Sim2DBackend
        return Sim2DBackend(perception=perception, obs_noise=obs_noise, **kw)
    if name == "gazebo":
        from recovery_gazebo import GazeboBackend  # needs ROS 2 + a running Gazebo
        if kw.pop("pusher", "paddles") != "paddles":
            raise ValueError("--pusher point is sim2d only (for now)")
        return GazeboBackend(perception=perception, **kw)
    raise ValueError(f"unknown backend {name!r}")


def trial_seed(seed, letter, trial) -> int:
    """Deterministic per-trial seed (independent of process / order)."""
    return int(np.random.SeedSequence([int(seed), ord(letter), int(trial)]).generate_state(1)[0])


def parse_params(items, max_pushes=None) -> Params:
    p = Params()
    if max_pushes is not None:
        p.max_pushes = int(max_pushes)
    types = {f.name: f.type for f in dataclasses.fields(Params)}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--params expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        k = k.strip().replace("-", "_")
        if k not in types:
            raise SystemExit(f"unknown Params field {k!r}; known: {', '.join(types)}")
        cur = getattr(p, k)
        if isinstance(cur, bool):
            val = v.strip().lower() in ("1", "true", "yes", "on")
        else:
            val = type(cur)(float(v)) if isinstance(cur, int) else type(cur)(v)
        setattr(p, k, val)
    return p


# ---------------------------------------------------------------- one trial

def _wait(backend, dt):
    w = getattr(backend, "wait", None)
    if w is not None:
        w(dt)
    else:
        time.sleep(dt)


def run_trial(backend, letter, trial, seed, disturb="teleport", params: Params | None = None,
              max_pushes=30, timeout=300.0, pool: PlannerPool | None = None,
              backend_name="sim2d", perception="truth", rehand_prob=0.0, rehand_when="any",
              trace=None, pusher=None):
    """Run one disturb -> recover trial. Returns a CSV row (dict).

    rehand_when: when the returning hand leaves its park pose: "push" (during
    push k), "settle" (after it, during the settle time) or "any" (either). trace: optional list, gets
    (t, event) for state changes, hand changes, pushes and push_done.
    pusher: "paddles" | "point" (default: the backend's `pusher`, else paddles)."""
    w0 = time.perf_counter()
    ts = trial_seed(seed, letter, trial)
    rng = np.random.default_rng(ts)
    rrng = np.random.default_rng([ts, 1])  # rehand draws: own stream (same disturbance either way)
    random.seed(ts)  # PushPlanner exploration uses the `random` module
    params = dataclasses.replace(params or Params())
    params.max_pushes = min(params.max_pushes, int(max_pushes))
    # A fresh planner per trial by default (PushSim rollouts depend slightly on
    # what was simulated before, so sharing one would couple the trials).
    pusher = pusher or getattr(backend, "pusher", "paddles")
    pool = pool or (PlannerPool(pusher="point") if pusher == "point" else PlannerPool())
    sc = load_scene()
    tgt = sc["target"]
    ws = sc["workspace"]["half_size"]
    # Beyond the paddle pivots no paddle can push the letter back in.
    reach = max(abs(c) for p in sc["pushers"]["list"] for c in p["pivot"]) + 0.01

    sm = PlannerStateMachine(params)
    pending: list = []
    done: list = []
    # hand bookkeeping (the hand as the state machine is told about it)
    hand = dict(present=False, last_present_t=-math.inf, left_t=-math.inf, wait_from=None, waits=[],
                violations=0)

    def log(t, what):
        if trace is not None:
            trace.append((round(t, 3), what))

    def handle(actions, now):
        for a in run_plan_sync(sm, pool, actions, now):
            if isinstance(a, Status):
                log(now, f"{a.state} {a.detail}".strip())
            if isinstance(a, PushCmd):
                st = getattr(a, "stroke", None)
                log(now, f"push {a.side} {a.angle_deg:.0f}" if st is None else
                    "push stroke " + " ".join(f"{v:.3f}" for v in st))
                if hand["present"] or now - hand["left_t"] < params.hand_clear_s - 1e-6:
                    hand["violations"] += 1
                pending.append(a)
            elif isinstance(a, LetterDone):
                done.append(a.letter)
            elif isinstance(a, Status) and a.state == RECOVER and hand["wait_from"] is not None:
                hand["waits"].append(now - hand["wait_from"])
                hand["wait_from"] = None

    def to_obs(raw):
        return [LetterObs(o[0], float(o[1]), float(o[2]), float(o[3]), float(o[4])) for o in raw]

    def feed_obs():
        now = backend.now()
        handle(sm.on_letters(to_obs(backend.observe()), now), now)

    def set_hand(present, t):
        if present != hand["present"]:
            log(t, f"hand {'present' if present else 'absent'}")
        if present:
            hand["last_present_t"] = t
        elif hand["present"]:
            # gone since its last "present" message (what hand_clear_s counts from)
            hand["left_t"] = hand["wait_from"] = hand["last_present_t"]
        hand["present"] = present
        handle(sm.on_hand(present, t), t)

    def push_done_(push_id, t):
        log(t, "push_done")
        handle(sm.on_push_done(push_id, t), t)

    def replay(events, push_done=None):
        """Feed a hand stroke timeline [(t, present, hand_pose, obs|None)] in time
        order; `push_done` = (push id, t) is delivered at its place in it."""
        for t, present, _hpos, obs in events:
            if push_done is not None and t >= push_done[1]:
                push_done_(*push_done)
                push_done = None
            set_hand(bool(present), t)
            if obs is not None:
                handle(sm.on_letters(to_obs(obs), t), t)
            handle(sm.tick(t), t)
        if push_done is not None:
            push_done_(*push_done)

    # 1. at the target, seen as aligned
    backend.reset(letter, (tgt["x"], tgt["y"], math.radians(tgt["theta"])), rng)
    t = backend.now()
    handle(sm.on_active_letter(letter, t), t)
    set_hand(False, t)
    feed_obs()
    # 2. disturb (the hand stroke, if recorded, is replayed into the state machine)
    start = backend.disturb(disturb, rng)
    info = getattr(backend, "last_disturb", None) or {}
    hand_flow = bool(info.get("events"))
    if hand_flow:
        replay(info["events"])
    t_start = backend.now()
    rehand_k = None
    if rehand_prob > 0 and rrng.random() < rehand_prob:
        rehand_k = int(rrng.integers(1, 4))
        if not hasattr(backend, "plan_hand_stroke"):
            rehand_k = None  # backend cannot do it (gazebo): plain trial
    rehand_at = None

    pushes, reason, lost_since = 0, None, None
    while True:
        now = backend.now()
        set_hand(False, now)
        feed_obs()
        handle(sm.tick(backend.now()), backend.now())
        if done:
            reason = "" if is_aligned(letter, *backend.true_pose()) else "false_done"
            break
        if sm.state == FAILED:
            reason = "max_pushes" if sm.rec.pushes >= params.max_pushes else "failed"
            break
        p = backend.true_pose()
        gone = getattr(backend, "out_of_workspace", None)  # gazebo: off table / fell / tipped
        if abs(p[0]) > reach or abs(p[1]) > reach or (gone is not None and gone()):
            reason = "out_of_workspace"
            break
        if sm.state == LOST:
            lost_since = now if lost_since is None else lost_since
            if now - lost_since > LOST_GIVEUP_S:
                reason = "lost"
                break
        else:
            lost_since = None
        if backend.now() - t_start > timeout:
            reason = "timeout"
            break
        if pending:
            if pushes >= max_pushes:
                reason = "max_pushes"
                break
            cmd = pending.pop(0)
            t_a = backend.now()
            st = getattr(cmd, "stroke", None)
            if st is not None:
                backend.push_stroke(*st)
            else:
                backend.push(cmd.side, cmd.angle_deg)
            pushes += 1
            t_b = backend.now()
            if rehand_k is not None and rehand_at is None and pushes == rehand_k:
                # The hand comes back: it leaves the park pose during this push
                # (still in the air when the push ends) or in the settle time after it.
                rehand_at = pushes
                plan = backend.plan_hand_stroke(rrng)
                lo = max(t_a, t_b - 0.9 * plan["air_s"])
                hi = t_b + 0.9 * params.settle_s
                lo, hi = {"push": (lo, t_b), "settle": (t_b + 1e-3, hi)}.get(rehand_when, (lo, hi))
                t_enter = float(rrng.uniform(lo, hi))
                if t_enter > t_b:
                    push_done_(cmd.id, t_b)
                    backend.wait(t_enter - t_b)
                    replay(backend.run_hand_stroke(plan))
                else:
                    replay(backend.run_hand_stroke(plan, t_start=t_enter), push_done=(cmd.id, t_b))
            else:
                push_done_(cmd.id, t_b)
        else:
            _wait(backend, CAMERA_DT)

    final = backend.true_pose()
    if reason and reason != "false_done" and (abs(final[0]) > ws or abs(final[1]) > ws):
        reason = "out_of_workspace"
    if hand["violations"]:
        reason = "pushed_with_hand"
    dp, da = pose_error(letter, *final)
    dp0, da0 = pose_error(letter, *start)
    return dict(trial=trial, backend=backend_name, perception=perception, letter=letter,
                seed=seed, disturb=disturb, start_x=round(start[0], 5), start_y=round(start[1], 5),
                start_theta=round(start[2], 4), success=int(reason == ""), fail_reason=reason,
                pushes=pushes, sim_time_s=round(backend.now() - t_start, 2),
                wall_time_s=round(time.perf_counter() - w0, 2),
                final_dp_mm=round(dp * 1000, 2), final_da_deg=round(da, 2),
                disturb_tries=info.get("tries", ""), disturb_dp_mm=round(dp0 * 1000, 2),
                disturb_da_deg=round(da0, 2),
                wait_hand_s=round(sum(hand["waits"]), 2) if hand["waits"] else "",
                rehand_at_push=rehand_at if rehand_at is not None else "",
                hand_violations=hand["violations"])


# ---------------------------------------------------------------- workers

_W: dict = {}


def _worker_init(cfg):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    _W["cfg"] = cfg
    _W["backend"] = make_backend(cfg["backend"], cfg["perception"], cfg["obs_noise"],
                                 **cfg.get("backend_kw", {}))


def _worker_run(job):
    letter, trial = job
    c = _W["cfg"]
    row = run_trial(_W["backend"], letter, trial, c["seed"], c["disturb"], c["params"],
                    c["max_pushes"], c["timeout"], None, c["backend"], c["perception"],
                    c.get("rehand_prob", 0.0), pusher=c.get("pusher", "paddles"))
    if c["verbose"]:
        s = "ok" if row["success"] else row["fail_reason"]
        print(f"  {letter}#{trial:02d} {s:16s} pushes={row['pushes']:2d} "
              f"sim={row['sim_time_s']:6.1f}s wall={row['wall_time_s']:5.1f}s "
              f"dp={row['final_dp_mm']:.1f}mm da={row['final_da_deg']:.1f}deg"
              + (f" wait_hand={row['wait_hand_s']}s" if row.get("wait_hand_s") != "" else "")
              + (f" rehand@{row['rehand_at_push']}" if row.get("rehand_at_push") != "" else ""),
              flush=True)
    return row


def run_eval(cfg, letters, trials, jobs=1):
    work = [(L, i) for L in letters for i in range(trials)]
    if jobs > 1 and cfg["backend"] == "sim2d":
        with Pool(min(jobs, len(work)), initializer=_worker_init, initargs=(cfg,)) as pool:
            rows = pool.map(_worker_run, work, chunksize=1)
    else:
        _worker_init(cfg)
        try:
            rows = [_worker_run(j) for j in work]
        finally:
            _W["backend"].close()
    return rows


# ---------------------------------------------------------------- summary

def summarize(rows, letters):
    out = []
    for L in letters:
        rs = [r for r in rows if r["letter"] == L]
        if not rs:
            continue
        ok = [r for r in rs if r["success"]]
        rate = len(ok) / len(rs)
        reasons = {}
        for r in rs:
            if not r["success"]:
                reasons[r["fail_reason"]] = reasons.get(r["fail_reason"], 0) + 1
        tgt = TARGETS.get(L)
        out.append(dict(
            letter=L, n_ok=len(ok), n=len(rs), rate=rate,
            pushes=(sum(r["pushes"] for r in ok) / len(ok)) if ok else float("nan"),
            sim_time=(sum(r["sim_time_s"] for r in ok) / len(ok)) if ok else float("nan"),
            wall=sum(r["wall_time_s"] for r in rs) / len(rs),
            reasons=", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) or "-",
            target=tgt, met=(tgt is None or rate >= tgt)))
    return out


def hand_summary(rows) -> str:
    """One line on the hand flow (empty if no trial had one)."""
    waits = [float(r["wait_hand_s"]) for r in rows if r.get("wait_hand_s", "") != ""]
    if not waits and not any(r.get("hand_violations") for r in rows):
        return ""
    re = [r for r in rows if r.get("rehand_at_push", "") != ""]
    viol = sum(int(r.get("hand_violations") or 0) for r in rows)
    tries = [int(r["disturb_tries"]) for r in rows if r.get("disturb_tries", "") != ""]
    return (f"hand: wait_hand_s mean {np.mean(waits):.2f} min {min(waits):.2f} max {max(waits):.2f} "
            f"(n={len(waits)}); pushes with hand present: {viol}; "
            f"rehand trials {len(re)} ({sum(r['success'] for r in re)} ok); "
            f"disturb tries mean {np.mean(tries) if tries else float('nan'):.2f}")


def format_table(summary, markdown=False):
    head = ["letter", "success", "rate", "target", "met", "mean pushes (ok)",
            "mean sim time s (ok)", "wall s/trial", "fail reasons"]
    lines = []
    for s in summary:
        lines.append([s["letter"], f"{s['n_ok']}/{s['n']}", f"{100 * s['rate']:.0f}%",
                      f"{100 * s['target']:.0f}%" if s["target"] else "-",
                      "yes" if s["met"] else "NO", f"{s['pushes']:.1f}", f"{s['sim_time']:.1f}",
                      f"{s['wall']:.1f}", s["reasons"]])
    if markdown:
        md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        md += ["| " + " | ".join(r) + " |" for r in lines]
        return "\n".join(md)
    widths = [max(len(h), *(len(r[i]) for r in lines)) for i, h in enumerate(head)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join([fmt.format(*head)] + [fmt.format(*r) for r in lines])


def write_csv(path, rows):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="sim2d", choices=["sim2d", "gazebo"])
    ap.add_argument("--perception", default=None,
                    choices=["truth", "classic", "cnn", "hybrid", "detector"],
                    help="default: truth (sim2d), detector = the running letter_detector (gazebo)")
    ap.add_argument("--obs-noise", nargs=2, type=float, default=[0.0, 0.0],
                    metavar=("SIGMA_POS_M", "SIGMA_ANG_RAD"), help="truth perception noise")
    ap.add_argument("--pusher", default="paddles", choices=["paddles", "point"],
                    help="point: Push-T style point pusher (docs/point_pusher.md), sim2d only")
    ap.add_argument("--letters", default="IUYK")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--disturb", default="teleport", choices=["teleport", "hand", "disc"],
                    help="hand: Gazebo-like hand stroke + hand flow in the state machine; "
                         "disc: old kinematic-disc sweep (sim2d only, no hand flow)")
    ap.add_argument("--rehand-prob", type=float, default=0.0,
                    help="probability that the hand comes back after push k=1..3 (sim2d)")
    ap.add_argument("--hand-vmax", type=float, default=0.15, help="hand speed limit [m/s] (sim2d)")
    ap.add_argument("--hand-tries", type=int, default=4,
                    help="hand strokes before falling back to a teleport (sim2d)")
    ap.add_argument("--max-pushes", type=int, default=30)
    ap.add_argument("--timeout", type=float, default=300.0, help="per trial, backend seconds")
    ap.add_argument("--jobs", type=int, default=2, help="parallel worker processes (sim2d)")
    ap.add_argument("--params", nargs="*", default=[], metavar="KEY=VALUE",
                    help="state machine Params overrides, e.g. settle_s=0.3 stall_pushes=2")
    ap.add_argument("--csv", help="write one row per trial")
    ap.add_argument("--markdown", action="store_true", help="print the table as Markdown")
    ap.add_argument("-q", "--quiet", action="store_true", help="no per-trial lines")
    args = ap.parse_args(argv)

    params = parse_params(args.params, args.max_pushes)
    perception = args.perception or ("truth" if args.backend == "sim2d" else "detector")
    if args.backend == "sim2d" and perception == "detector":
        raise SystemExit("--perception detector is gazebo only")
    if args.backend == "gazebo" and (args.disturb == "disc" or args.rehand_prob > 0):
        raise SystemExit("--disturb disc and --rehand-prob are sim2d only")
    if args.backend == "gazebo" and args.pusher != "paddles":
        raise SystemExit("--pusher point is sim2d only")
    backend_kw = (dict(hand_vmax=args.hand_vmax, hand_tries=args.hand_tries, pusher=args.pusher)
                  if args.backend == "sim2d" else {})
    cfg = dict(backend=args.backend, perception=perception,
               obs_noise=tuple(args.obs_noise), seed=args.seed, disturb=args.disturb, params=params,
               max_pushes=args.max_pushes, timeout=args.timeout, verbose=not args.quiet,
               rehand_prob=args.rehand_prob, backend_kw=backend_kw, pusher=args.pusher)
    t0 = time.perf_counter()
    rows = run_eval(cfg, args.letters, args.trials, args.jobs)
    wall = time.perf_counter() - t0
    if args.csv:
        write_csv(args.csv, rows)
    summary = summarize(rows, args.letters)
    print(f"\nbackend={args.backend} pusher={args.pusher} perception={cfg['perception']} disturb={args.disturb} "
          f"rehand_prob={args.rehand_prob} "
          f"trials={args.trials} seed={args.seed} max_pushes={args.max_pushes} "
          f"wall={wall:.0f}s")
    print(format_table(summary, markdown=args.markdown))
    hs = hand_summary(rows)
    if hs:
        print(hs)
    return rows, summary


if __name__ == "__main__":
    main()
