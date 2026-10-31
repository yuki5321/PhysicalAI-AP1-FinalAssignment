"""Closed-loop check of the planner against the 2D push model alone.

Random start poses -> plan -> push (in the same model) until aligned.
Prints success rate, pushes and planning time per letter; `--csv` also writes
one row per trial. Letters run in parallel with `--jobs`.
"""

import argparse
import csv
import math
from multiprocessing import Pool
import random
import time

from pa_core.config import load_scene
from pa_core.letters import is_aligned, pose_error
from pa_core.planner import PushPlanner
from pa_core.pushsim import PushSim

FIELDS = ["letter", "trial", "x0", "y0", "theta0_deg", "success", "pushes",
          "final_dp_mm", "final_da_deg", "out_of_workspace", "plan_ms_mean"]


def random_pose(rng, r=0.07):
    return rng.uniform(-r, r), rng.uniform(-r, r), rng.uniform(-math.pi, math.pi)


def run_letter(letter, trials, max_pushes, seed, verbose=True):
    # Start poses depend only on (seed, letter), so letters can run in any order.
    rng = random.Random(f"{seed}-{letter}")
    random.seed(f"{seed}-{letter}-explore")  # planner exploration uses `random`
    ws = load_scene()["workspace"]["half_size"]
    planner = PushPlanner(letter)
    world = PushSim(letter)
    rows = []
    for trial in range(trials):
        start = pose = random_pose(rng)
        world.set_pose(*pose)
        best, stall, t_plan = 1e9, 0, []
        success, n = False, 0
        for n in range(max_pushes + 1):
            if is_aligned(letter, *pose):
                success = True
                break
            if n == max_pushes:
                break
            t0 = time.perf_counter()
            p = planner.plan(pose, explore=0.3 if stall >= 3 else 0.0)
            t_plan.append(time.perf_counter() - t0)
            world.set_pose(*pose)
            pose = world.push(p.side, p.angle_deg)
            c = sum(pose_error(letter, *pose))
            stall = 0 if c < best - 1e-3 else stall + 1
            best = min(best, c)
        dp, da = pose_error(letter, *pose)
        out = abs(pose[0]) > ws or abs(pose[1]) > ws
        if verbose and not success:
            print(f"  {letter} fail: dp={dp*1000:.1f}mm da={da:.1f}deg"
                  + (" (left workspace)" if out else ""), flush=True)
        rows.append(dict(letter=letter, trial=trial, x0=round(start[0], 4), y0=round(start[1], 4),
                         theta0_deg=round(math.degrees(start[2]), 1), success=int(success),
                         pushes=n, final_dp_mm=round(dp * 1000, 2), final_da_deg=round(da, 2),
                         out_of_workspace=int(out),
                         plan_ms_mean=round(1000 * sum(t_plan) / max(1, len(t_plan)), 1)))
    return rows


def summarize(letter, rows):
    ok = [r for r in rows if r["success"]]
    mean = sum(r["pushes"] for r in ok) / len(ok) if ok else float("nan")
    n_plans = sum(r["pushes"] for r in rows)
    t = sum(r["plan_ms_mean"] * r["pushes"] for r in rows) / max(1, n_plans)
    out = sum(r["out_of_workspace"] for r in rows)
    return (f"{letter}: success {len(ok)}/{len(rows)}, mean pushes {mean:.1f}, "
            f"plan {t:.0f} ms, left workspace {out}")


def _job(args):
    return run_letter(*args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--letters", default="IUYK")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--max-pushes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=1, help="letters simulated in parallel")
    ap.add_argument("--csv", help="write one row per trial to this file")
    args = ap.parse_args()

    jobs = [(letter, args.trials, args.max_pushes, args.seed) for letter in args.letters]
    if args.jobs > 1:
        with Pool(min(args.jobs, len(jobs))) as pool:
            results = pool.map(_job, jobs)
    else:
        results = [_job(j) for j in jobs]

    for letter, rows in zip(args.letters, results):
        print(summarize(letter, rows))
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for rows in results:
                w.writerows(rows)


if __name__ == "__main__":
    main()
