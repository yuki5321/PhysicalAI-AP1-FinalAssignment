"""Fast tests of tools/eval_recovery.py + the sim2d backend (pytest, < 40 s).

    PYTHONPATH=ros2_ws/src/pa_core:ros2_ws/src/pa_planner:ros2_ws/src/pa_perception \\
        python3 -m pytest -q tools/test_eval_recovery.py
"""

import csv
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_recovery as ev  # noqa: E402
from pa_core.letters import is_disturbed, pose_error  # noqa: E402
from pa_planner.state_machine import Params, PlannerStateMachine  # noqa: E402
from recovery_sim2d import HAND_BOXES, Sim2DBackend, hand_world_boxes  # noqa: E402

CONTRACT_COLUMNS = """trial, backend, perception, letter, seed, disturb, start_x, start_y, start_theta,
success, fail_reason, pushes, sim_time_s, wall_time_s, final_dp_mm, final_da_deg"""


def _small_backend():
    # Small disturbances keep the test fast (few pushes).
    return Sim2DBackend("truth", teleport_range=0.025)


def test_truth_I_succeeds():
    be = _small_backend()
    for trial in range(3):
        row = ev.run_trial(be, "I", trial, seed=0)
        assert row["success"] == 1, row
        assert row["fail_reason"] == ""
        assert row["final_dp_mm"] < 8.0 and row["final_da_deg"] < 6.0
        assert row["pushes"] >= 1
        assert row["sim_time_s"] > 0


def test_csv_columns(tmp_path):
    expected = [c.strip() for c in CONTRACT_COLUMNS.replace("\n", " ").split(",")]
    assert ev.CONTRACT_FIELDS == expected
    assert ev.FIELDS[:len(expected)] == expected  # contract first, optional columns after
    assert ev.FIELDS[len(expected):] == ["disturb_tries", "disturb_dp_mm", "disturb_da_deg",
                                         "wait_hand_s", "rehand_at_push", "hand_violations"]
    row = ev.run_trial(_small_backend(), "I", 0, seed=0)
    assert row["wait_hand_s"] == "" and row["hand_violations"] == 0 and row["disturb_tries"] == 1
    path = tmp_path / "out.csv"
    ev.write_csv(str(path), [row])
    with open(path, newline="") as f:
        r = list(csv.reader(f))
    assert r[0][:len(expected)] == expected
    assert len(r) == 2


def test_deterministic_by_seed():
    be = _small_backend()
    a = ev.run_trial(be, "I", 1, seed=0)
    ev.run_trial(be, "I", 0, seed=1)  # something else in between
    b = ev.run_trial(be, "I", 1, seed=0)
    c = ev.run_trial(be, "I", 1, seed=1)
    a.pop("wall_time_s"), b.pop("wall_time_s"), c.pop("wall_time_s")
    assert a == b
    assert (a["start_x"], a["start_y"]) != (c["start_x"], c["start_y"])


class _BiasedPerception(Sim2DBackend):
    """Truth until the first push, then always reports the target pose."""

    def __init__(self):
        super().__init__("truth", teleport_range=0.025)
        self.pushed = False

    def reset(self, letter, pose, rng):
        self.pushed = False
        super().reset(letter, pose, rng)

    def push(self, side, angle_deg):
        self.pushed = True
        super().push(side, angle_deg)

    def observe(self):
        if self.pushed:
            return [(self.letter, 0.0, 0.0, 0.0, 1.0)]
        return super().observe()


def test_false_done():
    row = ev.run_trial(_BiasedPerception(), "I", 0, seed=0)
    assert row["success"] == 0
    assert row["fail_reason"] == "false_done"
    assert row["pushes"] == 1


class _Blind(Sim2DBackend):
    def observe(self):
        return []


def test_lost():
    row = ev.run_trial(_Blind("truth"), "U", 0, seed=0, timeout=60)
    assert row["fail_reason"] == "lost"  # never seen: IDLE -> LOST -> give up
    assert row["pushes"] == 0


def test_disturb_modes():
    import numpy as np
    be = Sim2DBackend("truth")
    for mode in ("teleport", "hand", "disc"):
        for s in range(3):
            rng = np.random.default_rng(s)
            be.reset("K", (0.0, 0.0, 0.0), rng)
            t0 = be.now()
            p = be.disturb(mode, rng)
            assert is_disturbed("K", *p), (mode, p)
            assert abs(p[0]) < 0.1 and abs(p[1]) < 0.1
            assert be.now() > t0
            assert p == be.true_pose()


def test_parse_params():
    p = ev.parse_params(["settle_s=0.2", "stall_pushes=2", "start_if_misaligned=false"], max_pushes=12)
    assert p.settle_s == 0.2 and p.stall_pushes == 2 and p.max_pushes == 12
    assert p.start_if_misaligned is False
    with pytest.raises(SystemExit):
        ev.parse_params(["nope=1"])


def test_summary_table():
    rows = [dict(letter="I", success=1, fail_reason="", pushes=4, sim_time_s=10.0, wall_time_s=1.0),
            dict(letter="I", success=0, fail_reason="max_pushes", pushes=30, sim_time_s=90.0,
                 wall_time_s=5.0)]
    s = ev.summarize(rows, "I")
    assert s[0]["n_ok"] == 1 and s[0]["n"] == 2 and not s[0]["met"]
    assert math.isclose(s[0]["pushes"], 4.0)
    assert "max_pushes 1" in ev.format_table(s, markdown=True)


# ------------------------------------------------------------ hand model / hand flow

def _intervals(trace):
    """[(t_present, t_last_present_or_absent)] of the hand in a run_trial trace."""
    out, t0 = [], None
    for t, e in trace:
        if e == "hand present":
            t0 = t
        elif e == "hand absent" and t0 is not None:
            out.append((t0, t))
            t0 = None
    return out


def test_hand_model_matches_gazebo_footprint():
    # palm 0.075 x 0.08 + 4 fingers + thumb (pa_description/models/hand/model.sdf)
    assert len(HAND_BOXES) == 6
    assert HAND_BOXES[0][2:4] == (0.075, 0.08)
    boxes = hand_world_boxes(0.01, -0.02, 0.0)
    palm = boxes[0][0]
    assert np.allclose(palm.min(axis=0), [0.01 - 0.0375, -0.02 - 0.04])
    assert all(zb < 0.012 for _, zb, _ in boxes)  # all touch a 12 mm letter at z_down


def test_hand_stroke_disturbs():
    be = Sim2DBackend("truth")
    for letter in "IK":
        for s in range(2):
            rng = np.random.default_rng(s)
            be.reset(letter, (0.0, 0.0, 0.0), rng)
            p = be.disturb("hand", rng)
            info = be.last_disturb
            assert info["mode"] == "hand" and 1 <= info["tries"] <= be.hand_tries
            assert is_disturbed(letter, *p) and max(abs(p[0]), abs(p[1])) < 0.1
            dp, _ = pose_error(letter, *p)
            assert dp > 0.015
            ev_ = info["events"]
            pres = [e[1] for e in ev_]
            # present from the start, then absent until the end (lift-off, flying away)
            k = pres.index(False)
            assert all(pres[:k]) and not any(pres[k:])
            ts = [e[0] for e in ev_]
            assert ts == sorted(ts) and math.isclose(ts[-1], be.now())
            assert ev_[-1][3] == [(letter, *p, 1.0)]  # final observation after the hand left
            # the hand is in the air at first: the letter is still at the target then
            assert ev_[2][3][0][1:4] == (0.0, 0.0, 0.0)


def test_hand_occludes_truth_observation():
    be = Sim2DBackend("truth")
    be.reset("I", (0.0, 0.0, 0.0), np.random.default_rng(0))
    be._hand = (0.0, 0.0, 0.003, 0.0)  # palm right over the letter
    assert be.occlusion() > 0.5 and be.observe() == []
    be._hand = (0.0, 0.0, 0.003, math.pi / 2)  # rotated: still covers the middle
    assert be.occlusion() > 0.2
    be._hand = (0.09, 0.0, 0.003, 0.0)  # beside it
    assert be.occlusion() == 0.0 and len(be.observe()) == 1


def test_hand_flow_waits_and_never_pushes_with_hand():
    trace = []
    row = ev.run_trial(Sim2DBackend("truth"), "K", 3, seed=0, disturb="hand", trace=trace)
    assert row["success"] == 1, row
    assert row["hand_violations"] == 0
    assert row["wait_hand_s"] >= Params().hand_clear_s - 1e-6
    assert row["disturb_dp_mm"] > 15.0
    (t_in, t_out), = _intervals(trace)
    states = [e for t, e in trace if t_in <= t <= t_out]
    assert "WAIT_HAND" in states  # disturbance seen while the hand was there
    pushes = [t for t, e in trace if e.startswith("push ")]
    assert pushes and min(pushes) >= t_out - 1 / 30 + Params().hand_clear_s - 1e-6


def test_pushed_with_hand_is_detected(monkeypatch):
    # A state machine that ignores /hand/state would push while the hand is there.
    monkeypatch.setattr(PlannerStateMachine, "on_hand", lambda self, present, now: self._flush())
    row = ev.run_trial(Sim2DBackend("truth"), "K", 0, seed=0, disturb="hand", max_pushes=2)
    assert row["hand_violations"] >= 1
    assert row["success"] == 0 and row["fail_reason"] == "pushed_with_hand"


@pytest.mark.parametrize("when", ["push", "settle"])
def test_rehand_pause_and_resume(when):
    trace = []
    row = ev.run_trial(Sim2DBackend("truth"), "I", 3, seed=0, disturb="hand", rehand_prob=1.0,
                       rehand_when=when, trace=trace)
    assert row["success"] == 1, row
    assert row["rehand_at_push"] == 2 and row["hand_violations"] == 0
    hc = Params().hand_clear_s
    assert row["wait_hand_s"] >= 2 * hc - 1e-6  # waited after each of the two strokes
    first, second = _intervals(trace)
    # the hand came back after the 2nd push was issued: during it, or after its push_done
    k = [t for t, e in trace if e.startswith("push ")][1]
    done = [t for t, e in trace if e == "push_done" and t > k][0]
    assert (second[0] < done) if when == "push" else (second[0] > done)
    assert [e for t, e in trace if t >= second[0]].count("WAIT_HAND") == 1
    later = [t for t, e in trace if e.startswith("push ") and t > second[0]]
    assert later and min(later) >= second[1] - 1 / 30 + hc - 1e-6


# ------------------------------------------------------------ point pusher (docs/point_pusher.md)

def test_point_backend_push_stroke():
    be = Sim2DBackend("truth", pusher="point")
    be.reset("I", (0.0, 0.0, 0.0), np.random.default_rng(0))
    assert be.sim.arms == []  # no paddles in the world
    t0 = be.now()
    be.push_stroke(-0.06, 0.0, 0.0, 0.0)
    assert be.true_pose()[0] > 0.005
    assert be.now() > t0 + 0.06 / 0.05  # stroke time at least
    with pytest.raises(RuntimeError):
        be.push(0, 30.0)
    with pytest.raises(RuntimeError):
        Sim2DBackend("truth").push_stroke(0, 0, 0.01, 0)
    with pytest.raises(ValueError):
        Sim2DBackend("truth", pusher="nope")


def test_point_trial_succeeds_with_strokes():
    be = Sim2DBackend("truth", teleport_range=0.04, pusher="point")
    trace = []
    for trial, letter in enumerate("UK"):
        row = ev.run_trial(be, letter, trial, seed=0, trace=trace)
        assert row["success"] == 1, row
        assert row["pushes"] <= 10
    assert any(e.startswith("push stroke ") for _, e in trace)
    assert not any(e.startswith("push ") and not e.startswith("push stroke") for _, e in trace)


def test_point_hand_flow():
    row = ev.run_trial(Sim2DBackend("truth", pusher="point"), "U", 1, seed=0, disturb="hand")
    assert row["success"] == 1 and row["hand_violations"] == 0, row


def test_cli_pusher_point(tmp_path, capsys):
    path = tmp_path / "p.csv"
    rows, summary = ev.main(["--pusher", "point", "--letters", "I", "--trials", "1", "--jobs", "1",
                             "-q", "--csv", str(path)])
    assert rows[0]["success"] == 1 and path.exists()
    assert "pusher=point" in capsys.readouterr().out
