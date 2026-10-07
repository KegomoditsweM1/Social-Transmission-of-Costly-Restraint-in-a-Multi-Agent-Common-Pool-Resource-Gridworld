"""Phase 2 tests: value shaping, decision biasing, student training, parameter selection."""

import json
import subprocess
import sys

import numpy as np

from agents.features import N_STATES
from agents.q_learner import QLearner
from commons_env.env import APPLE, PROTECTED_APPLE


def _obs(self_pos, other_pos=None):
    return {"self_pos": np.array(self_pos), "other_visible": int(other_pos is not None),
            "other_pos": np.array(other_pos if other_pos else (0, 0))}


def test_reconstruct_window_copies_seen_cells_and_marks_walls():
    """Student at (5,5) sees grid rows 1-9, cols 1-9 (radius 4).
    Teacher at (2,5): its window covers grid rows -2..6, cols 1..9."""
    from agents.social import reconstruct_window
    from commons_env.env import EMPTY, WALL
    sw = np.zeros((9, 9), dtype=np.int8)
    sw[1, 4] = APPLE  # grid (2,5): the teacher's own cell
    sw[2, 6] = PROTECTED_APPLE  # grid (3,7)
    w = reconstruct_window(sw, (5, 5), (2, 5), (10, 10))
    assert w[4, 4] == APPLE  # grid (2,5), seen -> copied
    assert w[5, 6] == PROTECTED_APPLE  # grid (3,7), seen -> copied
    assert w[0, 4] == WALL and w[1, 4] == WALL  # grid rows -2, -1: outside the grid
    assert w[2, 4] == EMPTY  # grid row 0: inside the grid but NOT seen -> assumed empty


def test_value_shaping_adds_bonus_to_teacher_state_action():
    from agents.social import ValueShaping, teacher_state
    from commons_env import CommonsHarvestEnv
    from commons_env.env import LEFT as ENV_LEFT, STAY
    env = CommonsHarvestEnv(agent_ids=("teacher", "student"))
    obs0, _ = env.reset(seed=0, options={"positions": {"teacher": (2, 4), "student": (3, 5)}})
    obs1, *_ = env.step({"teacher": ENV_LEFT, "student": STAY})  # teacher moves onto apple (2,3)
    pair = teacher_state(obs0["student"], obs1["student"], (10, 10))
    assert pair is not None and pair[1] == ENV_LEFT
    q = QLearner(N_STATES, q_init=0.0)
    vs = ValueShaping(kappa=0.5)
    vs.apply(q, obs0["student"], obs1["student"], (10, 10))
    assert q.Q[pair[0], ENV_LEFT] == 0.5 and q.Q.sum() == 0.5 and vs.applied == 1


def test_value_shaping_needs_the_social_channel():
    from agents.social import teacher_state
    from commons_env import CommonsHarvestEnv
    from commons_env.env import LEFT as ENV_LEFT, STAY
    env = CommonsHarvestEnv(agent_ids=("teacher", "student"))
    env.set_social_channel("student", False)
    obs0, _ = env.reset(seed=0, options={"positions": {"teacher": (2, 4), "student": (3, 5)}})
    obs1, *_ = env.step({"teacher": ENV_LEFT, "student": STAY})
    assert teacher_state(obs0["student"], obs1["student"], (10, 10)) is None


def test_social_action_moves_towards_teacher():
    from agents.social import DecisionBiasing
    db = DecisionBiasing(omega=1.0, seed=0)
    q = QLearner(N_STATES, q_init=0.0, seed=0)
    q.Q[:, 0] = 1.0  # on its own the student would STAY
    a = db.choose(q, 0, _obs((5, 5), (5, 8)), greedy=True)  # teacher 3 to the right
    assert a == 4  # RIGHT


def test_social_action_uses_last_observed_position():
    from agents.social import DecisionBiasing
    db = DecisionBiasing(omega=1.0, seed=0)
    q = QLearner(N_STATES, q_init=0.0, seed=0)
    db.choose(q, 0, _obs((5, 5), (2, 5)), greedy=True)  # teacher seen above
    a = db.choose(q, 0, _obs((5, 5), None), greedy=True)  # now out of view
    assert a == 1  # UP: towards where it was last seen
    db.reset()
    q.Q[:, 3] = 1.0  # own preference: LEFT
    assert db.choose(q, 0, _obs((5, 5), None), greedy=True) == 3  # no memory -> asocial


def test_omega_zero_is_asocial_and_adjacent_means_stay():
    from agents.social import DecisionBiasing
    q = QLearner(N_STATES, q_init=0.0, seed=0)
    q.Q[:, 2] = 1.0  # own preference: DOWN
    db0 = DecisionBiasing(omega=0.0, seed=0)
    assert all(db0.choose(q, 0, _obs((5, 5), (2, 5)), greedy=True) == 2 for _ in range(20))
    db1 = DecisionBiasing(omega=1.0, seed=0)
    assert db1.choose(q, 0, _obs((5, 5), (5, 6)), greedy=True) == 0  # adjacent: STAY


def test_training_episodes_run_for_every_mechanism():
    """End-to-end: one short training episode per condition must run cleanly
    (guards against bugs that only appear when a whole episode is played)."""
    from agents.runner import run_episode
    from agents.social import DecisionBiasing, ValueShaping
    from commons_env import CommonsHarvestEnv
    from commons_env.scripted import TeacherPolicy
    for condition in ("asocial", "value_shaping", "decision_biasing"):
        env = CommonsHarvestEnv(agent_ids=("teacher", "student"), max_steps=30)
        env.set_social_channel("student", condition != "asocial")
        q = QLearner(N_STATES, seed=0)
        q.mask_walls = True
        q.social = ValueShaping(1.0) if condition == "value_shaping" else None
        q.bias = DecisionBiasing(0.3, seed=0) if condition == "decision_biasing" else None
        rec = run_episode(env, {"teacher": TeacherPolicy(1.0, seed=0), "student": q}, seed=1, learn=True)
        assert rec["steps"] == 30
        rec = run_episode(env, {"teacher": TeacherPolicy(1.0, seed=0), "student": q}, seed=2,
                          learn=False, social={"student": condition != "asocial"}, switch_off_at=15)
        assert rec["steps"] == 30


def test_selection_rule_picks_the_value_closest_to_half(tmp_path):
    """Synthetic summaries with known restraint: A_solo = 0.1, E = 1.
    kappa 0.001 -> f = 0.11 (not partial), 0.01 -> f = 0.44, 0.1 -> f = 0.67,
    1.0 -> f = 1.0 (complete). Expected: choose 0.01, sensitivity 0.1."""
    def write(cond, value, restraint, seed):
        cfg = {"condition": cond, "lam": 0.0, "rho": 1.0, "seed": seed,
               "kappa": value if cond == "value_shaping" else None, "omega": None}
        fin = {"student_restraint": restraint, "student_protected_taken_per_ep": 1.0,
               "student_return": 10.0}
        (tmp_path / f"{cond}_{value}_{seed}_summary.json").write_text(json.dumps({"config": cfg, "final": fin}))
    for s in (100, 101, 102):
        write("asocial", None, 0.1, s)
        for k, r in ((0.001, 0.2), (0.01, 0.5), (0.1, 0.7), (1.0, 1.0)):
            write("value_shaping", k, r, s)
    out = subprocess.run([sys.executable, "scripts/select_social_param.py", str(tmp_path)],
                         capture_output=True, text=True, check=True).stdout
    assert "chosen kappa = 0.01;  sensitivity value kappa = 0.1" in out
