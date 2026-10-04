"""Phase 1 tests: state representation, Q-learner, wall mask, runner. Run: pytest -q"""

import numpy as np

from agents.features import N_STATES, decode, encode, features
from agents.q_learner import QLearner
from commons_env.env import APPLE, PROTECTED_APPLE, WALL

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def empty_window():
    return np.zeros((5, 5), dtype=np.int8)  # agent at the centre (2, 2)


# ---------------- features ----------------
def test_encode_decode_roundtrip():
    seen = set()
    for f in range(9):
        for d in range(5):
            for j in range(2):
                s = encode(f, d, j)
                assert 0 <= s < N_STATES and decode(s) == (f, d, j)
                seen.add(s)
    assert len(seen) == N_STATES == 90


def test_food_direction_points_to_nearest_ordinary_apple():
    w = empty_window()
    w[2, 4] = APPLE  # two cells to the right
    w[0, 2] = APPLE  # two cells up (same distance; earlier in scan order)
    food, _, _ = features(w, (4, 4))
    assert food == UP  # tie broken by row-major scan order
    w[0, 2] = 0
    assert features(w, (4, 4))[0] == RIGHT


def test_protected_apples_are_not_counted_as_food():
    w = empty_window()
    w[2, 3] = PROTECTED_APPLE  # adjacent, to the right
    food, prot_dir, prot_adj = features(w, (0, 0))  # no ordinary apple visible
    assert food >= 4  # falls back to the centre direction
    assert prot_dir == 1 + RIGHT and prot_adj == 1


def test_protected_apple_not_adjacent():
    w = empty_window()
    w[0, 2] = PROTECTED_APPLE  # two cells up
    _, prot_dir, prot_adj = features(w, (4, 4))
    assert prot_dir == 1 + UP and prot_adj == 0


def test_centre_fallback():
    w = empty_window()
    w[:, 0] = WALL
    assert features(w, (0, 0))[0] in (4 + DOWN, 4 + RIGHT)  # centre is down-right
    assert features(w, (9, 9))[0] in (4 + UP, 4 + LEFT)  # centre is up-left
    assert features(w, (4, 5))[0] == 8  # at the centre


# ---------------- learner ----------------
def test_q_update_matches_equation():
    q = QLearner(n_states=3, n_actions=2, alpha=0.5, gamma=0.9, q_init=0.0, seed=0)
    q.Q[1] = [2.0, 4.0]
    td = q.update(s=0, a=1, r=1.0, s_next=1)
    # target = 1 + 0.9 * 4 = 4.6 ; new Q = 0 + 0.5 * 4.6 = 2.3
    assert np.isclose(q.Q[0, 1], 2.3) and np.isclose(td, 4.6)


def test_terminal_update_does_not_bootstrap():
    q = QLearner(3, 2, alpha=1.0, gamma=0.9, q_init=0.0)
    q.Q[1] = [5.0, 5.0]
    q.update(0, 0, 1.0, 1, terminal=True)
    assert np.isclose(q.Q[0, 0], 1.0)


def test_frozen_learner_does_not_change():
    q = QLearner(3, 2, q_init=0.0)
    q.freeze()
    q.update(0, 0, 1.0, 1)
    assert q.Q.sum() == 0.0


def test_greedy_action_and_softmax():
    q = QLearner(1, 3, beta=50.0, q_init=0.0, seed=0)
    q.Q[0] = [0.0, 1.0, 0.0]
    assert q.act(0, greedy=True) == 1
    p = q.policy(0)
    assert np.isclose(p.sum(), 1.0) and p[1] > 0.99


def test_save_and_load(tmp_path):
    q = QLearner(4, 5, seed=0)
    q.Q[2, 3] = 7.0
    q.save(tmp_path / "q.npz")
    q2 = QLearner.load(tmp_path / "q.npz")
    assert np.array_equal(q.Q, q2.Q) and q2.gamma == q.gamma


# ---------------- E11: wall masking ----------------
def test_valid_actions_from_window():
    from agents.features import valid_actions
    w = np.zeros((9, 9), dtype=np.int8)
    w[:, :4] = WALL  # everything left of the agent is outside the grid
    w[:4, :] = WALL  # everything above it too
    assert valid_actions(w).tolist() == [True, False, True, False, True]


def test_masked_learner_never_walks_into_a_wall():
    q = QLearner(1, 5, beta=5.0, q_init=0.0, seed=0)
    q.Q[0] = [0.0, 9.0, 0.0, 0.0, 1.0]  # it "likes" UP, but UP is a wall
    mask = np.array([True, False, True, True, True])
    assert all(q.act(0, greedy=True, mask=mask) == 4 for _ in range(20))
    p = q.policy(0, mask)
    assert p[1] == 0.0 and np.isclose(p.sum(), 1.0)


def test_masked_target_ignores_invalid_next_actions():
    q = QLearner(2, 3, alpha=1.0, gamma=0.5, q_init=0.0)
    q.Q[1] = [1.0, 10.0, 2.0]  # the 10 belongs to an invalid action
    q.update(0, 0, 0.0, 1, next_mask=np.array([True, False, True]))
    assert np.isclose(q.Q[0, 0], 0.5 * 2.0)


# ---------------- CT evaluation: switching the channel off mid-episode ----------------
def test_channel_switch_off_mid_episode():
    from agents.runner import run_episode
    from commons_env import CommonsHarvestEnv

    class Recorder:  # stays still and records what it observes about the teacher
        def __init__(self):
            self.seen = []

        def act_from_obs(self, obs, env, agent):
            self.seen.append(int(obs["other_visible"]))
            return 0

    class Still:
        def act(self, env, agent):
            return 0

    env = CommonsHarvestEnv(agent_ids=("teacher", "student"), max_steps=6)
    rec = Recorder()
    env.reset(seed=0)  # positions are random; place the agents next to each other:
    orig_reset = env.reset
    env.reset = lambda seed=None, options=None: orig_reset(
        seed=seed, options={"positions": {"teacher": (4, 4), "student": (4, 5)}})
    out = run_episode(env, {"teacher": Still(), "student": rec}, seed=0, learn=False,
                      social={"student": True}, switch_off_at=3)
    assert rec.seen == [1, 1, 1, 0, 0, 0]  # observed before step 3, not after
    assert env._social_channel["student"] is True  # restored after the episode
    assert "student_encounters_first" in out


# ---------------- E14: evaluation epsilon never touches restraint decisions ----------------
def test_eval_epsilon_is_never_used_at_a_restraint_decision():
    from agents.runner import run_episode
    from commons_env import CommonsHarvestEnv
    from commons_env.env import LEFT as ENV_LEFT

    class Still:
        def act(self, env, agent):
            return 0

    def setup(student_pos):
        env = CommonsHarvestEnv(lam=0.0, p_bar=0.0, agent_ids=("teacher", "student"), max_steps=1)
        apples = np.zeros((10, 10), bool)
        apples[2, 2] = True  # the last apple of the top-left patch
        orig = env.reset
        env.reset = lambda seed=None, options=None: orig(
            seed=seed, options={"apples": apples, "positions": {"teacher": (9, 9), "student": student_pos}})
        return env

    q = QLearner(N_STATES, q_init=0.0, seed=0)
    q.Q[:, ENV_LEFT] = 1.0  # its learned choice: LEFT
    q.freeze()
    for seed in range(30):  # beside the protected apple: always greedy, even with epsilon = 1
        rec = run_episode(setup((2, 3)), {"teacher": Still(), "student": q}, seed=seed,
                          learn=False, greedy=True, eval_epsilon=1.0)
        assert rec["student_protected_harvests"] == 1
    moved = set()
    for seed in range(30):  # elsewhere, epsilon = 1 means random valid moves
        env = setup((5, 5))
        run_episode(env, {"teacher": Still(), "student": q}, seed=seed, learn=False, greedy=True,
                    eval_epsilon=1.0)
        moved.add(env.agent_position("student"))
    assert len(moved) > 2  # not always LEFT


# ---------------- Phase 1: the trained teacher (Eq. 9) ----------------
def _teacher_beside_last_apple():
    from commons_env import CommonsHarvestEnv
    env = CommonsHarvestEnv(lam=0.0, p_bar=0.0, agent_ids=("teacher",), max_steps=20)
    apples = np.zeros((10, 10), bool)
    apples[2, 2] = True  # one last apple, directly left of the teacher
    obs, _ = env.reset(seed=0, options={"apples": apples, "positions": {"teacher": (2, 3)}})
    return env, obs


def test_frozen_teacher_rho_one_follows_its_own_values():
    from agents.runner import FrozenTeacher
    env, obs = _teacher_beside_last_apple()
    q = QLearner(N_STATES, q_init=0.0, seed=0)
    q.Q[:, 2] = 1.0  # its own preference: DOWN (away from the apple)
    t = FrozenTeacher(q, rho=1.0, seed=0)
    assert all(t.act_from_obs(obs["teacher"], env, "teacher") == 2 for _ in range(30))
    assert q.frozen


def test_frozen_teacher_rho_zero_takes_an_adjacent_last_apple():
    from agents.runner import FrozenTeacher
    from commons_env.env import LEFT as ENV_LEFT
    env, obs = _teacher_beside_last_apple()
    q = QLearner(N_STATES, q_init=0.0, seed=0)
    q.Q[:, 2] = 1.0  # would go DOWN on its own
    t = FrozenTeacher(q, rho=0.0, seed=0)
    assert t.act_from_obs(obs["teacher"], env, "teacher") == ENV_LEFT  # takes the last apple


def test_teacher_trains_alone_with_wall_mask():
    """End-to-end: a few training episodes of a lone teacher run cleanly, and
    the wall mask means it never walks into the grid edge."""
    from agents.runner import run_episode
    from commons_env import CommonsHarvestEnv
    env = CommonsHarvestEnv(lam=1.0, agent_ids=("teacher",), max_steps=200)
    q = QLearner(N_STATES, seed=0)
    q.mask_walls = True
    for ep in range(3):
        rec = run_episode(env, {"teacher": q}, seed=ep, learn=True)
        assert rec["steps"] == 200
        assert env.episode_stats["wall_blocked"]["teacher"] == 0
    assert not np.allclose(q.Q, 1.0)  # it learned something
