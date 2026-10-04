"""Tests for the teacher restraint levels (proposal Eq. 9). Run: pytest -q"""

import numpy as np

from commons_env import CommonsHarvestEnv
from commons_env.env import STAY
from commons_env.scripted import RestrainedPolicy, TeacherPolicy

T, S = "teacher", "student"


def last_apple_world(teacher_pos=(2, 3)):
    """Still world (no regrowth) with one last apple at (2, 2), next to the teacher."""
    env = CommonsHarvestEnv(lam=0.0, p_bar=0.0, agent_ids=(T, S), max_steps=50)
    apples = np.zeros((10, 10), bool)
    apples[2, 2] = True
    env.reset(seed=0, options={"apples": apples, "positions": {T: teacher_pos, S: (9, 9)}})
    return env


def test_rho_one_never_takes_a_last_apple():
    env = last_apple_world()
    t = TeacherPolicy(1.0, seed=0)
    while env.agents:
        env.step({T: t.act(env, T), S: STAY})
    assert env.episode_stats["protected_harvests"][T] == 0


def test_rho_zero_takes_an_adjacent_last_apple_immediately():
    env = last_apple_world()
    env.step({T: TeacherPolicy(0.0, seed=0).act(env, T), S: STAY})
    assert env.episode_stats["protected_harvests"][T] == 1


def test_rho_half_takes_it_about_half_the_time():
    taken = 0
    for seed in range(400):
        env = last_apple_world()
        env.step({T: TeacherPolicy(0.5, seed=seed).act(env, T), S: STAY})
        taken += env.episode_stats["protected_harvests"][T]
    assert 160 < taken < 240  # 200 expected; about 4 standard deviations either side


def test_identical_to_restrained_policy_when_no_last_apple_is_adjacent():
    """Eq. 9: pi_R and pi_G differ only at restraint opportunities."""
    env = CommonsHarvestEnv(agent_ids=(T, S))
    env.reset(seed=3)  # all patches full: no protected apples anywhere at the start
    t = TeacherPolicy(0.0, seed=7)  # rho = 0 would take any adjacent last apple
    r = RestrainedPolicy(seed=int(np.random.default_rng(7).integers(2**31)))  # same inner seed
    for _ in range(5):
        if env.protected_apple_grid().any():
            break
        a_t, a_r = t.act(env, T), r.act(env, T)
        assert a_t == a_r
        env.step({T: a_t, S: STAY})


def test_discounted_returns_match_hand_calculation():
    import sys
    sys.path.insert(0, "scripts")
    from check_gate import returns_from_rewards
    # rewards on steps 1..4: 1, 0, 0, 1 ; gamma = 0.5
    r = returns_from_rewards(np.array([1.0, 0.0, 0.0, 1.0]), gamma=0.5)
    # G_3 = 1 ; G_2 = 0.5 ; G_1 = 0.25 ; G_0 = 1 + 0.125 = 1.125
    assert r["undiscounted"] == 2.0
    assert np.isclose(r["disc_start"], 1.125)
    assert np.isclose(r["disc_avg"], (1.125 + 0.25 + 0.5 + 1.0) / 4)


def test_dilemma_classification_distinguishes_no_from_unclear():
    import sys
    sys.path.insert(0, "scripts")
    from check_gate import classify
    assert classify("yes", "yes", "yes", "no") == "yes"        # fear-only dilemma
    assert classify("yes", "yes", "no", "yes") == "yes"        # greed-only dilemma
    assert classify("yes", "yes", "unclear", "no") == "unclear"  # not established, NOT ruled out
    assert classify("no", "yes", "yes", "yes") == "no"         # mutual restraint does not beat mutual greed
    assert classify("yes", "yes", "no", "no") == "no"          # no temptation at all


def test_dilemma_classification_distinguishes_no_from_unclear():
    import sys
    sys.path.insert(0, "scripts")
    from check_gate import classify
    assert classify("yes", "yes", "yes", "no") == "yes"        # fear-only dilemma
    assert classify("yes", "yes", "no", "yes") == "yes"        # greed-only dilemma
    assert classify("yes", "yes", "unclear", "no") == "unclear"  # not established, NOT "no"
    assert classify("no", "yes", "yes", "yes") == "no"         # condition (1) ruled out
    assert classify("yes", "yes", "no", "no") == "no"          # both temptations ruled out
    assert classify("unclear", "yes", "yes", "yes") == "unclear"
