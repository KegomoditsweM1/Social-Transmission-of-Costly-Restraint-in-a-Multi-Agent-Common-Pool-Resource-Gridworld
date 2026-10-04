"""
Run with:  pytest -q

Each test checks one claim the experiments depend on. If a test fails, do not
run experiments until you understand why.
"""

import numpy as np
import pytest
from pettingzoo.test import parallel_api_test, parallel_seed_test

from commons_env import CommonsHarvestEnv
from commons_env.env import DOWN, LEFT, OTHER_AGENT, PROTECTED_APPLE, RIGHT, STAY, UP

P0, P1 = "player_0", "player_1"
STAY_BOTH = {P0: STAY, P1: STAY}


# ---------------- Official PettingZoo checks ----------------
@pytest.mark.parametrize("lam", [0.0, 0.5, 1.0])
def test_pettingzoo_parallel_api(lam):
    parallel_api_test(CommonsHarvestEnv(lam=lam), num_cycles=1000)


def test_pettingzoo_seed_determinism():
    parallel_seed_test(lambda: CommonsHarvestEnv(lam=0.5), num_cycles=500)


# ---------------- Helpers ----------------
def one_apple_grid(env, cell):
    apples = np.zeros_like(env.patch_mask)
    apples[cell] = True
    return apples


# ---------------- Mechanics ----------------
def test_harvest_gives_reward_and_removes_apple():
    env = CommonsHarvestEnv()
    env.reset(seed=0, options={"positions": {P0: (0, 2), P1: (9, 9)}})
    assert env.apple_grid()[1, 2]
    _, rew, *_ = env.step({P0: DOWN, P1: STAY})
    assert rew[P0] == 1.0 and rew[P1] == 0.0
    assert env.agent_position(P0) == (1, 2)
    assert not env.apple_grid()[1, 2]


def test_walls_block_movement():
    env = CommonsHarvestEnv()
    env.reset(seed=0, options={"positions": {P0: (0, 0), P1: (9, 9)}})
    *_, info = env.step({P0: UP, P1: RIGHT})
    assert env.agent_position(P0) == (0, 0) and info[P0]["blocked"]
    assert env.agent_position(P1) == (9, 9) and info[P1]["blocked"]


def test_agents_never_share_a_cell():
    env = CommonsHarvestEnv()
    for seed in range(50):
        env.reset(seed=seed, options={"positions": {P0: (4, 3), P1: (4, 5)}})
        env.step({P0: RIGHT, P1: LEFT})  # both try to enter (4, 4)
        assert env.agent_position(P0) != env.agent_position(P1)


def test_episode_truncates_at_max_steps():
    env = CommonsHarvestEnv(max_steps=7)
    env.reset(seed=0)
    for t in range(7):
        _, _, term, trunc, _ = env.step(STAY_BOTH)
        assert not any(term.values())
        assert all(trunc.values()) == (t == 6)
    assert env.agents == []


# ---------------- Regrowth (the dilemma manipulation) ----------------
def test_empty_patch_never_regrows_at_lam_1():
    env = CommonsHarvestEnv(lam=1.0, max_steps=5000)
    env.reset(seed=0, options={"apples": np.zeros((10, 10), bool),
                               "positions": {P0: (4, 4), P1: (4, 5)}})
    while env.agents:
        env.step(STAY_BOTH)
    assert env.apple_grid().sum() == 0


def test_single_apple_can_seed_regrowth_at_lam_1():
    env = CommonsHarvestEnv(lam=1.0, max_steps=5000)
    env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                               "positions": {P0: (4, 4), P1: (4, 5)}})
    while env.agents:
        env.step(STAY_BOTH)
    assert env.apple_grid()[1:4, 1:4].sum() > 1  # the top-left patch recovered
    assert env.apple_grid()[5:, :].sum() == 0  # empty patches stayed empty


def test_empty_patch_regrows_at_lam_0():
    env = CommonsHarvestEnv(lam=0.0, p_bar=0.01, max_steps=2000)
    env.reset(seed=0, options={"apples": np.zeros((10, 10), bool),
                               "positions": {P0: (4, 4), P1: (4, 5)}})
    while env.agents:
        env.step(STAY_BOTH)
    assert env.apple_grid().sum() > 0


def test_regrowth_rate_matches_formula():
    """Empirical regrowth probability of one empty cell vs Eq. 8 (n = 3 case)."""
    lam, p_bar = 0.5, 0.01
    env = CommonsHarvestEnv(lam=lam, p_bar=p_bar)
    expected = (1 - lam) * p_bar + lam * 0.025
    apples = env.patch_mask.copy()
    apples[1, 2] = False  # the other 4 cells of the patch are within radius 2
    hits, trials = 0, 20000
    env.reset(seed=1, options={"apples": apples, "positions": {P0: (4, 4), P1: (4, 5)}})
    for _ in range(trials):
        env._apples[:] = apples  # restore the same configuration each trial
        env._regrow()
        hits += env.apple_grid()[1, 2]
    rate = hits / trials
    se = np.sqrt(expected * (1 - expected) / trials)
    assert abs(rate - expected) < 4 * se, (rate, expected)


# ---------------- Restraint bookkeeping ----------------
def test_last_apple_is_flagged_and_counted():
    env = CommonsHarvestEnv()
    obs, _ = env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                                        "positions": {P0: (2, 3), P1: (9, 9)}})
    r = env.view_radius
    assert obs[P0]["window"][r, r - 1] == PROTECTED_APPLE  # directly to the left
    _, rew, _, _, info = env.step({P0: LEFT, P1: STAY})
    assert rew[P0] == 1.0
    assert info[P0]["restraint_opportunity"] and info[P0]["harvested_protected"]
    assert info[P0]["depleted_patch"]
    assert not info[P1]["restraint_opportunity"]
    s = env.episode_stats
    assert s["protected_harvests"][P0] == 1 and s["restraint_opportunities"][P0] == 1


def test_declining_last_apple_counts_as_restraint():
    env = CommonsHarvestEnv()
    env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                               "positions": {P0: (2, 3), P1: (9, 9)}})
    _, _, _, _, info = env.step({P0: RIGHT, P1: STAY})
    assert info[P0]["restraint_opportunity"] and not info[P0]["harvested_protected"]


# ---------------- Social channel ----------------
def test_social_channel_switch():
    env = CommonsHarvestEnv()
    obs, _ = env.reset(seed=0, options={"positions": {P0: (4, 4), P1: (4, 5)}})
    assert obs[P1]["other_visible"] == 1
    assert tuple(obs[P1]["other_pos"]) == (4, 4)
    env.set_social_channel(P1, False)
    obs, *_ = env.step(STAY_BOTH)
    assert obs[P1]["other_visible"] == 0
    # physical presence remains in the window even with the channel off
    r = env.view_radius
    assert obs[P1]["window"][r, r - 1] == OTHER_AGENT


def test_other_agent_invisible_when_far():
    env = CommonsHarvestEnv(view_radius=2)
    obs, _ = env.reset(seed=0, options={"positions": {P0: (0, 0), P1: (9, 9)}})
    assert obs[P0]["other_visible"] == 0 and obs[P1]["other_visible"] == 0


def test_restraint_k_widens_protected_set():
    """With restraint_k = 3, an apple in a 3-apple patch is protected; in a 4-apple patch it is not."""
    env = CommonsHarvestEnv(restraint_k=3)
    apples = np.zeros((10, 10), bool)
    apples[1, 2] = apples[2, 1] = apples[2, 2] = True          # top-left patch: 3 apples
    apples[1, 7] = apples[2, 6] = apples[2, 7] = apples[2, 8] = True  # top-right: 4 apples
    env.reset(seed=0, options={"apples": apples, "positions": {P0: (4, 4), P1: (4, 5)}})
    prot = env.protected_apple_grid()
    assert prot[1, 2] and prot[2, 1] and prot[2, 2]
    assert not prot[1, 7] and not prot[2, 7]


def test_scripted_restrained_policy_never_fails_restraint():
    from commons_env.metrics import run_episode
    from commons_env.scripted import GreedyPolicy, RestrainedPolicy
    for k in (1, 3):
        env = CommonsHarvestEnv(lam=1.0, restraint_k=k, max_steps=300)
        for seed in range(10):
            rec = run_episode(env, {P0: RestrainedPolicy(seed=seed), P1: GreedyPolicy(seed=seed)}, seed=seed)
            assert rec[f"{P0}_protected_harvests"] == 0


def test_social_radius_limits_observation_not_presence():
    """The other agent is SEEN in the window up to view_radius, but its position
    and actions are only OBSERVED within social_radius."""
    env = CommonsHarvestEnv(view_radius=4, social_radius=2)
    obs, _ = env.reset(seed=0, options={"positions": {P0: (4, 4), P1: (4, 7)}})  # 3 apart
    r = env.view_radius
    assert obs[P0]["window"][r, r + 3] == OTHER_AGENT
    assert obs[P0]["other_visible"] == 0
    obs, _ = env.reset(seed=0, options={"positions": {P0: (4, 4), P1: (4, 6)}})  # 2 apart
    assert obs[P0]["other_visible"] == 1


# ---------------- per-encounter restraint (E8) ----------------
def _still_env():
    return CommonsHarvestEnv(lam=0.0, p_bar=0.0)  # no regrowth: fully deterministic


def test_waiting_beside_an_apple_is_one_encounter():
    from commons_env.metrics import restraint_rate
    env = _still_env()
    env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                               "positions": {P0: (2, 3), P1: (9, 9)}})
    for _ in range(5):
        env.step(STAY_BOTH)  # wait beside the last apple
    s = env.episode_stats
    assert s["restraint_opportunities"][P0] == 5  # per step: 5
    assert s["restraint_encounters"][P0] == 1  # per encounter: 1
    assert restraint_rate(s, P0) == 1.0


def test_leaving_and_returning_is_a_new_encounter_and_taking_counts_once():
    from commons_env.metrics import restraint_rate
    env = _still_env()
    env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                               "positions": {P0: (2, 3), P1: (9, 9)}})
    env.step({P0: STAY, P1: STAY})   # encounter 1 begins
    env.step({P0: RIGHT, P1: STAY})  # walk away
    env.step({P0: LEFT, P1: STAY})   # come back
    env.step({P0: STAY, P1: STAY})   # encounter 2 begins
    env.step({P0: LEFT, P1: STAY})   # take the last apple
    s = env.episode_stats
    assert s["restraint_encounters"][P0] == 2 and s["protected_harvests"][P0] == 1
    assert restraint_rate(s, P0) == 0.5  # 1 taken out of 2 encounters
    assert np.isclose(restraint_rate(s, P0, per="step"), 1 - 1 / 4)  # 4 adjacent steps


def test_encounter_ended_by_the_other_agent_is_preempted():
    import math
    from commons_env.metrics import restraint_rate
    env = _still_env()
    env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                               "positions": {P0: (2, 3), P1: (1, 2)}})
    env.step({P0: STAY, P1: DOWN})  # player_1 takes the apple player_0 was beside
    s = env.episode_stats
    assert s["restraint_encounters"][P0] == 1 and s["preempted_encounters"][P0] == 1
    assert math.isnan(restraint_rate(s, P0))  # no real choice was made
    assert s["restraint_encounters"][P1] == 1 and restraint_rate(s, P1) == 0.0


# ---------------- E11: blocked-by cause and patch deaths ----------------
def test_blocked_by_wall_and_by_agent_are_counted_separately():
    env = _still_env()
    env.reset(seed=0, options={"positions": {P0: (0, 0), P1: (0, 1)}})
    env.step({P0: UP, P1: STAY})    # into the top edge
    env.step({P0: RIGHT, P1: STAY})  # into the other agent
    s = env.episode_stats
    assert s["wall_blocked"][P0] == 1 and s["agent_blocked"][P0] == 1


def test_patch_death_by_lapse_is_attributed():
    env = _still_env()
    env.reset(seed=0, options={"apples": one_apple_grid(env, (2, 2)),
                               "positions": {P0: (2, 3), P1: (9, 9)}})
    env.step({P0: LEFT, P1: STAY})  # takes the protected last apple
    s = env.episode_stats
    assert s["patch_deaths_lapse"][P0] == 1 and s["patch_deaths_simultaneous"] == 0


def test_simultaneous_patch_death_is_detected():
    """Two apples left, neither protected (restraint_k = 1): both agents take
    one in the same step. No restraint rule is broken, but the patch dies."""
    env = _still_env()
    apples = np.zeros((10, 10), bool)
    apples[2, 1] = apples[2, 3] = True  # the last two apples of the top-left patch
    env.reset(seed=0, options={"apples": apples, "positions": {P0: (2, 0), P1: (2, 4)}})
    _, _, _, _, info = env.step({P0: RIGHT, P1: LEFT})
    s = env.episode_stats
    assert s["patch_deaths_simultaneous"] == 1
    assert s["patch_deaths_lapse"][P0] == 0 and s["patch_deaths_lapse"][P1] == 0
    assert s["protected_harvests"][P0] == 0 and s["protected_harvests"][P1] == 0
    assert set(info) == {P0, P1}  # internal bookkeeping never leaks into infos
