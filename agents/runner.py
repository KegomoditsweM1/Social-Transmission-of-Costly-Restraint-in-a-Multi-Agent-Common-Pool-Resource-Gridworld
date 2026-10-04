"""
Run episodes with any mix of learning agents (QLearner) and scripted policies.

    run_episode(env, controllers, seed, learn=True, greedy=False)
        One episode. Learners update their Q-tables if learn=True.
        With greedy=True learners pick argmax actions (exploration off).

    evaluate(env, controllers, seeds)
        Frozen, greedy evaluation over several episodes. Restraint is POOLED:
        total protected harvests / total (opportunities - pre-empted), summed
        over all evaluation episodes, which avoids unstable per-episode ratios.
"""

from __future__ import annotations

import math

import numpy as np

from agents.features import decode, state_of, valid_actions
from agents.q_learner import QLearner


class FrozenTeacher:
    """A trained, frozen Q-learner used as a teacher at restraint level rho
    (proposal Eq. 9): it acts on its own greedy values, EXCEPT that whenever a
    protected apple is adjacent it takes it with probability 1 - rho."""

    def __init__(self, q: QLearner, rho: float = 1.0, seed: int | None = None):
        q.freeze()
        self.q, self.rho = q, float(rho)
        self.rng = np.random.default_rng(seed)
        self.name = f"frozen_teacher_rho{rho}"

    def act_from_obs(self, obs: dict, env, agent: str) -> int:
        from commons_env.scripted import adjacent_protected_moves
        moves = adjacent_protected_moves(env, agent)
        if moves and self.rng.random() >= self.rho:
            return int(self.rng.choice(moves))
        mask = valid_actions(obs["window"]) if getattr(self.q, "mask_walls", False) else None
        return self.q.act(state_of(obs, (env.height, env.width)), greedy=True, mask=mask)


def _blank_social(o: dict) -> dict:
    """The same observation with the social channel emptied (as if switched off)."""
    from commons_env.env import NO_ACTION
    o = dict(o)
    o["other_visible"] = 0
    o["other_last_action"] = NO_ACTION
    o["other_pos"] = np.zeros(2, dtype=np.int64)
    o["other_prev_pos"] = np.zeros(2, dtype=np.int64)
    return o


def run_episode(env, controllers: dict, seed: int, learn: bool = True, greedy: bool = False,
                social: dict | None = None, switch_off_at: int | None = None,
                eval_epsilon: float = 0.0) -> dict:
    """social: {agent: bool} social-channel state at the START of the episode.
    switch_off_at: step at which those agents' channels are switched OFF (A_half);
    decision-biasing memory is cleared at that moment, so a removed teacher
    leaves no trace. Restraint counts at the switch are stored in the record.
    eval_epsilon (E14, evaluation only): with this probability a learner takes a
    random valid move, EXCEPT when it is next to a protected apple, where it
    always acts on its learned values. This breaks navigation loops of frozen
    greedy policies without ever counting a random move as a restraint failure.
    Uses its own random stream, seeded by the episode seed."""
    eps_rng = np.random.default_rng(seed)
    for a, on in (social or {}).items():
        env.set_social_channel(a, on)
    obs, _ = env.reset(seed=seed)
    snapshot = None
    grid = (env.height, env.width)
    learners = {a for a, c in controllers.items() if isinstance(c, QLearner)}
    states = {a: state_of(obs[a], grid) for a in learners}

    def mask_of(a, o):  # E11: wall masking, only for learners that use it
        return valid_actions(o["window"]) if getattr(controllers[a], "mask_walls", False) else None

    masks = {a: mask_of(a, obs[a]) for a in learners}
    still = {a: 0 for a in env.possible_agents}  # current run of steps without moving
    longest_still = {a: 0 for a in env.possible_agents}
    last_pos = {a: env.agent_position(a) for a in env.possible_agents}
    for a in learners:  # decision biasing remembers the teacher only within an episode
        if getattr(controllers[a], "bias", None) is not None:
            controllers[a].bias.reset()

    while env.agents:
        if switch_off_at is not None and env.episode_stats["steps"] == switch_off_at:
            st = env.episode_stats
            snapshot = {a: (st["protected_harvests"][a], st["restraint_encounters"][a],
                            st["preempted_encounters"][a]) for a in env.possible_agents}
            for a in (social or {}):
                env.set_social_channel(a, False)
                if getattr(controllers[a], "bias", None) is not None:
                    controllers[a].bias.reset()
                obs[a] = _blank_social(obs[a])  # the next choice already sees no teacher
        actions = {}
        for a in env.agents:
            c = controllers[a]
            if a in learners and eval_epsilon > 0 and decode(states[a])[2] == 0 \
                    and eps_rng.random() < eval_epsilon:
                valid = np.flatnonzero(masks[a]) if masks[a] is not None else np.arange(5)
                actions[a] = int(eps_rng.choice(valid))  # never at a restraint decision
            elif a in learners:
                bias = getattr(c, "bias", None)  # e.g. DecisionBiasing; None otherwise
                if bias is not None:
                    actions[a] = bias.choose(c, states[a], obs[a], greedy, mask=masks[a])
                else:
                    actions[a] = c.act(states[a], greedy=greedy, mask=masks[a])
            elif hasattr(c, "act_from_obs"):
                actions[a] = c.act_from_obs(obs[a], env, a)  # frozen teacher
            else:
                actions[a] = c.act(env, a)  # scripted policy
        prev_obs = obs
        obs, rewards, terms, truncs, _ = env.step(actions)
        for a in env.possible_agents:
            pos = env.agent_position(a)
            still[a] = still[a] + 1 if pos == last_pos[a] else 0
            longest_still[a] = max(longest_still[a], still[a])
            last_pos[a] = pos
        for a in learners:
            c = controllers[a]
            s_next = state_of(obs[a], grid)
            next_mask = mask_of(a, obs[a])
            if learn:
                # truncation is not termination: always bootstrap
                c.update(states[a], actions[a], rewards[a], s_next, terminal=terms[a],
                         next_mask=next_mask)
                mechanism = getattr(c, "social", None)  # e.g. ValueShaping; None = asocial
                if mechanism is not None:
                    mechanism.apply(c, prev_obs[a], obs[a], grid)
            states[a] = s_next
            masks[a] = next_mask
    rec = episode_record(env)
    for a in env.possible_agents:
        rec[f"{a}_longest_still"] = longest_still[a]
        if snapshot is not None:  # restraint counts in the first half (before the switch)
            rec[f"{a}_protected_harvests_first"], rec[f"{a}_encounters_first"], \
                rec[f"{a}_preempted_encounters_first"] = snapshot[a]
    for a, on in (social or {}).items():  # leave the channel as it was at the start
        env.set_social_channel(a, on)
    return rec


def episode_record(env) -> dict:
    s = env.episode_stats
    rec = {"survival_time": s["survival_time"], "steps": s["steps"]}
    for a in env.possible_agents:
        rec[f"{a}_return"] = s["returns"][a]
        rec[f"{a}_harvests"] = s["harvests"][a]
        rec[f"{a}_protected_harvests"] = s["protected_harvests"][a]
        rec[f"{a}_opportunities"] = s["restraint_opportunities"][a]
        rec[f"{a}_preempted"] = s["preempted_opportunities"][a]
        rec[f"{a}_encounters"] = s["restraint_encounters"][a]
        rec[f"{a}_preempted_encounters"] = s["preempted_encounters"][a]
        rec[f"{a}_wall_blocked"] = s["wall_blocked"][a]
        rec[f"{a}_agent_blocked"] = s["agent_blocked"][a]
        rec[f"{a}_patch_deaths_lapse"] = s["patch_deaths_lapse"][a]
    rec["patch_deaths_simultaneous"] = s["patch_deaths_simultaneous"]
    rec["patch_deaths_other"] = s["patch_deaths_other"]
    return rec


def evaluate(env, controllers: dict, seeds, greedy: bool = True,
             social: dict | None = None, switch_off_at: int | None = None,
             eval_epsilon: float = 0.0) -> dict:
    """Frozen evaluation. Returns per-agent means and POOLED restraint rates.

    <agent>_restraint       PRIMARY: per encounter (E8), pooled over episodes
    <agent>_restraint_step  original Eq. 9: per step adjacent (inflated by waiting)
    <agent>_protected_taken_per_ep  protected apples taken per episode
    """
    records = [run_episode(env, controllers, seed=s, learn=False, greedy=greedy,
                           social=social, switch_off_at=switch_off_at,
                           eval_epsilon=eval_epsilon) for s in seeds]
    out = {"episodes": len(records)}
    surv = [r["survival_time"] for r in records]
    out["fully_depleted_frac"] = float(np.mean([t is not None for t in surv]))
    n, steps = len(records), sum(r["steps"] for r in records)
    out["patch_deaths_simultaneous_per_ep"] = sum(r["patch_deaths_simultaneous"] for r in records) / n
    out["patch_deaths_other_per_ep"] = sum(r["patch_deaths_other"] for r in records) / n
    for a in env.possible_agents:
        ret = np.array([r[f"{a}_return"] for r in records], dtype=float)
        took = sum(r[f"{a}_protected_harvests"] for r in records)
        opps = sum(r[f"{a}_opportunities"] - r[f"{a}_preempted"] for r in records)
        enc = sum(r[f"{a}_encounters"] - r[f"{a}_preempted_encounters"] for r in records)
        out[f"{a}_return_mean"] = float(ret.mean())
        out[f"{a}_return_sd"] = float(ret.std(ddof=1)) if len(ret) > 1 else 0.0
        out[f"{a}_harvests_mean"] = float(np.mean([r[f"{a}_harvests"] for r in records]))
        out[f"{a}_opportunities_total"] = int(opps)
        out[f"{a}_encounters_total"] = int(enc)
        out[f"{a}_protected_taken_per_ep"] = took / len(records)
        out[f"{a}_restraint"] = (1.0 - took / enc) if enc > 0 else math.nan
        out[f"{a}_restraint_step"] = (1.0 - took / opps) if opps > 0 else math.nan
        out[f"{a}_patch_deaths_lapse_per_ep"] = sum(r[f"{a}_patch_deaths_lapse"] for r in records) / n
        out[f"{a}_wall_blocked_frac"] = sum(r[f"{a}_wall_blocked"] for r in records) / steps
        out[f"{a}_agent_blocked_frac"] = sum(r[f"{a}_agent_blocked"] for r in records) / steps
        out[f"{a}_longest_still_mean"] = float(np.mean([r[f"{a}_longest_still"] for r in records]))
        if switch_off_at is not None:  # restraint AFTER the switch only (second half)
            took2 = sum(r[f"{a}_protected_harvests"] - r[f"{a}_protected_harvests_first"] for r in records)
            enc2 = sum((r[f"{a}_encounters"] - r[f"{a}_preempted_encounters"])
                       - (r[f"{a}_encounters_first"] - r[f"{a}_preempted_encounters_first"])
                       for r in records)
            out[f"{a}_restraint_after_switch"] = (1.0 - took2 / enc2) if enc2 > 0 else math.nan
    return out


def student_measures(r: dict, S: str = "student", T: str = "teacher") -> dict:
    """The measures reported for a student run, from an evaluate() result.
    Shared by train_student.py and reevaluate_students.py so they cannot drift."""
    return {
        "student_return": r[f"{S}_return_mean"],
        "student_harvests": r[f"{S}_harvests_mean"],
        "student_restraint": r[f"{S}_restraint"],  # per encounter (primary, E8)
        "student_restraint_step": r[f"{S}_restraint_step"],  # original Eq. 9
        "student_protected_taken_per_ep": r[f"{S}_protected_taken_per_ep"],
        "student_encounters": r[f"{S}_encounters_total"],
        "student_opportunities": r[f"{S}_opportunities_total"],
        "teacher_return": r[f"{T}_return_mean"],
        "teacher_restraint": r[f"{T}_restraint"],
        "collective_return": r[f"{S}_return_mean"] + r[f"{T}_return_mean"],
        "fully_depleted_frac": r["fully_depleted_frac"],
        # E11: collective outcome by cause, and stall diagnostics
        "patch_deaths_by_student_per_ep": r[f"{S}_patch_deaths_lapse_per_ep"],
        "patch_deaths_by_teacher_per_ep": r[f"{T}_patch_deaths_lapse_per_ep"],
        "patch_deaths_simultaneous_per_ep": r["patch_deaths_simultaneous_per_ep"],
        "student_wall_blocked_frac": r[f"{S}_wall_blocked_frac"],
        "student_agent_blocked_frac": r[f"{S}_agent_blocked_frac"],
        "student_longest_still_mean": r[f"{S}_longest_still_mean"],
    }
