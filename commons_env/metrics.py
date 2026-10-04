"""Outcome measures (Section 6.6) and a simple rollout helper."""

from __future__ import annotations

import math

from .env import CommonsHarvestEnv


def restraint_rate(stats: dict, agent: str, exclude_preempted: bool = True,
                   per: str = "encounter") -> float:
    """Restraint rate:  1 - (protected apples taken) / (opportunities).

    per = "encounter" (PRIMARY, adopted in E8): one opportunity per ENCOUNTER,
        i.e. each time the agent becomes adjacent to a protected apple. Waiting
        beside the same apple is one decision, however many steps it lasts.
    per = "step": the original Eq. 9 of the proposal: every step spent
        adjacent to a protected apple counts. Inflated by waiting (E7).

    If exclude_preempted, opportunities ended by the OTHER agent taking the
    apple are removed from the denominator. Returns NaN with no opportunities.
    """
    if per == "encounter":
        opps = stats["restraint_encounters"][agent]
        if exclude_preempted:
            opps -= stats["preempted_encounters"][agent]
    elif per == "step":
        opps = stats["restraint_opportunities"][agent]
        if exclude_preempted:
            opps -= stats["preempted_opportunities"][agent]
    else:
        raise ValueError("per must be 'encounter' or 'step'")
    if opps <= 0:
        return math.nan
    return 1.0 - stats["protected_harvests"][agent] / opps


def run_episode(env: CommonsHarvestEnv, policies: dict, seed: int | None = None) -> dict:
    """Run one episode with fixed (non-learning) policies; return a flat record."""
    env.reset(seed=seed)
    while env.agents:
        actions = {a: policies[a].act(env, a) for a in env.agents}
        env.step(actions)
    s = env.episode_stats
    record = {
        "seed": seed,
        "lam": env.lam,
        "restraint_k": env.restraint_k,
        "max_steps": env.max_steps,
        "collective_return": sum(s["returns"].values()),
        "survival_time": s["survival_time"],  # None = never fully depleted
        "patches_ever_depleted": len(s["patch_depletion_steps"]),
    }
    for a in env.possible_agents:
        record[f"{a}_policy"] = getattr(policies[a], "name", type(policies[a]).__name__)
        record[f"{a}_return"] = s["returns"][a]
        record[f"{a}_opportunities"] = s["restraint_opportunities"][a]
        record[f"{a}_protected_harvests"] = s["protected_harvests"][a]
        record[f"{a}_restraint"] = restraint_rate(s, a)
    return record
