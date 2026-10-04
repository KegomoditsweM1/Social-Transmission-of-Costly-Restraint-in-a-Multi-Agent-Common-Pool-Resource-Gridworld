"""
Scripted (non-learning) policies for validating the environment.

These policies read the global apple grid directly, so they are PRIVILEGED:
they exist to test properties of the environment (e.g. the Schelling-diagram
gate of Section 6.2), not as learners. They are also the fallback teacher of
Section 6.3 if a trained tabular teacher is not reliably restrained.

pi_R = RestrainedPolicy (cooperator), pi_G = GreedyPolicy (defector),
pi_T^rho = TeacherPolicy (Eq. 9, mixes them only at restraint opportunities).

Every policy exposes  act(env, agent) -> action  and uses its own RNG, so its
randomness is independent of the environment's randomness.
"""

from __future__ import annotations

import numpy as np

from .env import MOVES, STAY, CommonsHarvestEnv


def _step_towards(env, agent, targets, forbidden, rng) -> int:
    """Move one step towards the nearest target cell (Manhattan distance),
    never stepping onto a forbidden cell. Returns STAY if no target exists."""
    if len(targets) == 0:
        return STAY
    y, x = env.agent_position(agent)
    d = np.abs(targets - np.array([y, x])).sum(axis=1)
    nearest = targets[rng.choice(np.flatnonzero(d == d.min()))]

    options = []
    for act, (dy, dx) in MOVES.items():
        ny, nx = y + dy, x + dx
        if act == STAY or not (0 <= ny < env.height and 0 <= nx < env.width):
            continue
        if forbidden[ny, nx]:
            continue
        if abs(nearest[0] - ny) + abs(nearest[1] - nx) < d.min():
            options.append(act)
    return int(rng.choice(options)) if options else STAY


class GreedyPolicy:
    """Defector: always heads for the nearest apple, including last apples."""

    name = "greedy"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, env: CommonsHarvestEnv, agent: str) -> int:
        targets = np.argwhere(env.apple_grid())
        forbidden = np.zeros((env.height, env.width), dtype=bool)
        return _step_towards(env, agent, targets, forbidden, self.rng)


class RestrainedPolicy:
    """Cooperator: never harvests a protected apple, i.e. an apple in a patch
    holding <= k apples. By default k = env.restraint_k, so the policy obeys
    exactly the definition that the restraint measure uses (k = 1: never take
    the last apple). Stays still when no harvestable apple exists."""

    name = "restrained"

    def __init__(self, min_left: int | None = None, seed: int | None = None):
        self.min_left = min_left  # None -> use env.restraint_k
        self.rng = np.random.default_rng(seed)

    def _forbidden(self, env: CommonsHarvestEnv) -> np.ndarray:
        k = env.restraint_k if self.min_left is None else self.min_left
        apples = env.apple_grid()
        counts = env.patch_counts()
        forbidden = np.zeros_like(apples)
        m = env.patch_mask
        forbidden[m] = counts[env.patch_id[m]] <= k
        return forbidden & apples

    def act(self, env: CommonsHarvestEnv, agent: str) -> int:
        forbidden = self._forbidden(env)
        targets = np.argwhere(env.apple_grid() & ~forbidden)
        return _step_towards(env, agent, targets, forbidden, self.rng)


def adjacent_protected_moves(env, agent) -> list[int]:
    """Moves that would step onto a protected apple from the agent's current cell
    (empty list = no restraint opportunity right now)."""
    y, x = env.agent_position(agent)
    prot = env.protected_apple_grid()
    out = []
    for act, (dy, dx) in MOVES.items():
        ny, nx = y + dy, x + dx
        if act != STAY and 0 <= ny < env.height and 0 <= nx < env.width and prot[ny, nx]:
            out.append(act)
    return out


class TeacherPolicy:
    """Teacher restraint levels, proposal Eq. 9:

        pi_T^rho = rho * pi_R + (1 - rho) * pi_G

    pi_R (restrained) and pi_G (greedy) differ only at a restraint opportunity,
    a step at which a protected (last) apple is adjacent. So the teacher behaves
    exactly like pi_R, EXCEPT that whenever a protected apple is adjacent it
    takes it with probability 1 - rho. rho = 1 is pi_R. The realised restraint
    rate is NOT rho (taking an apple changes later opportunities) and must be
    measured."""

    def __init__(self, rho: float, seed: int | None = None):
        self.rho = float(rho)
        self.name = f"teacher_rho{rho}"
        self.rng = np.random.default_rng(seed)
        self.restrained = RestrainedPolicy(seed=int(self.rng.integers(2**31)))

    def act(self, env: CommonsHarvestEnv, agent: str) -> int:
        moves = adjacent_protected_moves(env, agent)
        if moves and self.rng.random() >= self.rho:  # probability 1 - rho
            return int(self.rng.choice(moves))  # the defecting choice: take the last apple
        return self.restrained.act(env, agent)


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    def act(self, env: CommonsHarvestEnv, agent: str) -> int:
        return int(self.rng.integers(5))
