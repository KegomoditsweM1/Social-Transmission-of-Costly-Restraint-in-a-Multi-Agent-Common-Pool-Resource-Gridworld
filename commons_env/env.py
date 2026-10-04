"""
Commons Harvest-style common-pool resource gridworld (PettingZoo Parallel API).

Implements the environment of Section 6.2 of the proposal:

* A grid containing several apple patches, written as an ASCII layout.
* N agents (default 2) that move simultaneously: STAY, UP, DOWN, LEFT, RIGHT.
* Moving onto an apple harvests it for a reward of +1.
* Each step, every empty patch cell (not occupied by an agent) regrows with
  probability

      p(n; lam) = (1 - lam) * p_bar + lam * p_SJ(n)                      (Eq. 8)

  where n is the number of apples within `regrowth_radius` (Euclidean) of the
  cell and p_SJ is the SocialJax Commons Harvest rule
  (0 if n = 0, 0.001 if n = 1, 0.005 if n = 2, 0.025 if n >= 3).
* Episodes end by truncation after `max_steps`; there is no terminal state.

The environment also records the quantities needed for the restraint measure
(Eq. 9): restraint opportunities and protected-apple harvests, per agent.
An apple is 'protected' if its patch holds <= restraint_k apples; the
proposal's definition (never take the last apple) is restraint_k = 1.

Nothing in this file learns. Learners and scripted policies live elsewhere.
"""

from __future__ import annotations

import warnings
from collections import deque
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
ACTION_NAMES = ("STAY", "UP", "DOWN", "LEFT", "RIGHT")
MOVES = {STAY: (0, 0), UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}
NO_ACTION = 5  # used in observations when the other agent's action is not observed

# Cell codes used in each agent's egocentric observation window
# PROTECTED_APPLE = an apple in a patch holding <= restraint_k apples
# (restraint_k = 1: the last apple in its patch, the proposal's definition).
EMPTY, APPLE, PROTECTED_APPLE, WALL, OTHER_AGENT = 0, 1, 2, 3, 4

# SocialJax regrowth probabilities indexed by min(n, 3), as quoted in the proposal
SOCIALJAX_PROBS = (0.0, 0.001, 0.005, 0.025)

# 10 x 10 layout with four 5-cell "diamond" patches.
#   'A' = apple cell (a patch cell that starts with an apple)
#   'P' = possible spawn point for an agent
#   '.' = empty floor
# Patches are 3 cells apart, so with radius 2 no patch influences another.
DEFAULT_LAYOUT = (
    "..........",
    "..A....A..",
    ".AAA..AAA.",
    "..A....A..",
    "....PP....",
    "....PP....",
    "..A....A..",
    ".AAA..AAA.",
    "..A....A..",
    "..........",
)


class CommonsHarvestEnv(ParallelEnv):
    """Two-player (by default) common-pool resource gridworld."""

    metadata = {
        "name": "commons_restraint_v0",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": True,
    }

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        layout: tuple[str, ...] | list[str] = DEFAULT_LAYOUT,
        lam: float = 1.0,
        p_bar: float = 0.001,
        sj_probs: tuple[float, float, float, float] = SOCIALJAX_PROBS,
        regrowth_radius: float = 2.0,
        max_steps: int = 1500,
        restraint_k: int = 1,
        view_radius: int = 4,
        social_radius: int = 2,
        agent_ids: tuple[str, ...] = ("player_0", "player_1"),
        render_mode: str | None = None,
    ):
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"lam must be in [0, 1], got {lam}")
        if not 0.0 <= p_bar <= 1.0:
            raise ValueError(f"p_bar must be in [0, 1], got {p_bar}")
        if len(sj_probs) != 4 or any(not 0.0 <= p <= 1.0 for p in sj_probs):
            raise ValueError("sj_probs must be four probabilities for n = 0, 1, 2, >=3")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode {render_mode!r}")

        self.lam = float(lam)
        self.p_bar = float(p_bar)
        self.sj_probs = np.asarray(sj_probs, dtype=np.float64)
        self.regrowth_radius = float(regrowth_radius)
        self.max_steps = int(max_steps)
        if restraint_k < 1:
            raise ValueError('restraint_k must be >= 1')
        self.restraint_k = int(restraint_k)
        self.view_radius = int(view_radius)  # how far apples/cells are seen
        self.social_radius = int(social_radius)  # how far the other agent is OBSERVED
        if not 0 <= self.social_radius <= self.view_radius:
            raise ValueError('social_radius must be between 0 and view_radius')
        self.render_mode = render_mode

        self.possible_agents = list(agent_ids)
        self.agents: list[str] = []

        self._parse_layout(layout)
        self._build_regrowth_neighbours()

        # Spaces are created once and returned unchanged (PettingZoo requirement;
        # the seed test re-seeds these exact objects).
        H, W, k = self.height, self.width, 2 * self.view_radius + 1
        obs_space = spaces.Dict(
            {
                "window": spaces.Box(0, 4, shape=(k, k), dtype=np.int8),
                "self_pos": spaces.MultiDiscrete([H, W]),
                "other_visible": spaces.Discrete(2),
                "other_pos": spaces.MultiDiscrete([H, W]),
                "other_prev_pos": spaces.MultiDiscrete([H, W]),
                "other_last_action": spaces.Discrete(6),
            }
        )
        self.observation_spaces = {
            a: spaces.Dict(obs_space.spaces) for a in self.possible_agents
        }
        self.action_spaces = {a: spaces.Discrete(5) for a in self.possible_agents}
        self.state_space = spaces.Box(
            0, 4 + len(self.possible_agents), shape=(H, W), dtype=np.int8
        )

        # The social channel is an experiment-level switch (Section 6.5): when it
        # is off, the agent receives no information about the other agent's
        # position or action. It persists across resets until changed.
        self._social_channel = {a: True for a in self.possible_agents}

        self._rng: np.random.Generator | None = None

    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    # ------------------------------------------------------------------ #
    # Layout handling
    # ------------------------------------------------------------------ #
    def _parse_layout(self, layout) -> None:
        rows = [r for r in layout]
        if len({len(r) for r in rows}) != 1:
            raise ValueError("All layout rows must have the same length")
        self.height, self.width = len(rows), len(rows[0])
        grid = np.array([list(r) for r in rows])
        bad = set(np.unique(grid)) - {".", "A", "P"}
        if bad:
            raise ValueError(f"Unknown layout characters: {bad}")

        self.patch_mask = grid == "A"  # cells that can ever hold an apple
        spawn = grid == "P"
        if not spawn.any():
            spawn = ~self.patch_mask
        self.spawn_cells = np.argwhere(spawn)
        if len(self.spawn_cells) < len(self.possible_agents):
            raise ValueError("Not enough spawn cells for all agents")

        # Patches = 4-connected components of apple cells.
        self.patch_id = -np.ones((self.height, self.width), dtype=np.int64)
        pid = 0
        for r, c in np.argwhere(self.patch_mask):
            if self.patch_id[r, c] != -1:
                continue
            queue = deque([(r, c)])
            self.patch_id[r, c] = pid
            while queue:
                y, x = queue.popleft()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < self.height
                        and 0 <= nx < self.width
                        and self.patch_mask[ny, nx]
                        and self.patch_id[ny, nx] == -1
                    ):
                        self.patch_id[ny, nx] = pid
                        queue.append((ny, nx))
            pid += 1
        self.n_patches = pid

    def _build_regrowth_neighbours(self) -> None:
        """Precompute which patch cells count as neighbours for regrowth."""
        self.patch_cells = np.argwhere(self.patch_mask)  # (K, 2) array of (row, col)
        self.patch_cell_pid = self.patch_id[self.patch_mask]  # (K,) patch ids, same order
        diff = self.patch_cells[:, None, :] - self.patch_cells[None, :, :]
        dist = np.sqrt((diff**2).sum(axis=-1))
        # neighbour_matrix[i, j] = 1 if cell j is within the radius of cell i (j != i)
        self.neighbour_matrix = ((dist <= self.regrowth_radius) & (dist > 0)).astype(np.int64)

        cross = self.neighbour_matrix.astype(bool) & (
            self.patch_cell_pid[:, None] != self.patch_cell_pid[None, :]
        )
        if cross.any():
            warnings.warn(
                "Some patches lie within the regrowth radius of one another. "
                "Taking the 'last apple' of a patch will then not necessarily "
                "stop that patch from regrowing.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ #
    # Read-only helpers (used by scripted policies, tests and analysis)
    # ------------------------------------------------------------------ #
    def apple_grid(self) -> np.ndarray:
        return self._apples.copy()

    def patch_counts(self) -> np.ndarray:
        """Number of apples currently in each patch."""
        return np.bincount(
            self.patch_cell_pid[self._apples[self.patch_mask]], minlength=self.n_patches
        )

    def protected_apple_grid(self) -> np.ndarray:
        """Boolean grid: True where an apple lies in a patch holding <= restraint_k
        apples. Harvesting such an apple is a failure of restraint.
        With restraint_k = 1 these are exactly the last apples of their patches."""
        counts = self.patch_counts()
        prot = np.zeros_like(self._apples)
        pid = self.patch_id
        prot[self.patch_mask] = counts[pid[self.patch_mask]] <= self.restraint_k
        return prot & self._apples

    def agent_position(self, agent: str) -> tuple[int, int]:
        return tuple(int(v) for v in self._pos[agent])

    def set_social_channel(self, agent: str, on: bool) -> None:
        """Switch an agent's access to the other agent's position/action on or off.

        Can be called mid-episode (needed for the A_half condition of the CT metric).
        Takes effect from the next observation returned.
        """
        self._social_channel[agent] = bool(on)

    # ------------------------------------------------------------------ #
    # PettingZoo API
    # ------------------------------------------------------------------ #
    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None or self._rng is None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        self.agents = list(self.possible_agents)
        self._t = 0

        # Apples: by default every patch cell starts full.
        if "apples" in options:
            apples = np.asarray(options["apples"], dtype=bool)
            if apples.shape != self.patch_mask.shape or (apples & ~self.patch_mask).any():
                raise ValueError("options['apples'] must be a boolean grid inside patch cells")
            self._apples = apples.copy()
        else:
            self._apples = self.patch_mask.copy()

        # Positions: sampled from spawn cells unless given explicitly.
        if "positions" in options:
            self._pos = {a: np.array(options["positions"][a]) for a in self.agents}
            cells = [tuple(p) for p in self._pos.values()]
            if len(set(cells)) != len(cells):
                raise ValueError("Agents must start on distinct cells")
            for y, x in cells:
                if not (0 <= y < self.height and 0 <= x < self.width) or self._apples[y, x]:
                    raise ValueError(f"Invalid start position {(y, x)}")
        else:
            idx = self._rng.choice(len(self.spawn_cells), size=len(self.agents), replace=False)
            self._pos = {a: self.spawn_cells[i].copy() for a, i in zip(self.agents, idx)}

        self._prev_pos = {a: p.copy() for a, p in self._pos.items()}
        self._last_action = {a: NO_ACTION for a in self.agents}
        # protected-apple cells each agent was already next to (ongoing encounters)
        self._encounter_cells = {a: set() for a in self.agents}

        self.episode_stats = {
            "returns": {a: 0.0 for a in self.agents},
            "harvests": {a: 0 for a in self.agents},
            "protected_harvests": {a: 0 for a in self.agents},
            "restraint_opportunities": {a: 0 for a in self.agents},
            "preempted_opportunities": {a: 0 for a in self.agents},
            "restraint_encounters": {a: 0 for a in self.agents},
            "wall_blocked": {a: 0 for a in self.agents},   # moves into the grid edge
            "agent_blocked": {a: 0 for a in self.agents},  # moves into the other agent
            # every patch death (a patch going from >0 to 0 apples), by cause:
            "patch_deaths_lapse": {a: 0 for a in self.agents},  # a took a protected apple
            "patch_deaths_simultaneous": 0,  # several agents took the last apples together
            "patch_deaths_other": 0,
            "preempted_encounters": {a: 0 for a in self.agents},
            "survival_time": None,  # first step at which every patch is empty
            "patch_depletion_steps": {},  # patch id -> first step it became empty
            "steps": 0,
        }

        observations = {a: self._observe(a) for a in self.agents}
        infos = {a: {"position": self.agent_position(a)} for a in self.agents}
        if self.render_mode == "human":
            self.render()
        return observations, infos

    def step(self, actions: dict[str, int]):
        if not self.agents:
            return {}, {}, {}, {}, {}

        for a in self.agents:
            if a not in actions or not self.action_spaces[a].contains(actions[a]):
                raise ValueError(f"Missing or invalid action for {a}: {actions.get(a)!r}")

        self._t += 1
        rewards = {a: 0.0 for a in self.agents}

        # 1. Restraint opportunities are judged on the state at the START of the
        #    step, i.e. on what the agent could have observed when it chose.
        prot_at_start = self.protected_apple_grid()
        adjacent_prot = {a: self._adjacent_cells(a, prot_at_start) for a in self.agents}
        opportunity = {a: len(adjacent_prot[a]) > 0 for a in self.agents}
        # An ENCOUNTER starts when an agent becomes adjacent to a protected apple
        # it was not adjacent to on the previous step. Waiting beside the same
        # apple continues one encounter: one decision, not one per step.
        new_encounters = {a: set(adjacent_prot[a]) - self._encounter_cells[a] for a in self.agents}
        took_cell = {a: None for a in self.agents}
        counts_at_start = self.patch_counts()
        harvested_in_patch = {a: None for a in self.agents}

        # 2. Movement, resolved one agent at a time in a random order each step.
        #    A move fails if it leaves the grid or enters a cell occupied by
        #    another agent at that moment.
        info = {
            a: {
                "harvested": False,
                "harvested_protected": False,
                "depleted_patch": False,
                "blocked": False,
                "blocked_by": None,  # "wall" or "agent"
            }
            for a in self.agents
        }
        self._prev_pos = {a: p.copy() for a, p in self._pos.items()}
        order = self._rng.permutation(len(self.agents))
        for i in order:
            a = self.agents[i]
            act = int(actions[a])
            self._last_action[a] = act
            dy, dx = MOVES[act]
            y, x = self._pos[a][0] + dy, self._pos[a][1] + dx
            if act == STAY:
                continue
            occupied = any(
                (self._pos[b][0] == y and self._pos[b][1] == x) for b in self.agents if b != a
            )
            if not (0 <= y < self.height and 0 <= x < self.width) or occupied:
                info[a]["blocked"] = True
                info[a]["blocked_by"] = "agent" if occupied else "wall"
                continue
            self._pos[a] = np.array([y, x])
            if self._apples[y, x]:
                self._apples[y, x] = False
                rewards[a] = 1.0
                info[a]["harvested"] = True
                harvested_in_patch[a] = int(self.patch_id[y, x])
                info[a]["harvested_protected"] = bool(prot_at_start[y, x])
                if prot_at_start[y, x]:
                    took_cell[a] = (int(y), int(x))
                if self.patch_counts()[self.patch_id[y, x]] == 0:
                    info[a]["depleted_patch"] = True

        # An opportunity is "pre-empted" if the agent did not take a protected
        # apple but every protected apple next to it was taken by someone else.
        for a in self.agents:
            took = info[a]["harvested_protected"]
            gone = all(not self._apples[y, x] for (y, x) in adjacent_prot[a])
            info[a]["restraint_opportunity"] = opportunity[a]
            info[a]["preempted"] = bool(opportunity[a] and not took and gone)
            # encounters: new ones this step, and ones ended because the OTHER
            # agent took the apple (the agent never got to finish its choice)
            info[a]["new_encounters"] = len(new_encounters[a])
            info[a]["preempted_encounters"] = sum(
                1 for c in adjacent_prot[a] if not self._apples[c] and c != took_cell[a])
            self._encounter_cells[a] = {c for c in adjacent_prot[a] if self._apples[c]}

        # Classify every patch that was emptied by this step's harvesting.
        counts_now = self.patch_counts()
        info["_patch_deaths"] = []
        for pid in np.flatnonzero((counts_at_start > 0) & (counts_now == 0)):
            takers = [a for a in self.agents if harvested_in_patch[a] == pid]
            lapsed = [a for a in takers if info[a]["harvested_protected"]]
            if lapsed:
                cause = ("lapse", lapsed)
            elif len(takers) >= 2:
                cause = ("simultaneous", takers)
            else:
                cause = ("other", takers)
            info["_patch_deaths"].append(cause)

        # 3. Regrowth.
        self._regrow()

        # 4. Bookkeeping.
        self._update_stats(rewards, info)
        truncated = self._t >= self.max_steps
        terminations = {a: False for a in self.agents}
        truncations = {a: truncated for a in self.agents}
        observations = {a: self._observe(a) for a in self.agents}
        counts = self.patch_counts()
        for a in self.agents:
            info[a]["position"] = self.agent_position(a)
            info[a]["apples_remaining"] = int(self._apples.sum())
            info[a]["patches_alive"] = int((counts > 0).sum())

        info.pop("_patch_deaths", None)  # internal; infos must be keyed by agent only
        if self.render_mode == "human":
            self.render()
        if truncated:
            self.agents = []
        return observations, rewards, terminations, truncations, info

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _adjacent_cells(self, agent: str, mask: np.ndarray) -> list[tuple[int, int]]:
        """Cells at Manhattan distance 1 from `agent` where `mask` is True."""
        y0, x0 = self._pos[agent]
        out = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            y, x = y0 + dy, x0 + dx
            if 0 <= y < self.height and 0 <= x < self.width and mask[y, x]:
                out.append((int(y), int(x)))
        return out

    def _regrow(self) -> None:
        apples_k = self._apples[self.patch_mask].astype(np.int64)  # snapshot, shape (K,)
        n = self.neighbour_matrix @ apples_k  # apples within radius of each patch cell
        p = (1.0 - self.lam) * self.p_bar + self.lam * self.sj_probs[np.minimum(n, 3)]

        occupied = np.zeros(len(self.patch_cells), dtype=bool)
        for a in self.agents:
            occupied |= (self.patch_cells == self._pos[a]).all(axis=1)
        candidates = (apples_k == 0) & ~occupied

        # Draw for every patch cell every step so random-number use does not
        # depend on the state (keeps runs easier to reproduce and compare).
        draws = self._rng.random(len(self.patch_cells))
        grow = candidates & (draws < p)
        if grow.any():
            ys, xs = self.patch_cells[grow].T
            self._apples[ys, xs] = True

    def _update_stats(self, rewards, info) -> None:
        s = self.episode_stats
        s["steps"] = self._t
        for a in self.agents:
            s["returns"][a] += rewards[a]
            s["harvests"][a] += int(info[a]["harvested"])
            s["protected_harvests"][a] += int(info[a]["harvested_protected"])
            s["restraint_opportunities"][a] += int(info[a]["restraint_opportunity"])
            s["preempted_opportunities"][a] += int(info[a]["preempted"])
            s["restraint_encounters"][a] += info[a]["new_encounters"]
            s["wall_blocked"][a] += int(info[a]["blocked_by"] == "wall")
            s["agent_blocked"][a] += int(info[a]["blocked_by"] == "agent")
            s["preempted_encounters"][a] += info[a]["preempted_encounters"]
        for kind, agents in info.get("_patch_deaths", []):
            if kind == "lapse":
                for a in agents:
                    s["patch_deaths_lapse"][a] += 1
            elif kind == "simultaneous":
                s["patch_deaths_simultaneous"] += 1
            else:
                s["patch_deaths_other"] += 1
        counts = self.patch_counts()
        for pid in np.flatnonzero(counts == 0):
            s["patch_depletion_steps"].setdefault(int(pid), self._t)
        if s["survival_time"] is None and counts.sum() == 0:
            s["survival_time"] = self._t

    def _code_grid(self) -> np.ndarray:
        g = np.full((self.height, self.width), EMPTY, dtype=np.int8)
        g[self._apples] = APPLE
        g[self.protected_apple_grid()] = PROTECTED_APPLE
        return g

    def _observe(self, agent: str) -> dict[str, Any]:
        r = self.view_radius
        g = self._code_grid()
        for b in self.possible_agents:
            if b != agent and b in self._pos:
                y, x = self._pos[b]
                g[y, x] = OTHER_AGENT  # physical presence is always visible
        padded = np.pad(g, r, mode="constant", constant_values=WALL)
        y, x = self._pos[agent]
        window = padded[y : y + 2 * r + 1, x : x + 2 * r + 1].copy()

        # Social channel: position and last action of the other agent, only if
        # it is within social_radius (Chebyshev distance) AND this agent's social
        # channel is on. Its physical presence (code 4) is shown anywhere in the
        # window, but its position and actions are only observed when close.
        # (Written for two agents; with more, it reports the first visible one.)
        other_visible, other_pos, other_prev, other_act = 0, (0, 0), (0, 0), NO_ACTION
        if self._social_channel[agent]:
            for b in self.possible_agents:
                if b == agent:
                    continue
                by, bx = self._pos[b]
                if max(abs(by - y), abs(bx - x)) <= self.social_radius:
                    other_visible = 1
                    other_pos = (by, bx)
                    other_prev = tuple(self._prev_pos[b])
                    other_act = self._last_action[b]
                    break

        return {
            "window": window,
            "self_pos": np.array([y, x], dtype=np.int64),
            "other_visible": other_visible,
            "other_pos": np.array(other_pos, dtype=np.int64),
            "other_prev_pos": np.array(other_prev, dtype=np.int64),
            "other_last_action": other_act,
        }

    def state(self) -> np.ndarray:
        """Global view for logging/analysis (not given to the learners)."""
        g = self._code_grid()
        for i, a in enumerate(self.possible_agents):
            y, x = self._pos[a]
            g[y, x] = 5 + i
        return g

    def render(self):
        if self.render_mode is None:
            warnings.warn("render() called without a render_mode", stacklevel=2)
            return None
        chars = np.full((self.height, self.width), ".", dtype="<U1")
        chars[self.patch_mask] = ","
        chars[self._apples] = "a"
        chars[self.protected_apple_grid()] = "!"
        for i, a in enumerate(self.possible_agents):
            y, x = self._pos[a]
            chars[y, x] = str(i)
        text = f"t={self._t}\n" + "\n".join("".join(row) for row in chars)
        if self.render_mode == "human":
            print(text + "\n")
            return None
        return text

    def close(self) -> None:
        pass


def parallel_env(**kwargs) -> CommonsHarvestEnv:
    """PettingZoo-style constructor."""
    return CommonsHarvestEnv(**kwargs)
