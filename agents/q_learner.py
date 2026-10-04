"""
Tabular Q-learning with softmax action selection (proposal Section 5.1, Eq. 2).

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

    pi(a | s) = exp(beta * Q(s, a)) / sum_b exp(beta * Q(s, b))

Training vs evaluation:
    act(s)               samples from the softmax policy (exploration on)
    act(s, greedy=True)  picks argmax Q(s, .) with random tie-breaking
                         (exploration off; used for evaluation)

Episodes in this environment are TRUNCATED, never terminated, so the update
always bootstraps from the next state (terminal=False), including on the last
step. Treating a time limit as terminal would bias values downwards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class QLearner:
    def __init__(
        self,
        n_states: int,
        n_actions: int = 5,
        alpha: float = 0.1,
        gamma: float = 0.99,
        beta: float = 5.0,
        q_init: float = 1.0,
        seed: int | None = None,
    ):
        self.n_states, self.n_actions = n_states, n_actions
        self.alpha, self.gamma, self.beta = alpha, gamma, beta
        self.rng = np.random.default_rng(seed)
        # Optimistic initial values (as in Kessler et al.): untried actions look
        # good, which encourages the agent to try them.
        self.Q = np.full((n_states, n_actions), float(q_init))
        self.frozen = False

    # ---------------- acting ----------------
    def policy(self, s: int, mask=None) -> np.ndarray:
        """Softmax probabilities over actions in state s. Actions where
        mask is False get probability 0 (E11 wall masking)."""
        z = self.beta * self.Q[s].astype(float)
        if mask is not None:
            z = np.where(mask, z, -np.inf)
        z = z - z.max()  # subtract the max for numerical stability
        p = np.exp(z)
        return p / p.sum()

    def act(self, s: int, greedy: bool = False, mask=None) -> int:
        if greedy:
            q = self.Q[s] if mask is None else np.where(mask, self.Q[s], -np.inf)
            best = np.flatnonzero(q == q.max())
            return int(self.rng.choice(best))
        return int(self.rng.choice(self.n_actions, p=self.policy(s, mask)))

    # ---------------- learning ----------------
    def update(self, s: int, a: int, r: float, s_next: int, terminal: bool = False,
               next_mask=None) -> float:
        """One Q-learning update. Returns the TD error (useful for debugging).
        With next_mask, the max in the target is over valid next actions only."""
        if self.frozen:
            return 0.0
        q_next = self.Q[s_next] if next_mask is None else self.Q[s_next][next_mask]
        target = r if terminal else r + self.gamma * q_next.max()
        td = target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        return float(td)

    def freeze(self) -> None:
        """Stop learning (the teacher is frozen after training)."""
        self.frozen = True

    # ---------------- saving ----------------
    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, Q=self.Q, alpha=self.alpha, gamma=self.gamma, beta=self.beta)

    @classmethod
    def load(cls, path, seed: int | None = None) -> "QLearner":
        d = np.load(path)
        agent = cls(d["Q"].shape[0], d["Q"].shape[1], float(d["alpha"]),
                    float(d["gamma"]), float(d["beta"]), seed=seed)
        agent.Q = d["Q"].copy()
        return agent
