"""
Social learning mechanisms (proposal Section 6.3).

VALUE SHAPING (Eq. 6; Kessler et al.):
    whenever the teacher is observed acting, add a bonus kappa to the value
    of the teacher's (state, action) pair in the STUDENT's Q-table:

        Q(s_e, a_e) <- Q(s_e, a_e) + kappa

    The student then acts on its values as usual, and ordinary Q-learning
    continues to update them. The bonus is added only while learning.

How the student obtains (s_e, a_e) using ONLY its own observations:
    * a_e = other_last_action from the social channel.
    * The teacher acted from other_prev_pos, on the world as it was BEFORE the
      step. The student reconstructs the teacher's window around that position
      from its own PREVIOUS window:
          - cells the student could see are copied;
          - cells outside the grid are WALL;
          - cells inside the grid that the student could not see are treated
            as EMPTY  (PROVISIONAL design decision, experiment log E7).
      s_e = phi(reconstructed window, teacher's previous position).
    * The bonus is applied only if the social channel reports the teacher as
      visible (other_visible = 1), i.e. within social_radius and channel on.

The student never receives the teacher's rewards, values or parameters.
"""

from __future__ import annotations

import numpy as np

from agents.features import encode, features
from commons_env.env import EMPTY, NO_ACTION, WALL


def reconstruct_window(student_window: np.ndarray, student_pos, other_pos, grid_shape) -> np.ndarray:
    """Build the window centred on `other_pos` from what the student saw.

    Unseen in-grid cells are EMPTY; out-of-grid cells are WALL."""
    k = student_window.shape[0]
    r = k // 2
    H, W = grid_shape
    out = np.full((k, k), EMPTY, dtype=student_window.dtype)
    for i in range(k):
        for j in range(k):
            gy, gx = other_pos[0] - r + i, other_pos[1] - r + j  # grid cell
            if not (0 <= gy < H and 0 <= gx < W):
                out[i, j] = WALL
                continue
            sy, sx = gy - student_pos[0] + r, gx - student_pos[1] + r  # student window index
            if 0 <= sy < k and 0 <= sx < k:
                out[i, j] = student_window[sy, sx]
    return out


def teacher_state(prev_obs: dict, obs: dict, grid_shape):
    """Return (s_e, a_e) if the teacher was observed acting this step, else None."""
    if not obs["other_visible"] or obs["other_last_action"] == NO_ACTION:
        return None
    t_pos = tuple(int(v) for v in obs["other_prev_pos"])
    w = reconstruct_window(prev_obs["window"], tuple(prev_obs["self_pos"]), t_pos, grid_shape)
    s_e = encode(*features(w, t_pos, grid_shape))
    return s_e, int(obs["other_last_action"])


class ValueShaping:
    """Adds kappa to Q(s_e, a_e) each time the teacher is observed acting."""

    name = "value_shaping"

    def __init__(self, kappa: float = 1.0):
        self.kappa = float(kappa)
        self.applied = 0  # number of bonuses applied (for logging)

    def apply(self, learner, prev_obs: dict, obs: dict, grid_shape) -> None:
        if learner.frozen:
            return
        pair = teacher_state(prev_obs, obs, grid_shape)
        if pair is not None:
            s_e, a_e = pair
            learner.Q[s_e, a_e] += self.kappa
            self.applied += 1


class DecisionBiasing:
    """DECISION BIASING (Eq. 7; Kessler et al., model-free version):

        pi_DB = (1 - omega) * pi_asoc + omega * pi_soc

    With probability omega the student takes the SOCIAL action: the move that
    most reduces its Manhattan distance to the teacher's LAST OBSERVED position
    in this episode (ties broken at random; STAY if already adjacent).
    Otherwise it acts on its own Q-values (pi_asoc). Unlike value shaping, this
    changes what the student DOES, not what it has learned.

    "Last observed" means the most recent step on which the social channel
    reported the teacher (other_visible = 1). Before the teacher has been seen
    in an episode, the student acts asocially. reset() clears the memory; it is
    called at the start of every episode and when the social channel is
    switched off (the A_half condition), so a removed teacher leaves no trace.
    """

    name = "decision_biasing"

    def __init__(self, omega: float = 0.2, seed: int | None = None):
        self.omega = float(omega)
        self.rng = np.random.default_rng(seed)
        self.last_seen = None
        self.social_actions = 0  # number of social actions taken (for logging)

    def reset(self) -> None:
        self.last_seen = None

    def social_action(self, obs: dict):
        if self.last_seen is None:
            return None
        y, x = int(obs["self_pos"][0]), int(obs["self_pos"][1])
        ty, tx = self.last_seen
        d0 = abs(ty - y) + abs(tx - x)
        closer = [act for act, (dy, dx) in ((1, (-1, 0)), (2, (1, 0)), (3, (0, -1)), (4, (0, 1)))
                  if abs(ty - (y + dy)) + abs(tx - (x + dx)) < d0]
        if not closer or d0 <= 1:
            return 0  # STAY: already as close as possible
        return int(self.rng.choice(closer))

    def choose(self, learner, state: int, obs: dict, greedy: bool, mask=None) -> int:
        if obs["other_visible"]:
            self.last_seen = (int(obs["other_pos"][0]), int(obs["other_pos"][1]))
        if self.rng.random() < self.omega:
            a = self.social_action(obs)
            if a is not None:
                self.social_actions += 1
                return a
        return learner.act(state, greedy=greedy, mask=mask)
