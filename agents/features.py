"""
State representation for tabular learners:  s = phi(observation).

PROVISIONAL DESIGN (to be reviewed before Phase 2). Three features, all read
from the agent's own window (radius view_radius = 4, i.e. 9 x 9) plus its own position:

  1. food     (9 values)  first step towards the nearest ORDINARY apple
                           0-3: apple UP / DOWN / LEFT / RIGHT
                           if no ordinary apple is visible, the direction of
                           the grid centre instead:
                           4-7: centre UP / DOWN / LEFT / RIGHT,  8: at centre
  2. prot_dir (5 values)  first step towards the nearest PROTECTED apple
                           0: none visible, 1-4: UP / DOWN / LEFT / RIGHT
  3. prot_adj (2 values)  1 if that protected apple is one step away
                           (i.e. the agent has a restraint opportunity)

Number of states = 9 * 5 * 2 = 90.

Why these features:
  * food lets the agent find ordinary apples; the centre fallback stops it
    wandering blind when nothing is in view (all patches lie near the centre).
  * prot_dir and prot_adj let it tell protected apples from ordinary ones.
    Without this distinction restraint cannot be learned at all.
  * It deliberately does NOT include absolute position or the full window:
    those would make the table too large to learn in the available time.

Ties between equally near apples are broken by a fixed scan order, so phi is
deterministic: the same observation always gives the same state.
"""

from __future__ import annotations

import numpy as np

from commons_env.env import APPLE, PROTECTED_APPLE, WALL

N_FOOD, N_PROT_DIR, N_PROT_ADJ = 9, 5, 2
N_STATES = N_FOOD * N_PROT_DIR * N_PROT_ADJ  # 90

# Direction codes used inside the features. Feature code d corresponds to the
# environment action d + 1 (UP = 1, DOWN = 2, LEFT = 3, RIGHT = 4).
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3


def _first_step(dy: int, dx: int) -> int:
    """First move towards an offset (dy, dx) != (0, 0): the larger axis first,
    vertical on ties."""
    if abs(dy) >= abs(dx):
        return UP if dy < 0 else DOWN
    return LEFT if dx < 0 else RIGHT


def _nearest(window: np.ndarray, code: int):
    """Offset (dy, dx) of the nearest cell holding `code`, by Manhattan
    distance, or None. Ties are broken by row-major scan order."""
    r = window.shape[0] // 2
    cells = np.argwhere(window == code)
    if len(cells) == 0:
        return None
    offsets = cells - r
    d = np.abs(offsets).sum(axis=1)
    dy, dx = offsets[np.argmin(d)]  # argmin returns the first minimum
    return int(dy), int(dx)


def features(window: np.ndarray, self_pos, grid_shape=(10, 10)) -> tuple[int, int, int]:
    """Return (food, prot_dir, prot_adj) for an agent with this window and position."""
    apple = _nearest(window, APPLE)
    if apple is not None:
        food = _first_step(*apple)
    else:
        cy, cx = (grid_shape[0] - 1) / 2, (grid_shape[1] - 1) / 2  # 4.5, 4.5
        dy, dx = cy - self_pos[0], cx - self_pos[1]
        if abs(dy) <= 1 and abs(dx) <= 1:  # within the central 2 x 2 block
            food = 8
        else:
            food = 4 + _first_step(int(np.sign(dy) * np.ceil(abs(dy))),
                                   int(np.sign(dx) * np.ceil(abs(dx))))

    prot = _nearest(window, PROTECTED_APPLE)
    if prot is None:
        prot_dir, prot_adj = 0, 0
    else:
        prot_dir = 1 + _first_step(*prot)
        prot_adj = int(abs(prot[0]) + abs(prot[1]) == 1)
    return food, prot_dir, prot_adj


def encode(food: int, prot_dir: int, prot_adj: int) -> int:
    """Map the three features to a single state index in [0, N_STATES)."""
    return (food * N_PROT_DIR + prot_dir) * N_PROT_ADJ + prot_adj


def decode(state: int) -> tuple[int, int, int]:
    prot_adj = state % N_PROT_ADJ
    rest = state // N_PROT_ADJ
    return rest // N_PROT_DIR, rest % N_PROT_DIR, prot_adj


def state_of(observation: dict, grid_shape=(10, 10)) -> int:
    """phi: observation -> state index."""
    return encode(*features(observation["window"], observation["self_pos"], grid_shape))


def valid_actions(window: np.ndarray) -> np.ndarray:
    """Wall mask (E11): which of the 5 actions do not walk into the grid edge.

    Order: STAY, UP, DOWN, LEFT, RIGHT. STAY is always valid. The agent can see
    the edge in its own window (code WALL), so this uses only its observation.
    Moves into the OTHER AGENT are not masked: that obstacle is temporary.
    """
    r = window.shape[0] // 2
    return np.array([True,
                     window[r - 1, r] != WALL, window[r + 1, r] != WALL,
                     window[r, r - 1] != WALL, window[r, r + 1] != WALL])
