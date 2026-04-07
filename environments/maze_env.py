import numpy as np
from .base_env import BaseEnv


# 0 = open, 1 = wall
# ── 6×6 maze (legacy / small) ────────────────────────────
MAZE_6 = np.array([
    [0, 0, 1, 0, 0, 0],
    [1, 0, 1, 0, 1, 0],
    [0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0, 0],
], dtype=np.int8)

# ── 12×12 maze (default) ─────────────────────────────────
# Multiple paths, dead ends, and a guaranteed route from (0,0) to (11,11)
MAZE_12 = np.array([
    [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0],
    [0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0],
], dtype=np.int8)

DEFAULT_MAZE = MAZE_12  # used by notebook heatmap imports

# Actions
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def _default_maze_for_size(size: int) -> np.ndarray:
    if size == 6:  return MAZE_6
    if size == 12: return MAZE_12
    # For any other size: generate a random sparse maze (20% wall density)
    rng = np.random.default_rng(42)
    m = rng.choice([0, 1], size=(size, size), p=[0.8, 0.2]).astype(np.int8)
    m[0, 0] = 0          # keep start clear
    m[size-1, size-1] = 0  # keep goal clear
    return m


class MazeEnv(BaseEnv):
    """
    NxN grid maze (default 12×12). Agent starts at (0,0), goal at (N-1,N-1).
    Walls block movement — invalid moves keep agent in place with a small penalty.

    State:  integer index = row * size + col
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Rewards:
        Goal reached : +100
        Step penalty : -1
        Hit wall     : -5
    """

    def __init__(self, maze: np.ndarray = None, size: int = 12):
        self.maze = maze if maze is not None else _default_maze_for_size(size)
        self.size = size
        self.start = (0, 0)
        self.goal = (size - 1, size - 1)
        self.agent_pos = self.start

    # ── BaseEnv interface ─────────────────────────────────

    def reset(self):
        self.agent_pos = self.start
        return self._encode(self.agent_pos)

    def step(self, action: int):
        dr, dc = DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Out of bounds or wall
        if not (0 <= nr < self.size and 0 <= nc < self.size) or self.maze[nr, nc] == 1:
            reward = -5
            return self._encode(self.agent_pos), reward, False

        self.agent_pos = (nr, nc)

        if self.agent_pos == self.goal:
            return self._encode(self.agent_pos), 100, True

        return self._encode(self.agent_pos), -1, False

    @property
    def action_space(self) -> int:
        return 4

    @property
    def state_size(self) -> int:
        return self.size * self.size

    # ── Helpers ───────────────────────────────────────────

    def _encode(self, pos) -> int:
        return pos[0] * self.size + pos[1]

    def render(self):
        print()
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row += " A "
                elif (r, c) == self.goal:
                    row += " G "
                elif self.maze[r, c] == 1:
                    row += "███"
                else:
                    row += " . "
            print(row)
        print()
