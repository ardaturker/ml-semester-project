import numpy as np
from .base_env import BaseEnv


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class GridEnv(BaseEnv):
    """
    Open NxN grid with randomly placed obstacles.
    Agent spawns at (0,0), goal placed randomly (not on agent or obstacle).

    State:  integer index = row * size + col  (goal position is fixed per episode)
    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
    Rewards:
        Goal reached     : +100
        Step penalty     : -1
        Hit obstacle     : -10  (stay in place)
        Out of bounds    : -5   (stay in place)
    """

    def __init__(self, size: int = 8, obstacle_density: float = 0.15):
        self.size = size
        self.obstacle_density = obstacle_density
        self.obstacles: set = set()
        self.goal_pos = (size - 1, size - 1)
        self.agent_pos = (0, 0)

    # ── BaseEnv interface ─────────────────────────────────

    def reset(self):
        self.agent_pos = (0, 0)
        self._place_obstacles()
        self._place_goal()
        return self._encode(self.agent_pos)

    def step(self, action: int):
        dr, dc = DELTAS[action]
        r, c = self.agent_pos
        nr, nc = r + dr, c + dc

        # Out of bounds
        if not (0 <= nr < self.size and 0 <= nc < self.size):
            return self._encode(self.agent_pos), -5, False

        # Obstacle
        if (nr, nc) in self.obstacles:
            return self._encode(self.agent_pos), -10, False

        self.agent_pos = (nr, nc)

        if self.agent_pos == self.goal_pos:
            return self._encode(self.agent_pos), 100, True

        return self._encode(self.agent_pos), -1, False

    @property
    def action_space(self) -> int:
        return 4

    @property
    def state_size(self) -> int:
        return self.size * self.size

    # ── Helpers ───────────────────────────────────────────

    def _place_obstacles(self):
        n = int(self.size * self.size * self.obstacle_density)
        blocked = {(0, 0)}  # keep start clear
        self.obstacles = set()
        while len(self.obstacles) < n:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            if (r, c) not in blocked:
                self.obstacles.add((r, c))
                blocked.add((r, c))

    def _place_goal(self):
        blocked = {(0, 0)} | self.obstacles
        while True:
            r = np.random.randint(0, self.size)
            c = np.random.randint(0, self.size)
            if (r, c) not in blocked:
                self.goal_pos = (r, c)
                return

    def _encode(self, pos) -> int:
        return pos[0] * self.size + pos[1]

    def render(self):
        print()
        for r in range(self.size):
            row = ""
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row += " A "
                elif (r, c) == self.goal_pos:
                    row += " G "
                elif (r, c) in self.obstacles:
                    row += "███"
                else:
                    row += " . "
            print(row)
        print()
