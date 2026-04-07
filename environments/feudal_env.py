import numpy as np
from .base_env import BaseEnv


# Actions
ATTACK = 0
FORTIFY = 1
RECRUIT = 2
RAID = 3

N_ACTIONS = 4

# State bins: territories (0–4), resources (0–4) → 5*5 = 25 states
TERRITORY_BINS = 5   # 0..4 territories held
RESOURCE_BINS = 5    # 0..4 resource levels


class FeudalEnv(BaseEnv):
    """
    Simplified feudal territory-control game.

    The agent controls a lord competing against a rule-based enemy.
    Both sides start with 2 territories each (out of 8 total).

    State:  (agent_territories, resources) — binned to integers → encoded as flat index
    Actions:
        0 = ATTACK  — attempt to take an enemy territory (cost: 1 resource)
        1 = FORTIFY — reduce chance of losing territory next enemy turn (cost: 0)
        2 = RECRUIT — gain +2 resources (passive income)
        3 = RAID    — steal 1 resource from enemy, gain +1 (risky)

    Rewards:
        Capture territory   : +20
        Lose territory      : -20
        Win (all 8)         : +100, episode ends
        Lose (0 territories): -100, episode ends
        Survive a turn      : +1
        Out of resources    : -50, episode ends

    Enemy policy: always attacks if it has ≥ 2 territories and ≥ 1 resource.
    """

    TOTAL_TERRITORIES = 8

    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.agent_territories = 0
        self.enemy_territories = 0
        self.resources = 0
        self.fortified = False
        self.turn = 0

    # ── BaseEnv interface ─────────────────────────────────

    def reset(self):
        self.agent_territories = 2
        self.enemy_territories = 2
        self.resources = 3
        self.fortified = False
        self.turn = 0
        return self._encode()

    def step(self, action: int):
        self.fortified = False
        reward = 1  # survive bonus

        # ── Agent action ──────────────────────────────────
        if action == ATTACK:
            if self.resources < 1:
                reward -= 2  # can't attack without resources
            else:
                self.resources -= 1
                success = np.random.random() < 0.6
                if success and self.enemy_territories > 0:
                    self.agent_territories += 1
                    self.enemy_territories -= 1
                    reward += 20

        elif action == FORTIFY:
            self.fortified = True

        elif action == RECRUIT:
            self.resources = min(self.resources + 2, 10)

        elif action == RAID:
            success = np.random.random() < 0.5
            if success:
                self.resources = min(self.resources + 1, 10)
                reward += 5
            else:
                reward -= 3

        # ── Check win ────────────────────────────────────
        if self.agent_territories >= self.TOTAL_TERRITORIES:
            return self._encode(), reward + 100, True

        if self.enemy_territories == 0:
            return self._encode(), reward + 100, True

        # ── Enemy turn ───────────────────────────────────
        enemy_attack = (
            self.enemy_territories >= 1
            and np.random.random() < 0.5
        )
        if enemy_attack and self.agent_territories > 0:
            # Fortify reduces enemy success
            success_chance = 0.3 if self.fortified else 0.5
            if np.random.random() < success_chance:
                self.agent_territories -= 1
                self.enemy_territories += 1
                reward -= 20
        else:
            # Enemy recruits
            pass

        # ── Terminal checks ───────────────────────────────
        self.turn += 1

        if self.agent_territories == 0:
            return self._encode(), reward - 100, True

        if self.resources <= 0 and action == ATTACK:
            return self._encode(), reward - 50, True

        if self.turn >= self.max_turns:
            # Reward based on territory advantage at end
            final_bonus = (self.agent_territories - self.enemy_territories) * 5
            return self._encode(), reward + final_bonus, True

        return self._encode(), reward, False

    @property
    def action_space(self) -> int:
        return N_ACTIONS

    @property
    def state_size(self) -> int:
        return TERRITORY_BINS * RESOURCE_BINS

    # ── Helpers ───────────────────────────────────────────

    def _encode(self) -> int:
        t = min(self.agent_territories, TERRITORY_BINS - 1)
        r = min(self.resources // 2, RESOURCE_BINS - 1)  # bin resources 0–10 → 0–4
        return t * RESOURCE_BINS + r

    def render(self):
        bar = "█" * self.agent_territories + "░" * (self.TOTAL_TERRITORIES - self.agent_territories - self.enemy_territories) + "▒" * self.enemy_territories
        print(f"  Turn {self.turn:3d} | Agent: {self.agent_territories} terr | Enemy: {self.enemy_territories} terr | Resources: {self.resources} | [{bar}]")
