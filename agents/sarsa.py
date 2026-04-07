import numpy as np
import pickle
from collections import defaultdict
import config


class SARSAAgent:
    """
    SARSA — On-policy TD control.

    Update rule:
        Q(s,a) ← Q(s,a) + α * [r + γ * Q(s', a') - Q(s,a)]

    Key difference from Q-Learning:
        - Q-Learning (off-policy):  updates toward max_a' Q(s', a')  — the BEST possible next action
        - SARSA (on-policy):        updates toward Q(s', a')          — the action we ACTUALLY take next

    Effect: SARSA is more conservative. It accounts for the fact that the agent
    will sometimes explore (take random actions), so it avoids high-reward but
    risky paths that require always choosing optimally. Q-Learning ignores this
    exploration risk during updates.
    """

    def __init__(self, state_size: int, action_size: int):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = config.ALPHA
        self.gamma = config.GAMMA
        self.epsilon = config.EPSILON_START
        self.epsilon_end = config.EPSILON_END
        self.epsilon_decay = config.EPSILON_DECAY

        self.q_table = defaultdict(lambda: np.zeros(action_size))

    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return int(np.argmax(self.q_table[state]))

    def update(self, state: int, action: int, reward: float,
               next_state: int, next_action: int, done: bool):
        """
        SARSA update — uses the ACTUAL next action taken (not the max).
        The training loop must pass next_action explicitly.
        """
        next_q = 0.0 if done else float(self.q_table[next_state][next_action])
        td_target = reward + self.gamma * next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"q_table": dict(self.q_table), "epsilon": self.epsilon}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "q_table" in data:
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), data["q_table"])
            self.epsilon = data.get("epsilon", self.epsilon_end)
        else:
            self.q_table = defaultdict(lambda: np.zeros(self.action_size), data)
